"""Supervised-contrastive autoencoder for DeBERTa feature reduction.

The reducer is fitted only on a detector training split.  It standardizes and
compresses the ``feat_*`` semantic columns while handcrafted flow features are
kept unchanged for LightGBM.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from pipeline.config import FEATURES_FUSED_CSV, REDUCED_FEATURES_CSV, SUPCON_MODEL_PATH


DEFAULT_CONFIG = {
    "latent_dim": 64,
    "hidden_dims": [256, 128],
    "proj_dim": 64,
    "dropout": 0.1,
    "temperature": 0.1,
    "reconstruction_weight": 1.0,
    "contrastive_weight": 1.0,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 256,
    "epochs": 30,
    "early_stopping_patience": 5,
    "min_delta": 1e-4,
    "balance_classes": True,
    "seed": 42,
}


def semantic_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feat columns in stable numeric order."""

    columns = [column for column in df.columns if column.startswith("feat_")]

    def key(column: str):
        suffix = column.removeprefix("feat_")
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    return sorted(columns, key=key)


def _mlp(dims: Sequence[int], dropout: float, last_activation: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        is_last = index == len(dims) - 2
        if not is_last or last_activation:
            layers.extend([nn.LayerNorm(dims[index + 1]), nn.ReLU(inplace=True)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SupConAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: Sequence[int] = (512, 256),
        proj_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        self.encoder = _mlp([input_dim, *hidden_dims, latent_dim], dropout, False)
        self.decoder = _mlp([latent_dim, *hidden_dims[::-1], input_dim], dropout, False)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, proj_dim),
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def forward(self, values: torch.Tensor):
        latent = self.encoder(values)
        reconstructed = self.decoder(latent)
        projection = F.normalize(self.projector(latent), dim=1)
        return latent, reconstructed, projection


class SupConLoss(nn.Module):
    """Numerically stable supervised contrastive loss."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = projections.shape[0]
        if batch_size < 2:
            return projections.sum() * 0.0

        logits = projections @ projections.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        identity = torch.eye(batch_size, dtype=torch.bool, device=projections.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~identity
        denominator_mask = ~identity

        exp_logits = torch.exp(logits) * denominator_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            return projections.sum() * 0.0
        mean_log_prob = (log_prob * positive_mask).sum(dim=1) / positive_count.clamp_min(1)
        return -mean_log_prob[valid].mean()


class SupConAEReducer:
    def __init__(self, config: dict | None = None, device: str | None = None, verbose: bool = True):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self.model: SupConAE | None = None
        self.scaler: StandardScaler | None = None
        self.input_dim: int | None = None
        self.feature_columns: list[str] = []
        self.classes_: np.ndarray = np.array([])
        self.best_epoch: int | None = None
        self.best_val_loss: float | None = None

    def _build_model(self) -> SupConAE:
        assert self.input_dim is not None
        return SupConAE(
            input_dim=self.input_dim,
            latent_dim=int(self.config["latent_dim"]),
            hidden_dims=self.config["hidden_dims"],
            proj_dim=int(self.config["proj_dim"]),
            dropout=float(self.config["dropout"]),
        ).to(self.device)

    @staticmethod
    def _array(values) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            values = values.to_numpy()
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"SupCon-AE expects a 2-D feature matrix, got shape={values.shape}")
        return values

    def _loader(self, values, labels, training: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(values.astype(np.float32, copy=False)),
            torch.from_numpy(labels.astype(np.int64, copy=False)),
        )
        batch_size = min(int(self.config["batch_size"]), len(dataset))
        if batch_size < 2:
            raise ValueError("SupCon-AE requires at least two samples")

        sampler = None
        shuffle = training
        if training and self.config["balance_classes"]:
            counts = np.bincount(labels, minlength=len(self.classes_)).astype(np.float64)
            weights = 1.0 / np.maximum(counts, 1.0)
            sample_weights = torch.as_tensor(weights[labels], dtype=torch.double)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            shuffle = False
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=False)

    def _loss(self, batch, contrastive_loss):
        values, labels = (item.to(self.device) for item in batch)
        _, reconstructed, projection = self.model(values)
        reconstruction = F.mse_loss(reconstructed, values)
        contrastive = contrastive_loss(projection, labels)
        total = (
            float(self.config["reconstruction_weight"]) * reconstruction
            + float(self.config["contrastive_weight"]) * contrastive
        )
        return total, reconstruction.detach(), contrastive.detach()

    @torch.no_grad()
    def _evaluate_loss(self, loader, contrastive_loss) -> float:
        assert self.model is not None
        self.model.eval()
        losses = []
        for batch in loader:
            total, _, _ = self._loss(batch, contrastive_loss)
            losses.append(total.item())
        return float(np.mean(losses)) if losses else float("inf")

    def fit(
        self,
        train_values,
        train_labels,
        val_values,
        val_labels,
        feature_columns: Sequence[str],
        checkpoint_path: str | Path | None = None,
    ):
        train_values = self._array(train_values)
        val_values = self._array(val_values)
        if train_values.shape[1] != val_values.shape[1]:
            raise ValueError("SupCon-AE train/validation dimensions differ")

        self.feature_columns = list(feature_columns)
        self.input_dim = train_values.shape[1]
        if len(self.feature_columns) != self.input_dim:
            raise ValueError("feature_columns length does not match SupCon-AE input dimension")

        self.classes_, train_encoded = np.unique(np.asarray(train_labels), return_inverse=True)
        class_to_index = {label: index for index, label in enumerate(self.classes_.tolist())}
        try:
            val_encoded = np.asarray([class_to_index[label] for label in np.asarray(val_labels)], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"Validation contains a class absent from training: {exc.args[0]}") from exc

        torch.manual_seed(int(self.config["seed"]))
        np.random.seed(int(self.config["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config["seed"]))

        self.scaler = StandardScaler().fit(train_values)
        train_scaled = self.scaler.transform(train_values).astype(np.float32)
        val_scaled = self.scaler.transform(val_values).astype(np.float32)
        train_loader = self._loader(train_scaled, train_encoded, training=True)
        val_loader = self._loader(val_scaled, val_encoded, training=False)

        self.model = self._build_model()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        contrastive_loss = SupConLoss(float(self.config["temperature"]))
        best_state = None
        best_loss = float("inf")
        stale_epochs = 0

        for epoch in range(1, int(self.config["epochs"]) + 1):
            self.model.train()
            train_losses = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                total, _, _ = self._loss(batch, contrastive_loss)
                if not torch.isfinite(total):
                    raise FloatingPointError("SupCon-AE loss became NaN/Inf")
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(total.item())

            val_loss = self._evaluate_loss(val_loader, contrastive_loss)
            train_loss = float(np.mean(train_losses))
            if self.verbose:
                print(f"  SupCon-AE epoch {epoch:02d}: train={train_loss:.6f}, val={val_loss:.6f}")

            if val_loss < best_loss - float(self.config["min_delta"]):
                best_loss = val_loss
                self.best_epoch = epoch
                self.best_val_loss = val_loss
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in self.model.state_dict().items()})
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(self.config["early_stopping_patience"]):
                    break

        if best_state is None:
            raise RuntimeError("SupCon-AE training did not produce a valid checkpoint")
        self.model.load_state_dict(best_state)
        self.model.to(self.device).eval()
        if checkpoint_path is not None:
            self.save(checkpoint_path)
        return self

    @torch.no_grad()
    def transform(self, values, feature_columns: Sequence[str] | None = None) -> np.ndarray:
        if self.model is None or self.scaler is None or self.input_dim is None:
            raise RuntimeError("SupCon-AE reducer has not been fitted or loaded")
        if feature_columns is not None and list(feature_columns) != self.feature_columns:
            raise ValueError("SupCon-AE feature columns/order differ from the training checkpoint")
        values = self._array(values)
        if values.shape[1] != self.input_dim:
            raise ValueError(f"SupCon-AE input dimension mismatch: got {values.shape[1]}, expected {self.input_dim}")
        scaled = self.scaler.transform(values).astype(np.float32)
        self.model.eval()
        chunks = []
        batch_size = max(2, int(self.config["batch_size"]))
        for start in range(0, len(scaled), batch_size):
            batch = torch.from_numpy(scaled[start : start + batch_size]).to(self.device)
            chunks.append(self.model.encode(batch).cpu().numpy())
        if not chunks:
            return np.empty((0, int(self.config["latent_dim"])), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def save(self, path: str | Path):
        if self.model is None or self.scaler is None or self.input_dim is None:
            raise RuntimeError("Cannot save an unfitted SupCon-AE reducer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 2,
                "config": self.config,
                "input_dim": self.input_dim,
                "feature_columns": self.feature_columns,
                "classes": self.classes_.tolist(),
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_,
                "scaler_var": self.scaler.var_,
                "scaler_n_samples_seen": self.scaler.n_samples_seen_,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
                "state_dict": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None, verbose: bool = True):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SupCon-AE checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 2 or not checkpoint.get("feature_columns"):
            raise ValueError("Legacy SupCon-AE checkpoint is incompatible; retrain it with the current pipeline")

        reducer = cls(checkpoint["config"], device=device, verbose=verbose)
        reducer.input_dim = int(checkpoint["input_dim"])
        reducer.feature_columns = list(checkpoint["feature_columns"])
        reducer.classes_ = np.asarray(checkpoint["classes"])
        reducer.best_epoch = checkpoint.get("best_epoch")
        reducer.best_val_loss = checkpoint.get("best_val_loss")
        reducer.scaler = StandardScaler()
        reducer.scaler.mean_ = np.asarray(checkpoint["scaler_mean"])
        reducer.scaler.scale_ = np.asarray(checkpoint["scaler_scale"])
        reducer.scaler.var_ = np.asarray(checkpoint["scaler_var"])
        reducer.scaler.n_features_in_ = reducer.input_dim
        reducer.scaler.n_samples_seen_ = checkpoint["scaler_n_samples_seen"]
        reducer.model = reducer._build_model()
        reducer.model.load_state_dict(checkpoint["state_dict"])
        reducer.model.eval()
        return reducer


def replace_semantic_features(df: pd.DataFrame, reducer: SupConAEReducer) -> pd.DataFrame:
    columns = semantic_feature_columns(df)
    latent = reducer.transform(df[columns], feature_columns=columns)
    output = df.drop(columns=columns).copy()
    for index in range(latent.shape[1]):
        output[f"feat_{index}"] = latent[:, index]
    return output


def main():
    """Transform the fused CSV using a reducer already trained by detector.py."""
    dataframe = pd.read_csv(FEATURES_FUSED_CSV)
    reducer = SupConAEReducer.load(SUPCON_MODEL_PATH)
    output = replace_semantic_features(dataframe, reducer)
    Path(REDUCED_FEATURES_CSV).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(REDUCED_FEATURES_CSV, index=False)
    print(json.dumps({
        "input": str(FEATURES_FUSED_CSV),
        "output": str(REDUCED_FEATURES_CSV),
        "shape": list(output.shape),
        "best_epoch": reducer.best_epoch,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
