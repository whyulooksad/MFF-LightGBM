"""Developer-only SupCon-AE training built on the production reducer format."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from developer.config import EXPERIMENT_SUPCON_MODEL_PATH, FEATURES_FUSED_CSV
from user_app.inference.supcon_reducer import SupConAEReducer, semantic_feature_columns
from developer.data_prepare.group_split import predefined_split_indices


TRAINING_DEFAULTS = {
    "temperature": 0.1,
    "reconstruction_weight": 1.0,
    "contrastive_weight": 1.0,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 30,
    "early_stopping_patience": 5,
    "min_delta": 1e-4,
    "balance_classes": True,
    "seed": 42,
}


class SupConLoss(torch.nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections, labels):
        features = F.normalize(projections, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature
        labels = labels.view(-1, 1)
        self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        positive_mask = labels.eq(labels.T) & ~self_mask
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * ~self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            return projections.sum() * 0.0
        return -((log_prob * positive_mask).sum(dim=1) / positive_count.clamp_min(1))[valid].mean()


class TrainableSupConAEReducer(SupConAEReducer):
    """Adds fitting behavior that is intentionally absent from user inference."""

    def __init__(self, config=None, device=None, verbose=True):
        super().__init__({**TRAINING_DEFAULTS, **(config or {})}, device=device, verbose=verbose)

    def _loader(self, values, labels, training: bool) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(values), torch.from_numpy(labels.astype(np.int64)))
        batch_size = min(int(self.config["batch_size"]), len(dataset))
        if batch_size < 2:
            raise ValueError("SupCon-AE 至少需要两个样本")
        sampler = None
        shuffle = training
        if training and self.config["balance_classes"]:
            counts = np.bincount(labels, minlength=len(self.classes_)).astype(np.float64)
            weights = 1.0 / np.maximum(counts, 1.0)
            sampler = WeightedRandomSampler(torch.as_tensor(weights[labels], dtype=torch.double), len(labels), replacement=True)
            shuffle = False
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)

    def _loss(self, batch, contrastive_loss):
        values, labels = (item.to(self.device) for item in batch)
        _, reconstructed, projection = self.model(values)
        reconstruction = F.mse_loss(reconstructed, values)
        contrastive = contrastive_loss(projection, labels)
        total = self.config["reconstruction_weight"] * reconstruction + self.config["contrastive_weight"] * contrastive
        return total

    @torch.no_grad()
    def _validation_loss(self, loader, loss_fn) -> float:
        self.model.eval()
        values = [self._loss(batch, loss_fn).item() for batch in loader]
        return float(np.mean(values)) if values else float("inf")

    def fit(self, train_values, train_labels, val_values, val_labels, feature_columns: Sequence[str], checkpoint_path=None):
        train_values = self._array(train_values)
        val_values = self._array(val_values)
        self.feature_columns = list(feature_columns)
        self.input_dim = train_values.shape[1]
        if val_values.shape[1] != self.input_dim or len(self.feature_columns) != self.input_dim:
            raise ValueError("SupCon-AE 输入维度或特征契约不一致")
        self.classes_, train_labels = np.unique(np.asarray(train_labels), return_inverse=True)
        mapping = {label: index for index, label in enumerate(self.classes_.tolist())}
        val_labels = np.asarray([mapping[label] for label in np.asarray(val_labels)], dtype=np.int64)

        seed = int(self.config["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.scaler = StandardScaler().fit(train_values)
        train_scaled = self.scaler.transform(train_values).astype(np.float32)
        val_scaled = self.scaler.transform(val_values).astype(np.float32)
        train_loader = self._loader(train_scaled, train_labels, True)
        val_loader = self._loader(val_scaled, val_labels, False)
        self.model = self._build_model()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        loss_fn = SupConLoss(float(self.config["temperature"]))
        best_state, best_loss, stale = None, float("inf"), 0

        for epoch in range(1, int(self.config["epochs"]) + 1):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(batch, loss_fn)
                if not torch.isfinite(loss):
                    raise FloatingPointError("SupCon-AE loss 出现 NaN/Inf")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
            val_loss = self._validation_loss(val_loader, loss_fn)
            if self.verbose:
                print(f"SupCon-AE epoch {epoch:02d}: val={val_loss:.6f}")
            if val_loss < best_loss - float(self.config["min_delta"]):
                best_loss, self.best_epoch, self.best_val_loss, stale = val_loss, epoch, val_loss, 0
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in self.model.state_dict().items()})
            else:
                stale += 1
                if stale >= int(self.config["early_stopping_patience"]):
                    break
        if best_state is None:
            raise RuntimeError("SupCon-AE 未产生有效 checkpoint")
        self.model.load_state_dict(best_state)
        self.model.eval()
        if checkpoint_path is not None:
            self.save(checkpoint_path)
        return self


def train(input_csv: Path, output: Path, seed: int = 42) -> None:
    frame = pd.read_csv(input_csv)
    columns = semantic_feature_columns(frame)
    if "label" not in frame:
        raise ValueError("融合特征必须包含 label 列")
    train_idx, validation_idx, _ = predefined_split_indices(frame)
    train_df, validation_df = frame.iloc[train_idx], frame.iloc[validation_idx]
    TrainableSupConAEReducer(config={"seed": seed}).fit(
        train_df[columns], train_df["label"], validation_df[columns], validation_df["label"], columns, output
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=FEATURES_FUSED_CSV)
    parser.add_argument("--output", type=Path, default=EXPERIMENT_SUPCON_MODEL_PATH)
    args = parser.parse_args()
    train(args.input, args.output)
