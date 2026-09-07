"""Load a released SupCon-AE and reduce DeBERTa semantic features.
    总损失
    =
    重建损失 × 重建权重
    +
    对比损失 × 对比权重


        DeBERTa 语义特征
    每条流量 768 个数字
            │
            ▼
    训练时保存的 StandardScaler
    使用训练集平均值、标准差进行标准化
            │
            ▼
    Encoder
    768 → 256 → 128 → 64
            │
            ├───────────────┐
            │               │
            ▼               ▼
        Decoder         Projector
    64→128→256→768      64→64→64
            │               │
            ▼               ▼
    检查信息能否还原     同类靠近、异类远离
            │               │
            └────训练时使用──┘

    预测时只取 Encoder 的 64 维 latent
            │
            ▼
    与 80 个人工网络特征结合
            │
            ▼
    LightGBM 最终分类

"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG = {
    "latent_dim": 64,
    "hidden_dims": [256, 128],
    "proj_dim": 64,
    "dropout": 0.1,
    "batch_size": 256,
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

