# -*- coding: utf-8 -*-
"""
supcon_ae_inference.py  (纯推理版，无需训练)
==========================
使用预先训练好的 SupCon-AE 模型对 final_multiclass_features.csv 进行降维，
输出 reduced_features.csv。

用法：直接运行此脚本，无需命令行参数。
      请先修改下方的文件路径。
"""

import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from typing import Optional, Sequence, List

try:
    from .config import FEATURES_FUSED_CSV, REDUCED_FEATURES_CSV, SUPCON_MODEL_PATH
except ImportError:
    from config import FEATURES_FUSED_CSV, REDUCED_FEATURES_CSV, SUPCON_MODEL_PATH

# ======================== 可调配置 (直接修改此处) ========================
INPUT_CSV = str(FEATURES_FUSED_CSV)
OUTPUT_CSV = str(REDUCED_FEATURES_CSV)
MODEL_PATH = str(SUPCON_MODEL_PATH)

LABEL_COL = "label"
ID_COLS = ["flow_uid", "label_name"]

# 以下参数无需修改，它们已固化在模型中
# ======================== 数据预处理（与训练时保持一致）=====================

DROP_COLS = []


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # features_fused.csv contains flow_uid, feat_*, numeric flow features,
    # label, and label_name. Only numeric columns outside the protected set
    # participate in dimensionality reduction.
    protected = set(c for c in ID_COLS + [LABEL_COL] if c in df.columns)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns and c not in protected]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"已移除泄露列: {cols_to_drop}")

    non_num_cols = df.select_dtypes(include=['object']).columns.tolist()
    non_num_cols = [c for c in non_num_cols if c not in protected]
    if non_num_cols:
        df.drop(columns=non_num_cols, inplace=True)
        print(f"已删除非数值特征列: {non_num_cols}")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if df[num_cols].isnull().any().any():
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        print("缺失值已用中位数填充")

    feat_list = list(set(df.columns) - protected)
    print(f"最终保留特征列（不含标签和标识）: {feat_list}")
    return df


# ======================== 模型定义（需与训练时一致） ========================

def _mlp(dims: Sequence[int], dropout: float, last_activation: bool) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if (not is_last) or last_activation:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SupConAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Sequence[int] = (256, 128),
                 proj_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        hidden_dims = list(hidden_dims)
        self.encoder = _mlp([input_dim, *hidden_dims, latent_dim],
                            dropout, last_activation=False)
        # 解码器和投影头在推理时不使用，但为确保模型结构完整，依然保留
        self.decoder = _mlp([latent_dim, *hidden_dims[::-1], input_dim],
                            dropout, last_activation=False)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        # 推理时仅使用 encode，但 forward 保留以兼容加载
        z = self.encoder(x)
        x_hat = self.decoder(z)
        p = F.normalize(self.projector(z), dim=1)
        return z, x_hat, p


class SupConAEReducer:
    """纯推理加载器，用于加载已训练模型并执行降维。"""

    def __init__(self, model_path: str, device: Optional[str] = None, verbose: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self.model: Optional[SupConAE] = None
        self.scaler: Optional[StandardScaler] = None
        self.input_dim: Optional[int] = None
        self.latent_dim: Optional[int] = None
        self._load(model_path)

    def _load(self, path: str):
        """从 .pt 文件加载模型、scaler 和配置。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        self.input_dim = ckpt["input_dim"]
        self.latent_dim = cfg["latent_dim"]
        self.classes_ = np.array(ckpt["classes"])
        # 重建 scaler
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(ckpt["scaler_mean"])
        self.scaler.scale_ = np.array(ckpt["scaler_scale"])
        self.scaler.n_features_in_ = self.input_dim
        # 重建模型
        self.model = SupConAE(
            self.input_dim,
            cfg["latent_dim"],
            cfg["hidden_dims"],
            cfg["proj_dim"],
            cfg["dropout"]
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        if self.verbose:
            print(f"模型已从 {path} 加载，输入维度: {self.input_dim}, 隐空间维度: {self.latent_dim}")

    @staticmethod
    def _as_array(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.asarray(X, dtype=np.float32)

    @torch.no_grad()
    def transform(self, X) -> np.ndarray:
        """对输入数据执行降维，返回降维后的 numpy 数组。"""
        assert self.model is not None and self.scaler is not None, "模型未加载"
        X = self._as_array(X)
        Xs = self.scaler.transform(X).astype(np.float32)
        out = []
        for s in range(0, len(Xs), 4096):  # 分批防止显存溢出
            xb = torch.from_numpy(Xs[s:s + 4096]).to(self.device)
            out.append(self.model.encode(xb).cpu().numpy())
        return np.concatenate(out, axis=0)


# ======================== 主程序 ========================
if __name__ == "__main__":
    print("=" * 60)
    print(" 任务三 SupCon-AE 纯推理降维 ")
    print("=" * 60)

    # 1. 读取原始特征表
    print(f"\n[1/4] 读取 CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  原始形状: {df.shape}")

    # 2. 数据预处理（与训练时一致）
    print(f"\n[2/4] 数据预处理...")
    df = preprocess_dataframe(df)

    # 3. 确定特征列（排除标识列和标签列）
    id_cols_exist = [c for c in ID_COLS if c in df.columns]
    # 标签列可能存在，也可能不存在
    label_col_exist = LABEL_COL in df.columns
    feature_cols = [c for c in df.columns
                    if c not in id_cols_exist + ([LABEL_COL] if label_col_exist else [])
                    and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise RuntimeError("预处理后没有数值特征列可供降维，请检查数据。")
    print(f"  参与降维的特征列数: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[LABEL_COL].values if label_col_exist else None
    if y is not None:
        print(f"  样本数: {len(X)}, 标签类别数: {len(np.unique(y))}")

    # 4. 加载模型并降维
    print(f"\n[3/4] 加载模型并执行降维...")
    reducer = SupConAEReducer(MODEL_PATH, verbose=True)
    if X.shape[1] != reducer.input_dim:
        raise ValueError(
            f"SupCon-AE 输入维度不匹配: 当前特征维度={X.shape[1]}, "
            f"模型期望维度={reducer.input_dim}. 请确认 {INPUT_CSV} "
            "是否与训练该 supcon_model.pt 时使用的特征结构一致。"
        )
    Z = reducer.transform(X)
    print(f"  降维完成，输出维度: {Z.shape[1]}")

    # 5. 保存结果
    print(f"\n[4/4] 保存降维结果到 {OUTPUT_CSV}")
    out_df = pd.DataFrame(index=df.index)
    if id_cols_exist:
        out_df[id_cols_exist] = df[id_cols_exist]
    for i in range(Z.shape[1]):
        out_df[f"z_{i}"] = Z[:, i]
    if y is not None:
        out_df[LABEL_COL] = y
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("\n所有步骤完成！")
