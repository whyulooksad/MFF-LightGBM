# -*- coding: utf-8 -*-
"""
supcon_ae.py  (一键运行版)
==========================
任务三：基于监督对比约束自编码器 (SupCon-AE) 的多分类流量特征降维。

用法：直接运行此脚本即可，无需任何命令行参数。
它会自动处理 final_multiclass_features.csv 并输出降维结果。
"""
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter
from scipy.stats import entropy
from typing import Optional, Sequence, List

# ======================== 可调配置 ========================
# 输入 CSV 路径（相对于本脚本所在目录）
INPUT_CSV = "../final_multiclass_features.csv"
# 输出降维后 CSV 路径
OUTPUT_CSV = "reduced_features.csv"
# 模型保存路径
MODEL_PATH = "supcon_model.pt"

# 标签列名
LABEL_COL = "label"
# 需要保留的标识列（不参与训练，但会写入输出文件）
ID_COLS = ["flow_uid"]

# 降维相关参数
LATENT_DIM = 16         # 压缩到多少维（可调整，例如 32, 64）
EPOCHS = 100            # 训练轮数，真实项目建议 >=100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
HIDDEN_DIMS = [256, 128] # 编码器/解码器隐藏层维度
PROJ_DIM = 64           # 投影头维度
DROPOUT = 0.1
TEMPERATURE = 0.1
LAMBDA_SUPCON = 1.0     # 对比损失权重
BALANCE_CLASSES = True  # 是否使用类别均衡采样
SEED = 42

# ======================== 数据预处理 ========================

# 必须排除的泄露列（五元组、时间戳、日志等）
DROP_COLS = [
    'flow_uid', 'src_ip', 'src_port', 'dst_ip', 'dst_port',
    'protocol', 'timestamp', 'dataset_source', 'subfolder', 'pcap_filename',
    'zeek_conn_log', 'zeek_ssl_log', 'zeek_x509_log'
]

def extract_cn_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    将证书域名 cn_value 转换为多个数值统计特征。
    返回与 df 行数相同的 DataFrame。
    """
    if 'cn_value' not in df.columns:
        return pd.DataFrame(index=df.index)

    cn = df['cn_value'].fillna('').astype(str)
    feats = pd.DataFrame(index=df.index)

    # 长度
    feats['cn_len'] = cn.str.len()

    # 顶级域编码（出现次数>=5的保留，其余归为other）
    tld = cn.str.split('.').str[-1].str.lower()
    tld_counts = tld.value_counts()
    common_tlds = tld_counts[tld_counts >= 5].index.tolist()
    tld = tld.apply(lambda x: x if x in common_tlds else 'other')
    feats['cn_tld_encoded'] = LabelEncoder().fit_transform(tld)

    # 子域名数量
    def count_subdomains(domain):
        parts = domain.split('.')
        if parts and parts[-1] == '':
            parts = parts[:-1]
        return max(0, len(parts) - 1)
    feats['cn_subdomain_count'] = cn.apply(count_subdomains)

    # 是否为 IP 地址
    ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    feats['cn_has_ip'] = cn.apply(lambda x: 1 if ip_pattern.match(x) else 0)

    # 字符熵
    def calc_entropy(s):
        if len(s) == 0: return 0.0
        counts = Counter(s)
        probs = np.array(list(counts.values())) / len(s)
        return entropy(probs, base=2)
    feats['cn_entropy'] = cn.apply(calc_entropy)

    # 是否以 www 开头
    feats['cn_is_www'] = cn.str.startswith('www.').astype(int)

    # 数字比例（补充）
    feats['cn_digit_ratio_new'] = cn.apply(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))

    # 是否包含连字符
    feats['cn_has_hyphen'] = cn.str.contains('-').astype(int)

    return feats


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理 CSV：
      - 删除五元组、时间戳、日志等泄露列（但保护 ID_COLS 和 LABEL_COL）
      - 将 cn_value 转换为数值特征
      - 处理其他非数值列
      - 填充缺失值
    """
    protected = set(c for c in ID_COLS + [LABEL_COL] if c in df.columns)

    # 1. 删除显式泄露列
    cols_to_drop = [c for c in DROP_COLS if c in df.columns and c not in protected]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"已移除泄露列: {cols_to_drop}")

    # 2. 处理 cn_value
    cn_feats = extract_cn_features(df)
    if not cn_feats.empty:
        df = pd.concat([df, cn_feats], axis=1)
        if 'cn_value' in df.columns:
            df.drop(columns=['cn_value'], inplace=True)
        print(f"已从 cn_value 提取 {cn_feats.shape[1]} 个数值特征")

    # 3. 处理其他非数值列（排除受保护列）
    non_num_cols = df.select_dtypes(include=['object']).columns.tolist()
    non_num_cols = [c for c in non_num_cols if c not in protected]
    for col in non_num_cols:
        if df[col].nunique() < 50:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            print(f"非数值列 '{col}' 已标签编码")
        else:
            df.drop(columns=[col], inplace=True)
            print(f"非数值列 '{col}' 唯一值过多，已删除")

    # 4. 填充缺失值
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if df[num_cols].isnull().any().any():
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        print("缺失值已用中位数填充")

    print(f"最终保留特征列（不含标签和标识）: "
          f"{list(set(df.columns) - protected)}")
    return df


# ======================== 网络结构 ========================

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
        z = self.encoder(x)
        x_hat = self.decoder(z)
        p = F.normalize(self.projector(z), dim=1)
        return z, x_hat, p


# ======================== 监督对比损失 ========================

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                anchor_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = features.device
        features = F.normalize(features, dim=1)
        B = features.shape[0]

        sim = features @ features.t() / self.temperature
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        off_diag = (~self_mask).float()

        sim_masked = sim.masked_fill(self_mask, float("-inf"))
        logits = sim - sim_masked.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * off_diag
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.t()).float() * off_diag
        pos_count = pos_mask.sum(1)
        valid = pos_count > 0

        mean_log_prob_pos = torch.zeros(B, device=device)
        mean_log_prob_pos[valid] = (
            (pos_mask * log_prob).sum(1)[valid] / pos_count[valid])
        loss_per = -mean_log_prob_pos

        if not valid.any():
            return torch.zeros((), device=device, requires_grad=True)

        if anchor_weights is not None:
            w = anchor_weights[valid]
            return (loss_per[valid] * w).sum() / (w.sum() + 1e-12)
        return loss_per[valid].mean()


# ======================== 降维器封装 ========================

class SupConAEReducer:
    def __init__(self, latent_dim: int = 32,
                 hidden_dims: Sequence[int] = (256, 128),
                 proj_dim: int = 64, dropout: float = 0.1,
                 temperature: float = 0.1, lambda_supcon: float = 1.0,
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 batch_size: int = 256, epochs: int = 100,
                 balance_classes: bool = True, class_weighted_supcon: bool = False,
                 device: Optional[str] = None, seed: int = 42, verbose: bool = True):
        self.cfg = dict(latent_dim=latent_dim, hidden_dims=list(hidden_dims),
                        proj_dim=proj_dim, dropout=dropout, temperature=temperature,
                        lambda_supcon=lambda_supcon, lr=lr, weight_decay=weight_decay,
                        batch_size=batch_size, epochs=epochs,
                        balance_classes=balance_classes,
                        class_weighted_supcon=class_weighted_supcon, seed=seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self.model: Optional[SupConAE] = None
        self.scaler: Optional[StandardScaler] = None
        self.input_dim: Optional[int] = None
        self._class_weight_tensor: Optional[torch.Tensor] = None

    def _set_seed(self):
        torch.manual_seed(self.cfg["seed"])
        np.random.seed(self.cfg["seed"])

    @staticmethod
    def _as_array(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y) -> "SupConAEReducer":
        self._set_seed()
        X = self._as_array(X)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.input_dim = X.shape[1]

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X).astype(np.float32)

        counts = np.bincount(y_idx, minlength=len(self.classes_)).astype(np.float64)
        inv_freq = 1.0 / np.maximum(counts, 1.0)
        cls_w = inv_freq / inv_freq.sum() * len(self.classes_)
        self._class_weight_tensor = torch.tensor(cls_w, dtype=torch.float32,
                                                 device=self.device)

        dataset = TensorDataset(torch.from_numpy(Xs),
                                torch.from_numpy(y_idx.astype(np.int64)))
        if self.cfg["balance_classes"]:
            sample_w = inv_freq[y_idx]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_w, dtype=torch.double),
                num_samples=len(y_idx), replacement=True)
            loader = DataLoader(dataset, batch_size=self.cfg["batch_size"],
                                sampler=sampler, drop_last=True)
        else:
            loader = DataLoader(dataset, batch_size=self.cfg["batch_size"],
                                shuffle=True, drop_last=True)

        self.model = SupConAE(self.input_dim, self.cfg["latent_dim"],
                              self.cfg["hidden_dims"], self.cfg["proj_dim"],
                              self.cfg["dropout"]).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["lr"],
                                weight_decay=self.cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg["epochs"])
        recon_fn = nn.MSELoss()
        supcon_fn = SupConLoss(self.cfg["temperature"])

        self.model.train()
        for ep in range(self.cfg["epochs"]):
            tot, tot_r, tot_c = 0.0, 0.0, 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                z, x_hat, p = self.model(xb)
                loss_r = recon_fn(x_hat, xb)
                aw = (self._class_weight_tensor[yb]
                      if self.cfg["class_weighted_supcon"] else None)
                loss_c = supcon_fn(p, yb, anchor_weights=aw)
                loss = loss_r + self.cfg["lambda_supcon"] * loss_c
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item(); tot_r += loss_r.item(); tot_c += loss_c.item()
            sched.step()
            if self.verbose and (ep % max(1, self.cfg["epochs"] // 10) == 0
                                 or ep == self.cfg["epochs"] - 1):
                n = len(loader)
                print(f"epoch {ep+1:4d}/{self.cfg['epochs']}  "
                      f"loss={tot/n:.4f}  recon={tot_r/n:.4f}  supcon={tot_c/n:.4f}")
        return self

    @torch.no_grad()
    def transform(self, X) -> np.ndarray:
        assert self.model is not None and self.scaler is not None, "请先 fit 或 load"
        X = self._as_array(X)
        Xs = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        out = []
        for s in range(0, len(Xs), 4096):
            xb = torch.from_numpy(Xs[s:s+4096]).to(self.device)
            out.append(self.model.encode(xb).cpu().numpy())
        return np.concatenate(out, axis=0)

    def fit_transform(self, X, y) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, path: str):
        torch.save({
            "state_dict": self.model.state_dict(),
            "cfg": self.cfg,
            "input_dim": self.input_dim,
            "classes": self.classes_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }, path)
        if self.verbose:
            print(f"模型已保存到 {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None,
             verbose: bool = True) -> "SupConAEReducer":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(device=device, verbose=verbose, **ckpt["cfg"])
        obj.input_dim = ckpt["input_dim"]
        obj.classes_ = np.array(ckpt["classes"])
        obj.scaler = StandardScaler()
        obj.scaler.mean_ = np.array(ckpt["scaler_mean"])
        obj.scaler.scale_ = np.array(ckpt["scaler_scale"])
        obj.scaler.n_features_in_ = obj.input_dim
        obj.model = SupConAE(obj.input_dim, obj.cfg["latent_dim"],
                             obj.cfg["hidden_dims"], obj.cfg["proj_dim"],
                             obj.cfg["dropout"]).to(obj.device)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.model.eval()
        return obj


# ======================== 一键执行主程序 ========================
if __name__ == "__main__":
    print("=" * 60)
    print(" 任务三 SupCon-AE 一键降维流水线 ")
    print("=" * 60)

    # 1. 读取原始特征表
    print(f"\n读取 CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"原始形状: {df.shape}")

    # 2. 数据清洗与特征工程
    df = preprocess_dataframe(df)

    # 3. 分离特征、标签和标识列
    # 确定实际存在的标识列
    id_cols_exist = [c for c in ID_COLS if c in df.columns]
    # 特征列：排除标签和标识列，且只保留数值列
    feature_cols = [c for c in df.columns
                    if c not in id_cols_exist + [LABEL_COL]
                    and pd.api.types.is_numeric_dtype(df[c])]
    print(f"实际参与训练的特征列数: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[LABEL_COL].values
    print(f"样本数: {len(X)}, 类别数: {len(np.unique(y))}")

    # 4. 训练 SupCon-AE 并降维
    reducer = SupConAEReducer(
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        proj_dim=PROJ_DIM,
        dropout=DROPOUT,
        temperature=TEMPERATURE,
        lambda_supcon=LAMBDA_SUPCON,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        balance_classes=BALANCE_CLASSES,
        seed=SEED,
        verbose=True
    )
    print("\n开始训练...")
    Z = reducer.fit_transform(X, y)

    # 5. 保存模型
    reducer.save(MODEL_PATH)

    # 6. 输出降维结果 CSV
    # 构造输出 DataFrame
    out_df = pd.DataFrame(index=df.index)
    # 添加标识列（如果存在）
    if id_cols_exist:
        out_df[id_cols_exist] = df[id_cols_exist]
    # 添加降维特征
    for i in range(Z.shape[1]):
        out_df[f"z_{i}"] = Z[:, i]
    # 添加标签
    out_df[LABEL_COL] = y

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n降维结果（{Z.shape[1]} 维）已写入 {OUTPUT_CSV}")

    print("\n所有步骤完成！")