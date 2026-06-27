# -*- coding: utf-8 -*-
"""
demo_supcon_vs_original_optimized.py
=====================================
优化版 SupCon-AE 降维对比实验。
针对原版性能下降问题进行了如下改进：
  1. 提高 latent_dim 到 128
  2. 降低对比损失权重 lambda_supcon 到 2.0
  3. 简化投影头，proj_dim = latent_dim
  4. 增强编码器容量 ([512, 256])
  5. 增加训练轮次至 300

运行： python demo_supcon_vs_original_optimized.py
"""

import os, re, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import entropy
from typing import Sequence, List, Optional
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings('ignore')

# ======================== 实验配置 ========================
EXPERIMENT_CONFIG = {
    "INPUT_CSV": "D:/jinxian/Pycharm/比赛/data/output/features_fused.csv",
    "LABEL_COL": "label",
    "ID_COLS": ["flow_uid", "label_name"],  # 不参与降维的列
    "TARGET_DIM": 128,          # 优化1：提升隐空间维度
    "SUPCON_EPOCHS": 300,       # 优化2：增加训练轮次
    "SUPCON_BATCH_SIZE": 256,
    "SUPCON_LR": 5e-4,
    "SUPCON_HIDDEN_DIMS": [512, 256],  # 优化3：增强编码器容量
    "SUPCON_PROJ_DIM": 128,            # 优化4：投影头维度与隐空间相同
    "SUPCON_DROPOUT": 0.2,
    "SUPCON_TEMP": 0.2,
    "SUPCON_LAMBDA": 2.0,              # 优化5：降低对比损失权重
    "SUPCON_CLASS_WEIGHTED": True,
    "SUPCON_DYNAMIC_LAMBDA": False,    # 是否使用动态 lambda（逐步增加对比权重）
    "SUPCON_LAMBDA_START": 1.0,        # 若开启动态，起始 lambda
    "SUPCON_LAMBDA_END": 3.0,          # 若开启动态，结束 lambda
    "LGB_PARAMS": {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbose': -1,
        'n_estimators': 200,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    },
    "RANDOM_STATE": 42,
    "TEST_SIZE": 0.3,
    "DROP_COLS": [
        'flow_uid', 'src_ip', 'src_port', 'dst_ip', 'dst_port',
        'protocol', 'timestamp', 'dataset_source', 'subfolder', 'pcap_filename',
        'zeek_conn_log', 'zeek_ssl_log', 'zeek_x509_log'
    ]
}

# ======================== 预处理（保持不变） ========================
def extract_cn_features(df):
    if 'cn_value' not in df.columns:
        return pd.DataFrame(index=df.index)
    cn = df['cn_value'].fillna('').astype(str)
    feats = pd.DataFrame(index=df.index)
    feats['cn_len'] = cn.str.len()
    tld = cn.str.split('.').str[-1].str.lower()
    tld_counts = tld.value_counts()
    common_tlds = tld_counts[tld_counts >= 5].index.tolist()
    tld = tld.apply(lambda x: x if x in common_tlds else 'other')
    feats['cn_tld_encoded'] = LabelEncoder().fit_transform(tld)
    def count_subdomains(domain):
        parts = domain.split('.')
        if parts and parts[-1] == '': parts = parts[:-1]
        return max(0, len(parts) - 1)
    feats['cn_subdomain_count'] = cn.apply(count_subdomains)
    ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    feats['cn_has_ip'] = cn.apply(lambda x: 1 if ip_pattern.match(x) else 0)
    def calc_entropy(s):
        if len(s) == 0: return 0.0
        counts = Counter(s)
        probs = np.array(list(counts.values())) / len(s)
        return entropy(probs, base=2)
    feats['cn_entropy'] = cn.apply(calc_entropy)
    feats['cn_is_www'] = cn.str.startswith('www.').astype(int)
    feats['cn_digit_ratio_new'] = cn.apply(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
    feats['cn_has_hyphen'] = cn.str.contains('-').astype(int)
    return feats

def preprocess_dataframe(df, config):
    protected = set(config["ID_COLS"] + [config["LABEL_COL"]])
    existing_drop = [c for c in config["DROP_COLS"] if c in df.columns and c not in protected]
    if existing_drop:
        print(f"\n删除了 {len(existing_drop)} 个泄露列")
        df.drop(columns=existing_drop, inplace=True)

    cn_feats = extract_cn_features(df)
    if not cn_feats.empty:
        df = pd.concat([df, cn_feats], axis=1)
        if 'cn_value' in df.columns:
            df.drop(columns=['cn_value'], inplace=True)

    non_num = [c for c in df.select_dtypes(include=['object']).columns if c not in protected]
    for col in non_num:
        if df[col].nunique() < 50:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        else:
            df.drop(columns=[col], inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns
    if df[num_cols].isnull().any().any():
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df

# ======================== SupCon-AE 模型 ========================
def _mlp(dims, dropout, last_activation):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if not is_last or last_activation:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class SupConAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dims=(256, 128), proj_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = _mlp([input_dim, *hidden_dims, latent_dim], dropout, False)
        self.decoder = _mlp([latent_dim, *hidden_dims[::-1], input_dim], dropout, False)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, proj_dim)
        )
    def encode(self, x):
        return self.encoder(x)
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        p = F.normalize(self.projector(z), dim=1)
        return z, x_hat, p

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels, anchor_weights=None):
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
        labels = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.t()).float() * off_diag
        pos_count = pos_mask.sum(1)
        valid = pos_count > 0
        mean_log_prob_pos = torch.zeros(B, device=device)
        mean_log_prob_pos[valid] = (pos_mask * log_prob).sum(1)[valid] / pos_count[valid]
        loss_per = -mean_log_prob_pos
        if not valid.any():
            return torch.zeros((), device=device, requires_grad=True)
        if anchor_weights is not None:
            w = anchor_weights[valid]
            return (loss_per[valid] * w).sum() / (w.sum() + 1e-12)
        return loss_per[valid].mean()

class SupConAEReducer:
    def __init__(self, config, device=None, verbose=True):
        self.cfg = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = verbose
        self.model = None
        self.scaler = None
        self.input_dim = None
        self.classes_ = None
        self._class_weight_tensor = None
    def _set_seed(self):
        torch.manual_seed(self.cfg["RANDOM_STATE"])
        np.random.seed(self.cfg["RANDOM_STATE"])
    @staticmethod
    def _as_array(X):
        if isinstance(X, pd.DataFrame): X = X.values
        return np.asarray(X, dtype=np.float32)
    def fit(self, X, y):
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
        self._class_weight_tensor = torch.tensor(cls_w, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(y_idx.astype(np.int64)))
        if self.cfg["SUPCON_CLASS_WEIGHTED"]:
            sample_w = inv_freq[y_idx]
            sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                            num_samples=len(y_idx), replacement=True)
            loader = DataLoader(dataset, batch_size=self.cfg["SUPCON_BATCH_SIZE"], sampler=sampler, drop_last=True)
        else:
            loader = DataLoader(dataset, batch_size=self.cfg["SUPCON_BATCH_SIZE"], shuffle=True, drop_last=True)

        self.model = SupConAE(self.input_dim,
                              self.cfg["TARGET_DIM"],
                              self.cfg["SUPCON_HIDDEN_DIMS"],
                              self.cfg["SUPCON_PROJ_DIM"],
                              self.cfg["SUPCON_DROPOUT"]).to(self.device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["SUPCON_LR"], weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg["SUPCON_EPOCHS"])
        recon_fn = nn.MSELoss()
        supcon_fn = SupConLoss(self.cfg["SUPCON_TEMP"])

        # 动态 lambda 选项
        dynamic_lambda = self.cfg.get("SUPCON_DYNAMIC_LAMBDA", False)
        lam_start = self.cfg.get("SUPCON_LAMBDA_START", 1.0)
        lam_end = self.cfg.get("SUPCON_LAMBDA_END", 3.0)
        for ep in range(self.cfg["SUPCON_EPOCHS"]):
            self.model.train()
            tot_r, tot_c = 0.0, 0.0
            # 计算当前 lambda
            if dynamic_lambda:
                progress = ep / max(1, self.cfg["SUPCON_EPOCHS"] - 1)
                current_lambda = lam_start + (lam_end - lam_start) * progress
            else:
                current_lambda = self.cfg["SUPCON_LAMBDA"]
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                z, x_hat, p = self.model(xb)
                loss_r = recon_fn(x_hat, xb)
                aw = self._class_weight_tensor[yb] if self.cfg["SUPCON_CLASS_WEIGHTED"] else None
                loss_c = supcon_fn(p, yb, anchor_weights=aw)
                loss = loss_r + current_lambda * loss_c
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot_r += loss_r.item()
                tot_c += loss_c.item()
            sched.step()
            if self.verbose and (ep % max(1, self.cfg["SUPCON_EPOCHS"] // 10) == 0 or ep == self.cfg["SUPCON_EPOCHS"] - 1):
                print(f"  epoch {ep+1:4d}/{self.cfg['SUPCON_EPOCHS']}  "
                      f"recon={tot_r/len(loader):.4f}  supcon={tot_c/len(loader):.4f}  λ={current_lambda:.3f}")
        return self

    @torch.no_grad()
    def transform(self, X):
        assert self.model and self.scaler, "请先训练"
        X = self._as_array(X)
        Xs = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        out = []
        for s in range(0, len(Xs), 4096):
            xb = torch.from_numpy(Xs[s:s+4096]).to(self.device)
            out.append(self.model.encode(xb).cpu().numpy())
        return np.concatenate(out, axis=0)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# ======================== 分类器评估（修复版） ========================
def evaluate_lgbm(name, X_train, y_train, X_test, y_test, classes, config):
    params = config["LGB_PARAMS"].copy()
    params['num_class'] = len(classes)
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              eval_metric='multi_logloss', callbacks=[lgb.early_stopping(20)])
    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    str_classes = [str(c) for c in classes]  # 确保 target_names 为字符串
    report = classification_report(y_test, y_pred, target_names=str_classes, digits=4)
    print(f"\n[{name}] Macro F1 = {macro_f1:.4f}")
    print(report)
    return {"cm": cm, "macro_f1": macro_f1, "str_classes": str_classes}


# ======================== 主程序 ========================
def main():
    cfg = EXPERIMENT_CONFIG
    print("=" * 60)
    print(" 优化版 SupCon-AE 降维对比实验 ")
    print(f" 目标维度: {cfg['TARGET_DIM']}, lambda: {cfg['SUPCON_LAMBDA']}, epochs: {cfg['SUPCON_EPOCHS']}")
    print("=" * 60)

    # 1. 加载与预处理
    df = pd.read_csv(cfg["INPUT_CSV"])
    df = preprocess_dataframe(df, cfg)

    # 2. 排除标识列与标签列
    id_cols_exist = [c for c in cfg["ID_COLS"] if c in df.columns]
    non_feature_cols = set(id_cols_exist + [cfg["LABEL_COL"]])
    feature_cols = [c for c in df.columns
                    if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])]
    print(f"参与降维的特征列数: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[cfg["LABEL_COL"]].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    # 3. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=cfg["TEST_SIZE"], stratify=y_enc, random_state=cfg["RANDOM_STATE"])
    print(f"\n训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
    print(f"原始特征维度: {X.shape[1]}, 类别数: {len(classes)}")

    results = {}

    # 4. 基线：原始特征 + LightGBM
    print("\n--- 1. 原始特征 (LightGBM) ---")
    results["Original"] = evaluate_lgbm("Original", X_train, y_train,
                                        X_test, y_test, classes, cfg)

    # 5. 优化版 SupCon-AE 降维
    print(f"\n--- 2. 优化版 SupCon-AE 降维 ---")
    supcon = SupConAEReducer(cfg, verbose=True)
    X_train_supcon = supcon.fit_transform(X_train, y_train)
    X_test_supcon = supcon.transform(X_test)
    results["SupCon-AE_opt"] = evaluate_lgbm("SupCon-AE_opt", X_train_supcon, y_train,
                                             X_test_supcon, y_test, classes, cfg)

    # 6. 可视化
    print("\n生成对比图表...")
    method_names = ["Original", "SupCon-AE_opt"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, method_names):
        cm = results[name]["cm"]
        str_labels = results[name]["str_classes"]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=str_labels, yticklabels=str_labels, ax=ax, cbar=False)
        ax.set_title(f"{name} (Macro F1={results[name]['macro_f1']:.3f})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("supcon_optimized_vs_original_cm.png", dpi=150)
    plt.show()

    # F1 柱状图
    f1s = [results[n]["macro_f1"] for n in method_names]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(method_names, f1s, color=["#1f77b4", "#2ca02c"])
    for bar, score in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01, f"{score:.4f}", ha='center')
    plt.ylim(0, 1.05)
    plt.ylabel("Macro F1")
    plt.title("Original vs Optimized SupCon-AE")
    plt.tight_layout()
    plt.savefig("supcon_optimized_vs_original_f1.png", dpi=150)
    plt.show()

    # 结论
    orig_f1 = results["Original"]["macro_f1"]
    opt_f1 = results["SupCon-AE_opt"]["macro_f1"]
    print("\n" + "=" * 60)
    if opt_f1 > orig_f1:
        print(f"✅ 优化版 SupCon-AE 胜出！提升 {opt_f1 - orig_f1:.4f}")
    elif opt_f1 < orig_f1:
        print(f"⚠️ 优化版仍低于原始特征（差距 {orig_f1 - opt_f1:.4f}）。可尝试进一步调整或使用其他降维方法。")
    else:
        print("两者持平。")


if __name__ == "__main__":
    main()