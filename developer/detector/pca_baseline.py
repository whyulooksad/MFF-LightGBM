# -*- coding: utf-8 -*-
"""
PCA 降维:只对 feat_* 做 PCA,其余特征原样保留。
封装为内存方法供 detector2 调用,不单独落盘。

为什么只降 feat_*:
    当前模型输入由 768 维 DeBERTa 语义特征和 80 维严格解析的数值
    特征组成。PCA 只处理语义分支，避免改变 80 维人工特征的含义；
    768 维压到 64 维后，再与 80 维人工特征融合交给分类器。

纯 CPU,不依赖预训练模型,不占显存。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from user_app.inference.contract import NEW_FORMAT_NUM_FEATURES
from developer.config import FEATURES_FUSED_CSV
from developer.data_prepare.group_split import predefined_split_indices


def _pool_summary(n_feat: int, n_manual: int) -> str:
    total = n_feat + n_manual
    if not total:
        return "  (空)"
    return (
        f"  候选池(feature_fraction=0.8): 共 {total} 维, "
        f"feat 占 {n_feat / total:.1%}, manual 占 {n_manual / total:.1%}"
    )


def reduce_feat_in_memory(
    df: pd.DataFrame,
    n_components: int = 64,
    seed: int = 42,
    verbose: bool = True,
    fit_indices=None,
) -> pd.DataFrame:
    """对 df 的 feat_* 列做 PCA,其余列原样保留,返回新 DataFrame。

    返回列:原非 feat 列 + feat_0..feat_{n-1}。
    不落盘。供 detector2 在喂 LightGBM 前调用。

    manual 数值特征、flow_uid、label、label_name 等全部原样保留,
    后续 preprocess_detector_dataframe / detector_feature_columns 仍能
    正常把 feat_* 当 LLM 特征、其余当 manual。
    """
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols:
        if verbose:
            print("[PCA] 未找到 feat_* 列,跳过降维")
        return df

    other_cols = [c for c in df.columns if not c.startswith("feat_")]
    manual_cols = [c for c in NEW_FORMAT_NUM_FEATURES if c in df.columns]

    if fit_indices is None:
        raise ValueError("PCA必须显式提供train索引，禁止在全部数据上拟合")
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    n_components = int(min(n_components, len(feat_cols), len(fit_indices)))
    if verbose:
        print(f"[PCA] {len(feat_cols)} 维 feat_* -> {n_components} 维")
        print("[降维前]")
        print(_pool_summary(len(feat_cols), len(manual_cols)))

    train_medians = df.iloc[fit_indices][feat_cols].median(numeric_only=True).fillna(0.0)
    X = df[feat_cols].fillna(train_medians).to_numpy(np.float32)
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X[fit_indices])
    Z = pca.transform(X)

    out = df[other_cols].copy()
    for i in range(n_components):
        out[f"feat_{i}"] = Z[:, i]

    if verbose:
        evr = pca.explained_variance_ratio_
        print(
            f"[PCA] 累计解释方差: {evr.sum():.4f}  "
            f"(前5维: {evr[:5].round(4).tolist()})"
        )
        print("[降维后]")
        print(_pool_summary(n_components, len(manual_cols)))
        if evr.sum() < 0.7:
            print(
                f"[提示] 累计方差 {evr.sum():.2%} 偏低, {n_components} 维可能丢信息,"
                f" 可加大 n_components(如 128)再试。"
            )
    return out


def main():
    """单独运行只打印降维统计,不落盘。"""
    parser = argparse.ArgumentParser(description="PCA 降维 feat_*(只打印统计,不落盘)")
    parser.add_argument("--input", default=None, help="输入 CSV,默认 features_fused.csv")
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_csv = args.input or FEATURES_FUSED_CSV
    print(f"读取: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"形状: {df.shape}")
    train_idx, _, _ = predefined_split_indices(df)
    reduce_feat_in_memory(df, n_components=args.n_components, seed=args.seed, fit_indices=train_idx)


if __name__ == "__main__":
    main()
