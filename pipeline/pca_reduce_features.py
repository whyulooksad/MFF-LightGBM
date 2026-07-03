# -*- coding: utf-8 -*-
"""
PCA 降维:只对 feat_* 做 PCA,其余特征原样保留。
封装为内存方法供 detector2 调用,不单独落盘。

为什么只降 feat_*:
    features_fused.csv 共 846 维(768 feat + 78 manual)。LightGBM
    feature_fraction=0.8 按数量均匀采样,候选池里 feat 占 90.8%,
    且 feat 对 CIC 四类判别力弱,大量分裂落在噪声维度上,稀释了
    单独就能 0.9493 的 78 维 manual 强特征。
    PCA 把 768 压到 64,既去噪又把候选池 feat 占比降到约 45%,
    让 manual 恢复合理分裂机会。

纯 CPU,不依赖预训练模型,不占显存。
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from LLM_train.config import NEW_FORMAT_NUM_FEATURES
from pipeline.config import FEATURES_FUSED_CSV


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

    n_components = int(min(n_components, len(feat_cols), len(df)))
    if verbose:
        print(f"[PCA] {len(feat_cols)} 维 feat_* -> {n_components} 维")
        print("[降维前]")
        print(_pool_summary(len(feat_cols), len(manual_cols)))

    X = df[feat_cols].fillna(0).to_numpy(np.float32)
    pca = PCA(n_components=n_components, random_state=seed)
    Z = pca.fit_transform(X)

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
    reduce_feat_in_memory(df, n_components=args.n_components, seed=args.seed)


if __name__ == "__main__":
    main()
