"""Split final_multiclass_features.csv into task-2 input CSV files.

Default split:
    50% -> data/input/pretrain_flows.csv
    25% -> data/input/supervised_flows.csv
    25% -> data/input/feature_flows.csv

The split is stratified by the multiclass label so each subset keeps the same
class balance. The source CSV is already flow-level, so each row is kept intact.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

try:
    from .config import (
        FEATURE_INPUT_CSV,
        INPUT_DIR,
        LABEL2ID,
        PRETRAIN_INPUT_CSV,
        ROOT,
        SEED,
        SUPERVISED_INPUT_CSV,
    )
except ImportError:
    from config import (
        FEATURE_INPUT_CSV,
        INPUT_DIR,
        LABEL2ID,
        PRETRAIN_INPUT_CSV,
        ROOT,
        SEED,
        SUPERVISED_INPUT_CSV,
    )


def default_source_csv() -> str:
    path = os.path.join(ROOT, "final_multiclass_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing source CSV: {path}")
    return path


def validate_source(df: pd.DataFrame):
    if "label" not in df.columns:
        raise ValueError("source CSV must contain a label column")
    unknown = sorted(set(str(v).lower() for v in df["label"].dropna().unique()) - set(LABEL2ID))
    if unknown:
        raise ValueError(f"unknown labels in source CSV: {unknown}")


def split_by_label(
    source_csv: str | None = None,
    pretrain_ratio: float = 0.5,
    supervised_ratio: float = 0.25,
    seed: int = SEED,
):
    if source_csv is None:
        source_csv = default_source_csv()
    feature_ratio = 1.0 - pretrain_ratio - supervised_ratio
    if pretrain_ratio <= 0 or supervised_ratio <= 0 or feature_ratio <= 0:
        raise ValueError("split ratios must be positive and sum to less than 1")

    print(f"Read source CSV: {source_csv}")
    df = pd.read_csv(source_csv)
    validate_source(df)
    df["label"] = df["label"].astype(str).str.lower()

    pretrain_parts = []
    supervised_parts = []
    feature_parts = []

    print("Stratified split by label")
    for label, group in df.groupby("label", sort=True):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_pretrain = int(n * pretrain_ratio)
        n_supervised = int(n * supervised_ratio)

        pretrain_parts.append(group.iloc[:n_pretrain])
        supervised_parts.append(group.iloc[n_pretrain:n_pretrain + n_supervised])
        feature_parts.append(group.iloc[n_pretrain + n_supervised:])

        print(
            f"  {label}: total={n:,}, "
            f"pretrain={n_pretrain:,}, supervised={n_supervised:,}, "
            f"feature={n - n_pretrain - n_supervised:,}"
        )

    pretrain_df = pd.concat(pretrain_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    supervised_df = pd.concat(supervised_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    feature_df = pd.concat(feature_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
    pretrain_df.to_csv(PRETRAIN_INPUT_CSV, index=False)
    supervised_df.to_csv(SUPERVISED_INPUT_CSV, index=False)
    feature_df.to_csv(FEATURE_INPUT_CSV, index=False)

    print("Saved:")
    print(f"  {PRETRAIN_INPUT_CSV}: {len(pretrain_df):,}")
    print(f"  {SUPERVISED_INPUT_CSV}: {len(supervised_df):,}")
    print(f"  {FEATURE_INPUT_CSV}: {len(feature_df):,}")
    return pretrain_df, supervised_df, feature_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split task-1 multiclass CSV into task-2 input CSVs")
    parser.add_argument("--source-csv", default=None)
    parser.add_argument("--pretrain-ratio", type=float, default=0.5)
    parser.add_argument("--supervised-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    split_by_label(
        source_csv=args.source_csv,
        pretrain_ratio=args.pretrain_ratio,
        supervised_ratio=args.supervised_ratio,
        seed=args.seed,
    )
