"""Split final train flow-feature CSV into task-2 training input CSV files.

Default split:
    67% -> data/LLM_train/input/pretrain_flows.csv
    33% -> data/LLM_train/input/supervised_flows.csv

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
        FLOW_FEATURES_TRAIN_CSV,
        INPUT_DIR,
        LABEL2ID,
        PRETRAIN_INPUT_CSV,
        SEED,
        SUPERVISED_INPUT_CSV,
    )
except ImportError:
    from config import (
        FLOW_FEATURES_TRAIN_CSV,
        INPUT_DIR,
        LABEL2ID,
        PRETRAIN_INPUT_CSV,
        SEED,
        SUPERVISED_INPUT_CSV,
    )


def default_source_csv() -> str:
    path = FLOW_FEATURES_TRAIN_CSV
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
    pretrain_ratio: float = 0.67,
    seed: int = SEED,
):
    if source_csv is None:
        source_csv = default_source_csv()
    supervised_ratio = 1.0 - pretrain_ratio
    if pretrain_ratio <= 0 or supervised_ratio <= 0:
        raise ValueError("pretrain_ratio must be between 0 and 1")

    print(f"Read source CSV: {source_csv}")
    df = pd.read_csv(source_csv)
    validate_source(df)
    df["label"] = df["label"].astype(str).str.lower()

    pretrain_parts = []
    supervised_parts = []

    print("Stratified split by label")
    for label, group in df.groupby("label", sort=True):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(group)
        n_pretrain = int(n * pretrain_ratio)

        pretrain_parts.append(group.iloc[:n_pretrain])
        supervised_parts.append(group.iloc[n_pretrain:])

        print(
            f"  {label}: total={n:,}, "
            f"pretrain={n_pretrain:,}, supervised={n - n_pretrain:,}"
        )

    pretrain_df = pd.concat(pretrain_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    supervised_df = pd.concat(supervised_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
    pretrain_df.to_csv(PRETRAIN_INPUT_CSV, index=False)
    supervised_df.to_csv(SUPERVISED_INPUT_CSV, index=False)

    print("Saved:")
    print(f"  {PRETRAIN_INPUT_CSV}: {len(pretrain_df):,}")
    print(f"  {SUPERVISED_INPUT_CSV}: {len(supervised_df):,}")
    return pretrain_df, supervised_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split task-1 multiclass CSV into task-2 input CSVs")
    parser.add_argument("--source-csv", default=None)
    parser.add_argument("--pretrain-ratio", type=float, default=0.67)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    split_by_label(
        source_csv=args.source_csv,
        pretrain_ratio=args.pretrain_ratio,
        seed=args.seed,
    )
