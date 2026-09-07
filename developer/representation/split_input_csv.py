"""Create RTD and supervised inputs without inventing a second data split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from developer.data_prepare.group_split import predefined_split_indices
from developer.representation.config import PRETRAIN_INPUT_CSV, SUPERVISED_INPUT_CSV
from developer.config import FLOW_FEATURES_ALL_CSV


def create_inputs(source_csv: str | Path = FLOW_FEATURES_ALL_CSV):
    frame = pd.read_csv(source_csv)
    train_idx, _, _ = predefined_split_indices(frame)
    # RTD is restricted to training groups so reported test performance remains
    # inductive. Supervised JSONL keeps all rows and their fixed split marker;
    # Dataset/DataLoader decides which partition trains or evaluates.
    pretrain = frame.iloc[train_idx].reset_index(drop=True)
    supervised = frame.reset_index(drop=True)
    Path(PRETRAIN_INPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(SUPERVISED_INPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    pretrain.to_csv(PRETRAIN_INPUT_CSV, index=False)
    supervised.to_csv(SUPERVISED_INPUT_CSV, index=False)
    return pretrain, supervised


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, default=FLOW_FEATURES_ALL_CSV)
    args = parser.parse_args()
    train, all_rows = create_inputs(args.source_csv)
    print(f"RTD train rows={len(train):,}; supervised rows={len(all_rows):,}")
