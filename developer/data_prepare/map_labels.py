"""Map official dataset labels onto the project's stable eight-class contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from user_app.inference.contract import LABEL2ID


ALIASES = {
    "benign": "benign", "normal": "benign",
    "adware": "adware", "dns2tcp": "dns2tcp", "dnscat2": "dnscat2",
    "iodine": "iodine", "ransomware": "ransomware",
    "scareware": "scareware", "smsmalware": "smsmalware",
    "sms malware": "smsmalware",
}


def normalize_label(value: object) -> str:
    key = str(value).strip().lower().replace("_", " ")
    if key not in ALIASES:
        raise ValueError(f"未配置的官方标签: {value!r}")
    return ALIASES[key]


def map_csv(source: Path, output: Path, label_column: str = "label") -> None:
    frame = pd.read_csv(source)
    frame[label_column] = frame[label_column].map(normalize_label)
    frame["label_id"] = frame[label_column].map(LABEL2ID)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()
    map_csv(args.source, args.output, args.label_column)
