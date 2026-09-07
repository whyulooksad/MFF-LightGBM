"""Evaluate a labelled predictions CSV and persist machine-readable metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def evaluate(predictions: Path, output: Path) -> dict:
    frame = pd.read_csv(predictions)
    true_column = "label" if "label" in frame else "true_label"
    pred_column = "pred_label_name" if "pred_label_name" in frame else "pred_label"
    if true_column not in frame or pred_column not in frame:
        raise ValueError("预测文件必须包含 label/true_label 和 pred_label_name/pred_label")
    metrics = {
        "accuracy": accuracy_score(frame[true_column], frame[pred_column]),
        "macro_f1": f1_score(frame[true_column], frame[pred_column], average="macro"),
        "weighted_f1": f1_score(frame[true_column], frame[pred_column], average="weighted"),
        "classification_report": classification_report(frame[true_column], frame[pred_column], output_dict=True),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.predictions, args.output), ensure_ascii=False, indent=2))
