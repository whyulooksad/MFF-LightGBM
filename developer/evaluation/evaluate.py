"""Evaluate a labelled predictions CSV and persist machine-readable metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def evaluate(predictions: Path, output: Path) -> dict:
    frame = pd.read_csv(predictions)
    compatible_pairs = (
        ("true_label_name", "pred_label_name"),
        ("true_label", "pred_label"),
        ("label", "pred_label_name"),
        ("label", "pred_label"),
    )
    columns = next(((true, pred) for true, pred in compatible_pairs if true in frame and pred in frame), None)
    if columns is None:
        raise ValueError("预测文件必须包含 label/true_label 和 pred_label_name/pred_label")
    true_column, pred_column = columns
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
