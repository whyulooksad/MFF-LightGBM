# app/data_provider.py
import ast
import csv
import json
import os
import re
from typing import Any, Optional

import pandas as pd
import numpy as np

from tls_log_parser import parse_tls_logs

METADATA_COLS_BASIC = ["flow_uid", "src_ip", "src_port", "dst_ip", "dst_port",
                        "protocol", "duration", "total_packets", "total_bytes",
                        "label"]
METADATA_COLS_FULL = METADATA_COLS_BASIC + [
    "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "average_packet_size", "conn_state",
    "syn_count", "fin_count", "rst_count", "psh_count", "ack_count",
    "packet_time_offsets", "packet_directions", "packet_lengths",
]


class DataProvider:
    def __init__(self, flow_features_dir, pipeline_output_dir, runtime_dir):
        self.flow_features_dir = flow_features_dir
        self.pipeline_output_dir = pipeline_output_dir
        self.runtime_dir = runtime_dir
        self._metadata_df = None

    def _metadata_path(self):
        return os.path.join(self.flow_features_dir, "flow_metadata_temporal_test.csv")

    def _features_path(self):
        return os.path.join(self.flow_features_dir, "final_multiclass_features_test.csv")

    def _metadata(self):
        if self._metadata_df is None:
            # List/dashboard requests do not need the very large packet arrays.
            self._metadata_df = pd.read_csv(
                self._metadata_path(),
                usecols=lambda column: column in METADATA_COLS_BASIC,
                low_memory=False,
            )
        return self._metadata_df

    def read_metadata(self, limit=50, offset=0):
        df = self._metadata()[METADATA_COLS_BASIC]
        return df.iloc[offset:offset + limit].to_dict(orient="records")

    def read_metadata_detail(self, flow_uid):
        r = _find_csv_row(self._metadata_path(), flow_uid, METADATA_COLS_FULL)
        if r is None:
            return None
        metadata = {k: _native_value(r.get(k)) for k in METADATA_COLS_FULL if k in r}
        # TLS log: 来自 final_multiclass_features_test.csv
        tls = {}
        fr = _find_csv_row(
            self._features_path(),
            flow_uid,
            ["flow_uid", "zeek_conn_log", "zeek_ssl_log", "zeek_x509_log"],
        )
        if fr is not None:
            tls = parse_tls_logs({
                "zeek_conn_log": fr.get("zeek_conn_log"),
                "zeek_ssl_log": fr.get("zeek_ssl_log"),
                "zeek_x509_log": fr.get("zeek_x509_log"),
            })
        temporal = {
            "packet_lengths": _parse_array(r.get("packet_lengths")),
            "packet_directions": _parse_array(r.get("packet_directions")),
            "packet_time_offsets": _parse_array(r.get("packet_time_offsets")),
        }
        return {"flow_uid": flow_uid, "metadata": metadata, "tls": tls, "temporal": temporal}

    def read_predictions(self, source="runtime", limit=50, offset=0, label=None):
        path = self._predictions_path(source)
        if not os.path.exists(path):
            return {"rows": [], "total": 0}
        df = pd.read_csv(path)
        if label:
            df = df[df["pred_label_name"] == label]
        total = len(df)
        rows = df.iloc[offset:offset + limit].to_dict(orient="records")
        return {"rows": rows, "total": total}

    def _predictions_path(self, source):
        # The canonical pipeline writes its latest predictions here. Both names
        # remain accepted for compatibility with the existing frontend.
        return os.path.join(self.pipeline_output_dir, "predictions.csv")

    def read_evaluation(self, source="static"):
        summary_path = self._eval_path(source, "classification_report.txt")
        if not os.path.exists(summary_path):
            return {"algos": [], "test_set": {"samples": 0}}
        with open(summary_path, encoding="utf-8") as report_file:
            text = report_file.read()
        lightgbm = _parse_sklearn_classification_report(text, "lightgbm")
        algos = [lightgbm] if lightgbm else []

        comparison_path = os.path.join(
            self.pipeline_output_dir,
            "detector_report",
            "model_comparison_with_baselines.csv",
        )
        if os.path.exists(comparison_path):
            comparison = pd.read_csv(comparison_path)
            for _, row in comparison.iterrows():
                name = str(row.get("Model", "")).strip().lower()
                if name.startswith("lightgbm"):
                    continue
                algos.append({
                    "name": name,
                    "macro_f1": _number_or_none(row.get("F1")),
                    "accuracy": _number_or_none(row.get("Accuracy")),
                    "per_class": [],
                })
        # 评估元数据：测试集样本数从 confusion_matrix.csv 读
        cm_path = self._eval_path(source, "confusion_matrix.csv")
        samples = 0
        if os.path.exists(cm_path):
            cm_df = pd.read_csv(cm_path, index_col=0)
            samples = int(cm_df.values.sum())
        return {"algos": algos, "test_set": {"samples": samples}}

    def _eval_path(self, source, filename):
        return os.path.join(self.pipeline_output_dir, filename)

    def get_dashboard_stats(self):
        df = self._metadata()
        eval_data = self.read_evaluation(source="static")
        lgb = next((a for a in eval_data["algos"] if a["name"] == "lightgbm"), None)
        return {
            "test_samples": int(len(df)),
            "num_classes": 8,
            "macro_f1": lgb["macro_f1"] if lgb else 0.0,
            "class_distribution": df["label"].value_counts().to_dict() if "label" in df else {},
        }

    def read_image_bytes(self, name, source="static"):
        aliases = {
            "confusion_matrix.png": "confusion_matrix.png",
            "conf_matrices_compare.png": "weighted_metrics_comparison.png",
            "f1_comparison.png": "weighted_metrics_comparison.png",
            "roc_curves_compare.png": "roc_curve.png",
        }
        filename = aliases.get(name, name)
        path = os.path.join(self.pipeline_output_dir, "detector_report", filename)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()


def _parse_array(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, str):
        try:
            return [_native_value(item) for item in ast.literal_eval(v)]
        except (ValueError, SyntaxError):
            try:
                return json.loads(v)
            except Exception:
                return []
    return [_native_value(item) for item in v] if hasattr(v, "__iter__") else []


def _find_csv_row(path, flow_uid, columns, chunksize=512):
    """Find one flow without loading a large CSV and its packet arrays in full."""
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        matches = chunk[chunk["flow_uid"] == flow_uid]
        if not matches.empty:
            return matches.iloc[0]
    return None


def _native_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _number_or_none(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_sklearn_classification_report(text, name):
    per_class = []
    macro_f1 = None
    accuracy = None
    label_ids = {
        "benign": 0, "adware": 1, "dns2tcp": 2, "dnscat2": 3,
        "iodine": 4, "ransomware": 5, "scareware": 6, "smsmalware": 7,
    }
    for line in text.splitlines():
        match = re.match(r"\s*(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", line)
        if match and match.group(1).lower() in label_ids:
            label = match.group(1).lower()
            per_class.append({
                "label_id": label_ids[label],
                "precision": float(match.group(2)),
                "recall": float(match.group(3)),
                "f1": float(match.group(4)),
                "support": int(match.group(5)),
            })
            continue
        match = re.match(r"\s*accuracy\s+([\d.]+)\s+(\d+)\s*$", line)
        if match:
            accuracy = float(match.group(1))
            continue
        match = re.match(r"\s*macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+\d+\s*$", line)
        if match:
            macro_f1 = float(match.group(1))
    if not per_class and accuracy is None:
        return None
    return {"name": name, "macro_f1": macro_f1, "accuracy": accuracy, "per_class": per_class}


# classification_summary.txt 解析器
_ALGO_BLOCK_RE = re.compile(r"===\s*(\w+)\s*===.*?(?===\s*\w+\s*===|$)", re.DOTALL)


def _parse_classification_summary(text):
    algos = []
    blocks = re.split(r"(===\s*\w+\s*===)", text)
    cur_name = None
    for i, b in enumerate(blocks):
        m = re.match(r"===\s*(\w+)\s*===", b)
        if m:
            cur_name = m.group(1).lower()
            continue
        if cur_name and b.strip():
            algo = _parse_single_algo_block(cur_name, b)
            if algo:
                algos.append(algo)
            cur_name = None
    return algos


def _parse_single_algo_block(name, body):
    macro_m = re.search(r"Macro F1:\s*([\d.]+)", body)
    acc_m = re.search(r"accuracy\s+([\d.]+)", body)
    per_class = []
    for line in body.splitlines():
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
        if m:
            per_class.append({
                "label_id": int(m.group(1)),
                "precision": float(m.group(2)),
                "recall": float(m.group(3)),
                "f1": float(m.group(4)),
                "support": int(m.group(5)),
            })
    return {
        "name": name,
        "macro_f1": float(macro_m.group(1)) if macro_m else None,
        "accuracy": float(acc_m.group(1)) if acc_m else None,
        "per_class": per_class,
    }
