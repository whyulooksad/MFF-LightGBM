"""Read only task-scoped user results from data/runtime/tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from user_app.backend.tls_log_parser import parse_tls_logs


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{key: _native(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def _array(value: Any) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


class DataProvider:
    def __init__(self, runtime_dir: str | Path):
        self.tasks_root = (Path(runtime_dir) / "tasks").resolve()

    def task_dir(self, task_id: str) -> Path | None:
        # Task ids are opaque URL input. Resolve and contain them before I/O.
        candidate = (self.tasks_root / str(task_id)).resolve()
        try:
            candidate.relative_to(self.tasks_root)
        except ValueError:
            return None
        return candidate if candidate.is_dir() else None

    def read_task_summary(self, task_id: str) -> dict | None:
        task_dir = self.task_dir(task_id)
        path = task_dir / "summary.json" if task_dir else None
        if path is None or not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read_task_predictions(self, task_id: str, limit=50, offset=0, label=None) -> dict:
        task_dir = self.task_dir(task_id)
        path = task_dir / "predictions.csv" if task_dir else None
        if path is None or not path.is_file():
            return {"rows": [], "total": 0}
        frame = pd.read_csv(path)
        if label:
            frame = frame[frame["pred_label_name"] == label]
        return {"rows": _records(frame.iloc[offset : offset + limit]), "total": int(len(frame))}

    def read_task_flows(self, task_id: str, limit=50, offset=0) -> dict:
        task_dir = self.task_dir(task_id)
        if task_dir is None:
            return {"rows": [], "total": 0}
        temporal_path, features_path = task_dir / "flow_temporal.csv", task_dir / "flow_features.csv"
        if not temporal_path.is_file() or not features_path.is_file():
            return {"rows": [], "total": 0}
        temporal = pd.read_csv(temporal_path)
        features = pd.read_csv(features_path, usecols=lambda col: col in {"flow_uid", "tls_log"})
        tls_summary = []
        for value in features.get("tls_log", pd.Series(dtype=str)):
            tls_summary.append(parse_tls_logs({"tls_log": value}).get("sni"))
        features = features[["flow_uid"]].copy()
        features["sni"] = tls_summary
        frame = temporal.merge(features, on="flow_uid", how="left")
        columns = [
            "flow_uid", "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
            "duration", "total_packets", "total_bytes", "sni",
        ]
        frame = frame[[col for col in columns if col in frame.columns]]
        return {"rows": _records(frame.iloc[offset : offset + limit]), "total": int(len(frame))}

    def read_task_flow_detail(self, task_id: str, flow_uid: str) -> dict | None:
        task_dir = self.task_dir(task_id)
        if task_dir is None:
            return None
        temporal_path, features_path = task_dir / "flow_temporal.csv", task_dir / "flow_features.csv"
        if not temporal_path.is_file() or not features_path.is_file():
            return None
        temporal = pd.read_csv(temporal_path)
        match = temporal[temporal["flow_uid"].astype(str) == str(flow_uid)]
        if match.empty:
            return None
        time_row = match.iloc[0].to_dict()
        features = pd.read_csv(
            features_path,
            usecols=lambda col: col in {"flow_uid", "connection_log", "tls_log", "x509_log"},
        )
        feature_match = features[features["flow_uid"].astype(str) == str(flow_uid)]
        logs = feature_match.iloc[0].to_dict() if not feature_match.empty else {}
        temporal_arrays = {
            "packet_lengths": _array(time_row.pop("packet_lengths", None)),
            "packet_directions": _array(time_row.pop("packet_directions", None)),
            "packet_time_offsets": _array(time_row.pop("packet_time_offsets", None)),
        }
        metadata = {key: _native(value) for key, value in time_row.items()}
        return {
            "flow_uid": flow_uid,
            "metadata": metadata,
            "tls": parse_tls_logs(logs),
            "temporal": temporal_arrays,
        }
