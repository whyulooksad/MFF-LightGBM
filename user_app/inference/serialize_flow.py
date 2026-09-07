"""
Step 1: preprocess task-1 multiclass flow CSV.

Input format: canonical flow rows with parsed connection/TLS/X.509
JSON and the shared numeric feature contract.

Identifier/source columns such as five-tuple fields, timestamp, dataset_source,
subfolder, and pcap_filename are not used in model text or fused numeric
features.
    输入 flow_features.csv
        ↓
    pd.read_csv()
    读取为Pandas表格
        ↓
    validate_columns()
    检查基础列、80个特征列、特征版本
        ↓
    逐行处理
        ├── parse_label()
        │   标签名称转数字
        │
        ├── build_flow_text()
        │   connection_log → c事件
        │   tls_log        → s事件
        │   x509_log       → x事件
        │
        ├── extract_num_features()
        │   取出固定80个数值特征
        │
        └── 保留group_id、split等来源信息
        ↓
    构造一个流JSON对象
        ↓
    每行写一个JSON
        ↓
    输出 flows.jsonl
        ↓
    semantic_encoder.py读取
        ├── text进入DeBERTa
        └── num_features留给后续融合
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter

import pandas as pd
from tqdm import tqdm

from user_app.inference.contract import (
    FEATURE_SCHEMA_VERSION,
    LABEL2ID,
    NEW_FORMAT_NUM_FEATURES,
)

REQUIRED_COLUMNS = {
    "flow_uid",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "protocol",
    "timestamp",
    "label",
    "feature_schema_version",
    "connection_log",
    "tls_log",
    "x509_log",
}

CONN_FIELDS = {
    "proto": "proto",
    "service": "svc",
    "duration": "dur",
    "orig_bytes": "orig_bytes",
    "resp_bytes": "resp_bytes",
    "conn_state": "state",
    "missed_bytes": "missed",
    "history": "hist",
    "orig_pkts": "orig_pkts",
    "resp_pkts": "resp_pkts",
}

SSL_FIELDS = {
    "version": "ver",
    "cipher": "cipher",
    "curve": "curve",
    "server_name": "sni",
    "resumed": "resumed",
    "last_alert": "alert",
    "next_protocol": "next",
    "established": "est",
    "subject": "subj",
    "issuer": "issuer",
    "validation_status": "validation",
}

X509_FIELDS = {
    "certificate.version": "ver",
    "certificate.subject": "subj",
    "certificate.issuer": "issuer",
    "certificate.not_valid_before": "not_bef",
    "certificate.not_valid_after": "not_aft",
    "certificate.key_alg": "key_alg",
    "certificate.sig_alg": "sig",
    "certificate.key_type": "key_type",
    "certificate.key_length": "key_len",
    "certificate.curve": "curve",
    "san.dns": "san_dns",
    "basic_constraints.ca": "ca",
}


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value in {"", "-", "nan", "NaN", "None", "null"}:
            return None
        # Keep double quotes intact: connection_log/tls_log/x509_log are JSON
        # strings when they come back from CSV, and replacing JSON's quotes
        # before json.loads() makes every valid object fail to parse.  Values
        # are escaped safely later by json.dumps(), so only normalize line
        # breaks here.
        return value.replace("\r", " ").replace("\n", " ")
    return value


def parse_literal_cell(value):
    value = clean_value(value)
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return None


def compact_dict(data: dict, field_map: dict, event_type: str) -> dict:
    out = {"t": event_type}
    if not isinstance(data, dict):
        return out
    for src, dst in field_map.items():
        val = clean_value(data.get(src))
        if val is None:
            continue
        out[dst] = val
    return out


def compact_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def build_flow_text(row: pd.Series) -> tuple[str, int]:
    parts = []

    conn = parse_literal_cell(row.get("connection_log"))
    if isinstance(conn, dict):
        parts.append(compact_json(compact_dict(conn, CONN_FIELDS, "c")))

    ssl = parse_literal_cell(row.get("tls_log"))
    if isinstance(ssl, dict):
        parts.append(compact_json(compact_dict(ssl, SSL_FIELDS, "s")))

    x509 = parse_literal_cell(row.get("x509_log"))
    if isinstance(x509, dict):
        x509 = [x509]
    if isinstance(x509, list):
        for cert in x509:
            if isinstance(cert, dict):
                parts.append(compact_json(compact_dict(cert, X509_FIELDS, "x")))

    return " ".join(parts), len(parts)


def parse_label(label_value) -> tuple[int | None, str | None]:
    label_name = clean_value(label_value)
    if label_name is None:
        return None, None
    label_name = str(label_name).lower()
    if label_name not in LABEL2ID:
        raise ValueError(f"Unknown label: {label_value!r}. Expected one of {sorted(LABEL2ID)}")
    return LABEL2ID[label_name], label_name


def parse_numeric(value):
    value = clean_value(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def extract_num_features(row: pd.Series) -> dict:
    return {col: parse_numeric(row.get(col)) for col in NEW_FORMAT_NUM_FEATURES}


def canonical_flow_uid(row: pd.Series) -> str:
    if clean_value(row.get("flow_uid")) is not None:
        return str(row["flow_uid"])
    src_port = int(float(row["src_port"]))
    dst_port = int(float(row["dst_port"]))
    proto = str(row["protocol"]).lower()
    return f"{row['src_ip']}_{src_port}_{row['dst_ip']}_{dst_port}_{proto}_{row['timestamp']}"


def validate_columns(df: pd.DataFrame):
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    missing_features = [col for col in NEW_FORMAT_NUM_FEATURES if col not in df.columns]
    if missing_features:
        raise ValueError(f"Input CSV missing numeric feature columns: {missing_features}")
    versions = {str(value) for value in df["feature_schema_version"].dropna().unique()}
    if versions != {FEATURE_SCHEMA_VERSION}:
        raise ValueError(f"Expected feature schema {FEATURE_SCHEMA_VERSION}, got {sorted(versions)}")


def preprocess(
    csv_path: str | None = None,
    output_path: str | None = None,
    nrows: int | None = None,
    target: str = "supervised",
    allowed_splits: set[str] | None = None,
):
    if csv_path is None or output_path is None:
        raise ValueError("csv_path 与 output_path 必须由调用通道显式提供")

    print(f"[1/4] Read CSV: {csv_path}")
    if nrows:
        print(f"      nrows={nrows:,}")
    df = pd.read_csv(csv_path, nrows=nrows)
    validate_columns(df)
    if allowed_splits is not None:
        if "split" not in df.columns:
            raise ValueError("过滤训练来源需要 split 列")
        df = df[df["split"].astype(str).isin(allowed_splits)].reset_index(drop=True)
    print(f"      rows={len(df):,}, cols={len(df.columns)}")

    print("[2/4] Convert rows to flow JSONL entries")
    flows = []
    label_counts = Counter()
    empty_text = 0
    event_counts = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="preprocess", unit="row"):
        label_id, label_name = parse_label(row.get("label"))
        text, num_events = build_flow_text(row)
        if not text:
            empty_text += 1
        event_counts.append(num_events)
        label_counts[label_name] += 1

        flows.append(
            {
                "flow_uid": canonical_flow_uid(row),
                "src_ip": str(row["src_ip"]),
                "dst_ip": str(row["dst_ip"]),
                "text": text,
                "label": label_id,
                "label_name": label_name,
                "num_events": num_events,
                "num_features": extract_num_features(row),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "group_id": clean_value(row.get("group_id")),
                "split": clean_value(row.get("split")),
                "pcap_filename": clean_value(row.get("pcap_filename")),
                "dataset_source": clean_value(row.get("dataset_source")),
            }
        )

    print("[3/4] Stats")
    print(f"      flows={len(flows):,}")
    print(f"      labels={dict(label_counts)}")
    print(f"      empty_text={empty_text:,}")
    if event_counts:
        print(
            "      events_per_flow="
            f"min={min(event_counts)}, max={max(event_counts)}, "
            f"avg={sum(event_counts) / len(event_counts):.2f}"
        )

    print(f"[4/4] Save JSONL: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in flows:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return flows


def preprocess_supervised(csv_path: str, output_path: str, nrows: int | None = None):
    return preprocess(csv_path, output_path, nrows, target="supervised")


def preprocess_pretrain(csv_path: str, output_path: str, nrows: int | None = None):
    # Final test captures must not influence domain pretraining metrics.
    return preprocess(csv_path, output_path, nrows, target="pretrain", allowed_splits={"train"})


def preprocess_feature(csv_path: str, output_path: str, nrows: int | None = None):
    return preprocess(csv_path, output_path, nrows, target="feature")
