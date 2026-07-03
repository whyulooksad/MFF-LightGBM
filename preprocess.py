"""
Step 1: preprocess task-1 multiclass flow CSV.

Input format:
    current final_multiclass_features_train/test.csv rows with:
    flow_uid, five-tuple fields, label, flat numeric features,
    zeek_conn_log, zeek_ssl_log, zeek_x509_log.

Outputs:
    data/LLM_train/processed/pretrain_flows.jsonl
    data/LLM_train/processed/supervised_flows.jsonl
    data/pipeline/input/feature_flows.jsonl

Identifier/source columns such as five-tuple fields, timestamp, dataset_source,
subfolder, and pcap_filename are not used in model text or fused numeric
features.
"""

from __future__ import annotations

import ast
import json
import math
import os
from collections import Counter

import pandas as pd
from tqdm import tqdm

from pipeline.config import (
    FEATURE_FLOWS_JSONL,
    FLOW_FEATURES_TEST_CSV,
)
from LLM_train.config import (
    FLOW_FEATURES_TRAIN_CSV,
    LABEL2ID,
    NEW_FORMAT_NUM_FEATURES,
    PRETRAIN_FLOWS_JSONL,
    PRETRAIN_INPUT_CSV,
    SUPERVISED_FLOWS_JSONL,
    SUPERVISED_INPUT_CSV,
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
    "zeek_conn_log",
    "zeek_ssl_log",
    "zeek_x509_log",
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


def resolve_default_csv(target: str = "supervised") -> str:
    target_paths = {
        "pretrain": PRETRAIN_INPUT_CSV,
        "supervised": SUPERVISED_INPUT_CSV,
        "feature": str(FLOW_FEATURES_TEST_CSV),
    }
    preferred = target_paths[target]
    if os.path.exists(preferred):
        return preferred

    fallback_csv = FLOW_FEATURES_TRAIN_CSV if target in {"pretrain", "supervised"} else str(FLOW_FEATURES_TEST_CSV)
    if os.path.exists(fallback_csv):
        print(f"[INFO] {preferred} not found; using {fallback_csv}")
        return fallback_csv

    raise FileNotFoundError(
        f"Missing input CSV for target={target}. Expected {preferred} "
        f"or {fallback_csv}."
    )


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value in {"", "-", "nan", "NaN", "None", "null"}:
            return None
        return value.replace('"', "'").replace("\n", " ")
    return value


def parse_literal_cell(value):
    value = clean_value(value)
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
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

    conn = parse_literal_cell(row.get("zeek_conn_log"))
    if isinstance(conn, dict):
        parts.append(compact_json(compact_dict(conn, CONN_FIELDS, "c")))

    ssl = parse_literal_cell(row.get("zeek_ssl_log"))
    if isinstance(ssl, dict):
        parts.append(compact_json(compact_dict(ssl, SSL_FIELDS, "s")))

    x509 = parse_literal_cell(row.get("zeek_x509_log"))
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


def preprocess(
    csv_path: str | None = None,
    output_path: str | None = None,
    nrows: int | None = None,
    target: str = "supervised",
):
    if csv_path is None:
        csv_path = resolve_default_csv(target=target)

    if output_path is None:
        output_paths = {
            "pretrain": PRETRAIN_FLOWS_JSONL,
            "supervised": SUPERVISED_FLOWS_JSONL,
            "feature": str(FEATURE_FLOWS_JSONL),
        }
        output_path = output_paths[target]

    print(f"[1/4] Read CSV: {csv_path}")
    if nrows:
        print(f"      nrows={nrows:,}")
    df = pd.read_csv(csv_path, nrows=nrows)
    validate_columns(df)
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


def preprocess_supervised(csv_path: str | None = None, output_path: str | None = None, nrows: int | None = None):
    return preprocess(csv_path, output_path or SUPERVISED_FLOWS_JSONL, nrows, target="supervised")


def preprocess_pretrain(csv_path: str | None = None, output_path: str | None = None, nrows: int | None = None):
    return preprocess(csv_path, output_path or PRETRAIN_FLOWS_JSONL, nrows, target="pretrain")


def preprocess_feature(csv_path: str | None = None, output_path: str | None = None, nrows: int | None = None):
    return preprocess(csv_path, output_path or str(FEATURE_FLOWS_JSONL), nrows, target="feature")


def preprocess_all(nrows: int | None = None):
    print("=== Generate RTD pretraining corpus ===")
    pretrain_flows = preprocess_pretrain(nrows=nrows)
    print("\n=== Generate LoRA supervised corpus ===")
    supervised_flows = preprocess_supervised(nrows=nrows)
    print("\n=== Generate final feature extraction corpus ===")
    feature_flows = preprocess_feature(nrows=nrows)
    return pretrain_flows, supervised_flows, feature_flows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess task-1 multiclass flow CSV")
    parser.add_argument(
        "--target",
        choices=["all", "pretrain", "supervised", "feature"],
        default="all",
    )
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--csv-path", default=None)
    args = parser.parse_args()

    if args.target == "pretrain":
        preprocess_pretrain(csv_path=args.csv_path, nrows=args.nrows)
    elif args.target == "supervised":
        preprocess_supervised(csv_path=args.csv_path, nrows=args.nrows)
    elif args.target == "feature":
        preprocess_feature(csv_path=args.csv_path, nrows=args.nrows)
    else:
        if args.csv_path is not None:
            preprocess_pretrain(csv_path=args.csv_path, nrows=args.nrows)
            preprocess_supervised(csv_path=args.csv_path, nrows=args.nrows)
            preprocess_feature(csv_path=args.csv_path, nrows=args.nrows)
        else:
            preprocess_all(nrows=args.nrows)
