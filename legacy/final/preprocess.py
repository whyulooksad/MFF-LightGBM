"""
Preprocess a multiclass flow CSV into a single JSONL file.

Input format:
    final_multiclass_features.csv style rows with:
    flow_uid, five-tuple fields, label, flat numeric features,
    zeek_conn_log, zeek_ssl_log, zeek_x509_log.

Output:
    A single JSONL file where each line contains a flow object with text, label, etc.
"""

from __future__ import annotations

import ast
import json
import math
import os
from collections import Counter

import pandas as pd
from tqdm import tqdm

# ====================== 内联配置（不再依赖 config.py） ======================

# 八分类标签映射
CLASS_LABELS = [
    "benign",
    "adware",
    "dns2tcp",
    "dnscat2",
    "iodine",
    "ransomware",
    "scareware",
    "smsmalware",
]
LABEL2ID = {name: idx for idx, name in enumerate(CLASS_LABELS)}

# 需要保留的数值特征列名
NEW_FORMAT_NUM_FEATURES = [
    "pkts_forward",
    "pkts_backward",
    "pkts_total",
    "bytes_forward",
    "bytes_backward",
    "bytes_total",
    "ratio_bytes_back_to_forward",
    "pkt_len_max",
    "pkt_len_min",
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_fwd_mean",
    "pkt_len_fwd_std",
    "pkt_len_bwd_mean",
    "pkt_len_bwd_std",
    "iat_max",
    "iat_min",
    "iat_mean",
    "iat_std",
    "iat_fwd_max",
    "iat_fwd_min",
    "iat_fwd_mean",
    "iat_fwd_std",
    "iat_bwd_max",
    "iat_bwd_min",
    "iat_bwd_mean",
    "iat_bwd_std",
    "flag_syn_count",
    "flag_fin_count",
    "flag_rst_count",
    "flag_psh_count",
    "flag_ack_count",
    "rst_ratio",
    "handshake_fail_rate",
    "reconnect_count",
    "conn_count",
    "flow_interval_jitter",
    "flow_interval_diff_mean",
    "tcp_rst_count",
    "reconnection_flag",
    "unique_dst_count",
    "src_ip_abnormal_ratio",
    "duration_p25",
    "duration_p50",
    "duration_p75",
    "weighted_conn_count",
    "weighted_avg_duration",
    "abnormal_to_conn_ratio",
    "handshake_duration",
    "cn_vowel_ratio",
    "cn_digit_density",
    "cn_special_char_density",
    "cert_valid_days",
    "cert_age_at_capture",
    "cert_remaining_days",
    "cert_chain_depth",
]

# ---------- 修改的默认路径 ----------
DEFAULT_CSV_PATH = r"D:\jinxian\Pycharm\比赛\final test show\data\csv\final_multiclass_features_test.csv"
DEFAULT_OUTPUT_PATH = r"D:\jinxian\Pycharm\比赛\final test show\data\json\final_multiclass_features_test.jsonl"
# ------------------------------------

# ====================== 字段映射 ======================

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

# ====================== 辅助函数 ======================

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

def canonical_flow_id(row: pd.Series) -> str:
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

# ====================== 核心预处理 ======================

def preprocess(
    csv_path: str,
    output_path: str,
    nrows: int | None = None,
):
    """
    读取 csv_path，转换为 JSONL 并写入 output_path。
    返回生成的 flow 列表。
    """
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
                "flow_id": canonical_flow_id(row),
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

# ====================== 主程序 ======================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess a multiclass flow CSV to JSONL"
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to input CSV (default: specified inside script)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output JSONL (default: specified inside script)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit number of rows to read (for testing)",
    )

    args = parser.parse_args()

    # 确定 CSV 路径（如果命令行未提供，使用新的默认值）
    input_csv = args.csv_path or DEFAULT_CSV_PATH
    if not os.path.isfile(input_csv):
        print(f"Error: Input CSV not found: {input_csv}")
        exit(1)

    # 确定输出路径
    if args.output:
        output_path = args.output
    elif args.csv_path:
        # 如果用户通过命令行提供了输入文件，输出文件基于输入文件名放置在同目录下
        base = os.path.splitext(args.csv_path)[0]
        output_path = f"{base}.jsonl"
    else:
        output_path = DEFAULT_OUTPUT_PATH

    print(f"Input : {input_csv}")
    print(f"Output: {output_path}")
    preprocess(input_csv, output_path, nrows=args.nrows)