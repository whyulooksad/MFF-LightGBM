"""Global configuration for task 2."""

import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")

PRETRAIN_INPUT_CSV = os.path.join(INPUT_DIR, "pretrain_flows.csv")
SUPERVISED_INPUT_CSV = os.path.join(INPUT_DIR, "supervised_flows.csv")
FEATURE_INPUT_CSV = os.path.join(INPUT_DIR, "feature_flows.csv")

PRETRAIN_FLOWS_JSONL = os.path.join(PROCESSED_DIR, "pretrain_flows.jsonl")
SUPERVISED_FLOWS_JSONL = os.path.join(PROCESSED_DIR, "supervised_flows.jsonl")
FEATURE_FLOWS_JSONL = os.path.join(PROCESSED_DIR, "feature_flows.jsonl")

TRAIN_JSONL = os.path.join(SPLIT_DIR, "train.jsonl")
VAL_JSONL = os.path.join(SPLIT_DIR, "val.jsonl")
TEST_JSONL = os.path.join(SPLIT_DIR, "test.jsonl")

FEATURES_PURE_CSV = os.path.join(OUTPUT_DIR, "features_pure.csv")
FEATURES_FUSED_CSV = os.path.join(OUTPUT_DIR, "features_fused.csv")

# Model/checkpoint paths
MODEL_DIR = os.path.join(ROOT, "models", "deberta-v3-base")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
PRETRAIN_DIR = os.path.join(CHECKPOINT_DIR, "pretrain")
LORA_DIR = os.path.join(CHECKPOINT_DIR, "lora")

# Training hyperparameters
MAX_LENGTH = 512
PRETRAIN_MAX_LENGTH = 256
PRETRAIN_BATCH_SIZE = 1
PRETRAIN_EPOCHS = 3
PRETRAIN_LR = 3e-5
PRETRAIN_WARMUP = 1000

LORA_BATCH_SIZE = 16
LORA_EPOCHS = 5
LORA_LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query_proj", "value_proj", "key_proj"]

SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2

# Eight-class task labels. Keep this order stable because it is written into
# classifier configs and output labels.
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
ID2LABEL = {idx: name for name, idx in LABEL2ID.items()}
NUM_LABELS = len(CLASS_LABELS)

# Flat numeric columns from final_multiclass_features.csv. Source/leakage
# columns such as dataset_source, subfolder, pcap_filename, and raw Zeek text
# are intentionally excluded.
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
