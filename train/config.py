"""
任务二 全局配置
所有路径、超参数、常量都在这里，其他地方 import 即可
"""

import os

# ============================================================
# 路径配置（全部基于 E 盘项目根目录）
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据
DATA_DIR = os.path.join(ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")

# 样例数据（欧鲁金给的，先用这个开发，正式数据放 input/）
EXAMPLE_CSV = os.path.join(ROOT, "data_example", "merged_with_context_CIC-DoHBrw-2020.csv")

# 正式输入CSV
#   pretrain_flows.csv:   给RTD继续预训练，可以无label
#   supervised_flows.csv: 给LoRA分类器监督训练
#   feature_flows.csv:    欧鲁金任务一产出，给最终特征提取
SUPERVISED_INPUT_CSV = os.path.join(INPUT_DIR, "supervised_flows.csv")
PRETRAIN_INPUT_CSV = os.path.join(INPUT_DIR, "pretrain_flows.csv")
FEATURE_INPUT_CSV = os.path.join(INPUT_DIR, "feature_flows.csv")

# 兼容早期样例输入：如果 supervised_flows.csv 不存在，会回退到这个路径
INPUT_CSV = os.path.join(INPUT_DIR, "merged_with_context_CIC-DoHBrw-2020.csv")

# 预处理输出
PRETRAIN_FLOWS_JSONL = os.path.join(PROCESSED_DIR, "pretrain_flows.jsonl")
SUPERVISED_FLOWS_JSONL = os.path.join(PROCESSED_DIR, "supervised_flows.jsonl")
FEATURE_FLOWS_JSONL = os.path.join(PROCESSED_DIR, "feature_flows.jsonl")

# 固定监督训练划分
TRAIN_JSONL = os.path.join(SPLIT_DIR, "train.jsonl")
VAL_JSONL = os.path.join(SPLIT_DIR, "val.jsonl")
TEST_JSONL = os.path.join(SPLIT_DIR, "test.jsonl")

# 特征输出
FEATURES_PURE_CSV = os.path.join(OUTPUT_DIR, "features_pure.csv")
FEATURES_FUSED_CSV = os.path.join(OUTPUT_DIR, "features_fused.csv")

# 模型
MODEL_DIR = os.path.join(ROOT, "models", "deberta-v3-base")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
PRETRAIN_DIR = os.path.join(CHECKPOINT_DIR, "pretrain")
LORA_DIR = os.path.join(CHECKPOINT_DIR, "lora")

# ============================================================
# 训练超参数
# ============================================================
MAX_LENGTH = 512          # DeBERTa 最大序列长度

# RTD continued pretraining
PRETRAIN_BATCH_SIZE = 8
PRETRAIN_EPOCHS = 3
PRETRAIN_LR = 3e-5
PRETRAIN_WARMUP = 1000

# LoRA 微调
LORA_BATCH_SIZE = 16
LORA_EPOCHS = 5
LORA_LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query_proj", "value_proj", "key_proj"]

# 通用
SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
# TEST_RATIO = 0.2（剩下的）

# ============================================================
# 事件字段映射
# ============================================================

# SSL 事件中保留的字段：{原始字段名: 压缩后的字段名}
SSL_FIELDS = {
    "version": "ver",
    "cipher": "cipher",
    "curve": "curve",
    "server_name": "sni",
    "resumed": "resumed",
    "established": "est",
    "ssl_history": "hist",
}

# X509 事件中保留的字段
X509_FIELDS = {
    "certificate.subject": "subj",
    "certificate.issuer": "issuer",
    "certificate.key_length": "key_len",
    "certificate.sig_alg": "sig",
    "certificate.not_valid_before": "not_bef",
    "certificate.not_valid_after": "not_aft",
}

# summary_json 中提取的数值特征（用于融合 CSV）
# 格式: (json中的路径, 输出列名)
NUM_FEATURES = [
    # F_meta 字段
    (["F_meta", "handshake_duration"], "hs_duration"),
    (["F_meta", "key_length"], "key_length"),
    (["F_meta", "cert_valid_days"], "cert_valid_days"),
    (["F_meta", "cn_length"], "cn_length"),
    (["F_meta", "cert_chain_depth"], "cert_chain_depth"),
    (["F_meta", "session_id_len"], "session_id_len"),
    (["F_meta", "tls_ext_count"], "tls_ext_count"),
    (["F_meta", "first_cert_arrival_delay"], "cert_arrival_delay"),
    (["F_meta", "flow_interval_jitter"], "flow_jitter"),
    (["F_meta", "self_signed"], "self_signed"),
    (["F_meta", "has_unknown_ca"], "has_unknown_ca"),
    # F_agg_struct 字段
    (["F_agg_struct", "rst_ratio"], "rst_ratio"),
    (["F_agg_struct", "handshake_fail_rate"], "hs_fail_rate"),
    (["F_agg_struct", "reconnection_flag"], "reconn_flag"),
    (["F_agg_struct", "conn_count"], "conn_count"),
    (["F_agg_struct", "avg_duration"], "avg_duration"),
    (["F_agg_struct", "unique_dst_count"], "uniq_dst_count"),
    (["F_agg_struct", "src_ip_abnormal_ratio"], "src_abn_ratio"),
]
