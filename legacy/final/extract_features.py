"""
独立特征提取脚本（仅使用 checkpoints 下的微调模型）
完全忽略 models 目录。
"""

import csv
import json
import os
import sys

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==================== 路径配置（只使用 checkpoints） ====================
FEATURE_FLOWS_JSONL = r"D:\jinxian\Pycharm\比赛\final test show\data\json\final_multiclass_features_test.jsonl"
OUTPUT_DIR = r"D:\jinxian\Pycharm\比赛\final test show\data\csv"
FEATURES_PURE_CSV = os.path.join(OUTPUT_DIR, "features_pure.csv")
FEATURES_FUSED_CSV = os.path.join(OUTPUT_DIR, "features_fused.csv")

# 预训练基座（即最终微调时使用的基座）
PRETRAIN_CKPT_DIR = r"D:\jinxian\Pycharm\比赛\final\checkpoints\pretrain\checkpoint-epoch1"
# 微调后的 LoRA adapter
LORA_ADAPTER_DIR = r"D:\jinxian\Pycharm\比赛\final\checkpoints\lora\best"

# ==================== 推理参数 ====================
MAX_LENGTH = 512
BATCH_SIZE = 4

CLASS_LABELS = [
    "benign", "adware", "dns2tcp", "dnscat2",
    "iodine", "ransomware", "scareware", "smsmalware",
]
LABEL2ID = {name: idx for idx, name in enumerate(CLASS_LABELS)}
ID2LABEL = {idx: name for name, idx in LABEL2ID.items()}
NUM_LABELS = len(CLASS_LABELS)

NEW_FORMAT_NUM_FEATURES = [
    "pkts_forward", "pkts_backward", "pkts_total",
    "bytes_forward", "bytes_backward", "bytes_total",
    "ratio_bytes_back_to_forward",
    "pkt_len_max", "pkt_len_min", "pkt_len_mean", "pkt_len_std",
    "pkt_len_fwd_mean", "pkt_len_fwd_std",
    "pkt_len_bwd_mean", "pkt_len_bwd_std",
    "iat_max", "iat_min", "iat_mean", "iat_std",
    "iat_fwd_max", "iat_fwd_min", "iat_fwd_mean", "iat_fwd_std",
    "iat_bwd_max", "iat_bwd_min", "iat_bwd_mean", "iat_bwd_std",
    "flag_syn_count", "flag_fin_count", "flag_rst_count",
    "flag_psh_count", "flag_ack_count",
    "rst_ratio", "handshake_fail_rate", "reconnect_count",
    "conn_count", "flow_interval_jitter", "flow_interval_diff_mean",
    "tcp_rst_count", "reconnection_flag",
    "unique_dst_count", "src_ip_abnormal_ratio",
    "duration_p25", "duration_p50", "duration_p75",
    "weighted_conn_count", "weighted_avg_duration",
    "abnormal_to_conn_ratio", "handshake_duration",
    "cn_vowel_ratio", "cn_digit_density", "cn_special_char_density",
    "cert_valid_days", "cert_age_at_capture", "cert_remaining_days",
    "cert_chain_depth",
]

FEATURE_DIM = 768

# ==================== 数据加载 ====================
def load_flows(jsonl_path):
    flows = []
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    flows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"加载 {len(flows)} 条流 from {jsonl_path}")
    return flows

class FeatureDataset(Dataset):
    def __init__(self, flows, tokenizer, max_length=MAX_LENGTH):
        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        for flow in flows:
            encoded = tokenizer(
                flow["text"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.input_ids.append(encoded["input_ids"].squeeze(0))
            self.attention_mask.append(encoded["attention_mask"].squeeze(0))
            label = flow.get("label")
            if label is None:
                label = -1
            self.labels.append(int(label))
        self.input_ids = torch.stack(self.input_ids)
        self.attention_mask = torch.stack(self.attention_mask)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

# ==================== 模型加载（只依赖 checkpoints） ====================
def load_model():
    """加载微调模型：基座（仅 PRETRAIN_CKPT_DIR）+ LoRA adapter"""
    if not os.path.isdir(PRETRAIN_CKPT_DIR):
        raise FileNotFoundError(f"预训练 checkpoint 不存在: {PRETRAIN_CKPT_DIR}")
    print(f"使用预训练基座: {PRETRAIN_CKPT_DIR}")

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAIN_CKPT_DIR,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if not LORA_ADAPTER_DIR or not os.path.isdir(LORA_ADAPTER_DIR):
        raise FileNotFoundError(f"LoRA adapter 路径无效或不存在: {LORA_ADAPTER_DIR}")
    print(f"加载 LoRA adapter: {LORA_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_DIR)
    model.float()
    return model

# ==================== Tokenizer 加载（仅 checkpoints，含自动修复） ====================
def fix_tokenizer_config(path):
    """删除或修正 tokenizer_config.json 中可能导致冲突的字段"""
    config_file = os.path.join(path, "tokenizer_config.json")
    if not os.path.isfile(config_file):
        return
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    changed = False
    # 删除 extra_special_tokens（列表或 None 都可能引发问题）
    if "extra_special_tokens" in config:
        del config["extra_special_tokens"]
        changed = True
        print(f"  已移除 tokenizer_config.json 中的 extra_special_tokens 字段")

    if changed:
        # 备份一次（如果备份不存在）
        backup_file = config_file + ".backup"
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy2(config_file, backup_file)
            print(f"  原始配置已备份至 {backup_file}")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

def load_tokenizer():
    """仅从 checkpoints 目录寻找 tokenizer（优先级：LoRA > 预训练）"""
    candidates = [
        ("LoRA adapter", LORA_ADAPTER_DIR),
        ("预训练 checkpoint", PRETRAIN_CKPT_DIR),
    ]

    for name, path in candidates:
        if not os.path.isdir(path):
            continue
        # 检查是否有最基础的 spm.model
        spm_path = os.path.join(path, "spm.model")
        if not os.path.isfile(spm_path):
            print(f"  ⚠ {name} 缺少 spm.model，跳过")
            continue

        print(f"尝试从 {name} 加载 tokenizer: {path}")
        # 自动修复配置
        fix_tokenizer_config(path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            print(f"  ✅ 成功加载 tokenizer")
            return tokenizer
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")

    # 如果全部失败，建议用户从在线下载 tokenizer 文件
    print("无法从 checkpoints 目录加载 tokenizer。")
    print("请确保至少一个目录包含完整的 SentencePiece tokenizer 文件（spm.model, tokenizer_config.json, tokenizer.json）。")
    sys.exit(1)

# ==================== 特征提取 ====================
def extract_cls_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                cls = outputs.hidden_states[-1][:, 0, :].detach().cpu()
            else:
                cls = outputs.last_hidden_state[:, 0, :].detach().cpu()
            features.extend(cls.tolist())
            labels.extend(batch["labels"].tolist())
    return features, labels

# ==================== 写出 CSV ====================
def write_feature_csv(flows, features, output_path, fused=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feat_cols = [f"feat_{i}" for i in range(FEATURE_DIM)]
    num_cols = list(NEW_FORMAT_NUM_FEATURES)
    fieldnames = ["flow_id"] + feat_cols
    if fused:
        fieldnames += num_cols
    fieldnames += ["label", "label_name"]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for flow, vector in zip(flows, features):
            row = {
                "flow_id": flow.get("flow_id", ""),
                "label": flow.get("label"),
                "label_name": flow.get("label_name", ""),
            }
            for idx, value in enumerate(vector):
                row[f"feat_{idx}"] = value
            if fused:
                num_features = flow.get("num_features", {}) or {}
                for col in num_cols:
                    row[col] = num_features.get(col, "")
            writer.writerow(row)
    print(f"特征 CSV 已写出: {output_path}")

# ==================== 主流程 ====================
def main():
    print("=" * 60)
    print("特征提取开始（微调模型，仅使用 checkpoints）")

    print(f"[1/4] 读取 JSONL: {FEATURE_FLOWS_JSONL}")
    flows = load_flows(FEATURE_FLOWS_JSONL)
    if len(flows) == 0:
        print("错误：JSONL 中没有任何流，退出。")
        return
    print(f"  总流数: {len(flows)}")

    print("[2/4] 加载模型与 tokenizer...")
    model = load_model()
    tokenizer = load_tokenizer()

    print("[3/4] 提取 [CLS] 特征...")
    dataset = FeatureDataset(flows, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  推理设备: {device}")
    model.to(device)

    features, _ = extract_cls_features(model, dataloader, device)

    print("[4/4] 写出结果...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_feature_csv(flows, features, FEATURES_PURE_CSV, fused=False)
    write_feature_csv(flows, features, FEATURES_FUSED_CSV, fused=True)

    print("特征提取完成！")

if __name__ == "__main__":
    main()