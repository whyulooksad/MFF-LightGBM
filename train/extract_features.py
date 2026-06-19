"""
步骤5：特征提取

目的：
    使用RTD继续预训练后的DeBERTa encoder和LoRA分类训练后的adapter，
    为每条监督流提取[CLS] 768维特征。

输入：
    data/processed/feature_flows.jsonl
    checkpoints/pretrain/checkpoint-epoch*/
    checkpoints/lora/best/

输出：
    data/output/features_pure.csv
    data/output/features_fused.csv
"""

import csv
import os
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

try:
    from .config import (
        FEATURES_FUSED_CSV,
        FEATURES_PURE_CSV,
        FEATURE_FLOWS_JSONL,
        LORA_BATCH_SIZE,
        LORA_DIR,
        MAX_LENGTH,
        MODEL_DIR,
        NUM_FEATURES,
        OUTPUT_DIR,
        PRETRAIN_DIR,
    )
    from .dataset import load_flows
except ImportError:
    from config import (
        FEATURES_FUSED_CSV,
        FEATURES_PURE_CSV,
        FEATURE_FLOWS_JSONL,
        LORA_BATCH_SIZE,
        LORA_DIR,
        MAX_LENGTH,
        MODEL_DIR,
        NUM_FEATURES,
        OUTPUT_DIR,
        PRETRAIN_DIR,
    )
    from dataset import load_flows


FEATURE_DIM = 768


def load_tokenizer(model_dir: str):
    try:
        return AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_dir)


def latest_pretrain_checkpoint(pretrain_dir=PRETRAIN_DIR):
    root = Path(pretrain_dir)
    if not root.exists():
        return None
    checkpoints = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not checkpoints:
        return None

    def sort_key(path):
        digits = "".join(ch for ch in path.name if ch.isdigit())
        return int(digits) if digits else -1

    return str(sorted(checkpoints, key=sort_key)[-1])


def default_lora_adapter(lora_dir=LORA_DIR):
    adapter_dir = os.path.join(lora_dir, "best")
    return adapter_dir if os.path.exists(adapter_dir) else None


def build_model(base_model_dir=None, adapter_dir=None):
    if base_model_dir is None:
        base_model_dir = latest_pretrain_checkpoint()
        if base_model_dir is None:
            raise FileNotFoundError(
                "未找到RTD预训练checkpoint。请先运行 train/pretrain.py，"
                "或显式传入 base_model_dir=MODEL_DIR 做代码smoke test。"
            )

    if adapter_dir is None:
        adapter_dir = default_lora_adapter()

    if adapter_dir is None:
        raise FileNotFoundError(
            "未找到LoRA adapter: checkpoints/lora/best。请先运行 train/train_lora_classifier.py，"
            "或显式传入 adapter_dir=None 且 allow_no_lora=True 做代码smoke test。"
        )

    print(f"  base_model_dir={base_model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(base_model_dir, num_labels=2)
    print(f"  adapter_dir={adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.float()
    return model, base_model_dir


class FeatureDataset(Dataset):
    """Dataset for final feature extraction; label is optional here."""

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
            self.labels.append(-1 if label is None else int(label))

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


def build_feature_dataset(flows, tokenizer):
    return FeatureDataset(flows, tokenizer, MAX_LENGTH)


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
            if getattr(outputs, "hidden_states", None) is not None:
                cls = outputs.hidden_states[-1][:, 0, :].detach().cpu()
            else:
                cls = outputs.last_hidden_state[:, 0, :].detach().cpu()
            features.extend(cls.tolist())
            labels.extend(batch["labels"].tolist())

    return features, labels


def write_feature_csv(flows, features, output_path, fused=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    feat_cols = [f"feat_{i}" for i in range(FEATURE_DIM)]
    num_cols = [col_name for _, col_name in NUM_FEATURES]
    fieldnames = ["flow_id"] + feat_cols
    if fused:
        fieldnames += num_cols
    fieldnames += ["label"]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for flow, vector in zip(flows, features):
            row = {
                "flow_id": flow.get("flow_id", ""),
                "label": flow.get("label"),
            }
            for idx, value in enumerate(vector):
                row[f"feat_{idx}"] = value

            if fused:
                num_features = flow.get("num_features", {}) or {}
                for col in num_cols:
                    row[col] = num_features.get(col)

            writer.writerow(row)

    print(f"  写出: {output_path}")


def extract_features(
    base_model_dir=None,
    adapter_dir=None,
    batch_size=None,
    allow_no_lora=False,
    output_pure_csv=None,
    output_fused_csv=None,
    max_samples=None,
    jsonl_path=None,
):
    if batch_size is None:
        batch_size = LORA_BATCH_SIZE

    print("=" * 60)
    print("[1/4] 读取待提取特征的流")
    if jsonl_path is None:
        jsonl_path = FEATURE_FLOWS_JSONL
    flows = load_flows(jsonl_path)
    if max_samples is not None:
        flows = flows[:max_samples]
    print(f"  总流数: {len(flows):,}")

    print("\n[2/4] 加载模型和tokenizer")
    if base_model_dir is None:
        base_model_dir = latest_pretrain_checkpoint()
    if base_model_dir is None:
        base_model_dir = MODEL_DIR if allow_no_lora else None

    if adapter_dir is None and allow_no_lora:
        print("  [WARN] allow_no_lora=True，仅使用base encoder提取特征")
        model = AutoModel.from_pretrained(base_model_dir)
        model.float()
        tokenizer_dir = base_model_dir
    else:
        model, tokenizer_dir = build_model(base_model_dir=base_model_dir, adapter_dir=adapter_dir)

    tokenizer = load_tokenizer(tokenizer_dir)

    print("\n[3/4] 构造DataLoader并提取[CLS]")
    dataset = build_feature_dataset(flows, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    model.to(device)

    features, labels = extract_cls_features(model, dataloader, device)
    if len(features) != len(flows):
        raise RuntimeError("特征数量与流数量不一致")
    if features and len(features[0]) != FEATURE_DIM:
        raise RuntimeError(f"特征维度不是{FEATURE_DIM}: {len(features[0])}")

    print("\n[4/4] 写出CSV")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if output_pure_csv is None:
        output_pure_csv = FEATURES_PURE_CSV
    if output_fused_csv is None:
        output_fused_csv = FEATURES_FUSED_CSV

    write_feature_csv(flows, features, output_pure_csv, fused=False)
    write_feature_csv(flows, features, output_fused_csv, fused=True)

    return output_pure_csv, output_fused_csv


if __name__ == "__main__":
    print("=== 特征提取 ===\n")
    extract_features()
