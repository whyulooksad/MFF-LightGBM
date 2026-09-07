"""Shared DeBERTa/LoRA loading and CLS feature extraction primitives.

    输入 flows.jsonl
    每条流包含：
    text、label、num_features、来源信息
        ↓
    load_flows()
    读取所有流
        ↓
    latest_pretrain_checkpoint()
    选择最新RTD Encoder检查点
        ↓
    default_lora_adapter()
    选择lora/best
        ↓
    build_model()
    基础DeBERTa分类模型 + LoRA适配器
        ↓
    load_tokenizer()
    加载与基础模型一致的分词器
        ↓
    FeatureDataset
    每条text：
        切成token
        转成input_ids
        补齐/截断到512
        生成attention_mask
        ↓
    DataLoader
    按16条一批，不打乱顺序
        ↓
    自动选择CPU或CUDA GPU
        ↓
    model.eval() + torch.no_grad()
    执行稳定的纯推理
        ↓
    最后一层hidden_states
    形状：
    [batch, 512, 768]
        ↓
    取[:, 0, :]
    取得每条流的CLS
    形状：
    [batch, 768]
        ↓
    汇总全部流的向量
        ↓
    写出两个CSV
    ├── features_pure.csv
    │   768维CLS
    │
    └── features_fused.csv
        768维CLS + 80维人工特征
"""

import csv
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from user_app.inference.contract import (
    MAX_LENGTH,
    NEW_FORMAT_NUM_FEATURES,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
)


FEATURE_DIM = 768
LORA_BATCH_SIZE = 16


def load_tokenizer(model_dir: str):
    try:
        return AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_dir)


def latest_pretrain_checkpoint(pretrain_dir):
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


def default_lora_adapter(lora_dir):
    adapter_dir = os.path.join(lora_dir, "best")
    return adapter_dir if os.path.exists(adapter_dir) else None


def build_model(base_model_dir=None, adapter_dir=None):
    if base_model_dir is None:
        raise FileNotFoundError("未显式提供 RTD encoder checkpoint")

    if adapter_dir is None:
        raise FileNotFoundError("未显式提供 LoRA adapter")

    print(f"  base_model_dir={base_model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_dir,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    print(f"  adapter_dir={adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.float()
    return model, base_model_dir


class FeatureDataset(Dataset):
    """Dataset for final feature extraction; label is optional here."""

    def __init__(self, flows, tokenizer, max_length=MAX_LENGTH):
        self.flows = flows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = []

        for flow in flows:
            label = flow.get("label")
            self.labels.append(-1 if label is None else int(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.flows[idx]["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
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
    num_cols = list(NEW_FORMAT_NUM_FEATURES)
    provenance_cols = ["feature_schema_version", "group_id", "split", "pcap_filename", "dataset_source"]
    fieldnames = ["flow_uid"] + provenance_cols + feat_cols
    if fused:
        fieldnames += num_cols
    fieldnames += ["label", "label_name"]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for flow, vector in zip(flows, features):
            row = {
                "flow_uid": flow.get("flow_uid", ""),
                "label": flow.get("label"),
                "label_name": flow.get("label_name", ""),
            }
            for col in provenance_cols:
                row[col] = flow.get(col)
            for idx, value in enumerate(vector):
                row[f"feat_{idx}"] = value

            if fused:
                num_features = flow.get("num_features", {}) or {}
                for col in num_cols:
                    row[col] = num_features.get(col)

            writer.writerow(row)

    print(f"  写出: {output_path}")


def _flow_chunks(jsonl_path: str | Path, chunk_size: int, max_samples: int | None = None):
    chunk = []
    emitted = 0
    with Path(jsonl_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if max_samples is not None and emitted >= max_samples:
                break
            chunk.append(json.loads(line))
            emitted += 1
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def _feature_row(flow: dict, vector: list[float], fused: bool) -> dict:
    row = {
        "flow_uid": flow.get("flow_uid", ""),
        "label": flow.get("label"),
        "label_name": flow.get("label_name", ""),
    }
    for col in ("feature_schema_version", "group_id", "split", "pcap_filename", "dataset_source"):
        row[col] = flow.get(col)
    for idx, value in enumerate(vector):
        row[f"feat_{idx}"] = value
    if fused:
        num_features = flow.get("num_features", {}) or {}
        for col in NEW_FORMAT_NUM_FEATURES:
            row[col] = num_features.get(col)
    return row


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
    print("[1/4] 检查待提取特征的流")
    if jsonl_path is None:
        raise ValueError("jsonl_path 必须由调用通道显式提供")
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"流 JSONL 不存在: {jsonl_path}")

    print("\n[2/4] 加载模型和tokenizer")
    if base_model_dir is None:
        raise ValueError("base_model_dir 必须由调用通道显式提供")

    if adapter_dir is None and allow_no_lora:
        print("  [WARN] allow_no_lora=True，仅使用base encoder提取特征")
        model = AutoModel.from_pretrained(base_model_dir)
        model.float()
        tokenizer_dir = base_model_dir
    else:
        model, tokenizer_dir = build_model(base_model_dir=base_model_dir, adapter_dir=adapter_dir)

    tokenizer = load_tokenizer(tokenizer_dir)

    print("\n[3/4] 分块构造DataLoader并提取[CLS]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    model.to(device)

    print("\n[4/4] 写出CSV")
    if output_pure_csv is None or output_fused_csv is None:
        raise ValueError("两个输出 CSV 路径必须由调用通道显式提供")

    pure_path, fused_path = Path(output_pure_csv), Path(output_fused_csv)
    pure_path.parent.mkdir(parents=True, exist_ok=True)
    fused_path.parent.mkdir(parents=True, exist_ok=True)
    pure_tmp = pure_path.with_suffix(pure_path.suffix + ".tmp")
    fused_tmp = fused_path.with_suffix(fused_path.suffix + ".tmp")
    feat_cols = [f"feat_{i}" for i in range(FEATURE_DIM)]
    provenance_cols = ["feature_schema_version", "group_id", "split", "pcap_filename", "dataset_source"]
    pure_fields = ["flow_uid", *provenance_cols, *feat_cols, "label", "label_name"]
    fused_fields = ["flow_uid", *provenance_cols, *feat_cols, *NEW_FORMAT_NUM_FEATURES, "label", "label_name"]
    written = 0
    try:
        with pure_tmp.open("w", encoding="utf-8", newline="") as pure_stream, fused_tmp.open(
            "w", encoding="utf-8", newline=""
        ) as fused_stream:
            pure_writer = csv.DictWriter(pure_stream, fieldnames=pure_fields)
            fused_writer = csv.DictWriter(fused_stream, fieldnames=fused_fields)
            pure_writer.writeheader()
            fused_writer.writeheader()
            chunk_size = max(256, int(batch_size) * 32)
            for flows in _flow_chunks(jsonl_path, chunk_size, max_samples):
                dataset = build_feature_dataset(flows, tokenizer)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                features, _ = extract_cls_features(model, dataloader, device)
                if len(features) != len(flows):
                    raise RuntimeError("特征数量与流数量不一致")
                if features and len(features[0]) != FEATURE_DIM:
                    raise RuntimeError(f"特征维度不是{FEATURE_DIM}: {len(features[0])}")
                pure_writer.writerows(_feature_row(flow, vector, False) for flow, vector in zip(flows, features))
                fused_writer.writerows(_feature_row(flow, vector, True) for flow, vector in zip(flows, features))
                written += len(flows)
        if written == 0:
            raise ValueError("流 JSONL 中没有可提取的记录")
        os.replace(pure_tmp, pure_path)
        os.replace(fused_tmp, fused_path)
    finally:
        pure_tmp.unlink(missing_ok=True)
        fused_tmp.unlink(missing_ok=True)
    print(f"  总流数: {written:,}")
    print(f"  写出: {pure_path}")
    print(f"  写出: {fused_path}")

    return output_pure_csv, output_fused_csv
