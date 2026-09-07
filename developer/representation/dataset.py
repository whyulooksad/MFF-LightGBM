"""
步骤2：PyTorch Dataset 封装
输入：data/developer/workspace/corpus/supervised/supervised_flows.jsonl
输出：DataLoader（训练/验证/测试）

做的事：
1. 读取 JSONL，加载所有流
2. 用 DeBERTa tokenizer 对每条流的 text 分词
3. 按 manifest 中已经固定的 split 分发 train/validation/test
4. 封装成 PyTorch Dataset → DataLoader
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from developer.data_prepare.group_split import split_flow_records
from user_app.inference.flow_io import load_flows

from developer.representation.config import (
    MODEL_DIR,
    MAX_LENGTH,
    TEST_JSONL,
    TRAIN_JSONL,
    VAL_JSONL,
    SUPERVISED_FLOWS_JSONL,
    LORA_BATCH_SIZE,
)


class FlowDataset(Dataset):
    """
    流序列 Dataset。

    每条流返回:
        input_ids:       token ID 序列  [max_length]
        attention_mask:  注意力掩码      [max_length]
        label:           八分类 ID（0..7）
    """

    def __init__(self, flows, tokenizer, max_length=MAX_LENGTH):
        self.flows = flows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = []

        for flow in flows:
            if flow.get("label") is None:
                raise ValueError("FlowDataset requires label; use supervised_flows.jsonl for LoRA training")
            self.labels.append(flow["label"])

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


def save_flows_jsonl(flows, jsonl_path):
    """保存流列表到JSONL。"""
    import os

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for flow in flows:
            f.write(json.dumps(flow, ensure_ascii=False) + "\n")


def save_splits(train, val, test):
    """保存固定 train/val/test 划分。"""
    save_flows_jsonl(train, TRAIN_JSONL)
    save_flows_jsonl(val, VAL_JSONL)
    save_flows_jsonl(test, TEST_JSONL)
    print("已保存固定划分:")
    print(f"  train: {TRAIN_JSONL}")
    print(f"  val:   {VAL_JSONL}")
    print(f"  test:  {TEST_JSONL}")


def materialize_authoritative_splits(flows=None):
    """
    按当前权威 flow 中的 split 字段生成三个派生 JSONL。

    每次运行都会覆盖派生文件，防止数据集或 manifest 已变化、训练却静默
    复用上一次运行留下的旧划分。
    """
    if flows is None:
        flows = load_flows(SUPERVISED_FLOWS_JSONL)

    # split 字段来自权威 manifest。每次都从当前输入重建三个派生文件，
    # 避免更换数据或 manifest 后静默复用旧 JSONL。
    train, val, test = split_flows(flows)
    save_splits(train, val, test)
    return train, val, test


def split_flows(flows):
    """
    划分 train / val / test。

    返回: train_flows, val_flows, test_flows
    """
    train, val, test = split_flow_records(flows)
    print(f"划分: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


def create_dataloaders(flows=None, batch_size=None):
    """
    一站式函数：加载数据 → 划分 → 创建 DataLoader。

    参数:
        flows: 流列表，None 则从 JSONL 读取
        batch_size: 批次大小，None 则用 config 默认值

    返回:
        train_dl, val_dl, test_dl, tokenizer
    """
    if flows is None:
        flows = load_flows(SUPERVISED_FLOWS_JSONL)

    if batch_size is None:
        batch_size = LORA_BATCH_SIZE

    # 划分。监督训练默认使用固定split，避免不同脚本临时划分不一致。
    train_flows, val_flows, test_flows = materialize_authoritative_splits(flows=flows)

    # 加载 tokenizer
    print(f"加载 tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # 创建 Dataset
    print("创建 Dataset...")
    train_ds = FlowDataset(train_flows, tokenizer, MAX_LENGTH)
    val_ds = FlowDataset(val_flows, tokenizer, MAX_LENGTH)
    test_ds = FlowDataset(test_flows, tokenizer, MAX_LENGTH)

    # 创建 DataLoader
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"DataLoader 创建完毕: batch_size={batch_size}")
    print(f"  train batches: {len(train_dl)}")
    print(f"  val batches:   {len(val_dl)}")
    print(f"  test batches:  {len(test_dl)}")

    return train_dl, val_dl, test_dl, tokenizer


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    print("=== 测试 Dataset 加载 ===\n")

    # 只测加载，不划分（用全部数据快速验证）
    flows = load_flows(SUPERVISED_FLOWS_JSONL)  # 当前只有测试数据
    train_dl, val_dl, test_dl, tokenizer = create_dataloaders(flows)

    # 检查一个 batch
    print("\n=== 检查第一个 batch ===")
    batch = next(iter(train_dl))
    print(f"input_ids shape:      {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"labels shape:         {batch['labels'].shape}")
    print(f"labels 值:            {batch['labels'].tolist()[:10]}...")

    # 解码一条看看
    print("\n=== 解码第一条(前200 token) ===")
    decoded = tokenizer.decode(batch["input_ids"][0][:50], skip_special_tokens=False)
    print(decoded[:300])

    # 统计序列长度（非 padding 部分）
    lengths = batch["attention_mask"].sum(dim=1)
    print(f"\n第一个 batch 的实际序列长度: min={lengths.min().item()}, "
          f"max={lengths.max().item()}, avg={lengths.float().mean().item():.0f}")

    print("\nDataset 加载测试通过!")
