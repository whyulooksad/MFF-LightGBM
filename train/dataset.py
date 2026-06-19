"""
步骤2：PyTorch Dataset 封装
输入：data/processed/supervised_flows.jsonl
输出：DataLoader（训练/验证/测试）

做的事：
1. 读取 JSONL，加载所有流
2. 用 DeBERTa tokenizer 对每条流的 text 分词
3. 划分 train/val/test
4. 封装成 PyTorch Dataset → DataLoader
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

try:
    from .config import (
        MODEL_DIR,
        MAX_LENGTH,
        SEED,
        TEST_JSONL,
        TRAIN_RATIO,
        TRAIN_JSONL,
        VAL_RATIO,
        VAL_JSONL,
        SUPERVISED_FLOWS_JSONL,
        LORA_BATCH_SIZE,
    )
except ImportError:
    from config import (
        MODEL_DIR,
        MAX_LENGTH,
        SEED,
        TEST_JSONL,
        TRAIN_RATIO,
        TRAIN_JSONL,
        VAL_RATIO,
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
        label:           0 正常 / 1 恶意
    """

    def __init__(self, flows, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        for flow in flows:
            # 分词
            encoded = tokenizer(
                flow["text"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.input_ids.append(encoded["input_ids"].squeeze(0))
            self.attention_mask.append(encoded["attention_mask"].squeeze(0))
            if flow.get("label") is None:
                raise ValueError("FlowDataset requires label; use supervised_flows.jsonl for LoRA training")
            self.labels.append(flow["label"])

        # 转成大 Tensor，比逐条取快
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


def load_flows(jsonl_path=None):
    """从 JSONL 加载所有流。"""
    if jsonl_path is None:
        jsonl_path = SUPERVISED_FLOWS_JSONL

    flows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            flows.append(json.loads(line))

    print(f"加载 {len(flows):,} 条流 from {jsonl_path}")

    # 统计
    labels = {}
    for f in flows:
        lbl = f.get("label")
        labels[lbl] = labels.get(lbl, 0) + 1
    print(f"  label 分布: {labels}")

    return flows


def save_flows_jsonl(flows, jsonl_path):
    """保存流列表到JSONL。"""
    import os

    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for flow in flows:
            f.write(json.dumps(flow, ensure_ascii=False) + "\n")


def split_files_exist():
    """检查固定划分文件是否已经存在。"""
    import os

    return all(os.path.exists(p) for p in (TRAIN_JSONL, VAL_JSONL, TEST_JSONL))


def save_splits(train, val, test):
    """保存固定 train/val/test 划分。"""
    save_flows_jsonl(train, TRAIN_JSONL)
    save_flows_jsonl(val, VAL_JSONL)
    save_flows_jsonl(test, TEST_JSONL)
    print("已保存固定划分:")
    print(f"  train: {TRAIN_JSONL}")
    print(f"  val:   {VAL_JSONL}")
    print(f"  test:  {TEST_JSONL}")


def load_splits():
    """读取固定 train/val/test 划分。"""
    train = load_flows(TRAIN_JSONL)
    val = load_flows(VAL_JSONL)
    test = load_flows(TEST_JSONL)
    return train, val, test


def get_or_create_splits(flows=None, force_recreate=False):
    """
    获取固定划分。

    如果 data/splits/train.jsonl、val.jsonl、test.jsonl 已存在，默认直接读取；
    否则从 supervised_flows.jsonl 创建划分并保存。
    """
    if split_files_exist() and not force_recreate:
        print("发现已保存的固定划分，直接读取 data/splits/*.jsonl")
        return load_splits()

    if flows is None:
        flows = load_flows()

    train, val, test = split_flows(flows)
    save_splits(train, val, test)
    return train, val, test


def split_flows(flows):
    """
    划分 train / val / test。

    返回: train_flows, val_flows, test_flows
    """
    labels = [f["label"] for f in flows]
    unique_labels = set(labels)

    # 类别样本过少时跳过 stratify，避免 train_test_split 报错
    if can_stratify(labels):
        stratify_kwargs = {"stratify": labels}
    else:
        stratify_kwargs = {}
        print(f"  (类别过少或样本不足 {unique_labels}，跳过 stratify 分层)")

    # 先分出 test
    train_val, test = train_test_split(
        flows,
        test_size=1 - TRAIN_RATIO - VAL_RATIO,
        random_state=SEED,
        **stratify_kwargs,
    )
    # 再从 train_val 里分出 val
    val_ratio_of_train_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_val_labels = [f["label"] for f in train_val]
    if can_stratify(train_val_labels):
        val_stratify_kwargs = {"stratify": train_val_labels}
    else:
        val_stratify_kwargs = {}
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_of_train_val,
        random_state=SEED,
        **val_stratify_kwargs,
    )

    print(f"划分: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


def can_stratify(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return len(counts) > 1 and min(counts.values()) >= 2


def create_dataloaders(flows=None, batch_size=None, force_recreate_splits=False):
    """
    一站式函数：加载数据 → 划分 → 创建 DataLoader。

    参数:
        flows: 流列表，None 则从 JSONL 读取
        batch_size: 批次大小，None 则用 config 默认值

    返回:
        train_dl, val_dl, test_dl, tokenizer
    """
    if flows is None:
        flows = load_flows()

    if batch_size is None:
        batch_size = LORA_BATCH_SIZE

    # 划分。监督训练默认使用固定split，避免不同脚本临时划分不一致。
    train_flows, val_flows, test_flows = get_or_create_splits(
        flows=flows,
        force_recreate=force_recreate_splits,
    )

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
    flows = load_flows()  # 当前只有测试数据
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
