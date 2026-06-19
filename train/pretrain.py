"""
步骤3：SimCSE 对比学习预训练
目的：让 DeBERTa-v3 Encoder 适应加密流量领域的数据分布

输入：全部流的文本序列（不需要 label）
输出：checkpoints/pretrain/ 下的 Encoder 权重

方法：SimCSE（Simple Contrastive Sentence Embedding，ACL 2021）
  - 同一条流文本过 Encoder 两遍（不同 dropout），[CLS] 输出形成正样本对
  - 同一个 batch 内其他流的 [CLS] 是负样本
  - 对比损失拉近正样本对、推远负样本对
  - Encoder 学会在特征空间中区分不同流量的"语义"

为什么不用 MLM：
  DeBERTa-v3 是 ELECTRA/RTD 预训练的，checkpoint 中 MLM 头 key 名与
  AutoModelForMaskedLM 期望不兼容。SimCSE 只使用 Encoder，天然避免此问题。
  且 SimCSE 直接在 [CLS] 768 维空间中训练，与最终交付物（[CLS] 特征向量）同空间。
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

from config import (
    MODEL_DIR,
    FLOWS_JSONL,
    PRETRAIN_DIR,
    MAX_LENGTH,
    PRETRAIN_BATCH_SIZE,
    PRETRAIN_EPOCHS,
    PRETRAIN_LR,
    PRETRAIN_WARMUP,
    SIMCSE_TEMPERATURE,
    SEED,
)


# SimCSE 专属参数
TEMPERATURE = SIMCSE_TEMPERATURE


class SimCSEDataset(Dataset):
    """最简单的文本 Dataset，返回 input_ids + attention_mask。"""

    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.input_ids = []
        self.attention_mask = []
        # 分批 tokenize 避免爆内存，但数据量不大时一次做完更快
        for i in range(0, len(texts), 2048):
            chunk = texts[i : i + 2048]
            tok = tokenizer(
                chunk,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.input_ids.append(tok["input_ids"])
            self.attention_mask.append(tok["attention_mask"])

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_mask = torch.cat(self.attention_mask, dim=0)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def load_all_texts(jsonl_path=None):
    """加载全部流的文本（预训练不需要 label 和划分）。"""
    if jsonl_path is None:
        jsonl_path = FLOWS_JSONL

    texts = []
    label_counts = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            lbl = obj["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(f"加载 {len(texts):,} 条流文本")
    print(f"  label 分布: {label_counts}")
    return texts


def simcse_loss(z1, z2, temperature=TEMPERATURE):
    """
    SimCSE InfoNCE loss。

    参数:
        z1: 第一遍的 [CLS] 向量，shape [N, D]
        z2: 第二遍的 [CLS] 向量，shape [N, D]
        temperature: 温度系数

    Returns:
        loss: 标量

    计算逻辑：
        z1 和 z2 拼成 [2N, D]，算相似矩阵 [2N, 2N]。
        z1[i] 的正样本是 z2[i]（即矩阵中行 i 对应列 i+N）。
        z2[i] 的正样本是 z1[i]（即矩阵中行 i+N 对应列 i）。
        对角位置 (i, i) 是自身，mask 掉。
        CrossEntropy 会自动在 2N-1 个负类中找到正类。
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    N = z1.size(0)
    z_all = torch.cat([z1, z2], dim=0)          # [2N, D]
    sim = torch.mm(z_all, z_all.t()) / temperature  # [2N, 2N]

    # 正样本标签：z1[i]→i+N,  z2[i]→i
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z1.device)

    # mask 掉自身（对角线），避免自己和自己的无穷大相似
    mask = torch.eye(2 * N, dtype=torch.bool, device=z1.device)
    sim = sim.masked_fill(mask, float("-inf"))

    loss = F.cross_entropy(sim, labels)
    return loss


def pretrain(jsonl_path=None, epochs=None, batch_size=None, lr=None):
    """
    SimCSE 预训练主函数。

    参数可选，方便调参；None 则用 config 默认值。
    """
    if epochs is None:
        epochs = PRETRAIN_EPOCHS
    if batch_size is None:
        batch_size = PRETRAIN_BATCH_SIZE
    if lr is None:
        lr = PRETRAIN_LR

    # ========================================
    # 1. 加载数据
    # ========================================
    print("=" * 60)
    print("[1/5] 加载数据")
    texts = load_all_texts(jsonl_path)

    # ========================================
    # 2. 加载 tokenizer 和 Encoder
    # ========================================
    print("\n[2/5] 加载 tokenizer & Encoder")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    # SimCSE 只需要 Encoder，不碰任何 MLM/ELECTRA 头，不存在 key 不匹配问题
    model = AutoModel.from_pretrained(MODEL_DIR)

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================
    # 3. 准备 DataLoader
    # ========================================
    print("\n[3/5] 分词 & 准备数据")

    dataset = SimCSEDataset(texts, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # 丢掉不够一个 batch 的尾巴，避免 batch_size=1
    )

    print(f"  batch 数: {len(dataloader)} (batch_size={batch_size})")

    # ========================================
    # 4. 训练
    # ========================================
    print(f"\n[4/5] 开始 SimCSE 对比学习预训练")
    print(f"  epochs={epochs}, lr={lr}, temperature={TEMPERATURE}, warmup={PRETRAIN_WARMUP}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=PRETRAIN_WARMUP,
        num_training_steps=total_steps,
    )

    os.makedirs(PRETRAIN_DIR, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 第一遍前向（dropout 随机开启一组）
            out1 = model(input_ids=input_ids, attention_mask=attention_mask)
            z1 = out1.last_hidden_state[:, 0, :]  # [CLS] token, [N, 768]

            # 第二遍前向（dropout 不同 → 输出略有差异）
            out2 = model(input_ids=input_ids, attention_mask=attention_mask)
            z2 = out2.last_hidden_state[:, 0, :]  # [CLS] token, [N, 768]

            loss = simcse_loss(z1, z2, TEMPERATURE)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / epoch_steps
        print(f"  Epoch {epoch} 完成: avg_loss={avg_loss:.4f}")

        # 保存最佳
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(PRETRAIN_DIR, f"checkpoint-epoch{epoch}")
            print(f"  -> 保存最佳模型到 {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    # ========================================
    # 5. 完成
    # ========================================
    print(f"\n[5/5] 预训练完成!")
    print(f"  最佳 loss: {best_loss:.4f}")
    print(f"  模型保存在: {PRETRAIN_DIR}")

    return model, tokenizer


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    print("=== SimCSE 对比学习预训练 测试 ===\n")
    print("注意：当前测试数据只有 2457 条 + 无 label=1，仅验证流程不崩溃")
    print("正式训练需要全量数据 + GPU\n")

    pretrain(epochs=1, batch_size=8)
