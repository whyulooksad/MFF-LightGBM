"""
步骤3：MLM 预训练
目的：让 DeBERTa 适应加密流量领域的数据分布

输入：全部流的文本序列（不需要 label）
输出：checkpoints/pretrain/ 下的模型权重

方法：Masked Language Modeling
  - 随机遮住 15% 的 token
  - 让模型预测被遮住的是什么
  - 反复几轮后，模型学会了流量序列的"语法"
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from config import (
    MODEL_DIR,
    FLOWS_JSONL,
    PRETRAIN_DIR,
    MAX_LENGTH,
    MLM_PROB,
    PRETRAIN_BATCH_SIZE,
    PRETRAIN_EPOCHS,
    PRETRAIN_LR,
    PRETRAIN_WARMUP,
    SEED,
)


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


def pretrain(jsonl_path=None, epochs=None, batch_size=None, lr=None):
    """
    MLM 预训练主函数。

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
    # 2. 加载 tokenizer 和模型
    # ========================================
    print("\n[2/5] 加载 tokenizer & 模型")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR)

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================
    # 3. 分词 + DataCollator
    # ========================================
    print("\n[3/5] 分词 & 准备数据")

    # 分批 tokenize，避免一次性加载全部到内存
    tokenized = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    print(f"  input_ids shape: {input_ids.shape}")

    # DataCollator 自动处理随机 mask
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB,
    )

    # 简单的 list of dict 格式给 collator
    dataset = [
        {"input_ids": input_ids[i], "attention_mask": attention_mask[i]}
        for i in range(len(input_ids))
    ]

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )

    print(f"  batch 数: {len(dataloader)} (batch_size={batch_size})")

    # ========================================
    # 4. 训练
    # ========================================
    print(f"\n[4/5] 开始 MLM 预训练")
    print(f"  epochs={epochs}, lr={lr}, warmup={PRETRAIN_WARMUP}")

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

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / epoch_steps
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        print(f"  Epoch {epoch} 完成: avg_loss={avg_loss:.4f}, perplexity={perplexity:.1f}")

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
    print("=== MLM 预训练测试 ===\n")
    print("注意：当前测试数据只有 2457 条 + 无 label=1，仅验证流程不崩溃")
    print("正式训练需要全量数据 + GPU\n")

    pretrain(epochs=1, batch_size=4)
