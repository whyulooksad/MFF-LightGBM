"""
步骤4：LoRA分类器监督训练

目的：
    在有标签流量数据上训练一个轻量分类头和LoRA adapter，让encoder的
    表征空间更利于区分正常/恶意流量。这个分类器不是最终交付检测器，
    它服务于后续[CLS]特征提取。

输入：
    1. data/processed/flows.jsonl
    2. checkpoints/pretrain/checkpoint-epoch*/ 中的RTD继续预训练encoder

输出：
    checkpoints/lora/best/ 下的LoRA adapter和tokenizer。
"""

import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .config import (
        LORA_ALPHA,
        LORA_BATCH_SIZE,
        LORA_DIR,
        LORA_DROPOUT,
        LORA_EPOCHS,
        LORA_LR,
        LORA_R,
        LORA_TARGET_MODULES,
        MAX_LENGTH,
        MODEL_DIR,
        PRETRAIN_DIR,
        SEED,
    )
    from .dataset import FlowDataset, get_or_create_splits, load_flows
except ImportError:
    from config import (
        LORA_ALPHA,
        LORA_BATCH_SIZE,
        LORA_DIR,
        LORA_DROPOUT,
        LORA_EPOCHS,
        LORA_LR,
        LORA_R,
        LORA_TARGET_MODULES,
        MAX_LENGTH,
        MODEL_DIR,
        PRETRAIN_DIR,
        SEED,
    )
    from dataset import FlowDataset, get_or_create_splits, load_flows


NUM_LABELS = 2


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def build_model(base_model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_dir,
        num_labels=NUM_LABELS,
        problem_type="single_label_classification",
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=["classifier", "pooler"],
    )
    model = get_peft_model(model, peft_config)
    # 本地DeBERTa权重可能是fp16；LoRA和分类头训练统一用fp32，避免AdamW更新后NaN。
    model.float()
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    preds = []
    labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=y)
        logits = outputs.logits
        total_loss += outputs.loss.item()
        steps += 1
        preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())

    if not labels:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    return {
        "loss": total_loss / max(1, steps),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_lora_classifier(jsonl_path=None, base_model_dir=None, epochs=None, batch_size=None, lr=None):
    set_seed(SEED)

    if epochs is None:
        epochs = LORA_EPOCHS
    if batch_size is None:
        batch_size = LORA_BATCH_SIZE
    if lr is None:
        lr = LORA_LR

    if base_model_dir is None:
        base_model_dir = latest_pretrain_checkpoint()
        if base_model_dir is None:
            raise FileNotFoundError(
                "未找到RTD预训练checkpoint。请先运行 train/pretrain.py，"
                "或显式传入 base_model_dir=MODEL_DIR 做代码smoke test。"
            )

    print("=" * 60)
    print("[1/5] 加载数据")
    flows = load_flows(jsonl_path)
    if not flows:
        raise ValueError("没有可用于LoRA分类训练的有标签流")
    train_flows, val_flows, test_flows = get_or_create_splits(flows=flows)

    print("\n[2/5] 加载tokenizer和Dataset")
    tokenizer = load_tokenizer(base_model_dir)
    train_ds = FlowDataset(train_flows, tokenizer, MAX_LENGTH)
    val_ds = FlowDataset(val_flows, tokenizer, MAX_LENGTH)
    test_ds = FlowDataset(test_flows, tokenizer, MAX_LENGTH)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"  train batches={len(train_dl)}, val batches={len(val_dl)}, test batches={len(test_dl)}")

    print("\n[3/5] 构建LoRA分类模型")
    print(f"  base_model_dir={base_model_dir}")
    model = build_model(base_model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    model.to(device)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    os.makedirs(LORA_DIR, exist_ok=True)
    best_dir = os.path.join(LORA_DIR, "best")
    best_f1 = -1.0
    best_acc = -1.0

    print("\n[4/5] 开始LoRA分类器监督训练")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError("LoRA分类训练loss出现NaN/Inf，已停止训练")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(1, steps)
        val_metrics = evaluate(model, val_dl, device)
        print(
            f"  Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        should_save = (
            val_metrics["f1"] > best_f1
            or (val_metrics["f1"] == best_f1 and val_metrics["accuracy"] > best_acc)
        )
        if should_save:
            best_f1 = val_metrics["f1"]
            best_acc = val_metrics["accuracy"]
            print(f"  -> 保存最佳LoRA adapter到 {best_dir}")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    print("\n[5/5] 测试集评估")
    if os.path.exists(best_dir):
        # 当前model已经是PEFT模型；训练结束后直接评估最后加载中的最佳不必要。
        # 指标主要用于快速检查流程，正式报告建议固定checkpoint重新评估。
        pass
    test_metrics = evaluate(model, test_dl, device)
    print(
        f"  test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_precision={test_metrics['precision']:.4f} "
        f"test_recall={test_metrics['recall']:.4f} "
        f"test_f1={test_metrics['f1']:.4f}"
    )
    print(f"  LoRA输出目录: {best_dir}")

    return model, tokenizer, test_metrics


if __name__ == "__main__":
    print("=== LoRA分类器监督训练 ===\n")
    train_lora_classifier()
