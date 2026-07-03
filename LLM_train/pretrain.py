"""
步骤3：DeBERTa-v3 RTD 继续预训练

目的：
    让 DeBERTa-v3-base 继续学习加密流量文本的领域分布。

为什么用 RTD：
    DeBERTa-v3 的原始预训练目标是 ELECTRA-style Replaced Token
    Detection (RTD)，不是传统 MLM。这里按 RTD 思路做继续预训练：

        1. generator 对随机 mask 位置做 MLM 预测
        2. 用 generator 采样结果替换原 token，构造 corrupted input
        3. discriminator 判断每个 token 是否被替换
        4. 保存继续预训练后的 discriminator encoder，供后续 LoRA 使用

输入：
    data/LLM_train/processed/pretrain_flows.jsonl，每行包含一条流的 text。
    该文件可以无label，专门作为RTD继续预训练语料。

输出：
    checkpoints/pretrain/checkpoint-epoch*/ 下的 DeBERTa encoder 权重。
"""

import copy
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from .config import (
        MAX_LENGTH,
        MODEL_DIR,
        PRETRAIN_BATCH_SIZE,
        PRETRAIN_EARLY_STOP_PATIENCE,
        PRETRAIN_DIR,
        PRETRAIN_EPOCHS,
        PRETRAIN_FLOWS_JSONL,
        PRETRAIN_LR,
        PRETRAIN_MAX_LENGTH,
        PRETRAIN_MIN_DELTA,
        PRETRAIN_WARMUP,
        SEED,
    )
except ImportError:
    from config import (
        MAX_LENGTH,
        MODEL_DIR,
        PRETRAIN_BATCH_SIZE,
        PRETRAIN_EARLY_STOP_PATIENCE,
        PRETRAIN_DIR,
        PRETRAIN_EPOCHS,
        PRETRAIN_FLOWS_JSONL,
        PRETRAIN_LR,
        PRETRAIN_MAX_LENGTH,
        PRETRAIN_MIN_DELTA,
        PRETRAIN_WARMUP,
        SEED,
    )


MLM_PROBABILITY = 0.15
GENERATOR_LAYERS = 6
GENERATOR_LOSS_WEIGHT = 1.0
DISCRIMINATOR_LOSS_WEIGHT = 50.0
MAX_GRAD_NORM = 1.0


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FlowTextDataset(Dataset):
    """只用于预训练的文本 Dataset。"""

    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.input_ids = []
        self.attention_mask = []

        for i in range(0, len(texts), 2048):
            chunk = texts[i : i + 2048]
            encoded = tokenizer(
                chunk,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.input_ids.append(encoded["input_ids"])
            self.attention_mask.append(encoded["attention_mask"])

        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_mask = torch.cat(self.attention_mask, dim=0)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


class LazyFlowTextDataset(Dataset):
    """Text dataset for RTD pretraining with lazy tokenization."""

    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class RTDHead(nn.Module):
    """
    Port of DeBERTa's RTD prediction head structure.
    Official LMMaskPredictionHead uses CLS context + token state, then
    LayerNorm -> dense -> activation -> token-level binary classifier.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        cls_states = hidden_states[:, 0, :].unsqueeze(1)
        seq_states = self.layer_norm(hidden_states + cls_states)
        seq_states = self.dense(seq_states)
        seq_states = self.activation(seq_states)
        return self.classifier(seq_states).squeeze(-1)


class MLMHead(nn.Module):
    """Generator 的 MLM head，decoder 权重与 word embedding 绑定。"""

    def __init__(self, config, embedding_weight):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.embedding_weight = embedding_weight

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return F.linear(hidden_states, self.embedding_weight, self.bias)


def load_local_lm_head(head: MLMHead, model_dir: str):
    """
    本地 microsoft/deberta-v3-base checkpoint 中保留了官方风格的
    lm_predictions.lm_head.* 权重。transformers 的 DebertaV2ForMaskedLM
    头部命名不同，所以这里手动加载，避免 generator head 完全随机初始化。
    """
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    pytorch_path = os.path.join(model_dir, "pytorch_model.bin")

    state = None
    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file

            state = load_file(safetensors_path)
        except Exception as exc:
            print(f"  [WARN] 读取 safetensors 失败，generator MLM head 将随机初始化: {exc}")
    elif os.path.exists(pytorch_path):
        state = torch.load(pytorch_path, map_location="cpu")

    if state is None:
        print("  [WARN] 未找到本地模型权重文件，generator MLM head 将随机初始化")
        return

    mapping = {
        "lm_predictions.lm_head.dense.weight": head.dense.weight,
        "lm_predictions.lm_head.dense.bias": head.dense.bias,
        "lm_predictions.lm_head.LayerNorm.weight": head.layer_norm.weight,
        "lm_predictions.lm_head.LayerNorm.bias": head.layer_norm.bias,
        "lm_predictions.lm_head.bias": head.bias,
    }

    loaded = 0
    with torch.no_grad():
        for key, param in mapping.items():
            tensor = state.get(key)
            if tensor is not None and tensor.shape == param.shape:
                param.copy_(tensor)
                loaded += 1

    print(f"  generator MLM head 加载 {loaded}/{len(mapping)} 个本地权重")


class DebertaV3RTDPretrainer(nn.Module):
    """ELECTRA / DeBERTa-v3 风格的 generator + discriminator 双分支。"""

    def __init__(self, model_dir=MODEL_DIR, generator_layers=GENERATOR_LAYERS):
        super().__init__()

        disc_config = AutoConfig.from_pretrained(model_dir)
        gen_config = copy.deepcopy(disc_config)
        gen_config.num_hidden_layers = min(generator_layers, disc_config.num_hidden_layers)

        self.generator = AutoModel.from_pretrained(
            model_dir,
            config=gen_config,
            ignore_mismatched_sizes=True,
        )
        self.generator_lm_head = MLMHead(
            gen_config,
            self.generator.embeddings.word_embeddings.weight,
        )
        load_local_lm_head(self.generator_lm_head, model_dir)

        self.discriminator = AutoModel.from_pretrained(model_dir, config=disc_config)
        self.rtd_head = RTDHead(disc_config)

        # 本地权重可能是 fp16；继续预训练先统一用 fp32，优先保证数值稳定。
        self.float()

    def forward(self, input_ids, attention_mask, special_tokens_mask, mask_token_id):
        masked_input_ids, mlm_labels, mlm_mask = mask_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            mask_token_id=mask_token_id,
        )

        generator_outputs = self.generator(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
        )
        gen_logits = self.generator_lm_head(generator_outputs.last_hidden_state)
        gen_loss = F.cross_entropy(
            gen_logits.view(-1, gen_logits.size(-1)),
            mlm_labels.view(-1),
            ignore_index=-100,
        )

        with torch.no_grad():
            sampled_ids = sample_generator_tokens(gen_logits)
            corrupted_input_ids = input_ids.clone()
            corrupted_input_ids[mlm_mask] = sampled_ids[mlm_mask]
            rtd_labels = ((corrupted_input_ids != input_ids) & mlm_mask).float()

        disc_outputs = self.discriminator(
            input_ids=corrupted_input_ids,
            attention_mask=attention_mask,
        )
        disc_logits = self.rtd_head(disc_outputs.last_hidden_state)

        active = attention_mask.bool() & ~special_tokens_mask.bool()
        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[active],
            rtd_labels[active],
        )

        loss = GENERATOR_LOSS_WEIGHT * gen_loss + DISCRIMINATOR_LOSS_WEIGHT * disc_loss
        with torch.no_grad():
            predictions = (torch.sigmoid(disc_logits[active]) > 0.5).float()
            disc_acc = (predictions == rtd_labels[active]).float().mean()
            replaced_rate = rtd_labels[active].mean()

        return {
            "loss": loss,
            "gen_loss": gen_loss.detach(),
            "disc_loss": disc_loss.detach(),
            "disc_acc": disc_acc.detach(),
            "replaced_rate": replaced_rate.detach(),
        }


def load_all_texts(jsonl_path=None):
    """加载全部流文本。RTD 预训练不使用 label，但打印分布方便检查数据。"""
    if jsonl_path is None:
        jsonl_path = PRETRAIN_FLOWS_JSONL

    texts = []
    label_counts = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                texts.append(text)
            lbl = obj.get("label")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(f"加载 {len(texts):,} 条流文本")
    print(f"  label 分布: {label_counts}")
    return texts


def load_tokenizer():
    """兼容新版 transformers 对 SentencePiece regex 的提示。"""
    try:
        return AutoTokenizer.from_pretrained(MODEL_DIR, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(MODEL_DIR)


def build_special_tokens_mask(input_ids, tokenizer):
    """标记 padding/CLS/SEP/MASK 等特殊 token，避免被随机 mask。"""
    special_ids = set(tokenizer.all_special_ids)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        mask |= input_ids.eq(token_id)
    return mask


def mask_inputs(input_ids, attention_mask, special_tokens_mask, mask_token_id):
    """
    构造 generator 的 MLM 输入。

    mlm_labels 中未 mask 的位置为 -100，只有被选中的普通 token 参与 MLM loss。
    """
    probability_matrix = torch.full(input_ids.shape, MLM_PROBABILITY, device=input_ids.device)
    probability_matrix = probability_matrix.masked_fill(~attention_mask.bool(), 0.0)
    probability_matrix = probability_matrix.masked_fill(special_tokens_mask.bool(), 0.0)

    mlm_mask = torch.bernoulli(probability_matrix).bool()
    valid_tokens = attention_mask.bool() & ~special_tokens_mask.bool()
    for row_idx in range(mlm_mask.size(0)):
        if not mlm_mask[row_idx].any() and valid_tokens[row_idx].any():
            candidates = valid_tokens[row_idx].nonzero(as_tuple=False).view(-1)
            chosen = candidates[torch.randint(0, candidates.numel(), (1,), device=input_ids.device)]
            mlm_mask[row_idx, chosen] = True

    mlm_labels = input_ids.clone()
    mlm_labels[~mlm_mask] = -100

    masked_input_ids = input_ids.clone()
    masked_input_ids[mlm_mask] = mask_token_id

    return masked_input_ids, mlm_labels, mlm_mask


def sample_generator_tokens(logits):
    """从 generator 分布中采样替换 token。"""
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(logits.shape[:2])


def pretrain(jsonl_path=None, epochs=None, batch_size=None, lr=None):
    """RTD 继续预训练主函数。"""
    set_seed(SEED)

    if epochs is None:
        epochs = PRETRAIN_EPOCHS
    if batch_size is None:
        batch_size = PRETRAIN_BATCH_SIZE
    if lr is None:
        lr = PRETRAIN_LR

    print("=" * 60)
    print("[1/5] 加载数据")
    texts = load_all_texts(jsonl_path)
    if not texts:
        raise ValueError("没有可用于预训练的文本，请先生成 data/LLM_train/processed/pretrain_flows.jsonl")

    print("\n[2/5] 加载 tokenizer、generator、discriminator")
    tokenizer = load_tokenizer()
    if tokenizer.mask_token_id is None:
        raise ValueError("当前 tokenizer 没有 mask_token_id，无法执行 RTD 预训练")

    model = DebertaV3RTDPretrainer(MODEL_DIR)
    print(f"  generator layers: {model.generator.config.num_hidden_layers}")
    print(f"  discriminator layers: {model.discriminator.config.num_hidden_layers}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n[3/5] 分词 & 准备 DataLoader")
    dataset = LazyFlowTextDataset(texts, tokenizer, PRETRAIN_MAX_LENGTH)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    print(f"  batch 数: {len(dataloader)} (batch_size={batch_size})")

    print("\n[4/5] 开始 RTD 继续预训练")
    print(
        f"  epochs={epochs}, batch_size={batch_size}, max_length={PRETRAIN_MAX_LENGTH}, "
        f"lr={lr}, warmup={PRETRAIN_WARMUP}, "
        f"mlm_prob={MLM_PROBABILITY}, disc_weight={DISCRIMINATOR_LOSS_WEIGHT}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    warmup_steps = min(PRETRAIN_WARMUP, max(0, total_steps // 10))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    os.makedirs(PRETRAIN_DIR, exist_ok=True)
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            special_tokens_mask = build_special_tokens_mask(input_ids, tokenizer).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                mask_token_id=tokenizer.mask_token_id,
            )
            loss = outputs["loss"]

            if not torch.isfinite(loss):
                raise FloatingPointError("RTD loss 出现 NaN/Inf，已停止训练")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            epoch_steps += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                gen=f"{outputs['gen_loss'].item():.4f}",
                disc=f"{outputs['disc_loss'].item():.4f}",
                acc=f"{outputs['disc_acc'].item():.3f}",
                repl=f"{outputs['replaced_rate'].item():.3f}",
            )

        avg_loss = epoch_loss / max(1, epoch_steps)
        print(f"  Epoch {epoch} 完成: avg_loss={avg_loss:.4f}")

        improved = avg_loss < best_loss - PRETRAIN_MIN_DELTA
        if improved:
            best_loss = avg_loss
            stale_epochs = 0
            save_path = os.path.join(PRETRAIN_DIR, f"checkpoint-epoch{epoch}")
            print(f"  -> 保存最佳 discriminator encoder 到 {save_path}")
            model.discriminator.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            stale_epochs += 1
            print(
                f"  -> loss 未明显降低: stale={stale_epochs}/"
                f"{PRETRAIN_EARLY_STOP_PATIENCE}"
            )
            if stale_epochs >= PRETRAIN_EARLY_STOP_PATIENCE:
                print("  -> 触发预训练早停")
                break

    print("\n[5/5] RTD 继续预训练完成")
    print(f"  最佳 loss: {best_loss:.4f}")
    print(f"  模型保存在: {PRETRAIN_DIR}")

    return model.discriminator, tokenizer


if __name__ == "__main__":
    print("=== DeBERTa-v3 RTD 继续预训练 ===\n")
    pretrain(batch_size=PRETRAIN_BATCH_SIZE)
