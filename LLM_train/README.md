# LLM Train

本目录只放 DeBERTa-v3 / LoRA 表征学习相关的离线训练代码。它不属于 `main.py --stage all` 默认执行的最终检测 pipeline。

## 流程

```text
data/flow_features/final_multiclass_features_train.csv
    -> LLM_train.split_input_csv
    -> data/LLM_train/input/pretrain_flows.csv
    -> data/LLM_train/input/supervised_flows.csv
    -> preprocess.py --target pretrain
    -> data/LLM_train/processed/pretrain_flows.jsonl
    -> preprocess.py --target supervised
    -> data/LLM_train/processed/supervised_flows.jsonl
    -> LLM_train.dataset
    -> data/LLM_train/splits/train.jsonl
    -> data/LLM_train/splits/val.jsonl
    -> data/LLM_train/splits/test.jsonl
    -> LLM_train.pretrain
    -> checkpoints/pretrain/checkpoint-epoch*/
    -> LLM_train.train_lora_classifier
    -> checkpoints/lora/best/
```

## 命令

```powershell
python -m LLM_train.split_input_csv
python preprocess.py --target pretrain
python preprocess.py --target supervised
python -m LLM_train.dataset --force
python -m LLM_train.pretrain
python -m LLM_train.train_lora_classifier
```

## 文件

```text
LLM_train/
├── config.py                   # LLM 训练路径、超参数、标签映射、数值特征列配置
├── split_input_csv.py          # 将训练流特征 CSV 切分为 pretrain/supervised 输入 CSV
├── dataset.py                  # JSONL 加载、train/val/test 划分、PyTorch Dataset
├── pretrain.py                 # DeBERTa-v3 RTD/ELECTRA-style 继续预训练
└── train_lora_classifier.py    # LoRA 分类器监督训练
```
