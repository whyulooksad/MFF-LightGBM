# 任务2：加密流量 DeBERTa 表征学习与特征提取

本仓库负责实验中的任务2：把加密流量 CSV 转成适合 DeBERTa-v3-base 阅读的流文本，先做 RTD/ELECTRA 风格继续预训练，再用 LoRA 分类训练塑造特征空间，最后为后续降维和 LightGBM 分类输出特征 CSV。

这部分不是最终检测器。LoRA 分类训练的目的，是让 Encoder 学到更适合区分正常流量和恶意流量的表示；最终判断仍交给后续同学的降维模块和 LightGBM。

## 整体流程

```text
data/input/pretrain_flows.csv
    -> train.preprocess
    -> data/processed/pretrain_flows.jsonl
    -> train.pretrain
    -> checkpoints/pretrain/checkpoint-epoch*

data/input/supervised_flows.csv
    -> train.preprocess
    -> data/processed/supervised_flows.jsonl
    -> train.dataset
    -> data/splits/train.jsonl, val.jsonl, test.jsonl
    -> train.train_lora_classifier
    -> checkpoints/lora/best

data/input/feature_flows.csv
    -> train.preprocess
    -> data/processed/feature_flows.jsonl
    -> extract_features.py
    -> data/output/features_pure.csv, features_fused.csv
```

## 快速运行

安装依赖：

```powershell
uv sync
```

预处理三份输入 CSV：

```powershell
.\.venv\Scripts\python.exe -m train.preprocess --target all
```

划分 LoRA 监督训练集：

```powershell
.\.venv\Scripts\python.exe -m train.dataset --force
```

RTD 继续预训练：

```powershell
.\.venv\Scripts\python.exe -m train.pretrain
```

LoRA 分类训练：

```powershell
.\.venv\Scripts\python.exe -m train.train_lora_classifier
```

提取最终特征：

```powershell
.\.venv\Scripts\python.exe extract_features.py
```

## 输入输出约定

- `pretrain_flows.csv`：给 RTD 继续预训练使用，可以无 `label`。
- `supervised_flows.csv`：给 LoRA 分类训练使用，必须有 `label`。
- `feature_flows.csv`：欧鲁金任务一产出的待提取特征数据，是最终交给后续同学的输入来源。
- `features_pure.csv`：`flow_id + 768维DeBERTa特征 + label`。
- `features_fused.csv`：`flow_id + 768维DeBERTa特征 + 18个原始数值特征 + label`。

## 项目结构

```text
.
├── checkpoints/                         # 训练过程中保存的中间模型与 LoRA adapter
│   ├── pretrain/                        # RTD 继续预训练后的 DeBERTa encoder checkpoint
│   └── lora/                            # LoRA 分类训练产物
├── data/                                # 本任务的数据入口、中间文件和最终输出
│   ├── input/                           # 原始输入 CSV
│   │   ├── pretrain_flows.csv           # RTD 继续预训练数据，可以无 label
│   │   ├── supervised_flows.csv         # LoRA 监督训练数据，必须有 label
│   │   └── feature_flows.csv            # 最终特征提取输入，来自任务一产出
│   ├── processed/                       # 预处理后的流级 JSONL
│   │   ├── pretrain_flows.jsonl         # pretrain_flows.csv 的预处理结果
│   │   ├── supervised_flows.jsonl       # supervised_flows.csv 的预处理结果
│   │   └── feature_flows.jsonl          # feature_flows.csv 的预处理结果
│   ├── splits/                          # 只从 supervised_flows.jsonl 划分，用于 LoRA 训练
│   │   ├── train.jsonl                  # LoRA 训练集
│   │   ├── val.jsonl                    # LoRA 验证集
│   │   └── test.jsonl                   # LoRA 测试集
│   └── output/                          # 提供给后续降维和 LightGBM 的最终特征 CSV
│       ├── features_pure.csv            # 纯 DeBERTa 768维特征
│       └── features_fused.csv           # DeBERTa 特征 + 原始数值特征
├── models/                              # 本地基座模型目录
│   ├── deberta-v3-base/                 # 当前主线使用的 DeBERTa-v3-base
│   └── deberta-base/                    # 备用/历史基座模型
└── train/                               # 任务2核心代码
    ├── config.py                        # 路径、超参、字段映射和数值特征配置
    ├── preprocess.py                    # CSV -> 流级 JSONL，生成三份 processed 文件
    ├── dataset.py                       # 读取 supervised_flows.jsonl 并划分 train/val/test
    ├── pretrain.py                      # DeBERTa-v3 RTD/ELECTRA 风格继续预训练
    ├── train_lora_classifier.py         # LoRA 分类训练，用于塑造 encoder 特征空间
    └── __init__.py                      # Python 包标记
├── extract_features.py                  # 加载预训练 checkpoint + LoRA，输出最终特征 CSV
```
