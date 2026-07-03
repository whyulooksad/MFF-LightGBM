# 恶意加密流量检测

本项目从 `pcap/pcapng` 加密流量包出发，提取双向流统计特征和 TLS/X509 文本特征，再使用 DeBERTa-v3 + LoRA 提取语义表征，最终用 LightGBM 检测器输出 8 分类恶意加密流量检测结果。

当前标签：

```text
benign, adware, dns2tcp, dnscat2, iodine, ransomware, scareware, smsmalware
```

## 主线 Pipeline

`main.py` 是唯一主入口，`pipeline/` 只放被 `main.py` 调用的阶段实现。

```text
data/pcap/raw/
    -> pipeline.truncate_pcap
    -> data/pcap/truncated/
    -> pipeline.extract_flow_features
    -> data/flow_features/final_multiclass_features_test.csv
    -> preprocess.py --target feature
    -> data/pipeline/input/feature_flows.jsonl
    -> pipeline.LLM_extract_features
    -> data/pipeline/features/features_fused.csv
    -> pipeline.detector
    -> data/pipeline/output/
```

可选 SupCon-AE：

```text
data/pipeline/features/features_fused.csv
    -> pipeline.supcon_ae
    -> data/pipeline/features/reduced_features.csv
```

默认完整流程不运行 SupCon-AE。

## 快速运行

```powershell
uv sync
python main.py --stage all
```

跳过 pcap 截断：

```powershell
python main.py --stage all --skip-truncate
```

单独运行阶段：

```powershell
python main.py --stage truncate
python main.py --stage flow_features
python main.py --stage preprocess
python main.py --stage extract
python main.py --stage detector
```

## 离线训练

### LLM 表征训练

```powershell
python -m LLM_train.split_input_csv
python preprocess.py --target pretrain
python preprocess.py --target supervised
python -m LLM_train.dataset --force
python -m LLM_train.pretrain
python -m LLM_train.train_lora_classifier
```

产物：

```text
checkpoints/pretrain/
checkpoints/lora/best/
```

### LightGBM 检测器训推一体

在已有 `data/pipeline/features/features_fused.csv` 后运行：

```powershell
python main.py --stage detector
```

该阶段会在 `features_fused.csv` 内部切分训练集、验证集和测试集：

```text
70% train  # 训练 LightGBM
10% val    # early stopping
20% test   # 测试集推理评估
```

产物：

```text
checkpoints/detector/best_lgb_model.pkl
checkpoints/detector/best_lgb_model.txt
checkpoints/detector/feature_columns.json
data/pipeline/output/detection_results.csv
data/pipeline/output/confusion_matrix.csv
data/pipeline/output/classification_report.txt
data/pipeline/output/detector_report/model_comparison_with_baselines.csv
```

## 数据生命周期

正式主线接收的是 pcap/pcapng，不接收原始公开 CSV。

```text
data/pcap/raw/
    # 原始 pcap/pcapng

    -> pipeline.truncate_pcap

data/pcap/truncated/
    # 每条双向流截断前 N 个包后的 pcap/pcapng

    -> pipeline.extract_flow_features

data/flow_features/
    # pcap 提取出的流统计特征和 TLS/X509 文本字段

    -> preprocess.py --target feature

data/pipeline/input/
    # LLM 可读 JSONL

    -> pipeline.LLM_extract_features

data/pipeline/features/
    # features_pure.csv：DeBERTa/LoRA 提取的纯 768 维语义特征
    # features_fused.csv：768 维语义特征 + 原始数值流特征

    -> pipeline.detector

data/pipeline/output/
    # LightGBM 训练/测试报告、检测结果、混淆矩阵、前端可视化数据
```

队长实验里的 `processed_csv/` 是公开 CSV 清洗产物，不属于正式 pcap 主线，保留在 `legacy/processed_csv/`。

## 目录结构

```text
.
├── main.py                                      # 唯一主入口：调度完整 pipeline 或指定阶段
├── preprocess.py                                # 公共预处理脚本：流特征 CSV -> LLM JSONL
├── pyproject.toml                               # 项目依赖配置
├── uv.lock                                      # uv 锁文件
├── README.md                                    # 项目说明文档
├── .gitignore                                   # Git 忽略规则
├── .python-version                              # Python 版本声明
│
├── pipeline/                                    # main.py 调用的各阶段实现，不是项目入口
│   ├── __init__.py                              # Python 包标记
│   ├── config.py                                # pipeline 路径配置
│   ├── truncate_pcap.py                         # 阶段：pcap/pcapng 按双向流截断
│   ├── extract_flow_features.py                 # 阶段：pcap -> 流统计特征 CSV
│   ├── LLM_extract_features.py                  # 阶段：JSONL -> DeBERTa/LoRA 特征 CSV
│   ├── detector.py                              # 阶段：LightGBM 训推一体，内部切分 train/val/test
│   └── supcon_ae.py                             # 可选阶段：SupCon-AE 降维推理，默认不走
│
├── LLM_train/                                   # LLM 离线训练，不由 main.py 默认调用
│   ├── __init__.py                              # Python 包标记
│   ├── config.py                                # LLM 训练路径、超参数、标签映射、数值特征列配置
│   ├── split_input_csv.py                       # 将训练流特征 CSV 切分为 pretrain/supervised 输入 CSV
│   ├── dataset.py                               # JSONL 加载、train/val/test 划分、PyTorch Dataset
│   ├── pretrain.py                              # DeBERTa-v3 RTD/ELECTRA-style 继续预训练
│   ├── train_lora_classifier.py                 # LoRA 分类器监督训练
│   └── README.md                                # LLM 训练子流程说明
│
├── data/                                        # 正式 pipeline 和离线训练使用的数据
│   ├── pcap/                                    # pcap/pcapng 流量文件
│   │   ├── raw/                                 # 原始 pcap/pcapng 输入
│   │   └── truncated/                           # 按双向流截断后的 pcap/pcapng
│   ├── flow_features/                           # 从 pcap 提取出的流特征 CSV
│   │   ├── final_multiclass_features_train.csv  # 训练集流特征
│   │   ├── final_multiclass_features_test.csv   # 测试/最终检测流特征
│   │   ├── flow_metadata_temporal_train.csv     # 训练集时序元数据
│   │   └── flow_metadata_temporal_test.csv      # 测试/最终检测时序元数据
│   ├── LLM_train/                               # LLM 离线训练数据
│   │   ├── input/                               # LLM 训练输入 CSV
│   │   ├── processed/                           # LLM 训练 JSONL
│   │   └── splits/                              # LoRA 监督训练固定划分
│   └── pipeline/                                # 最终 pipeline 运行时输入/输出
│       ├── input/                               # feature_flows.jsonl
│       ├── features/                            # features_pure.csv / features_fused.csv / reduced_features.csv
│       └── output/                              # LightGBM 训推报告、检测结果和前端可视化数据
│
├── checkpoints/                                 # 所有训练产物
│   ├── pretrain/                                # DeBERTa RTD 继续预训练 checkpoint
│   ├── lora/                                    # LoRA 微调产物
│   ├── detector/                                # LightGBM 检测器模型产物
│   └── supcon_ae/                               # SupCon-AE 降维模型产物
│
├── models/                                      # 本地基座模型和外部模型资产
│   ├── deberta-v3-base/                         # 当前主线使用的 DeBERTa-v3-base
│   └── deberta-base/                            # 备用/历史 DeBERTa base
│
├── docs/                                        # 文档资料
│
└── legacy/                                      # 历史参考、队长临时实验和非正式数据准备产物
    ├── processed_csv/                           # 队长清洗公开 CSV 后的结果，不进入正式 pcap 主线
    ├── raw_data_prepare/                        # 队长原始数据准备脚本，仅作参考
    └── final/                                   # 旧版交付/实验代码
```

## 注意

- `main.py` 是唯一主入口。
- `pipeline/` 中的文件是阶段实现，不是项目入口。
- `LLM_train/` 只放 DeBERTa/LoRA 训练。
- `pipeline/detector.py` 是当前 LightGBM 训推一体阶段。
- `legacy/processed_csv/` 不属于正式 pcap 主线。
