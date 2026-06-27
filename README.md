# 恶意加密流量检测

本项目面向“在不解密 SSL/TLS 等加密通信内容的前提下识别恶意行为”的实验场景。项目从原始 `pcap/pcapng` 流量包出发，提取双向流统计特征和 TLS/X509 文本特征，再使用继续预训练后的 DeBERTa-v3 + LoRA 提取语义表征，最终将融合特征送入检测器，输出多分类恶意流量检测结果和前端可视化数据。

当前主线支持 8 类流量标签：

```text
benign, adware, dns2tcp, dnscat2, iodine, ransomware, scareware, smsmalware
```

## 项目流程

### 最终检测 Pipeline

```text
data/pcap/raw/
    -> pipeline.truncate_pcap
    -> data/pcap/truncated/
    -> pipeline.extract_flow_features
    -> data/flow_features/final_multiclass_features_test.csv
    -> preprocess.py --target feature
    -> data/pipeline/input/feature_flows.jsonl
    -> pipeline.LLM_extract_features
    -> data/pipeline/features/features_pure.csv
    -> data/pipeline/features/features_fused.csv
    -> pipeline.detector
    -> data/pipeline/output/detection_results.csv
    -> data/pipeline/output/detection_assets/
```

可选 SupCon-AE 降维阶段：

```text
data/pipeline/features/features_fused.csv
    -> pipeline.supcon_ae
    -> data/pipeline/features/reduced_features.csv
```

默认 `main.py --stage all` 不运行 SupCon-AE；需要时使用 `--use-supcon`。

### 离线训练流程

DeBERTa/LoRA 训练是离线步骤，不会被 `main.py` 默认触发。

```text
data/flow_features/final_multiclass_features_train.csv
    -> train.split_input_csv
    -> data/train/input/pretrain_flows.csv
    -> data/train/input/supervised_flows.csv
    -> preprocess.py --target pretrain
    -> preprocess.py --target supervised
    -> train.dataset
    -> data/train/splits/train.jsonl
    -> data/train/splits/val.jsonl
    -> data/train/splits/test.jsonl
    -> train.pretrain
    -> checkpoints/pretrain/checkpoint-epoch*/
    -> train.train_lora_classifier
    -> checkpoints/lora/best/
```

训练产物会被最终特征提取阶段加载：

- `checkpoints/pretrain/checkpoint-epoch*/`：RTD 继续预训练后的 DeBERTa encoder。
- `checkpoints/lora/best/`：LoRA adapter、分类头和 tokenizer 文件。

## 环境准备

项目使用 Python 3.12 和 `uv` 管理依赖。

```powershell
uv sync
```

主要依赖包括：

- `torch` / `transformers` / `peft`
- `pandas` / `numpy` / `scikit-learn`
- `lightgbm` / `xgboost`
- `tqdm` / `joblib` / `safetensors`

本地模型文件默认放在：

```text
models/deberta-v3-base/
models/deberta-base/
```

其中当前主线使用 `models/deberta-v3-base/`。

## 快速运行

运行完整检测流程：

```powershell
python main.py --stage all
```

如果已经有截断后的 pcap，可以跳过截断：

```powershell
python main.py --stage all --skip-truncate
```

如果需要在特征提取后运行 SupCon-AE：

```powershell
python main.py --stage all --use-supcon
```

单独运行某个阶段：

```powershell
python main.py --stage truncate
python main.py --stage flow_features
python main.py --stage preprocess
python main.py --stage extract
python main.py --stage supcon
python main.py --stage detector
```

## 离线训练

训练数据默认来自：

```text
data/flow_features/final_multiclass_features_train.csv
```

完整训练步骤：

```powershell
python -m train.split_input_csv
python preprocess.py --target pretrain
python preprocess.py --target supervised
python -m train.dataset --force
python -m train.pretrain
python -m train.train_lora_classifier
```

也可以一次生成全部 JSONL：

```powershell
python preprocess.py --target all
```

注意：`preprocess.py --target all` 会同时尝试生成训练用和最终检测用 JSONL，因此需要对应输入 CSV 都存在。

## 输入输出约定

### 原始流量输入

原始 pcap/pcapng 放在：

```text
data/pcap/raw/
```

截断后输出到：

```text
data/pcap/truncated/
```

`pipeline.extract_flow_features` 会扫描截断目录中的 `*_train.pcap` 和 `*_test.pcap` 文件，并根据文件名生成训练集和测试集特征 CSV。

### 流特征 CSV

输出位置：

```text
data/flow_features/final_multiclass_features_train.csv
data/flow_features/final_multiclass_features_test.csv
data/flow_features/flow_metadata_temporal_train.csv
data/flow_features/flow_metadata_temporal_test.csv
```

核心字段包括：

- 标识字段：`flow_uid`, `src_ip`, `src_port`, `dst_ip`, `dst_port`, `protocol`, `timestamp`
- 数据来源字段：`dataset_source`, `subfolder`, `pcap_filename`
- 标签字段：`label`
- 数值统计特征：包数、字节数、IAT、方向统计、TCP flag、窗口聚合、证书结构等
- 文本事件字段：`zeek_conn_log`, `zeek_ssl_log`, `zeek_x509_log`

### JSONL 中间文件

训练用：

```text
data/train/processed/pretrain_flows.jsonl
data/train/processed/supervised_flows.jsonl
data/train/splits/train.jsonl
data/train/splits/val.jsonl
data/train/splits/test.jsonl
```

最终检测用：

```text
data/pipeline/input/feature_flows.jsonl
```

每行是一条流，典型结构如下：

```json
{
  "flow_uid": "10.42.0.1_12345_8.8.8.8_443_tcp_1497000000.0",
  "src_ip": "10.42.0.1",
  "dst_ip": "8.8.8.8",
  "text": "{\"t\":\"c\",\"proto\":\"tcp\"} {\"t\":\"s\",\"ver\":\"TLSv1.2\"}",
  "label": 0,
  "label_name": "benign",
  "num_events": 2,
  "num_features": {
    "pkts_forward": 3.0,
    "bytes_total": 295.0
  }
}
```

### DeBERTa 特征输出

```text
data/pipeline/features/features_pure.csv
data/pipeline/features/features_fused.csv
```

- `features_pure.csv`：`flow_uid + 768维 DeBERTa [CLS] 特征 + label + label_name`
- `features_fused.csv`：`flow_uid + 768维 DeBERTa [CLS] 特征 + 原始数值流特征 + label + label_name`

### 检测输出

```text
data/pipeline/output/detection_results.csv
data/pipeline/output/detection_assets/
```

`detection_results.csv` 保存每条流的模型预测结果。`detection_assets/` 中保存前端展示用数据，例如：

- `detection_summary.csv`
- `risk_distribution.csv`
- `malicious_details.csv`
- `trend_data.csv`
- `vote_consistency.csv`
- `latent_2d.csv`

## 目录结构

```text
.
├── main.py                         # 最终检测 pipeline 入口，按阶段调度截断、特征提取、预处理、表征提取和检测器
├── preprocess.py                   # CSV -> JSONL 预处理脚本，构造 DeBERTa 可读的流文本和数值特征字典
├── pyproject.toml                  # Python 项目依赖与 uv 配置
├── README.md                       # 项目说明文档
├── pipeline/                       # 最终检测主流程代码
│   ├── config.py                   # pipeline 路径配置，集中定义 data、models、checkpoints、输出目录
│   ├── truncate_pcap.py            # pcap/pcapng 截断工具，按双向流保留前 N 个包
│   ├── extract_flow_features.py    # 从截断后的 pcap 中提取流统计特征、TLS/X509 文本字段和时序元数据
│   ├── LLM_extract_features.py     # 加载 DeBERTa checkpoint + LoRA adapter，提取 768 维 [CLS] 特征
│   ├── supcon_ae.py                # 可选 SupCon-AE 推理降维模块，输出 reduced_features.csv
│   └── detector.py                 # 传统机器学习检测器推理脚本，输出检测结果和前端可视化数据
├── train/                          # DeBERTa 继续预训练与 LoRA 监督训练代码
│   ├── config.py                   # 训练路径、超参数、标签映射和数值特征列配置
│   ├── split_input_csv.py          # 将训练流特征 CSV 分成 RTD 预训练输入和 LoRA 监督训练输入
│   ├── dataset.py                  # JSONL 数据加载、train/val/test 固定划分和 PyTorch Dataset 封装
│   ├── pretrain.py                 # DeBERTa-v3 RTD/ELECTRA-style 继续预训练
│   └── train_lora_classifier.py    # LoRA 分类器监督训练，用于塑造 encoder 表征空间
├── data/                           # 数据工作区，存放原始流量、中间文件和最终输出
│   ├── pcap/                       # pcap/pcapng 流量文件目录
│   │   ├── raw/                    # 原始 pcap/pcapng 输入
│   │   └── truncated/              # 截断后的 pcap/pcapng 输出
│   ├── flow_features/              # 流统计特征 CSV 和时序元数据 CSV
│   ├── train/                      # 离线训练数据目录
│   │   ├── input/                  # 训练输入 CSV，如 pretrain_flows.csv、supervised_flows.csv
│   │   ├── processed/              # 训练用 JSONL，如 pretrain_flows.jsonl、supervised_flows.jsonl
│   │   └── splits/                 # LoRA 监督训练固定划分 train/val/test JSONL
│   └── pipeline/                   # 最终检测流程的数据目录
│       ├── input/                  # 最终特征提取输入 JSONL
│       ├── features/               # DeBERTa 特征、融合特征和可选降维特征输出
│       └── output/                 # 检测结果 CSV 与前端可视化数据
├── models/                         # 本地模型和检测器资产目录
│   ├── deberta-v3-base/            # 当前主线使用的 DeBERTa-v3-base 本地基座模型
│   ├── deberta-base/               # 备用/历史 DeBERTa base 模型
│   └── detector/                   # 检测器模型文件，如 scaler、RandomForest、ExtraTrees、LightGBM、XGBoost
├── checkpoints/                    # 训练产物和中间模型目录
│   ├── pretrain/                   # RTD 继续预训练后的 DeBERTa encoder checkpoint
│   ├── lora/                       # LoRA adapter、分类头和 tokenizer 文件
│   └── supcon_ae/                  # SupCon-AE 降维模型 checkpoint
├── docs/                           # 项目设计文档、赛题附件和历史方案说明
└── legacy/                         # 历史实验/交付代码，部分脚本包含旧路径，仅作参考
```

## 模块说明

### `main.py`

最终检测 pipeline 的入口。支持按阶段运行，也支持一键运行完整流程。

### `preprocess.py`

将流特征 CSV 转换为模型可读的 JSONL。它会：

1. 校验输入 CSV 必需字段。
2. 解析 `zeek_conn_log`、`zeek_ssl_log`、`zeek_x509_log`。
3. 压缩成紧凑 JSON 文本序列。
4. 映射 8 分类标签。
5. 提取配置中声明的数值特征。
6. 写出训练或推理所需 JSONL。

### `pipeline/truncate_pcap.py`

纯 Python pcap/pcapng 截断工具。按双向流保留前 `DEFAULT_MAX_PKTS_PER_FLOW` 个包，默认值为 200。

### `pipeline/extract_flow_features.py`

从 pcap/pcapng 中提取双向流特征，输出多分类特征 CSV 和时序元数据 CSV。该模块包含：

- pcap/pcapng 解析器
- IPv4 TCP/UDP 流识别
- 包长、字节数、方向、IAT、active/idle 统计
- TCP flag 统计
- TLS/X509 轻量提取
- Zeek 风格日志字段构造
- 多进程单文件处理和临时 CSV 合并

### `pipeline/LLM_extract_features.py`

加载 RTD 继续预训练 checkpoint 和 LoRA adapter，为每条流提取 768 维 `[CLS]` 表征，并输出纯特征和融合特征 CSV。

### `pipeline/supcon_ae.py`

可选 SupCon-AE 推理降维模块。默认读取 `features_fused.csv`，输出 `reduced_features.csv`。

### `pipeline/detector.py`

加载传统机器学习检测器，对 `features_fused.csv` 中的特征执行推理，并生成检测结果和前端可视化数据。

### `train/pretrain.py`

实现 DeBERTa-v3 的 RTD/ELECTRA-style 继续预训练：

1. 构造 generator 和 discriminator。
2. 随机 mask 普通 token。
3. generator 预测替换 token。
4. discriminator 判断 token 是否被替换。
5. 保存继续预训练后的 discriminator encoder。

### `train/train_lora_classifier.py`

在有标签流量文本上训练 LoRA 分类器。该分类器主要用于塑造 encoder 表征空间，最终检测仍由后续传统模型完成。

## 常见注意事项

1. `models/`、`checkpoints/`、`data/` 体积较大，默认不适合提交到 Git。
2. `legacy/` 是历史实验和交付代码，部分脚本包含硬编码旧路径，主线运行请优先使用根目录、`pipeline/` 和 `train/`。
3. `docs/GPT生成任务二数据提示词.md` 是早期二分类模拟数据说明，和当前 8 分类、94 列流特征格式不完全一致。
4. LoRA 训练和特征提取依赖 `checkpoints/pretrain/checkpoint-epoch*/` 与 `checkpoints/lora/best/`。
5. 如果只想测试代码路径，可以在部分函数中传入小样本或显式使用 `allow_no_lora=True` 做 smoke test。
6. 训练脚本默认 batch size 较小，优先保证显存和数值稳定；如果显存充足，可以在 `train/config.py` 中调整。

## Git 忽略策略

`.gitignore` 默认忽略：

- 虚拟环境和缓存：`.venv/`, `__pycache__/`, `.pytest_cache/`
- 本地环境：`.env`
- 大模型和训练产物：`models/`, `checkpoints/`, `*.safetensors`, `*.bin`, `*.pt`
- 数据和中间产物：`data/`, `*.csv`, `*.jsonl`, `*.pcap`, `*.pcapng`
- 文档和实验输出：`docs/`, `*.pdf`, `*.docx`, `*.png`

如需提交示例数据或说明文档，请单独调整忽略规则或放入专门的轻量样例目录。
