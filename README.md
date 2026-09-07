# MFF-LightGBM：异常加密流量八分类系统

这是一个在**不解密应用正文**的前提下，对 PCAP/PCAPNG 中的网络连接进行八分类检测的项目。

系统把每条双向流表示为两类信息：

- 80 维流量统计、TCP、TLS 和 X.509 数值特征；
- 由连接、TLS、证书结构化事件序列产生的 DeBERTa-v3 语义特征。

语义特征经过 SupCon-AE 压缩后与 80 维数值特征融合，最终由 LightGBM 输出八分类结果。

> 当前状态：源码重构和严格解析器已经完成，旧代码与旧模型已经归档；公开数据集仍在下载，因此兼容新特征契约的生产模型尚未重新训练。用户页面可以启动，但上传检测会在模型发布前明确提示“检测模型尚未就绪”。

---

## 1. 第一次回来时，按什么顺序阅读

如果已经忘记了项目，不建议从训练代码一头扎进去。按下面顺序读最快：

1. 先读本 README 的“项目全貌”和“完整目录树”，建立整体地图。
2. 读 [`docs/系统复习文档.md`](docs/系统复习文档.md)，补回流、TLS、X.509、CLS、LoRA、SupCon-AE 等概念。
3. 读 [`user_app/inference/contract.py`](user_app/inference/contract.py)，明确八个标签和 80 个特征的唯一契约。
4. 依次读抓包解析主链：`pcap_reader.py → tcp_reassembly.py → tls_parser.py → x509_parser.py → extract_flow_features.py`。
5. 再读模型主链：`serialize_flow.py → semantic_encoder.py → supcon_reducer.py → lightgbm_detector.py`。
6. 读 [`developer/pipeline.py`](developer/pipeline.py)，理解开发者如何串起数据、训练与评估。
7. 读 [`user_app/inference/pipeline.py`](user_app/inference/pipeline.py)，理解用户上传后如何只做推理。
8. 最后读后端和前端；它们只负责提交用户任务并展示结果。

如果只想启动或继续训练，直接看本文第 8～13 节即可。

---

## 2. 项目全貌：只有两条正式通道

### 2.1 开发者通道

```text
公开数据集下载与解压
→ 来源清单、SHA-256 和标签核实
→ 按 PCAP / APK / sample / capture 分组
→ 固定 train / validation / test
→ 统一特征提取
→ RTD 领域继续预训练
→ LoRA 八分类训练
→ 提取 768 维 CLS
→ SupCon-AE 压缩为 64 维
→ 与 80 维数值特征融合
→ LightGBM 训练
→ 独立测试、消融实验与资源测试
→ 校验并发布冻结模型
```

对应目录是 `developer/`、`data/developer/` 和 `models/experiments/`。

### 2.2 用户通道

```text
上传 PCAP/PCAPNG
→ 等待检测
→ 查看风险概览、逐流结果和 TLS 详情
```

用户通道不训练、不调参、不读取开发者评估结果，也不会把用户上传自动加入训练集。对应目录是 `user_app/` 和 `data/runtime/`。

### 2.3 依赖方向

```text
developer ──────→ user_app.inference（复用唯一的解析与特征实现）
user_app  ──X──→ developer（禁止反向依赖）
正式源码 ──X──→ legacy（禁止使用旧代码）
```

离线训练和在线检测共同调用 `extract_pcap()`，不会出现“训练时一套特征、用户上传时另一套特征”。这些边界由 `tests/test_architecture.py` 自动检查。

---

## 3. 八分类任务

| ID | 标签 | 中文含义 | 典型行为 |
|---:|---|---|---|
| 0 | `benign` | 正常流量 | 正常浏览、应用通信、合法 DNS/HTTPS 等 |
| 1 | `adware` | 广告软件流量 | 高频广告、追踪、推广服务器通信 |
| 2 | `dns2tcp` | dns2tcp 隧道 | 把其他协议的数据封装进 DNS 查询与响应 |
| 3 | `dnscat2` | dnscat2 隧道 | 通过 DNS 建立隐蔽命令控制或数据通道 |
| 4 | `iodine` | iodine 隧道 | 使用 DNS 承载 IP 数据形成隐蔽隧道 |
| 5 | `ransomware` | 勒索软件流量 | 与勒索软件基础设施或控制端相关的通信 |
| 6 | `scareware` | 恐吓软件流量 | 虚假告警、诱导付费或伪安全软件通信 |
| 7 | `smsmalware` | 短信恶意软件流量 | 与恶意短信发送、窃取或控制行为相关的通信 |

标签顺序不能随意调整，因为它会同时写入 LoRA 分类头、LightGBM 标签和生产模型契约。

---

## 4. 完整逻辑目录树

下面列出项目中需要维护和理解的文件。`.venv/`、`__pycache__/`、模型权重、下载中的大型压缩包以及自动生成的 CSV/图片不会逐个展开，但其存放位置和用途均已列出。

```text
E:\Work\ccb
├─ README.md                              项目接管、复习和运行总入口
├─ pyproject.toml                         Python 版本、依赖、pytest 与 uv 配置
├─ uv.lock                                uv 锁定的精确依赖版本
├─ .python-version                        本机默认 Python 版本提示
├─ .gitignore                             忽略数据、权重、缓存和生成产物
│
├─ developer/                             面向开发者：数据、制模、评估、发布
│  ├─ __init__.py                         Python 包声明
│  ├─ README.md                           开发者通道简要说明
│  ├─ config.py                           开发数据、实验模型和报告的路径定义
│  ├─ pipeline.py                         开发者离线流水线统一入口
│  ├─ data_prepare/
│  │  ├─ __init__.py                     子包声明
│  │  ├─ build_manifest.py               扫描 source，记录来源、类型与 SHA-256
│  │  ├─ map_labels.py                   把外部标签规范到项目八分类
│  │  ├─ split_dataset.py                按来源组划分并建立 prepared 硬链接
│  │  ├─ group_split.py                  校验 group_id 不跨三个集合
│  │  └─ batch_extract.py                批量调用唯一 extract_pcap()
│  ├─ representation/
│  │  ├─ __init__.py                     子包声明
│  │  ├─ config.py                       DeBERTa、RTD、LoRA 路径和超参数
│  │  ├─ split_input_csv.py              生成预训练与监督训练输入表
│  │  ├─ dataset.py                      加载固定划分并构造 DataLoader
│  │  ├─ pretrain.py                     DeBERTa-v3 RTD 领域继续预训练
│  │  └─ train_lora.py                   LoRA 八分类训练和最佳适配器保存
│  ├─ detector/
│  │  ├─ __init__.py                     子包声明
│  │  ├─ train_supcon.py                 训练 SupCon-AE，保存 reducer.pt
│  │  ├─ train_lightgbm.py               训练 LightGBM 并生成测试报告
│  │  └─ pca_baseline.py                 PCA 对照实验，不进入正式推理链
│  ├─ evaluation/
│  │  ├─ __init__.py                     子包声明
│  │  ├─ evaluate.py                     从逐流预测计算分类指标
│  │  ├─ feature_ablation.py             三组特征消融实验
│  │  ├─ resource_benchmark.py            耗时、吞吐、内存和显存测试
│  │  └─ compare_versions.py              比较多个候选发布的 metrics.json
│  └─ release/
│     ├─ __init__.py                     子包声明
│     ├─ build_contract.py               生成特征、标签和发布清单
│     ├─ validate_bundle.py              校验文件、字段和标签契约
│     └─ publish.py                      不可覆盖地发布并可激活模型
│
├─ user_app/                              面向用户：上传、推理、展示
│  ├─ __init__.py                         Python 包声明
│  ├─ README.md                           用户通道简要说明
│  ├─ inference/
│  │  ├─ __init__.py                     子包声明
│  │  ├─ config.py                       用户运行时与生产模型路径
│  │  ├─ contract.py                     八个标签、80 维特征和契约标识
│  │  ├─ pcap_reader.py                  Scapy 读取抓包、链路层和 IP 层
│  │  ├─ tcp_reassembly.py               TCP 排序、去重、重传与缺口处理
│  │  ├─ tls_parser.py                   TLS record、Hello、SNI、ALPN 等
│  │  ├─ x509_parser.py                  用 cryptography 解析 DER 证书
│  │  ├─ extract_flow_features.py        唯一双向流和 80 维特征实现
│  │  ├─ serialize_flow.py               结构化事件转 DeBERTa 输入文本
│  │  ├─ flow_io.py                      读取和检查流 JSONL
│  │  ├─ semantic_encoder.py             Encoder/LoRA 与 768 维 CLS
│  │  ├─ supcon_reducer.py               加载 reducer.pt，压缩至 64 维
│  │  ├─ lightgbm_detector.py            加载 LightGBM 和训练期预处理量
│  │  ├─ model_loader.py                 解析并校验生产模型包
│  │  └─ pipeline.py                     单个 PCAP 到最终结果的总入口
│  ├─ backend/
│  │  ├─ __init__.py                     子包声明
│  │  ├─ server.py                       FastAPI、上传、任务、结果和 SSE API
│  │  ├─ orchestrator.py                 排队、运行、取消和进度转发
│  │  ├─ pipeline_runner.py              启动推理子进程并生成用户进度
│  │  ├─ tasks_store.py                  SQLite 任务状态读写
│  │  ├─ data_provider.py                只读取指定用户任务的结果
│  │  └─ tls_log_parser.py               整理 TLS/X.509 详情数据
│  └─ frontend/
│     ├─ index.html                      首页、服务状态和最近检测
│     ├─ upload.html                     抓包上传和四步用户进度
│     ├─ detection.html                  风险概览与逐流结果
│     ├─ alarms.html                     历史检测记录
│     ├─ tls-analysis.html               指定任务的 TLS/证书详情
│     └─ shared/
│        ├─ real-data.js                 API、SSE、标签和公共视图函数
│        └─ styles.css                   用户界面公共样式
│
├─ data/                                  开发数据与用户数据严格分离
│  ├─ README.txt                          data 总规则
│  ├─ developer/
│  │  ├─ datasets/
│  │  │  ├─ source/
│  │  │  │  ├─ readme.txt                官网原件区总说明
│  │  │  │  ├─ CIC-AndMal2017/
│  │  │  │  │  ├─ archives/readme.txt    原始压缩包
│  │  │  │  │  └─ extracted/readme.txt   原样解压内容
│  │  │  │  └─ CIRA-CIC-DoHBrw-2020/
│  │  │  │     ├─ archives/readme.txt    原始压缩包
│  │  │  │     └─ extracted/readme.txt   原样解压内容
│  │  │  ├─ manifests/readme.txt          哈希、标签、group_id、split
│  │  │  └─ prepared/
│  │  │     ├─ train/readme.txt           固定训练组 PCAP 硬链接
│  │  │     ├─ validation/readme.txt      固定验证组 PCAP 硬链接
│  │  │     └─ test/readme.txt            最终独立测试组
│  │  └─ workspace/
│  │     ├─ features/
│  │     │  ├─ flow/readme.txt            80 维特征与时序元数据
│  │     │  ├─ semantic/readme.txt        768 维 CLS 特征
│  │     │  └─ fused/readme.txt           融合和 64 维降维特征
│  │     ├─ corpus/
│  │     │  ├─ pretrain/readme.txt        仅 train 组的 RTD 语料
│  │     │  └─ supervised/splits/readme.txt  固定三集合 JSONL
│  │     └─ evaluation/
│  │        ├─ figures/readme.txt          混淆矩阵、ROC 和指标图
│  │        ├─ predictions/                独立测试逐流预测和展示资产
│  │        ├─ reports/                    分类报告和消融报告
│  │        └─ benchmarks/                 时间、内存、显存和吞吐报告
│  └─ runtime/
│     ├─ uploads/readme.txt                用户上传短暂暂存区
│     ├─ tasks/readme.txt                  每个任务的输入、预测和摘要
│     └─ state/readme.txt                  tasks.db 等服务状态
│
├─ models/
│  ├─ README.md                            模型目录规则
│  ├─ base/
│  │  ├─ deberta-v3-base/                  当前训练起点
│  │  └─ deberta-base/                     历史下载，不是当前默认
│  ├─ experiments/current/readme.txt        当前候选实验输出位置
│  └─ production/
│     ├─ active.json                       当前激活的 release_id
│     └─ <release_id>/
│        ├─ encoder/                       RTD Encoder
│        ├─ lora/                          最佳 LoRA 适配器
│        ├─ supcon/reducer.pt              SupCon-AE 及输入列契约
│        ├─ detector/                      LightGBM、列顺序和中位数
│        ├─ manifest.json                  发布身份和状态
│        ├─ feature_schema.json            特征契约
│        └─ label_mapping.json             标签顺序
│
├─ docs/
│  ├─ 项目结构.md                          两条通道的简明结构说明
│  ├─ 模型训练说明.md                      数据准备、训练和发布命令
│  ├─ 系统复习文档.md                      概念和源码阅读顺序
│  ├─ archive/
│  │  ├─ 任务二实施指南.md                 早期任务二的实施记录
│  │  ├─ 预训练问题分析.md                 早期预训练故障与判断记录
│  │  ├─ GPT生成任务二数据提示词.md        早期数据生成提示词，仅追溯
│  │  └─ 最初思路.docx                    项目最初设计思路
│  ├─ assets/
│  │  ├─ dashboard_reference.png           早期仪表盘参考图
│  │  ├─ dashboards_discover.png           早期界面探索图
│  │  └─ dashboards_discover_table.png     早期表格界面探索图
│  └─ reports/
│     ├─ 1780056556282726.pdf              原项目报告 PDF
│     └─ 成都信息工程大学-欧鲁金-基于多维特征的异常加密流量检测系统-作品报告.docx
│                                            原项目作品报告 Word 文档
│
├─ tests/
│  ├─ test_architecture.py                依赖边界和目录结构
│  ├─ unit/
│  │  ├─ test_tcp_reassembly.py           乱序、重传、重叠、缺口
│  │  ├─ test_tls_parser.py               ClientHello 与截断 record
│  │  ├─ test_x509_parser.py              真实 DER 与畸形证书
│  │  └─ test_group_split.py              数据泄漏拦截
│  ├─ integration/
│  │  ├─ test_pcap_extractor.py           抓包格式、链路层和 IP 分片
│  │  └─ test_production_bundle.py        未发布时失败关闭
│  ├─ developer/
│  │  ├─ test_data_prepare.py             清单、标签和确定性划分
│  │  └─ test_supcon_training.py          SupCon 保存、加载和列契约
│  └─ backend/
│     ├─ test_data_provider.py             任务结果和越界保护
│     ├─ test_orchestrator.py              任务提交与取消
│     ├─ test_pipeline_runner.py           推理子进程和进度事件
│     ├─ test_tasks_store.py               SQLite 任务记录
│     └─ test_tls_log_parser.py            TLS 详情整理
│
└─ legacy/                                遗产区，不参与正式运行
   ├─ README.md                            遗产边界说明
   ├─ code/
   │  ├─ extract_flow_features_legacy.py   重构前的近似特征提取器
   │  └─ truncate_flow_legacy.py           重构前的逐流截断器
   ├─ models/production/v1/               不兼容的旧生产模型
   ├─ docs/
   │  ├─ README_legacy.md                  重构前的根说明
   │  ├─ 系统复习文档_旧实现.md            旧系统复习说明
   │  └─ MFF-LightGBM项目源码解析长文_旧实现.md  旧源码长文
   ├─ raw_data_prepare/
   │  ├─ csv_preprocessor.py               早期 CSV 预处理
   │  ├─ pcap.py                           早期抓包处理
   │  └─ pcap -200.py                      早期前 200 包处理
   ├─ processed_csv/                      早期 CSV 产物
   ├─ data/                               其他旧数据产物
   └─ final/                              早期比赛/交付副本
      ├─ extract_features.py               早期语义特征提取
      ├─ feature show.py                   早期特征展示
      ├─ LightGBM_Detector.py              早期检测器入口
      ├─ pcap -200.py                      早期截断脚本副本
      ├─ preprocess.py                     早期预处理副本
      ├─ supcon_ae.py                      早期 SupCon-AE 副本
      ├─ readme.txt                        早期交付说明
      ├─ LightGBM_Detector/                早期检测代码、指标与图
      ├─ supcon_ae/                        早期 SupCon 代码、权重与图
      └─ data/csv/                         早期交付数据
```

`__init__.py` 通常没有业务逻辑，只用于声明 Python 包。每个数据叶子目录中的 `readme.txt` 是该目录允许存放什么的最终说明。

---

## 5. 抓包解析链做了什么

### 5.1 PCAP/PCAPNG 与网络层

`pcap_reader.py` 使用 Scapy 读取抓包，而不是自己猜文件头。当前覆盖：

- PCAP 与 PCAPNG；
- 大端、小端以及微秒/纳秒时间戳；
- Ethernet、VLAN、Linux SLL、SLL2；
- IPv4、IPv6、TCP、UDP；
- IPv4/IPv6 分片重组。

### 5.2 双向流与 TCP 重组

同一 TCP/UDP 会话的两个方向合并成一条双向流。TCP 根据 sequence number 恢复乱序、去掉重复重传、裁剪重叠，并在捕获缺口处分段。它允许一个 TLS record 跨多个 TCP 包，也允许一个 TCP 包携带多个 TLS record。

包长、IAT、Active/Idle 等统计量使用 Welford 单遍在线算法更新均值与方差，不需要把整条流的数值再复制一份到内存；双向总量和两个方向仍分别统计。

### 5.3 TLS 与 X.509

TLS 识别基于二进制 record/handshake，不依赖端口 443。当前解析 ClientHello、ServerHello、SNI、ALPN、supported_versions、cipher suites、extensions 和可见的 Certificate。

证书 DER 交给 `cryptography.x509.load_der_x509_certificate()`，提取 Subject/CN、SAN、Issuer、Serial Number、有效期、签名算法、公钥和 SHA-256 指纹。

TLS 1.3 的 ServerHello 后通常进入加密握手。没有会话密钥时看不到证书是正常情况，代码记录为缺失，不会伪造字段。SNI 是客户端想访问的主机名；CN/SAN 是证书声明的名称；Issuer 是签发者，三者不能混用。

### 5.4 缺失值语义

- `0`：确实观测到零；
- `NaN/null`：没有观测到、无法解析或无法计算；
- 不会为了填满表格而“近似造字段”。

当前特征契约是 `strict-tls-x509-2026-09`。模型声明的契约不同，用户推理就拒绝加载。

---

## 6. 模型链怎么理解

### 6.1 结构化文本和 CLS

每条流的连接、TLS 和 X.509 信息先压缩为结构化事件序列。它不是让模型写文章，而是让编码器学习字段组合关系。

`[CLS]` 是序列开头的特殊标记。经过 Transformer 后，它的隐藏向量可视为整条输入的摘要。本项目提取 768 维 CLS 作为流的语义特征。

### 6.2 为什么是 DeBERTa-v3-base，不是生成式模型

这里需要固定类别判别和稳定向量，不需要生成文本。编码式模型更适合分类与特征提取，训练、显存和推理成本也更可控。

### 6.3 RTD、LoRA、SupCon-AE 和 LightGBM

- RTD：用流量语料继续预训练，让 Encoder 熟悉 TLS、连接和证书字段分布。
- LoRA：只训练少量低秩参数完成八分类适配。
- SupCon-AE：把 768 维语义特征压缩为 64 维，使同类靠近、异类分离。
- LightGBM：把 64 维语义特征与 80 维数值特征融合后完成八分类。

语义分支是否真正有效必须由 `fused / semantic-only / manual-only` 消融证明，不能只凭模型结构下结论。

---

## 7. 为什么不用 Zeek

项目目标环境是 Windows，因此正式链采用：

- Scapy：抓包、链路层、IP/TCP/UDP；
- 自有严格实现：双向流、TCP 重组和 TLS handshake；
- cryptography：X.509 DER。

不用 Zeek 是可行的，但意味着边界条件必须由测试保证。不要把 `legacy/` 中的近似解析器搬回正式代码。

---

## 8. 环境准备

要求：Windows、Python 3.12、`uv`。训练 DeBERTa 建议使用 NVIDIA GPU；只运行解析器和测试不要求 GPU。

```powershell
cd E:\Work\ccb
uv sync
uv run pytest -q
```

回归测试数量会随功能增加，以本机 `uv run pytest -q` 的实时结果为准，不在文档里保存容易过期的数字。

数据和模型很大，默认被 `.gitignore` 排除。代码、契约、目录说明和测试才应提交 Git。

---

## 9. 数据集下载位置

当前优先使用：

1. CIC-AndMal2017：`benign`、`adware`、`ransomware`、`scareware`、`smsmalware` 的主要来源。
2. CIRA-CIC-DoHBrw-2020：`dns2tcp`、`dnscat2`、`iodine` 及相应正常场景。

压缩包放到：

```text
E:\Work\ccb\data\developer\datasets\source\CIC-AndMal2017\archives\
E:\Work\ccb\data\developer\datasets\source\CIRA-CIC-DoHBrw-2020\archives\
```

原样解压到对应的：

```text
E:\Work\ccb\data\developer\datasets\source\<数据集>\extracted\
```

规则：

- 正式端到端训练需要 PCAP/PCAPNG；CSV 用于辅助核对标签，不能替代抓包重新提取当前特征。
- 保留官方目录层级与文件名，不要混合两个数据集。
- 不要放到旧的 `data/pcap/raw`。
- 用户上传不能放进 `data/developer`。
- 下载未完成时不要建立清单或训练，避免扫描到半个压缩包。

---

## 10. 下载完成后的数据准备

### 10.1 建来源清单

```powershell
uv run python -m developer.data_prepare.build_manifest `
  --provenance data/developer/datasets/manifests/provenance.json `
  --output data/developer/datasets/manifests/source_manifest.json
```

清单需要核实路径、数据集、SHA-256 和八分类标签；尽可能补充 `sample_id`、`capture_id`、`apk_sha256`、family、设备、捕获批次、DoH 工具和 resolver。`provenance.json` 以 source 相对路径为键；需要整体划分的多个文件填写相同 `group_id`，并用 `group_basis` 记录依据。不能自动判断的标签必须人工核实。

### 10.2 固定分组划分

```powershell
uv run python -m developer.data_prepare.split_dataset `
  data/developer/datasets/manifests/source_manifest.json `
  data/developer/datasets/manifests/split_manifest.json `
  --materialize-root data/developer/datasets/prepared
```

分组优先级：

```text
sample_id → capture_id → apk_sha256 → 文件 SHA-256
```

程序会在每个“数据集 + 类别”内按完整来源组做 70/10/20 分配。同组只能位于 train、validation、test 之一，每层少于三个独立组会拒绝划分。`prepared/` 尽量使用同盘硬链接，不复制第二份巨大 PCAP。

自动字段只能提供保底分组；如果同一 APK、恶意软件 family、捕获批次或 DoH 场景横跨多个 PCAP，应在 `provenance.json` 中为它们设置共同的显式 `group_id`，以数据集说明为准。

禁止把全部 flow 随机打散后再切分；相同 PCAP、APK 或实验场景的流高度相似，这会造成数据泄漏和虚高指标。

---

## 11. 开发者训练命令

第一次建议逐阶段执行：

```powershell
# prepared PCAP → 80 维流特征和结构化日志
uv run python -m developer.pipeline --stage flow_features

# 流特征 → RTD、LoRA、CLS 输入 JSONL
uv run python -m developer.pipeline --stage preprocess

# DeBERTa-v3 RTD 领域继续预训练
uv run python -m developer.pipeline --stage pretrain

# LoRA 八分类训练
uv run python -m developer.pipeline --stage lora

# 提取 768 维 CLS 并与数值特征对齐
uv run python -m developer.pipeline --stage extract

# SupCon-AE，输出 models/experiments/current/supcon/reducer.pt
uv run python -m developer.pipeline --stage supcon

# LightGBM 与固定 test 结果
uv run python -m developer.pipeline --stage detector

# fused / semantic-only / manual-only 消融
uv run python -m developer.pipeline --stage evaluate
```

下面命令运行默认训练主链，但**不包含额外的 `evaluate` 消融阶段**：

```powershell
uv run python -m developer.pipeline --stage all
```

训练前检查 [`developer/representation/config.py`](developer/representation/config.py) 中的 batch size、epoch、学习率和显存参数。当前默认值偏向低显存 Windows 环境，不代表最终最优参数。

---

## 12. 评估与发布

### 12.1 最低评估要求

- 独立 test 的 accuracy、macro F1、weighted F1；
- 每类 precision、recall、F1、support；
- 混淆矩阵和 one-vs-rest ROC/AUC；
- 三组特征消融；
- 按数据来源观察泛化表现；
- 解析吞吐、推理延迟、内存与显存；
- 三个集合的 `group_id` 无交集。

### 12.2 完整模型包

```text
encoder/
lora/
supcon/reducer.pt
detector/best_lgb_model.pkl
detector/feature_columns.json
detector/imputation_medians.json
manifest.json
feature_schema.json
label_mapping.json
```

`imputation_medians.json` 只能由 train 拟合，在线不能拿用户当前上传重新计算。特征列、字段顺序和标签顺序必须与运行时契约一致。

### 12.3 构建、校验和发布

```powershell
uv run python -m developer.release.build_contract `
  models/experiments/<run_id> 2026-09-strict-parser

uv run python -m developer.release.validate_bundle `
  models/experiments/<run_id>

uv run python -m developer.release.publish `
  models/experiments/<run_id> 2026-09-strict-parser --activate
```

发布标识只能含字母、数字、点、下划线和连字符。已发布目录不可覆盖。

当前 [`models/production/active.json`](models/production/active.json) 是：

```json
{
  "release_id": "unpublished"
}
```

这是有意的失败关闭：旧模型由旧近似字段训练，不能读取当前严格特征。

---

## 13. 用户系统

兼容模型发布并激活后启动：

```powershell
uv run uvicorn user_app.backend.server:app
```

访问 `http://127.0.0.1:8000/`。

用户页面只显示：读取抓包、分析连接、识别风险、生成结果。不存在训练入口、开发者评估接口或 demo 回放。

也可直接调用冻结推理：

```powershell
uv run python -m user_app.inference.pipeline `
  --pcap <抓包文件> `
  --output-dir <任务目录>
```

主要 API：

```text
GET  /api/status
POST /api/analyze
GET  /api/tasks
GET  /api/tasks/{task_id}
GET  /api/tasks/{task_id}/stream
GET  /api/tasks/{task_id}/summary
GET  /api/tasks/{task_id}/predictions
GET  /api/tasks/{task_id}/flows
GET  /api/tasks/{task_id}/flows/{flow_uid}
POST /api/tasks/{task_id}/cancel
GET  /api/labels
```

---

## 14. 用户任务产物

任务目录位于 `data/runtime/tasks/<task_id>/`，典型内容如下：

```text
input.pcap / input.pcapng       上传抓包
flow_features.csv               流级数值特征和结构化日志
flow_temporal.csv               时序/窗口统计辅助信息
flows.jsonl                     DeBERTa 逐流事件序列
predictions.csv                 逐流类别和置信度
summary.json                    文件级检测摘要
readme.txt                      任务目录说明
```

用户推理中的 768 维 CLS 和 64 维降维特征在内存中流转，不额外长期保存一份语义特征 CSV；开发者离线流程才会把它们写入 `data/developer/workspace/features/`。

实际文件名以 `user_app/inference/pipeline.py` 为准。用户任务不会进入开发者训练集。

---

## 15. 测试

```powershell
uv run pytest -q
uv run pytest tests/unit -q
uv run pytest tests/integration -q
uv run pytest tests/developer -q
uv run pytest tests/backend -q
uv run pytest tests/test_architecture.py -q
```

测试验证：TCP 乱序和重传、捕获缺口、非 443 TLS、截断 record、畸形证书、多种抓包/链路格式、分组泄漏、目录越界、开发/用户边界，以及未发布模型时的失败关闭。

---

## 16. 必须守住的规则

1. **只有一份特征提取器。** 不为训练和 Web 各复制一份。
2. **不恢复抓包前 N 包截断。** 它会破坏 TCP/TLS 重组。
3. **不按端口判断 TLS。** 443 不一定是 TLS，TLS 也不限于 443。
4. **不伪造 X.509。** 看不到或解析失败就是缺失。
5. **不随机拆 flow。** 必须按 PCAP/APK/sample/capture 分组。
6. **预训练不能看 test。** RTD 也只能使用 train 组。
7. **预处理量只在 train 拟合。** 包括中位数、PCA 和其他统计量。
8. **用户上传不自动训练。** 两类数据物理隔离。
9. **生产模型不可覆盖。** 新发布使用新的 release_id。
10. **legacy 只追溯。** 旧指标和行为不代表当前系统。

---

## 17. 常见问题

### 为什么现在页面不能完成检测？

旧模型已经归档，新数据尚未完成下载和训练。让旧模型读取新特征会得到无意义结果，因此系统明确拒绝，而不是假装可用。

### `encrypted_traffic_detection.egg-info` 是什么？

它是 Python 安装或构建工具生成的包元数据，不是业务源码。当前 `uv` 配置为 `package = false`，该目录不应属于项目结构，`.gitignore` 也会忽略 `*.egg-info/`。

### 为什么没有 `src/mff_lightgbm/`？

当前项目不是准备发布到 PyPI 的通用库。直接使用 `developer/` 和 `user_app/` 更直观地表达开发者通道与用户通道。

### 为什么没有 `scripts/`？

稳定入口已经是 Python 模块命令。暂时不维护一批 `.ps1` 包装脚本；真正出现重复、稳定的部署操作时再增加。

### 外部 CSV 能不能直接训练？

如果字段与当前 80 维特征契约不完全一致，就不能直接训练当前 LightGBM。CSV 可用于研究或标签核对，但正式端到端模型应从 PCAP 经唯一提取器重新生成特征。

### TLS 1.3 为什么经常没有证书？

ServerHello 后通常已经进入加密握手。没有会话密钥时无法看到 Certificate 是正常结果，不一定是解析错误。

---

## 18. 当前待办

1. 等两个数据集完整下载。
2. 校验压缩包并原样解压。
3. 建立 `source_manifest.json`。
4. 人工核对八类标签和来源分组。
5. 生成固定 `split_manifest.json` 和 prepared 硬链接。
6. 分阶段训练并检查每步输出。
7. 在固定 test 上做指标、消融和资源测试。
8. 构建模型契约，校验并发布 production release。
9. 完成真实用户上传的端到端验收。

专项说明：

- [`docs/项目结构.md`](docs/项目结构.md)
- [`docs/模型训练说明.md`](docs/模型训练说明.md)
- [`docs/系统复习文档.md`](docs/系统复习文档.md)
- [`developer/README.md`](developer/README.md)
- [`user_app/README.md`](user_app/README.md)
- [`models/README.md`](models/README.md)
- [`legacy/README.md`](legacy/README.md)
