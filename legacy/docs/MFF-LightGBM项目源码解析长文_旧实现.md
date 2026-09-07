# MFF-LightGBM：从 PCAP 到 Web 可视化的异常加密流量检测系统源码解析

> 目录更新说明（2026-09-06）：本文的算法讲解仍可阅读，但旧的 `main.py`、`pipeline/`、`LLM_train/`、`app/` 路径已经统一迁移到 `src/mff_lightgbm`。运行命令和最新文件位置以 [`项目目录结构.md`](项目目录结构.md) 为准。

> 基于 DeBERTa-v3 与 LightGBM 的恶意加密流量检测算法——在不解密通信内容的前提下，融合流量统计特征与 TLS/X509 行为语义特征，实现正常流量、广告软件、DNS 隧道、勒索软件等八类加密流量检测。

## 前言

如今 HTTPS、TLS 等加密协议已经成为网络通信的默认选择。加密保护了用户隐私，也让传统依赖载荷关键字、明文协议字段和内容规则的检测方法受到限制。更棘手的是，恶意软件同样可以使用 TLS 隐藏命令控制通信，DNS 隧道可以把数据编码到域名中，勒索软件也可能通过加密连接与外部服务器交互。

这并不意味着加密流量完全不可分析。即使不解密应用层内容，我们仍然能够观察流量的包长、方向、时间间隔、连接状态、TLS 握手参数、服务器名称以及 X509 证书等侧面信息。这些信息不会直接泄露通信正文，却能反映一条连接的行为模式。

本文完整讲解我实现的 **MFF-LightGBM 异常加密流量检测系统**。MFF 是 Multi-dimensional Feature Fusion，即“多维特征融合”。系统从原始 PCAP/PCAPNG 文件开始，依次完成双向流截断、统计特征提取、TLS/X509 日志序列化、DeBERTa-v3 领域语义建模、LoRA 分类适配、SupCon-AE 监督对比降维、LightGBM 八分类，以及 FastAPI 与前端可视化。

这不是一篇只罗列模型名称和最终指标的项目介绍。下面会按照真实源码的执行顺序，说明每一步为什么存在、输入输出是什么、关键代码如何工作，以及实现过程中有哪些值得注意的问题。

---

## 一、任务定义与总体架构

### 1.1 整体解题思路

这个项目要解决的核心问题可以概括为：**当 TLS 已经把通信正文加密后，如何只利用仍然可观察的信息判断一条流量是否恶意，以及它属于哪一种恶意行为？**

项目的基本判断是：加密隐藏了“传输了什么内容”，但没有完全隐藏“这次通信是怎样发生的”。即使看不到明文，仍然可以观察到两类信息：

1. **流量统计与时序行为**：一条连接包含多少个包、上下行各有多少字节、包长如何分布、包与包之间间隔多久、是否频繁重连、TCP 握手是否异常等；
2. **TLS/X509 握手语义**：使用什么 TLS 版本和密码套件、访问什么服务器名称、证书由谁签发、证书有效期多长、域名结构是否异常等。

这两类信息具有互补性。人工统计特征擅长表达明确的数值规律，例如 DNS 隧道可能产生大量长度相近、时间间隔稳定的查询；TLS/X509 字段之间则存在组合关系，例如某种 TLS 版本、密码套件、SNI 和证书签发信息共同出现时，可能比单独观察其中一个字段更有意义。

因此，项目没有选择“只使用一个深度模型端到端分类”，而是把问题拆成两条特征分支：

```text
分支A：数据包和双向流
       → 包数、字节数、IAT、TCP状态、域名结构等人工特征

分支B：连接日志、TLS握手和X509证书
       → 结构化文本
       → DeBERTa-v3
       → 深度语义特征
```

在语义分支中，DeBERTa-v3 并不是直接读取加密载荷，而是把一条流的连接、TLS 和证书字段编码成768维向量。为了让通用语言模型理解这些不同于普通自然语言的字段，先使用 RTD 进行领域继续预训练，再通过 LoRA 让模型适应八分类任务。

但是，直接把768维语义向量与约80维人工特征拼接，会出现两个问题：一是总维度较高、存在冗余；二是语义特征数量远多于人工特征，可能使融合结果被语义分支主导。于是项目在融合前加入 SupCon-AE：

- AutoEncoder 通过重构任务尽量保留原始语义信息；
- 监督对比学习利用标签让同类流量在低维空间中更接近、不同类别更分离；
- 最终把768维语义向量压缩为64维。

降维后的64维语义特征与约80维人工特征拼接，形成约144维最终特征，再交给 LightGBM 完成八分类。选择 LightGBM，是因为融合后的数据已经是典型的表格型数据：既包含连续统计量，也包含神经网络生成的稠密向量。LightGBM 能表达非线性特征组合，训练和推理成本也低于继续堆叠大型神经网络。

因此，各模块的分工可以概括为：

| 模块 | 解决的问题 |
|---|---|
| 双向流截断 | 控制每条连接的观测范围和处理成本 |
| 人工特征提取 | 描述包长、方向、时序、连接状态等显式行为 |
| TLS/X509 序列化 | 把结构化握手字段转换成模型可读取的文本 |
| DeBERTa-v3 RTD | 学习加密流量领域文本的分布 |
| LoRA | 以较低训练成本注入八分类监督信息 |
| SupCon-AE | 保留语义信息并增强类别可分性，将768维降到64维 |
| 特征融合 | 组合深度语义特征与人工统计特征 |
| LightGBM | 根据约144维融合特征输出八类概率 |
| FastAPI 与前端 | 执行任务、推送进度并展示检测结果 |

如果用一条样本概括整个过程，就是：

```text
一条PCAP中的双向连接
    → 截取前N个包
    → 提取约80维人工特征和TLS/X509日志
    → 日志经过DeBERTa得到768维语义向量
    → SupCon-AE将语义向量压缩到64维
    → 64维语义特征与约80维人工特征拼接
    → LightGBM输出8个类别概率
    → 取最大概率作为预测类别，并在前端展示
```

整个方案的重点不是简单叠加多个算法，而是让每个模型负责它更擅长的部分：DeBERTa 学习字段之间难以手工定义的语义关系，人工特征保留明确、可解释的流量行为，SupCon-AE负责建立更紧凑且具有类别区分度的表示，LightGBM完成最终的表格特征决策。

### 1.2 八分类检测任务

系统当前识别八类加密流量：

| 标签 ID | 类别 | 含义 |
|---:|---|---|
| 0 | benign | 正常流量 |
| 1 | adware | 广告软件流量 |
| 2 | dns2tcp | dns2tcp DNS 隧道 |
| 3 | dnscat2 | dnscat2 DNS 隧道 |
| 4 | iodine | iodine DNS 隧道 |
| 5 | ransomware | 勒索软件流量 |
| 6 | scareware | 恐吓软件流量 |
| 7 | smsmalware | 短信恶意软件流量 |

标签顺序定义在 `src/mff_lightgbm/training/config.py` 中，并在 LoRA 分类头、SupCon-AE 和 LightGBM 之间保持稳定：

```python
CLASS_LABELS = [
    "benign",
    "adware",
    "dns2tcp",
    "dnscat2",
    "iodine",
    "ransomware",
    "scareware",
    "smsmalware",
]

LABEL2ID = {name: idx for idx, name in enumerate(CLASS_LABELS)}
ID2LABEL = {idx: name for name, idx in LABEL2ID.items()}
NUM_LABELS = len(CLASS_LABELS)
```

这里不能随意改变列表顺序。模型输出的第 0 个概率对应 benign，第 1 个概率对应 adware；如果训练和推理阶段映射不同，即使模型文件能够正常加载，最终类别名称也会全部错位。

### 1.3 完整数据链路

项目的整体数据流如下：

```text
原始 PCAP / PCAPNG
        │
        ▼
按双向流保留前 N 个包
        │
        ▼
流统计特征 + Zeek 风格 TLS/X509 日志
        │
        ├──────────────► 80 维人工数值特征 ──────────────┐
        │                                                │
        ▼                                                │
TLS/X509 结构化日志序列化                                │
        │                                                │
        ▼                                                │
DeBERTa-v3 RTD 领域继续预训练                            │
        │                                                │
        ▼                                                │
LoRA 八分类监督适配                                      │
        │                                                │
        ▼                                                │
提取每条流的 768 维 [CLS] 语义向量                       │
        │                                                │
        ▼                                                │
SupCon-AE：768 维 → 64 维                                │
        │                                                │
        └──────────────► 与人工特征拼接 ◄────────────────┘
                                │
                                ▼
                       LightGBM 八分类检测
                                │
                                ▼
                  指标、混淆矩阵、ROC、逐流预测
                                │
                                ▼
                    FastAPI + SSE + 前端可视化
```

需要注意，DeBERTa 并没有读取被 TLS 加密后的应用正文。它读取的是 TLS、X509 和连接行为字段序列化形成的文本，因此整个过程仍然属于“不解密载荷”的检测。

### 1.4 流水线入口

`mff-offline`（实现位于 `src/mff_lightgbm/cli/offline.py`）是正式离线流水线入口：

```python
DEFAULT_PIPELINE_STAGES = [
    "truncate",
    "flow_features",
    "preprocess",
    "extract",
    "detector",
]

STAGE_RUNNERS = {
    "truncate": run_truncate,
    "flow_features": run_flow_features,
    "preprocess": run_preprocess,
    "extract": run_extract,
    "supcon": run_supcon,
    "detector": run_detector,
}
```

`detector` 阶段内部默认训练 SupCon-AE，因此执行完整流水线时不需要再单独插入一次 `supcon`。单独的 `supcon` 阶段主要用于加载已经训练好的检查点，对融合特征重新执行变换。

启动最终检测流水线：

```powershell
cd E:\Work\ccb
.\.venv\Scripts\Activate.ps1
.\scripts\run_offline.ps1 --stage all
```

这里的 `all` 会使用已经存在的 RTD 和 LoRA 检查点完成最终特征抽取与检测，不会从头重新训练 DeBERTa。RTD 继续预训练和 LoRA 监督训练属于离线模型训练流程，需要使用后文给出的独立命令执行。

已有截断后的 PCAP 时可以跳过第一步：

```powershell
.\scripts\run_offline.ps1 --stage all --skip-truncate
```

也可以只运行某一个阶段：

```powershell
.\scripts\run_offline.ps1 --stage flow_features
.\scripts\run_offline.ps1 --stage preprocess
.\scripts\run_offline.ps1 --stage extract
.\scripts\run_offline.ps1 --stage detector
```

---

## 二、PCAP/PCAPNG 解析与双向流截断

### 2.1 为什么先做流截断

原始抓包文件中可能包含大量长连接。如果把每条连接的全部数据都交给后续流程，不仅处理速度慢，而且不同流之间长度差异非常大。项目先按双向流聚合数据包，然后为每条流保留前 N 个包，主要有三个目的：

1. 限制单条流的计算量和内存占用；
2. 让不同流具有更接近的观测窗口；
3. 尽可能利用连接早期特征完成检测。

这种做法也有代价：只在长连接后期出现的行为可能被截掉。因此，“前 N 包”应当被视为效率与完整性之间的工程折中，而不是适用于所有数据集的固定真理。

### 2.2 包与双向流

一个网络包具有源地址、目的地址、源端口、目的端口和协议。最直接的五元组可以表示为：

```python
forward = (src_ip, src_port, dst_ip, dst_port, protocol)
reverse = (dst_ip, dst_port, src_ip, src_port, protocol)
```

如果只按 `forward` 建立键，那么请求和响应会被拆成两条单向流。项目在 `src/mff_lightgbm/data/truncate_pcap.py` 的 `extract_four_tuple()` 与 `truncate_flows_in_pcap()` 中对正向键和反向键进行归一化，使下面两个方向归入同一条连接：

```text
192.168.1.10:51000 → 8.8.8.8:443
8.8.8.8:443 → 192.168.1.10:51000
```

处理逻辑可以概括为：

```python
key = extract_four_tuple(raw_packet)
reverse_key = reverse_tuple(key)

if key in flow_counts:
    canonical_key = key
elif reverse_key in flow_counts:
    canonical_key = reverse_key
else:
    canonical_key = key
    flow_counts[canonical_key] = 0

if flow_counts[canonical_key] < max_pkts:
    writer.writepkt(raw_packet, pkt_ts)
    flow_counts[canonical_key] += 1
```

核心不是代码写法，而是“先确定规范化流键，再限制该流写出的包数”。

### 2.3 PCAP 与 PCAPNG

`fast_pcap_iter()` 直接处理 PCAP 和 PCAPNG 文件。解析时需要关注：

- 文件魔数决定格式和字节序；
- PCAP 包头记录秒、微秒或纳秒时间戳；
- PCAPNG 由 Section、Interface、Enhanced Packet 等 Block 组成；
- 链路层类型决定原始数据从哪里开始解析 Ethernet/IP；
- 截断包长度和原始包长度含义不同，读取时必须检查边界。

解析器最终统一产出：

```python
(packet_timestamp, raw_packet, linktype)
```

这样，后续代码不用再区分输入来自 PCAP 还是 PCAPNG。

---

## 三、流量统计特征与 TLS/X509 行为特征

### 3.1 特征分组

`src/mff_lightgbm/data/extract_flow_features.py` 将包级数据聚合为流级特征。最终用于融合的人工数值特征约 80 维，可以分为以下几类。

下面只整理真正进入融合 CSV 的 **80个数值字段**。TLS 版本、密码套件、SNI、证书 Subject 和 Issuer 等原始字符串不属于这80维数值特征，它们会在下一章被序列化为文本并输入 DeBERTa。

| 特征类别 | 包含的数值信息 | 对应源码字段 | 直观含义 |
|---|---|---|---|
| 流量规模、方向与速率 | 正向/反向/总包数，正向/反向/总字节数，上下行比例，字节速率与包速率，TCP头长度，平均Segment大小 | `pkts_forward`、`pkts_backward`、`pkts_total`、`bytes_forward`、`bytes_backward`、`bytes_total`、`ratio_bytes_back_to_forward`、`flow_bytes_s`、`flow_pkts_s`、`fwd_pkts_s`、`bwd_pkts_s`、`fwd_header_len`、`bwd_header_len`、`down_up_ratio`、`avg_fwd_segment_size`、`avg_bwd_segment_size` | 描述一条流有多大、传输有多快，以及数据主要流向哪一边 |
| 包长统计 | 全部包、正向包和反向包的长度均值、最大值、最小值、标准差和方差 | `pkt_len_max`、`pkt_len_min`、`pkt_len_mean`、`pkt_len_std`、`pkt_len_var`、`pkt_len_fwd_mean`、`pkt_len_fwd_std`、`pkt_len_bwd_mean`、`pkt_len_bwd_std` | 描述数据包通常有多大、长度是否固定，以及请求与响应的包长是否对称 |
| 包到达时间与活跃/空闲行为 | 全部、正向和反向IAT统计，Active/Idle统计 | `iat_max`、`iat_min`、`iat_mean`、`iat_std`、`iat_fwd_max`、`iat_fwd_min`、`iat_fwd_mean`、`iat_fwd_std`、`iat_bwd_max`、`iat_bwd_min`、`iat_bwd_mean`、`iat_bwd_std`、`active_max`、`active_min`、`active_mean`、`active_std`、`idle_max`、`idle_min`、`idle_mean`、`idle_std` | 描述数据包发送节奏，以及流量是连续传输、间歇传输还是周期性唤醒 |
| TCP标志与子流 | SYN、FIN、RST、PSH、ACK计数，以及子流两个方向的包数和字节数 | `flag_syn_count`、`flag_fin_count`、`flag_rst_count`、`flag_psh_count`、`flag_ack_count`、`subflow_fwd_pkts`、`subflow_fwd_bytes`、`subflow_bwd_pkts`、`subflow_bwd_bytes` | 描述TCP建连、数据推送、断开和重置过程，以及子流规模 |
| 连接窗口与异常行为 | RST比例、握手失败、重连、连接数量、连接间隔抖动、目标数量、源地址异常比例、持续时间分位数和加权统计 | `rst_ratio`、`handshake_fail_rate`、`reconnect_count`、`conn_count`、`flow_interval_jitter`、`flow_interval_diff_mean`、`tcp_rst_count`、`reconnection_flag`、`unique_dst_count`、`src_ip_abnormal_ratio`、`duration_p25`、`duration_p50`、`duration_p75`、`weighted_conn_count`、`weighted_avg_duration`、`abnormal_to_conn_ratio`、`handshake_duration` | 从一组相关连接的角度描述频繁重连、异常目标访问、连接持续时间和握手异常 |
| CN与X509数值特征 | CN字符组成、CN哈希、证书有效期、采集时证书年龄、剩余有效期和证书链深度 | `cn_vowel_ratio`、`cn_digit_density`、`cn_special_char_density`、`cn_length`、`cn_hash`、`cert_valid_days`、`cert_age_at_capture`、`cert_remaining_days`、`cert_chain_depth` | 描述域名或证书CN的字符结构，以及证书生命周期和信任链结构 |

表中描述的是特征可能反映的行为，不是固定检测规则。例如，频繁重连、较高的CN数字比例或较短的证书有效期都可能出现在合法业务中，模型需要结合多项特征共同判断。

为了避免混淆，两条特征路径可以明确区分为：

| 特征路径 | 典型内容 | 进入模型的方式 |
|---|---|---|
| 人工数值特征 | 包数、字节数、IAT、TCP标志、重连行为、CN字符比例、证书有效期 | 作为数值列保留，后续与64维SupCon-AE输出拼接后输入LightGBM |
| TLS/X509语义字段 | TLS版本、密码套件、曲线、SNI、证书Subject、Issuer、SAN等 | 序列化成结构化文本，输入DeBERTa得到768维语义向量 |

上表给出了80维人工特征的全貌。下面的3.2～3.4不是新的流水线阶段，也不是另外增加的特征，而是从这80维中选择三类有代表性的实现，进一步说明均值和方差、包到达间隔以及CN字符特征具体如何计算。

### 3.2 Welford 在线均值与方差

如果一条流包含很多数据包，最简单的统计方式是先保存所有包长，再调用 NumPy 计算均值和方差。但当同时维护大量活跃流时，这会消耗很多内存。

项目使用 Welford 在线算法逐个更新统计量：

```python
def update_welford(stats, value):
    stats["count"] += 1
    delta = value - stats["mean"]
    stats["mean"] += delta / stats["count"]
    delta2 = value - stats["mean"]
    stats["m2"] += delta * delta2
```

其均值更新公式为：

\[
\mu_n=\mu_{n-1}+\frac{x_n-\mu_{n-1}}{n}
\]

二阶矩更新为：

\[
M_{2,n}=M_{2,n-1}+(x_n-\mu_{n-1})(x_n-\mu_n)
\]

最后通过 `M2 / count` 或 `M2 / (count - 1)` 得到总体方差或样本方差。它只保存计数、均值和二阶矩，不需要保留完整历史数组。

### 3.3 IAT、活跃时间和空闲时间

包长描述的是“每次传输了多少数据”，IAT、Active 和 Idle 描述的则是“这些数据在时间上如何出现”。正常网页访问通常具有明显的突发性：页面加载时短时间内集中传输大量数据，完成后连接逐渐安静；周期性 C2 心跳可能每隔固定时间发送少量报文；DNS 隧道则可能连续发起间隔较短、节奏相似的查询。因此，时间特征能够补充包长和字节数无法表达的通信节奏。

#### 3.3.1 IAT的定义

IAT 是 Inter-Arrival Time，即相邻两个数据包到达时间之差。设一条流按时间排序后的数据包时间戳为：

\[
t_1,t_2,\ldots,t_n
\]

那么第 \(i\) 个到达间隔为：

\[
IAT_i=t_i-t_{i-1},\quad i=2,3,\ldots,n
\]

项目分别维护三组 IAT：

| IAT类型 | 计算范围 | 输出字段 |
|---|---|---|
| 整体IAT | 双向流中所有相邻数据包 | `iat_max`、`iat_min`、`iat_mean`、`iat_std` |
| 正向IAT | 只观察正向数据包之间的间隔 | `iat_fwd_max`、`iat_fwd_min`、`iat_fwd_mean`、`iat_fwd_std` |
| 反向IAT | 只观察反向数据包之间的间隔 | `iat_bwd_max`、`iat_bwd_min`、`iat_bwd_mean`、`iat_bwd_std` |

在逐包解析时，代码保存整条流以及两个方向最近一次出现的时间戳。每读到一个新包，就用当前时间减去相应的上一个时间：

```python
# 整条双向流的IAT
iat_total = pkt_ts - flow["last_ts_total"]
update_welford(flow["iat_total"], iat_total)
flow["last_ts_total"] = pkt_ts

if is_forward:
    # 当前包属于正向；只有存在上一个正向包时才能计算正向IAT
    if flow["last_ts_fwd"] is not None:
        iat_fwd = pkt_ts - flow["last_ts_fwd"]
        update_welford(flow["iat_fwd"], iat_fwd)
    flow["last_ts_fwd"] = pkt_ts
else:
    # 当前包属于反向；只有存在上一个反向包时才能计算反向IAT
    if flow["last_ts_bwd"] is not None:
        iat_bwd = pkt_ts - flow["last_ts_bwd"]
        update_welford(flow["iat_bwd"], iat_bwd)
    flow["last_ts_bwd"] = pkt_ts
```

这里继续使用上一节介绍的 Welford 在线统计，因此不需要保存全部 IAT 数组。流处理结束后，再读取每组统计量：

```python
iat_max, iat_min, iat_mean, iat_std = get_welford_metrics(
    conn_entry["iat_total"]
)

iat_fwd_max, iat_fwd_min, iat_fwd_mean, iat_fwd_std = (
    get_welford_metrics(conn_entry["iat_fwd"])
)

iat_bwd_max, iat_bwd_min, iat_bwd_mean, iat_bwd_std = (
    get_welford_metrics(conn_entry["iat_bwd"])
)
```

例如，一条流的包到达时间为：

```text
0.0秒、0.2秒、0.8秒、7.0秒、7.4秒
```

那么整体 IAT 为：

```text
0.2秒、0.6秒、6.2秒、0.4秒
```

相应统计量约为：

```text
iat_min  = 0.2
iat_max  = 6.2
iat_mean = 1.85
iat_std  ≈ 2.52
```

其中6.2秒的长间隔明显区别于其余短间隔，它也会成为划分 Active 和 Idle 的依据。

#### 3.3.2 Active和Idle区间

仅使用 IAT 可以观察单次间隔，却不能直接概括一条流经历了多少段连续活动。项目进一步使用5秒阈值，把时间轴划分为 Active 和 Idle：

- 相邻包间隔不超过5秒：仍然处于当前 Active 区间；
- 相邻包间隔超过5秒：当前 Active 区间结束，这段长间隔记为一次 Idle，新包开始下一段 Active。

逐包处理代码如下：

```python
gap = pkt_ts - flow["last_ts_active"]

if gap > 5.0:
    # 上一段连续活动的持续时间
    active_duration = (
        flow["last_ts_active"] - flow["active_start"]
    )
    update_welford(flow["act_welford"], active_duration)

    # 两段活动之间的空闲时间
    update_welford(flow["idl_welford"], gap)

    # 当前包是新Active区间的起点
    flow["active_start"] = pkt_ts

flow["last_ts_active"] = pkt_ts
```

对于需要根据完整时间戳序列离线计算的输出，项目还提供了 `get_active_idle_metrics_func()`：

```python
def get_active_idle_metrics_func(pkt_times, threshold=5.0):
    if len(pkt_times) < 2:
        return (0.0,) * 8

    times = sorted(pkt_times)
    active_intervals = []
    idle_intervals = []
    active_start = times[0]

    for index in range(len(times) - 1):
        gap = times[index + 1] - times[index]

        if gap > threshold:
            active_intervals.append(
                times[index] - active_start
            )
            idle_intervals.append(gap)
            active_start = times[index + 1]

    # 保存最后一段Active区间
    active_intervals.append(times[-1] - active_start)

    active = get_stats_metrics_func(active_intervals)[:4]
    idle = (
        get_stats_metrics_func(idle_intervals)[:4]
        if idle_intervals
        else (0.0, 0.0, 0.0, 0.0)
    )

    return (*active, *idle)
```

仍以前面的时间戳为例：

```text
0.0 ── 0.2 ── 0.8 ────────── 7.0 ── 7.4
└──── Active 1 ────┘  Idle   └─ Active 2 ─┘
```

划分结果为：

```text
Active 1：0.8 - 0.0 = 0.8秒
Idle：    7.0 - 0.8 = 6.2秒
Active 2：7.4 - 7.0 = 0.4秒
```

最终分别对 Active 和 Idle 持续时间计算最大值、最小值、均值和标准差，形成8维特征：

```text
active_max、active_min、active_mean、active_std
idle_max、idle_min、idle_mean、idle_std
```

加上整体、正向和反向三组 IAT 的12维，本节一共对应20维人工特征。固定周期的流量通常会表现出较稳定的 IAT 或 Idle 分布，而突发式交互流量的分布可能更加离散。不过，5秒只是当前实验采用的特征工程阈值，并不是 TCP 或 TLS 协议规定；更换数据集或部署环境时，应通过验证集重新评估这一阈值。

### 3.4 CN字符结构特征

CN 是 Common Name 的缩写，是 X509 证书 Subject 中用于表示证书主体名称的字段。在 TLS 流量中，它通常与服务器域名有关。普通域名往往包含容易阅读的单词或品牌名称，而某些自动生成域名、DNS 隧道标识和恶意基础设施可能包含较长的随机字符串、大量数字或特殊字符。

例如下面两个名称在字符结构上就有明显区别：

```text
www.example.com
aj39dk2m91f0.example.com
```

模型不能直接把字符串交给 LightGBM，因此项目从 CN 中提取5个数值特征：

| 字段 | 计算方式 | 表达的信息 |
|---|---|---|
| `cn_vowel_ratio` | 元音字母数 ÷ CN长度 | 字符串中自然语言式字母组合的比例 |
| `cn_digit_density` | 数字字符数 ÷ CN长度 | CN中数字的密集程度 |
| `cn_special_char_density` | 非字母且非数字字符数 ÷ CN长度 | 点、连字符等特殊字符的比例 |
| `cn_length` | `len(cn_value)` | CN整体长度 |
| `cn_hash` | `MD5(CN) mod 1024` | 将完整CN稳定映射为一个有限范围整数 |

前三个比例由 `analyze_cn_structure()` 计算：

```python
def analyze_cn_structure(cn_str):
    if not cn_str:
        return 0.0, 0.0, 0.0

    length = len(cn_str)

    vowel_ratio = (
        len(re.findall(r"[aeiouAEIOU]", cn_str))
        / length
    )
    digit_density = (
        len(re.findall(r"\d", cn_str))
        / length
    )
    special_char_density = (
        len(re.findall(r"[^a-zA-Z0-9]", cn_str))
        / length
    )

    return (
        round(vowel_ratio, 4),
        round(digit_density, 4),
        round(special_char_density, 4),
    )
```

提取器从 TLS/证书日志中取得 CN 后，调用该函数并计算长度和哈希：

```python
cn_value = ssl_entry.get("cn")

if cn_value:
    (
        cn_vowel_ratio,
        cn_digit_density,
        cn_special_char_density,
    ) = analyze_cn_structure(cn_value)

    cn_length = len(cn_value)
    cn_hash = int(
        hashlib.md5(cn_value.encode()).hexdigest(),
        16,
    ) % 1024
```

以 `www.example.com` 为例，代码会依次统计：

```text
字符串总长度
元音字母 a、e、i、o、u 出现的次数
数字字符出现的次数
点号等非字母、非数字字符出现的次数
```

再分别除以总长度，得到0到1之间的比例。比例特征比直接使用字符数量更容易比较不同长度的CN。例如两个CN都包含4个数字，但一个长度为10、另一个长度为40，它们的数字密度显然不同。

`cn_hash` 的作用与字符比例不同。它将完整CN映射到0～1023，使相同CN得到相同数值，给模型提供一个粗粒度的身份标记。但哈希值的大小没有语义顺序，数值接近也不表示两个CN相似；取模后还可能发生碰撞。因此它只能作为辅助特征，不能单独用于判断域名或证书是否恶意。

CN字符特征同样不是检测规则。合法CDN、对象存储和自动生成的云服务域名也可能很长、包含数字或特殊字符。它们需要与包长、IAT、重连行为、证书有效期以及DeBERTa提取的TLS/X509语义特征共同使用。

---

## 四、把 TLS/X509 日志转换为模型文本

### 4.1 为什么需要序列化

DeBERTa 是文本编码器，原始 CSV 中的 Zeek 风格日志却是字典或字典列表。`preprocess.py` 负责把连接、TLS 和证书字段压缩成结构稳定的文本。

项目分别定义字段映射：

```python
CONN_FIELDS = {
    "proto": "proto",
    "service": "svc",
    "duration": "dur",
    "orig_bytes": "orig_bytes",
    "resp_bytes": "resp_bytes",
    "conn_state": "state",
}

SSL_FIELDS = {
    "version": "ver",
    "cipher": "cipher",
    "curve": "curve",
    "server_name": "sni",
    "established": "est",
    "issuer": "issuer",
}
```

`build_flow_text()` 解析三类日志并拼接：

```python
def build_flow_text(row):
    parts = []

    conn = parse_literal_cell(row.get("zeek_conn_log"))
    if isinstance(conn, dict):
        parts.append(compact_json(compact_dict(conn, CONN_FIELDS, "c")))

    ssl = parse_literal_cell(row.get("zeek_ssl_log"))
    if isinstance(ssl, dict):
        parts.append(compact_json(compact_dict(ssl, SSL_FIELDS, "s")))

    x509 = parse_literal_cell(row.get("zeek_x509_log"))
    if isinstance(x509, dict):
        x509 = [x509]
    if isinstance(x509, list):
        for cert in x509:
            parts.append(compact_json(compact_dict(cert, X509_FIELDS, "x")))

    return " ".join(parts), len(parts)
```

一条序列化结果大致类似：

```text
{"t":"c","proto":"tcp","svc":"ssl","dur":1.27,"state":"SF"}
{"t":"s","ver":"TLSv12","cipher":"TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256","sni":"api.example.com"}
{"t":"x","issuer":"Let's Encrypt","key_alg":"rsaEncryption","key_len":2048}
```

这里的 `c`、`s`、`x` 分别表示 connection、SSL/TLS 和 X509 事件。使用紧凑字段名能减少 Token 数量，使有限的最大序列长度容纳更多有用信息。

### 4.2 JSONL 中保存什么

每条流最后写成一行 JSON：

```python
flows.append({
    "flow_uid": canonical_flow_uid(row),
    "src_ip": str(row["src_ip"]),
    "dst_ip": str(row["dst_ip"]),
    "text": text,
    "label": label_id,
    "label_name": label_name,
    "num_events": num_events,
    "num_features": extract_num_features(row),
})
```

其中：

- `text` 进入 DeBERTa；
- `label` 用于 LoRA、SupCon-AE 和 LightGBM；
- `num_features` 保存约 80 维人工特征；
- `flow_uid` 用于将预测结果重新关联到原始流。

源/目的 IP 等标识字段用于追踪和展示，不作为模型训练特征，避免模型记忆某个数据集中的固定地址。

### 4.3 Tokenizer 与 Dataset

`FlowDataset` 使用本地 DeBERTa tokenizer：

```python
encoded = tokenizer(
    flow["text"],
    max_length=MAX_LENGTH,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)
```

几个参数的作用如下：

| 参数 | 作用 |
|---|---|
| `max_length=512` | 每条流最多保留 512 个 Token |
| `truncation=True` | 超长序列截断 |
| `padding="max_length"` | 短序列补齐到固定长度 |
| `input_ids` | Token 对应的整数编号 |
| `attention_mask` | 区分真实 Token 和 Padding |

固定长度便于组成 Batch，但会增加短文本的 Padding 计算量；动态 Padding 更节省计算，但实现和复现会稍复杂。

---

## 五、DeBERTa-v3 的 RTD 领域继续预训练

### 5.1 为什么还要继续预训练

通用 DeBERTa-v3 学习的是自然语言分布，而本项目输入包含 TLS 版本、密码套件、证书字段、连接状态和域名。继续预训练的目的是让编码器适应这种领域文本，而不是重新从零训练一个语言模型。

DeBERTa-v3 使用 ELECTRA 风格的 Replaced Token Detection，简称 RTD。它由 generator 和 discriminator 构成：

1. 随机选择一部分普通 Token；
2. 将这些位置替换为 `[MASK]`；
3. generator 预测原 Token 并从分布中采样；
4. 用采样 Token 构造 corrupted input；
5. discriminator 判断每个 Token 是否被替换。

### 5.2 Generator 与 Discriminator

`DebertaV3RTDPretrainer` 使用较浅的 generator 和完整 discriminator：

```python
disc_config = AutoConfig.from_pretrained(model_dir)
gen_config = copy.deepcopy(disc_config)
gen_config.num_hidden_layers = min(generator_layers, disc_config.num_hidden_layers)

self.generator = AutoModel.from_pretrained(
    model_dir,
    config=gen_config,
    ignore_mismatched_sizes=True,
)
self.discriminator = AutoModel.from_pretrained(
    model_dir,
    config=disc_config,
)
```

Generator 预测词表概率，Discriminator 输出每个位置的二分类 Logit。训练前向过程的核心代码为：

```python
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
```

联合损失为：

\[
L_{RTD}=\lambda_g L_{generator}+\lambda_d L_{discriminator}
\]

代码只保存验证表现最佳的 discriminator encoder，因为后续 LoRA 需要的是完成领域适配后的编码器，而不是用于制造替换 Token 的 generator。

### 5.3 特殊 Token 为什么不能被替换

Padding、CLS、SEP 和 MASK 等特殊 Token 不参与随机 Mask：

```python
active = attention_mask.bool() & ~special_tokens_mask.bool()
```

否则模型可能把 Padding 的固定规律当成简单答案，或者破坏序列边界，导致损失看似下降但没有学到有效领域知识。

---

## 六、LoRA 八分类监督适配

### 6.1 为什么不用全参数微调

全参数微调会更新 DeBERTa 的全部权重，需要更多显存，也会为每个任务保存完整模型。LoRA 将某些线性层的增量写成低秩矩阵：

\[
W'=W+\Delta W=W+BA
\]

其中 `W` 冻结，只训练较小的 `A` 和 `B`。

项目配置为：

```python
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query_proj", "value_proj", "key_proj"]
```

模型构建代码：

```python
model = AutoModelForSequenceClassification.from_pretrained(
    base_model_dir,
    num_labels=NUM_LABELS,
    problem_type="single_label_classification",
    id2label=ID2LABEL,
    label2id=LABEL2ID,
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
```

`classifier` 和 `pooler` 必须跟随 Adapter 保存，因为它们承担当前八分类任务，不能只保存注意力层的低秩参数。

### 6.2 梯度累积

当前单步 Batch Size 为 2，梯度累积步数为 8，因此有效 Batch Size 为：

\[
2\times 8=16
\]

```python
(loss / grad_accum_steps).backward()

if step_idx % grad_accum_steps == 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

梯度累积可以在显存有限时模拟更大 Batch。除以 `grad_accum_steps` 是为了避免累积后的梯度整体被放大。

### 6.3 验证与早停

LoRA 训练最多 20 轮，以验证集 Macro F1 为主要保存标准，Accuracy 为并列判断：

```python
should_save = (
    val_f1 > best_f1 + min_delta
    or (
        abs(val_f1 - best_f1) <= min_delta
        and val_accuracy > best_accuracy + min_delta
    )
)
```

类别不平衡时，仅依靠 Accuracy 可能让模型偏向 benign，因此用 Macro F1 作为主要标准更合理。

LoRA 数据集使用固定随机种子和分层划分，比例为60%训练、20%验证、20%测试。这个划分用于评估 DeBERTa/LoRA 分类适配；最终 SupCon-AE + LightGBM 会对融合特征重新执行70%/10%/20%的划分，两者不能混为同一次实验划分。

---

## 七、提取 768 维语义特征并完成初次融合

### 7.1 为什么提取隐藏状态

LoRA 分类头能够直接输出八分类结果，但本项目没有把它作为最终检测器。它的另一个作用是让 DeBERTa 编码空间适应监督分类任务，然后从最后一层提取 `[CLS]` 表示，交给后续 SupCon-AE 和 LightGBM。

`src/mff_lightgbm/representation/extract_features.py` 的核心逻辑可以概括为：

```python
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    cls_features = outputs.hidden_states[-1][:, 0, :]
```

张量切片 `[:, 0, :]` 的含义是：

- 第一维选择 Batch 中全部样本；
- 第二维选择序列第 0 个位置，即 `[CLS]`；
- 第三维保留全部隐藏维度。

DeBERTa-v3-base 的隐藏维度为 768，所以每条流得到：

```text
[batch_size, sequence_length, 768]
                    ↓ 取第0个Token
[batch_size, 768]
```

### 7.2 纯语义特征与融合特征

提取阶段生成两个文件：

```text
features_pure.csv   只包含 feat_0 ... feat_767
features_fused.csv  768维语义特征 + 人工数值特征 + 标识/标签
```

融合不是把原始 CSV 所有列不加选择地拼接，而是只加入配置中明确列出的数值特征。IP、端口、文件名和原始日志等字段用于追踪，不进入训练特征。

---

## 八、SupCon-AE：监督对比学习与自动编码器降维

### 8.1 PCA 为什么不够

PCA 寻找数据总体方差最大的线性方向，但它不知道标签。最大方差方向不一定是最适合区分 benign、DNS 隧道和恶意软件的方向。

SupCon-AE 同时加入两个目标：

1. AutoEncoder 尽可能保留原始语义信息；
2. Supervised Contrastive Learning 让同类靠近、异类远离。

最终将 768 维 DeBERTa 特征压缩为 64 维。

### 8.2 模型结构

`SupConAE` 包含 Encoder、Decoder 和 Projector：

```python
class SupConAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, proj_dim, dropout):
        super().__init__()
        self.encoder = _mlp(
            [input_dim, *hidden_dims, latent_dim],
            dropout,
            last_activation=False,
        )
        self.decoder = _mlp(
            [latent_dim, *reversed(hidden_dims), input_dim],
            dropout,
            last_activation=False,
        )
        self.projector = _mlp(
            [latent_dim, latent_dim, proj_dim],
            dropout,
            last_activation=False,
        )

    def forward(self, values):
        latent = self.encoder(values)
        reconstructed = self.decoder(latent)
        projection = self.projector(latent)
        return latent, reconstructed, projection
```

三个输出的用途不同：

| 输出 | 用途 |
|---|---|
| `latent` | 64维最终表示，输入 LightGBM |
| `reconstructed` | 重构768维输入，计算重构损失 |
| `projection` | 计算监督对比损失，推理时不用 |

这里使用 LayerNorm 而不是 BatchNorm，使较小 Batch 或最后一个不完整 Batch 的训练更加稳定。

### 8.3 重构损失

AutoEncoder 的重构损失为：

\[
L_{recon}=\frac{1}{N}\sum_{i=1}^{N}\|x_i-\hat{x}_i\|_2^2
\]

```python
reconstruction_loss = F.mse_loss(reconstructed, values)
```

它要求低维表示仍然保留足够信息，使 Decoder 能够近似恢复原来的 768 维向量。

### 8.4 监督对比损失

对于锚点样本 `i`，设同类别样本集合为 `P(i)`：

\[
L_i=-\frac{1}{|P(i)|}\sum_{p\in P(i)}
\log\frac{\exp(z_i\cdot z_p/\tau)}
{\sum_{a\ne i}\exp(z_i\cdot z_a/\tau)}
\]

其中 `τ` 是温度参数。实现首先做 L2 归一化并计算 Batch 内两两相似度：

```python
features = F.normalize(projections, dim=1)
logits = torch.matmul(features, features.T) / self.temperature
```

随后构造同类掩码并排除样本自身：

```python
labels = labels.view(-1, 1)
positive_mask = labels.eq(labels.T)
self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
positive_mask = positive_mask & ~self_mask
```

如果某个样本在当前 Batch 中没有同类样本，它无法产生有效正样本对，因此类均衡采样和合理 Batch Size 对 SupCon 很重要。

### 8.5 联合目标

最终损失为：

\[
L=\lambda_rL_{recon}+\lambda_cL_{supcon}
\]

```python
loss = (
    reconstruction_weight * reconstruction_loss
    + contrastive_weight * contrastive_loss
)
```

重构约束防止表示只追求类别分离而丢失结构信息，对比约束则使降维结果更适合后续分类。

### 8.6 最重要的数据泄漏问题

SupCon 使用标签，因此不能在全量数据上先降维再划分测试集。正确顺序是：

```python
train_val_idx, test_idx = train_test_split(
    indices,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

train_idx, val_idx = train_test_split(
    train_val_idx,
    y_train_val,
    test_size=0.125,
    random_state=42,
    stratify=y_train_val,
)
```

最终比例为 70% train、10% validation、20% test。然后只用训练和验证部分拟合：

```python
reducer.fit(
    semantic_values[train_idx],
    y[train_idx],
    semantic_values[val_idx],
    y[val_idx],
    feature_columns=semantic_cols,
    checkpoint_path=SUPCON_MODEL_PATH,
)
```

StandardScaler 同样只在训练特征上 `fit`。验证集只用于早停，测试集直到最终评估才使用。

### 8.7 检查点兼容性

检查点不仅保存神经网络权重，还保存：

- 格式版本；
- 输入维度与64维输出配置；
- 训练特征列名和顺序；
- StandardScaler 的均值、方差和缩放参数；
- 类别列表；
- 最佳 Epoch 和验证损失。

加载时执行严格校验：

```python
if checkpoint.get("format_version") != 2:
    raise ValueError("Legacy SupCon-AE checkpoint is incompatible")

if list(feature_columns) != self.feature_columns:
    raise ValueError("SupCon-AE feature columns/order differ")
```

仅检查“都是768维”是不够的。若 `feat_1` 和 `feat_100` 顺序颠倒，模型仍能运行，但结果已经失去意义。

---

## 九、多维特征融合与 LightGBM 八分类

### 9.1 最终维度

SupCon-AE 只替换语义列，人工特征保持不变：

```python
def replace_semantic_features(df, reducer):
    columns = semantic_feature_columns(df)
    latent = reducer.transform(df[columns], feature_columns=columns)
    output = df.drop(columns=columns).copy()
    for index in range(latent.shape[1]):
        output[f"feat_{index}"] = latent[:, index]
    return output
```

维度变化为：

```text
原融合特征：768维 DeBERTa + 约80维人工特征 = 848维
SupCon-AE 后：64维语义特征 + 约80维人工特征 = 144维
```

这就是 MFF-LightGBM 中“多维特征融合”的具体含义。

### 9.2 检测前预处理

`preprocess_detector_dataframe()` 负责：

1. 将字符串标签映射为整数；
2. 删除不应参与训练的标识列和高基数字符串列；
3. 从证书 CN 继续生成长度、熵、子域数量等特征；
4. 对低基数字符串进行编码；
5. 填补数值缺失值。

模型训练特征通过 `detector_feature_columns()` 筛选：

```python
return [
    col
    for col in df.columns
    if col not in protected
    and pd.api.types.is_numeric_dtype(df[col])
]
```

标签、flow UID、源目的地址等 protected 字段不会进入 LightGBM。

从更严格的实验规范看，当前低基数字符串编码和缺失值中位数填补是在完整 DataFrame 上执行的。虽然 SupCon-AE 的 Scaler 和监督训练已经严格限制在训练集，但这些数据驱动的预处理参数也最好进一步改造成 `fit(train) / transform(val, test)`，并随模型保存。这是当前实现仍可继续改进的一处潜在无监督数据泄漏，而不是应当忽略的细节。

### 9.3 类别不平衡

测试集中 benign 数量明显多于各恶意类别。项目通过类别平衡权重训练：

```python
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

dtrain = lgb.Dataset(
    X_train,
    label=y_train,
    weight=sample_weights,
)
```

类别越少，单个样本获得的权重通常越高，从而降低模型只追求正常流量准确率的倾向。

### 9.4 LightGBM 参数与早停

```python
params = {
    "objective": "multiclass",
    "num_class": 8,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 50,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
}
```

`feature_fraction` 每轮只抽取部分特征，`bagging_fraction` 对样本进行子采样，L1/L2 正则降低过拟合。训练最多2000轮，验证集连续100轮不提升则停止：

```python
model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dval],
    callbacks=[
        lgb.early_stopping(100, verbose=False),
        lgb.log_evaluation(100),
    ],
)
```

正式训练中最佳迭代轮数为 761。

### 9.5 分批推理

为了避免一次性构造过大的预测结果，测试集按4096条分批：

```python
def predict_in_batches(model, X):
    chunks = []
    for start in range(0, len(X), 4096):
        batch = X[start:start + 4096]
        proba = model.predict(batch, num_iteration=model.best_iteration)
        chunks.append(proba)
    return np.vstack(chunks)
```

每条流得到8个类别概率，最大概率对应预测类别，最大值作为置信度。

---

## 十、实验结果与指标解释

正式融合特征共 40,052 条，最终测试集为 8,011 条。SupCon-AE 在第11轮得到最佳验证损失，LightGBM 最佳迭代为761。

### 10.1 分类结果

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| benign | 0.9665 | 0.9749 | 0.9707 | 4946 |
| adware | 0.8718 | 0.8313 | 0.8511 | 581 |
| dns2tcp | 1.0000 | 0.9875 | 0.9937 | 240 |
| dnscat2 | 0.9563 | 0.9837 | 0.9698 | 245 |
| iodine | 0.9872 | 0.9627 | 0.9748 | 241 |
| ransomware | 0.8479 | 0.8333 | 0.8406 | 582 |
| scareware | 0.8805 | 0.8477 | 0.8638 | 591 |
| smsmalware | 0.8408 | 0.8667 | 0.8535 | 585 |

整体指标：

| 指标 | 数值 |
|---|---:|
| Accuracy | 0.9372 |
| Macro F1 | 0.9148 |
| Weighted F1 | 0.9369 |

### 10.2 Precision、Recall 与 F1

\[
Precision=\frac{TP}{TP+FP}
\]

Precision 高表示被模型判为某类的样本中，真正属于该类的比例高。

\[
Recall=\frac{TP}{TP+FN}
\]

Recall 高表示真实属于某类的样本很少漏检。

\[
F1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall}
\]

Macro F1 对八个类别等权平均，更能反映少数类表现；Weighted F1 按类别样本数加权，更接近总体样本表现。因为 benign 占比很大，单看 Accuracy 或 Weighted F1 可能掩盖少数类问题，所以博客和报告中应同时给出 Macro F1。

### 10.3 如何理解结果

DNS 隧道三类的表现较好，说明它们在域名结构、时间行为或 TLS/X509 语义上具有较明显模式。Adware、Ransomware、Scareware 和 SMSMalware 之间更容易混淆，说明这些恶意软件类别的行为边界没有 DNS 隧道那么清晰。

不能因为测试集准确率达到 93.72% 就声称系统在所有真实网络中都具有同等表现。当前结果成立于给定数据集、固定划分和八个已知类别下。跨网络、跨时间和未知类别仍需额外实验。

### 10.4 消融实验应如何补充

为了证明每个模块都有贡献，后续应在同一划分上比较：

| 实验 | 输入特征/降维方式 |
|---|---|
| Manual-only | 只使用人工统计特征 |
| Semantic-only | 只使用64维语义特征 |
| Fused | 64维语义 + 人工特征 |
| PCA + LightGBM | PCA 线性降维 |
| AE + LightGBM | 仅重构损失 |
| SupCon-AE + LightGBM | 重构 + 监督对比损失 |

当前正式重训为了缩短时间使用了 `--skip-baselines`，所以现有正式报告只记录 LightGBM 最终模型。最终发表实验型文章前，应该重新运行完整基线与消融，避免只凭最终指标推断每个模块的提升。

---

## 十一、FastAPI 后端与任务调度

### 11.1 启动方式

项目只保留根目录一个虚拟环境：

```powershell
cd E:\Work\ccb
.\.venv\Scripts\Activate.ps1
uvicorn server:app --app-dir app --port 8000 --reload --reload-dir app
```

然后访问：

```text
http://127.0.0.1:8000/
```

`--reload-dir app` 将重载监听范围限制在后端代码，避免监听 `.venv`、大模型和大型 CSV 导致服务反复重启或卡顿。

### 11.2 静态前端与根路由

FastAPI 将前端目录挂载到 `/ui`：

```python
app.mount(
    "/ui",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui/index.html")
```

这样前端与 API 同源，不需要单独启动前端开发服务器，也避免浏览器直接打开 `file://` 页面时出现跨域问题。

### 11.3 主要接口

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/api/analyze` | 提交演示或真实检测任务 |
| GET | `/api/tasks/{task_id}/stream` | SSE 进度流 |
| GET | `/api/tasks` | 查询任务列表 |
| GET | `/api/tasks/{task_id}` | 查询任务详情 |
| POST | `/api/tasks/{task_id}/cancel` | 取消任务 |
| GET | `/api/metadata` | 分页读取流元数据 |
| GET | `/api/metadata/{flow_uid}` | 查询单条流详情 |
| GET | `/api/predictions` | 查询逐流预测 |
| GET | `/api/evaluation` | 获取评估结果 |
| GET | `/api/evaluation/image/{name}` | 获取评估图片 |
| GET | `/api/dashboard` | 首页统计指标 |
| GET | `/api/labels` | 八类标签列表 |

上传接口会检查模式和扩展名：

```python
if mode not in ("demo", "real_test", "real_unknown"):
    raise HTTPException(400, "非法模式")

if not filename.lower().endswith((".pcap", ".pcapng")):
    raise HTTPException(400, "仅支持 .pcap / .pcapng 文件")
```

当前正式流水线需要从文件名获取测试标签，因此 `real_unknown` 明确拒绝，而不是悄悄把未知样本分到任意类别。

### 11.4 Orchestrator 与并发控制

任务提交后生成唯一 ID，并创建队列：

```python
task_id = (
    f"T-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    f"-{uuid.uuid4().hex[:6]}"
)
self._queues[task_id] = asyncio.Queue()
```

真实任务通过 `asyncio.Lock` 串行执行：

```python
if mode in ("real_test", "real_unknown"):
    await self._real_lock.acquire()
```

这是因为完整模型流水线会大量占用 GPU、CPU、内存并修改公共输出文件。若同时运行两个任务，不仅资源竞争严重，还可能互相覆盖结果。

任务状态写入 SQLite，而进度事件保存在内存队列。数据库保存 `running`、`done`、`failed` 和 `cancelled` 等持久状态；队列负责当前连接的实时事件。

### 11.5 SSE 进度推送

服务端不断从队列读取事件：

```python
async def gen():
    while True:
        event_name, data = await q.get()
        if event_name == "_close":
            break
        yield {
            "event": event_name,
            "data": json.dumps(data, ensure_ascii=False),
        }

return EventSourceResponse(gen())
```

SSE 适合“服务器持续向浏览器单向推送进度”的场景。与轮询相比，它不需要浏览器每隔几秒重复请求；与 WebSocket 相比，实现更简单。

需要准确描述：当前 Web 系统实现的是“异步执行检测流水线并实时推送进度”，而不是毫秒级在线流式分类。完整 PCAP 解析和特征抽取仍然需要一定时间。

---

## 十二、前端如何上传 PCAP 并展示进度

前端是原生 HTML、CSS 和 JavaScript，实现上传、任务状态、TLS 分析、预测结果和模型评估页面。

### 12.1 提交任务

```javascript
async function startAnalysis() {
  const mode = document.querySelector(
    'input[name=mode]:checked'
  ).value;

  const formData = new FormData();
  formData.append('mode', mode);

  if (mode !== 'demo') {
    formData.append('file', fileInput.files[0]);
  }

  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  const data = await response.json();
  streamTaskProgress(data.task_id, handlers);
}
```

上传文件必须使用 `FormData`，浏览器会自动生成 multipart boundary，因此不要手动设置 `Content-Type`。

### 12.2 更新流水线 UI

后端发送的事件包括：

```text
task_start
stage_start
stage_progress
stage_done
task_done
task_error
```

前端按照阶段编号找到对应节点：

```javascript
function updatePipelineUI(eventName, data) {
  if (!data.stage) return;

  const step = document.querySelectorAll(
    '.pipeline-step'
  )[data.stage - 1];

  if (eventName === 'stage_start') {
    step.className = 'pipeline-step active';
  }
  if (eventName === 'stage_done') {
    step.className = 'pipeline-step done';
  }
  if (eventName === 'stage_progress' && data.progress != null) {
    step.querySelector('.progress-fill').style.width =
      `${data.progress * 100}%`;
  }
}
```

模型计算与页面展示是解耦的：前端只根据 API 和 SSE 数据渲染，不直接加载 PyTorch 或 LightGBM 模型。

---

## 十三、项目整合过程中遇到的问题

### 13.1 两个虚拟环境导致启动方式混乱

早期交付目录中的后端自带 `app/.venv`，根目录也有 `.venv`。这会导致：

- 同一个命令在不同环境中依赖不一致；
- `uv run --project app` 自动选择 app 环境；
- 模型依赖和 Web 依赖被分开维护；
- Uvicorn 重载可能监听 app 虚拟环境。

最终删除了 `app/.venv`，将依赖合并到根目录 `pyproject.toml`，统一使用：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 13.2 根路由返回 404

Uvicorn 启动成功不代表 `/` 一定存在。最初只有 API，没有根路由，所以浏览器访问首页得到404。解决方法是挂载静态目录，并将 `/` 重定向到 `/ui/index.html`。

### 13.3 页面很卡

卡顿不完全来自模型。前端早期依赖外部 Google Fonts 和 ECharts CDN，网络不可用或较慢时浏览器会等待。后来移除外部字体，把关键图表改为本地 SVG，同时把后端 CSV 详情查询改成分块读取。

### 13.4 旧 SupCon-AE 检查点不能直接使用

旧检查点输入维度为128或86，而当前 DeBERTa 语义特征是768维；旧文件还没有保存特征列名和完整 Scaler 状态。强行加载会产生维度错误，甚至出现“能运行但语义错位”的隐患。

最终建立格式版本2检查点并重新训练正式模型：

```text
input_dim      = 768
latent_dim     = 64
classes        = 0...7
best_epoch     = 11
format_version = 2
```

### 13.5 PCA 和 SupCon-AE 曾经没有真正接入最终检测器

仅仅存在一个 `supcon_ae.py` 文件并不代表最终模型使用了它。必须沿着数据流检查：

```text
SupCon-AE 输出文件
    ↓
detector 是否读取
    ↓
resource benchmark 是否加载
    ↓
Web 推理是否使用同一检查点
```

现在 `src/mff_lightgbm/training/detector.py` 默认在划分数据后训练 SupCon-AE，并将64维结果交给 LightGBM；资源基准也加载相同检查点。

### 13.6 前端算法文案与真实代码不一致

早期原型曾写成 Longformer、Center-Loss AE 和 MFF-IKNN，但正式源码实际使用 DeBERTa-v3、LoRA、SupCon-AE 和 LightGBM。最终将前端品牌统一为 MFF-LightGBM，并保留早期设计文档的废弃说明。

技术博客必须以可执行代码为准，不能因为旧页面里出现过一个算法名，就把它写进最终方案。

---

## 十四、当前系统的局限与改进方向

### 14.1 未知类别检测

当前 LightGBM 是闭集八分类。输入任何样本都会在八类中选择概率最大的一个，即使它属于训练中从未出现的新攻击。后续可以加入：

- 置信度拒绝阈值；
- 能量分数或距离分数；
- One-Class 模型；
- OpenMax 或其他开放集识别方法。

### 14.2 跨数据集泛化

同一数据集随机划分的测试结果不能完全代表跨网络性能。更严格的实验可以按以下方式划分：

- 按 PCAP 文件划分，避免同源流分散到训练和测试；
- 按采集时间划分，测试时间漂移；
- 按网络环境划分，测试域迁移；
- 使用另一公开数据集进行外部验证。

### 14.3 概率校准

树模型最大概率不一定等于真实正确概率。可以使用 Temperature Scaling、Platt Scaling 或 Isotonic Regression 进行校准，并用 ECE、可靠性曲线评估。

### 14.4 可解释性

LightGBM 可以加入 SHAP 分析，回答：

- 某条流为什么被判为 DNS 隧道；
- 哪些人工特征贡献最大；
- SupCon-AE 语义维度和人工特征各贡献多少；
- 错误样本主要由哪些特征推动。

### 14.5 真正的在线推理服务

当前真实任务仍按流水线运行。更适合生产的结构应当把“训练”和“推理”分开：

```text
离线训练：
数据集 → DeBERTa/LoRA → SupCon-AE → LightGBM → 固化模型

在线推理：
单个PCAP → 特征提取 → 已训练DeBERTa → 已训练SupCon-AE
         → 已训练LightGBM → 返回结果
```

在线路径绝不能每次上传都重新训练模型。后续可为它编写独立 `predict_pipeline.py`，统一加载四类模型与特征 Schema。

---

## 十五、复现实验与常用命令

### 15.1 安装依赖

```powershell
cd E:\Work\ccb
uv sync
.\.venv\Scripts\Activate.ps1
```

### 15.2 数据预处理

```powershell
python preprocess.py --target all
```

### 15.3 RTD 继续预训练

```powershell
python -m LLM_train.pretrain
```

### 15.4 LoRA 八分类训练

```powershell
python -m LLM_train.train_lora_classifier
```

### 15.5 语义特征提取

```powershell
python -m pipeline.LLM_extract_features
```

### 15.6 SupCon-AE + LightGBM 正式训练

```powershell
python -m pipeline.detector
```

只训练最终模型、跳过其他基线：

```powershell
python -m pipeline.detector --skip-baselines
```

### 15.7 启动 Web 系统

```powershell
uvicorn server:app --app-dir app --port 8000 --reload --reload-dir app
```

### 15.8 执行测试

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -p pytest_asyncio.plugin app\tests -q
```

当前自动化测试覆盖数据读取、任务调度、任务存储、TLS 日志解析和 SupCon-AE 的训练、保存、加载、列顺序校验。正式改造完成后的结果为：

```text
19 passed
```

---

## 总结

MFF-LightGBM 不是单一模型名称，而是一条完整的加密流量检测链路：

1. 从 PCAP/PCAPNG 中解析并聚合双向流；
2. 提取包长、方向、IAT、连接状态、TLS 和 X509 等人工特征；
3. 将 TLS/X509 结构化日志序列化为 DeBERTa 可处理的文本；
4. 通过 RTD 继续预训练适应加密流量领域；
5. 使用 LoRA 完成八分类监督适配；
6. 提取每条流的768维 DeBERTa 语义向量；
7. 使用 SupCon-AE 在监督对比约束下压缩为64维；
8. 将64维语义特征与约80维人工特征融合；
9. 使用 LightGBM 完成最终八分类；
10. 通过 FastAPI、SSE 和前端页面展示任务、流量行为和检测结果。

这个项目最值得总结的不是“叠加了多少算法”，而是如何保证不同模块之间的数据接口一致：标签映射必须一致，训练与测试必须隔离，特征列顺序必须固定，标准化参数必须随模型保存，Web 展示也必须与真实代码一致。

目前模型在8,011条测试流上取得93.72%的 Accuracy、93.69%的 Weighted F1 和91.48%的 Macro F1。这个结果证明多维特征融合方案在当前数据集上具有较好的检测能力，但仍需要跨数据集验证、开放集检测、概率校准和独立在线推理链路，才能进一步接近真实网络部署。

从工程角度看，这套系统已经完成了从原始抓包、特征工程、深度语义建模、监督降维、树模型分类到 Web 交付的闭环；从研究角度看，它也留下了足够多可继续深入的问题。这正是重新阅读和讲解源码的意义：不仅知道系统“能跑”，更清楚每一步为什么存在、依赖什么假设，以及下一步应该往哪里改。
