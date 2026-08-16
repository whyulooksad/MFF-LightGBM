# MFF-LightGBM：基于多维特征融合的异常加密流量检测模型

> 基于 DeBERTa-v3 与 LightGBM 的恶意加密流量检测算法——在不解密通信内容的前提下，融合流量统计特征与 TLS/X509 行为语义特征，实现正常流量、广告软件、DNS 隧道、勒索软件等八类加密流量检测。

如今 HTTPS、TLS 等加密协议已经成为网络通信的默认选择。加密保护了用户隐私，也让传统依赖载荷关键字、明文协议字段和内容规则的检测方法受到限制。更棘手的是，恶意软件同样可以使用 TLS 隐藏命令控制通信，DNS 隧道可以把数据编码到域名中，勒索软件也可能通过加密连接与外部服务器交互。

这并不意味着加密流量完全不可分析。即使不解密应用层内容，仍然能够观察流量的包长、方向、时间间隔、连接状态、TLS 握手参数、服务器名称以及 X509 证书等侧面信息。这些信息不会直接泄露通信正文，却能反映一条连接的行为模式。

本文完整讲解我实现的 **MFF-LightGBM 异常加密流量检测系统**。系统从原始 PCAP/PCAPNG 文件开始，依次完成数据处理、统计特征提取、语义特征提取、特征降维融合和 LightGBM 检测。

![image-20260808035210747](https://fastly.jsdelivr.net/gh/whyulooksad/image_bed@main/images/20260808035211197.png)

# 一、任务定义与总体架构

### 1.1 八分类检测任务

系统当前识别八类加密流量：

| 标签 ID | 类别       | 含义             |
| ------: | ---------- | ---------------- |
|       0 | benign     | 正常流量         |
|       1 | adware     | 广告软件流量     |
|       2 | dns2tcp    | dns2tcp DNS 隧道 |
|       3 | dnscat2    | dnscat2 DNS 隧道 |
|       4 | iodine     | iodine DNS 隧道  |
|       5 | ransomware | 勒索软件流量     |
|       6 | scareware  | 恐吓软件流量     |
|       7 | smsmalware | 短信恶意软件流量 |

### 1.2 项目整体思路

这个项目要解决的核心问题可以概括为：**当 TLS 已经把通信正文加密后，如何只利用仍然可观察的信息判断一条流量是否恶意，以及它属于哪一种恶意行为？**

我认为的基本判断是：加密隐藏了“传输了什么内容”，但没有完全隐藏“这次通信是怎样发生的”。即使看不到明文，仍然可以观察到两类信息：

1. **流量统计与时序行为**：一条连接包含多少个包、上下行各有多少字节、包长如何分布、包与包之间间隔多久、是否频繁重连、TCP 握手是否异常等；
2. **TLS/X509 握手语义**：使用什么 TLS 版本和密码套件、访问什么服务器名称、证书由谁签发、证书有效期多长、域名结构是否异常等。

两类特征分别描述流量的不同方面，各自能发现一些另一类特征不容易发现的信息；把它们组合起来，通常比单独使用其中一类更完整。

因此，我们没有选择“只使用一个深度模型端到端分类”，而是把问题拆成两条特征分支：

```text
分支A：数据包和双向流
       → 包数、字节数、IAT、TCP状态、域名结构等统计特征

分支B：连接日志、TLS握手和X509证书
       → 结构化文本
       → DeBERTa-v3
       → 深度语义特征
```

在分支B中，DeBERTa-v3 不是直接读取加密载荷，是把一条流的连接、TLS 和证书字段这些信息编码成768维的向量。为了让通用语言模型理解这些不同于普通自然语言的字段，需要先使用 RTD 进行领域继续预训练，再通过 LoRA 让模型适应八分类任务。

但是，直接把768维语义向量与约80维统计特征拼接，会出现两个问题：一是总维度较高、存在冗余；二是语义特征数量远多于统计特征，可能使融合结果被语义分支主导。于是在融合前加入 SupCon-AE：

- AutoEncoder 通过重构任务尽量保留原始语义信息；
- 监督对比学习利用标签让同类流量在低维空间中更接近、不同类别更分离；
- 最终把768维语义向量压缩为64维。

降维后的64维语义特征与约80维统计特征拼接，形成约144维最终特征，再交给 LightGBM 完成八分类。选择 LightGBM，是因为融合后的数据已经是典型的表格型数据：既包含连续统计量，也包含神经网络生成的稠密向量。LightGBM 能表达非线性特征组合，训练和推理成本也低于继续堆叠大型神经网络。

整个方案的思路不是说多个算法的叠加，而是让每个模型负责它更擅长的部分：DeBERTa 学习字段之间难以手工定义的语义关系，统计特征保留明确、可解释的流量行为，SupCon-AE负责建立更紧凑且具有类别区分度的表示，LightGBM完成最终的表格特征决策。

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
        ├──────────────► 80 维统计数值特征 ──────────────┐
        │                                              │
        ▼                                              │
TLS/X509 结构化日志序列化                                 │
        │                                              │
        ▼                                              │
DeBERTa-v3 RTD 领域继续预训练                            │
        │                                              │
        ▼                                              │
LoRA 八分类监督适配                                      │
        │                                              │
        ▼                                              │
提取每条流的 768 维 [CLS] 语义向量                         │
        │                                              │
        ▼                                              │
SupCon-AE：768 维 → 64 维                               │ 
        │                                              │
        └──────────────► 与人工特征拼接 ◄────────────────┘
                                │
                                ▼
                       LightGBM 八分类检测
                                │
                                ▼
                  指标、混淆矩阵、ROC、逐流预测
```

DeBERTa 并没有读取被 TLS 加密后的应用正文，而是读取 TLS、X509 和连接行为字段序列化形成的文本，整个过程属于“不解密载荷”的检测。

# 二、数据预处理

原始抓包文件中可能包含大量长连接。如果把每条连接的全部数据都交给后续流程，不仅处理速度慢，而且不同流之间长度差异非常大。所以我们先按双向流聚合数据包，然后为每条流保留前 N 个包，主要有三个目的：

1. 限制单条流的计算量和内存占用；
2. 让不同流具有更接近的观测窗口；
3. 尽可能利用连接早期特征完成检测。

这种做法也有代价：只在长连接后期出现的行为可能被截掉。因此，这种做法是效率与完整性之间的工程折中，不适用于所有数据集。

### 2.1 包与双向流

一个网络包具有源地址、目的地址、源端口、目的端口和协议。最直接的五元组可以表示为：

```python
forward = (src_ip, src_port, dst_ip, dst_port, protocol)
reverse = (dst_ip, dst_port, src_ip, src_port, protocol)
```

如果只按 `forward` 建立键，那么请求和响应会被拆成两条单向流。 所以我们对正向键和反向键进行归一化，使类似下面两个方向归入同一条连接：

```text
192.168.1.10:51000 → 8.8.8.8:443
8.8.8.8:443 → 192.168.1.10:51000
```

处理逻辑可以概括为：

```
def reverse_tuple(flow_key):
    src_ip, src_port, dst_ip, dst_port, protocol = flow_key

    return (
        dst_ip,
        dst_port,
        src_ip,
        src_port,
        protocol,
    )
```

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

### 2.2 PCAP 与 PCAPNG

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

下面是省略异常处理和部分兼容逻辑后的核心实现：

```
import struct


def fast_pcap_iter(file_path):
    """逐包读取 PCAP/PCAPNG。

    每次返回：
        packet_timestamp：数据包时间戳
        raw_packet：原始报文字节
        linktype：链路层类型
    """

    with open(file_path, "rb") as file:
        file_header = file.read(24)

        if len(file_header) < 24:
            return

        magic = file_header[:4]

        # ==================================================
        # 1. 标准 PCAP
        # ==================================================
        if magic in (
            b"\xa1\xb2\xc3\xd4",  # 大端 PCAP
            b"\xd4\xc3\xb2\xa1",  # 小端 PCAP
        ):
            endian = (
                ">"
                if magic == b"\xa1\xb2\xc3\xd4"
                else "<"
            )

            # PCAP 全局头的第20~24字节保存链路层类型
            linktype = struct.unpack(
                endian + "I",
                file_header[20:24],
            )[0] & 0xFFFF

            while True:
                # 每个 PCAP 数据包具有16字节记录头
                packet_header = file.read(16)

                if len(packet_header) < 16:
                    break

                timestamp_seconds, timestamp_microseconds, captured_length, original_length = (
                    struct.unpack(
                        endian + "IIII",
                        packet_header,
                    )
                )

                # 检查抓包文件中实际保存的数据长度
                if (
                    captured_length <= 0
                    or captured_length > 262144
                ):
                    break

                raw_packet = file.read(captured_length)

                # 文件提前结束，说明报文数据不完整
                if len(raw_packet) < captured_length:
                    break

                packet_timestamp = (
                    timestamp_seconds
                    + timestamp_microseconds / 1_000_000
                )

                yield (
                    packet_timestamp,
                    raw_packet,
                    linktype,
                )

        # ==================================================
        # 2. PCAPNG
        # ==================================================
        elif magic == b"\x0a\x0d\x0d\x0a":
            # PCAPNG 需要从第一个 Block 重新读取
            file.seek(0)

            # 当前项目数据默认采用 Ethernet
            linktype = 1

            while True:
                # 每个 PCAPNG Block 前8字节：
                # Block Type + Block Total Length
                block_header = file.read(8)

                if len(block_header) < 8:
                    break

                block_type, block_length = struct.unpack(
                    "<II",
                    block_header,
                )

                # 一个 Block 至少包含：
                # 8字节头 + 4字节尾部长度
                if (
                    block_length < 12
                    or block_length > 262144
                ):
                    break

                body_length = block_length - 12
                block_body = file.read(body_length)
                trailing_length_data = file.read(4)

                if (
                    len(block_body) < body_length
                    or len(trailing_length_data) < 4
                ):
                    break

                trailing_length = struct.unpack(
                    "<I",
                    trailing_length_data,
                )[0]

                # PCAPNG 的 Block 首尾都会记录长度
                if trailing_length != block_length:
                    break

                # ------------------------------------------
                # Interface Description Block
                # 保存该接口使用的链路层类型
                # ------------------------------------------
                if (
                    block_type == 0x00000001
                    and len(block_body) >= 8
                ):
                    linktype = struct.unpack(
                        "<H",
                        block_body[:2],
                    )[0]

                # ------------------------------------------
                # Enhanced Packet Block
                # 保存时间戳、抓取长度和原始报文
                # ------------------------------------------
                elif (
                    block_type == 0x00000006
                    and len(block_body) >= 20
                ):
                    timestamp_high = struct.unpack(
                        "<I",
                        block_body[4:8],
                    )[0]

                    timestamp_low = struct.unpack(
                        "<I",
                        block_body[8:12],
                    )[0]

                    captured_length = struct.unpack(
                        "<I",
                        block_body[12:16],
                    )[0]

                    original_length = struct.unpack(
                        "<I",
                        block_body[16:20],
                    )[0]

                    if (
                        captured_length <= 0
                        or captured_length > body_length - 20
                    ):
                        continue

                    timestamp_value = (
                        timestamp_high << 32
                    ) + timestamp_low

                    # 当前项目数据按照微秒换算
                    packet_timestamp = (
                        timestamp_value / 1_000_000
                    )

                    raw_packet = block_body[
                        20:20 + captured_length
                    ]

                    yield (
                        packet_timestamp,
                        raw_packet,
                        linktype,
                    )

        else:
            raise ValueError(
                f"无法识别抓包文件格式：{magic!r}"
            )
```

# 三、流量统计特征提取

### 3.1 特征分组

在对 PCAP/PCAPNG包进行处理后，我们需要将包级数据聚合为流级特征。最终用于融合的统计数值特征约 80 维，可以分为以下几类。

| 特征类别         | 包含的数值信息                                               | 对应源码字段                                                 | 直观含义                                                     |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 规模与方向特征   | 正向/反向/总包数，正向/反向/总字节数，上下行比例，字节速率与包速率，TCP头长度，平均Segment大小 | `pkts_forward`、`pkts_backward`、`pkts_total`、`bytes_forward`、`bytes_backward`、`bytes_total`、`ratio_bytes_back_to_forward`、`flow_bytes_s`、`flow_pkts_s`、`fwd_pkts_s`、`bwd_pkts_s`、`fwd_header_len`、`bwd_header_len`、`down_up_ratio`、`avg_fwd_segment_size`、`avg_bwd_segment_size` | 描述一条流有多大、传输有多快，以及数据主要流向哪一边         |
| 包长统计特征     | 全部包、正向包和反向包的长度均值、最大值、最小值、标准差和方差 | `pkt_len_max`、`pkt_len_min`、`pkt_len_mean`、`pkt_len_std`、`pkt_len_var`、`pkt_len_fwd_mean`、`pkt_len_fwd_std`、`pkt_len_bwd_mean`、`pkt_len_bwd_std` | 描述数据包通常有多大、长度是否固定，以及请求与响应的包长是否对称 |
| 时间行为特征     | 全部、正向和反向IAT统计，Active/Idle统计                     | `iat_max`、`iat_min`、`iat_mean`、`iat_std`、`iat_fwd_max`、`iat_fwd_min`、`iat_fwd_mean`、`iat_fwd_std`、`iat_bwd_max`、`iat_bwd_min`、`iat_bwd_mean`、`iat_bwd_std`、`active_max`、`active_min`、`active_mean`、`active_std`、`idle_max`、`idle_min`、`idle_mean`、`idle_std` | 描述数据包发送节奏，以及流量是连续传输、间歇传输还是周期性唤醒 |
| TCP 状态特征     | SYN、FIN、RST、PSH、ACK计数，握手失败率、重连次数与重连标记、RST 数量与比例 | `flag_syn_count`、`flag_fin_count`、`flag_rst_count`、`flag_psh_count`、`flag_ack_count`、`subflow_fwd_pkts`、`subflow_fwd_bytes`、`subflow_bwd_pkts`、`subflow_bwd_bytes`、`rst_ratio`、`handshake_fail_rate`、`reconnect_count`、`conn_count`、`flow_interval_jitter`、`flow_interval_diff_mean`、`tcp_rst_count`、`reconnection_flag`、`unique_dst_count`、`src_ip_abnormal_ratio`、`duration_p25`、`duration_p50`、`duration_p75`、`weighted_conn_count`、`weighted_avg_duration`、`abnormal_to_conn_ratio`、`handshake_duration` | 描述TCP建连、数据推送、断开和重置过程，以及子流规模          |
| CN与X509数值特征 | CN字符组成、CN哈希、证书有效期、采集时证书年龄、剩余有效期和证书链深度 | `cn_vowel_ratio`、`cn_digit_density`、`cn_special_char_density`、`cn_length`、`cn_hash`、`cert_valid_days`、`cert_age_at_capture`、`cert_remaining_days`、`cert_chain_depth` | 描述域名或证书CN的字符结构，以及证书生命周期和信任链结构     |

表中描述的是特征可能反映的行为，不是固定检测规则。例如，频繁重连、较高的CN数字比例或较短的证书有效期都可能出现在合法业务中，模型需要结合多项特征共同判断。

为了避免混淆，两条特征路径可以明确区分为：

| 特征路径         | 典型内容                                                     | 进入模型的方式                                            |
| ---------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| 统计数值特征     | 包数、字节数、IAT、TCP标志、重连行为、CN字符比例、证书有效期 | 作为数值列保留，后续与64维SupCon-AE输出拼接后输入LightGBM |
| TLS/X509语义字段 | TLS版本、密码套件、曲线、SNI、证书Subject、Issuer、SAN等     | 序列化成结构化文本，输入DeBERTa得到768维语义向量          |

篇幅限制很难完整讲解这80维的数值特征。下面3.2会挑一种重要的统计计算方法讲解，3.3会挑时间行为特征和CN字符特征讲解。

### 3.2 Welford 在线均值与方差

如果一条流包含很多数据包，最简单的统计方式是先保存所有包长，再调用 NumPy 计算均值和方差。但当同时维护大量活跃流时，这会消耗很多内存。

所以我们使用 Welford 在线算法逐个更新统计量：

```python
def update_welford(stats, value):
    stats["count"] += 1
    delta = value - stats["mean"]
    stats["mean"] += delta / stats["count"]
    delta2 = value - stats["mean"]
    stats["m2"] += delta * delta2
```

其均值更新公式为：

$$
\mu_n=\mu_{n-1}+\frac{x_n-\mu_{n-1}}{n}
$$
二阶矩更新为：

$$
M_{2,n}=M_{2,n-1}+(x_n-\mu_{n-1})(x_n-\mu_n)
$$
最后通过 `M2 / count` 或 `M2 / (count - 1)` 得到总体方差或样本方差。它只保存计数、均值和二阶矩，不需要保留完整历史数组。

### 3.3 时间行为特征和CN字符结构特征

时间行为特征：

IAT、Active（活跃时间） 和 Idle（空闲时间） 描述的是“数据在时间上如何出现”。正常网页访问通常具有明显的突发性：页面加载时短时间内集中传输大量数据，完成后连接逐渐安静；周期性 C2 心跳可能每隔固定时间发送少量报文；DNS 隧道则可能连续发起间隔较短、节奏相似的查询。因此，时间特征能够补充包长和字节数无法表达的通信节奏。

IAT 是 Inter-Arrival Time，即相邻两个数据包到达时间之差。设一条流按时间排序后的数据包时间戳为：

$$
t_1,t_2,\ldots,t_n
$$
那么第 \(i\) 个到达间隔为：

$$
IAT_i=t_i-t_{i-1},\quad i=2,3,\ldots,n
$$
当前是维护了三组 IAT：

| IAT类型 | 计算范围                   | 输出字段                                                    |
| ------- | -------------------------- | ----------------------------------------------------------- |
| 整体IAT | 双向流中所有相邻数据包     | `iat_max`、`iat_min`、`iat_mean`、`iat_std`                 |
| 正向IAT | 只观察正向数据包之间的间隔 | `iat_fwd_max`、`iat_fwd_min`、`iat_fwd_mean`、`iat_fwd_std` |
| 反向IAT | 只观察反向数据包之间的间隔 | `iat_bwd_max`、`iat_bwd_min`、`iat_bwd_mean`、`iat_bwd_std` |

做法上，我们在逐包解析时，保存整条流以及两个方向最近一次出现的时间戳。每读到一个新包，就用当前时间减去相应的上一个时间：

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

举个例子，一条流的包到达时间为：

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

其中6.2秒的长间隔明显区别于其余短间隔，它也会成为划分 Active 和 Idle 的依据。因为仅使用 IAT 可以观察单次间隔，却不能直接概括一条流经历了多少段连续活动。所以进一步使用5秒阈值，把时间轴划分为 Active 和 Idle：

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

对于需要根据完整时间戳序列离线计算的输出，可以使用下面的方法：

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

加上整体、正向和反向三组 IAT 的12维，本节一共对应20维统计特征。固定周期的流量通常会表现出较稳定的 IAT 或 Idle 分布，而突发式交互流量的分布可能更加离散。不过，5秒只是当前我们做实验采用的特征工程阈值，并不是 TCP 或 TLS 协议规定；更换数据集或部署环境时，需要通过验证集重新评估这一阈值。

CN字符结构特征：

CN 是 Common Name 的缩写，是 X509 证书 Subject 中用于表示证书主体名称的字段。在 TLS 流量中，它通常与服务器域名有关。普通域名往往包含容易阅读的单词或品牌名称，而某些自动生成域名、DNS 隧道标识和恶意基础设施可能包含较长的随机字符串、大量数字或特殊字符。

例如下面两个名称在字符结构上就有明显区别：

```text
www.example.com
aj39dk2m91f0.example.com
```

模型不能直接把字符串交给 LightGBM，因此从 CN 中提取5个数值特征：

| 字段                      | 计算方式                      | 表达的信息                         |
| ------------------------- | ----------------------------- | ---------------------------------- |
| `cn_vowel_ratio`          | 元音字母数 ÷ CN长度           | 字符串中自然语言式字母组合的比例   |
| `cn_digit_density`        | 数字字符数 ÷ CN长度           | CN中数字的密集程度                 |
| `cn_special_char_density` | 非字母且非数字字符数 ÷ CN长度 | 点、连字符等特殊字符的比例         |
| `cn_length`               | `len(cn_value)`               | CN整体长度                         |
| `cn_hash`                 | `MD5(CN) mod 1024`            | 将完整CN稳定映射为一个有限范围整数 |

前三个比例可以根据字符密度来计算：

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

下面以 `www.example.com` 为例，依次统计：

```text
字符串总长度
元音字母 a、e、i、o、u 出现的次数
数字字符出现的次数
点号等非字母、非数字字符出现的次数
```

再分别除以总长度，得到0到1之间的比例。比例特征比直接使用字符数量更容易比较不同长度的CN。例如两个CN都包含4个数字，但一个长度为10、另一个长度为40，它们的数字密度显然不同。

`cn_hash` 的作用与字符比例不同。它将完整CN映射到0～1023，使相同CN得到相同数值，给模型提供一个粗粒度的身份标记。但哈希值的大小没有语义顺序，数值接近也不表示两个CN相似；取模后还可能发生碰撞。因此它只能作为辅助特征，不能单独用于判断域名或证书是否恶意。

不过CN字符特征同样不是检测规则。合法CDN、对象存储和自动生成的云服务域名也可能很长、包含数字或特殊字符。它需要与包长、IAT、重连行为、证书有效期以及DeBERTa提取的TLS/X509语义特征共同使用。

# 四、深层语义特征提取

### 4.1 TLS/X509 日志序列化

DeBERTa 是文本编码器， 所以需要先对 Zeek 风格日志进行一个序列化的处理，把连接、TLS 和证书字段压缩成结构稳定的文本。

一条序列化结果大致类似：

```text
{"t":"c","proto":"tcp","svc":"ssl","dur":1.27,"state":"SF"}
{"t":"s","ver":"TLSv12","cipher":"TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256","sni":"api.example.com"}
{"t":"x","issuer":"Let's Encrypt","key_alg":"rsaEncryption","key_len":2048}
```

这里的 `c`、`s`、`x` 分别表示 connection、SSL/TLS 和 X509 事件。使用紧凑字段名能减少 Token 数量，使有限的最大序列长度容纳更多有用信息。

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

- `text` 为序列化后的日志进入 DeBERTa；
- `label` 用于 LoRA、SupCon-AE 和 LightGBM；
- `num_features` 保存约 80 维人工特征；
- `flow_uid` 用于将预测结果重新关联到原始流。

源/目的 IP 等标识字段用于追踪和展示，不作为模型训练特征，避免模型记忆某个数据集中的固定地址。

### 4.2 DeBERTa-v3 领域继续预训练

通用 DeBERTa-v3 学习的是自然语言分布，而用于本次实验的输入包含 TLS 版本、密码套件、证书字段、连接状态和域名等等领域专用表示，所以在使用它前需要先继续预训练让编码器适应这种领域文本。

这里我曾经尝试过用 MLM 和 SimCS 对 DeBERTa-v3 进行预训练，但发现训练过程中都会报NaN的错误，后面调研了才发现DeBERTa-v3 是使用 ELECTRA 风格的 RTD（Replaced Token Detection）训练的。它由 generator 和 discriminator 构成：

1. 随机选择一部分普通 Token；
2. 将这些位置替换为 `[MASK]`；
3. generator 预测原 Token 并从分布中采样；
4. 用采样 Token 构造 corrupted input；
5. discriminator 判断每个 Token 是否被替换。

这里`DebertaV3RTDPretrainer` 使用较浅的 generator 和完整 discriminator：

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

Generator 预测词表概率，Discriminator 输出每个位置的二分类 Logit。forward 过程的核心代码如下：

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

$$
L_{RTD}=\lambda_g L_{generator}+\lambda_d L_{discriminator}
$$
保存最佳的 discriminator encoder。

注意：Padding、CLS、SEP 和 MASK 等特殊 Token 不能参与随机 Mask，否则模型可能把 Padding 的固定规律当成简单答案，或者破坏序列边界，导致损失看似下降但没有学到有效领域知识。

### 4.3 LoRA 八分类监督适配

这里可能会有点疑问，最终不是 LightGBM 完成流量的分类吗？为什么这里还要让 DeBERTa-v3 完成一次分类呢？

我来解释一下：如果只使用未经监督分类训练的 DeBERTa，它生成的向量主要表达：两段TLS/X509文本在一般语义或字段结构上是否相似。

但我需要的是：哪些TLS/X509字段组合有助于区分正常流量、DNS隧道和恶意软件？

例如下面两条流的文本结构可能非常接近：

```
TLS版本：TLS 1.2
密码套件：AES_128_GCM
SNI：www.example.com
证书签发者：Let's Encrypt
```

```
TLS版本：TLS 1.2
密码套件：AES_128_GCM
SNI：aj39dk2m91.example.com
证书签发者：Unknown CA
```

未进行分类适配的 DeBERTa 更关注：

```
两条文本都包含TLS版本、密码套件、SNI和签发者
```

LoRA 分类训练则通过标签告诉模型：

```
哪些字段相同并不重要
哪些字段差异对区分类别更加重要
```

训练完成后，DeBERTa 的隐藏空间会更偏向当前检测任务。

完整过程其实是两级监督：

```
TLS/X509结构化文本
        ↓
DeBERTa-v3
        ↓
LoRA分类训练
        ↓
得到经过分类任务适配的DeBERTa编码器
        ↓
丢掉/不使用LoRA分类结果
        ↓
提取768维[CLS]语义向量
        ↓
SupCon-AE压缩到64维
        ↓
与80维人工特征融合
        ↓
LightGBM完成最终分类
```

模型构建代码如下：

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

训练参数可以根据自己的配置来，这里就不多说。

最后从最后一层提取 `[CLS]` 表示，核心逻辑可以概括为：

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

# 五、特征降维融合

前面说过直接把768维语义向量与约80维统计特征拼接，会出现两个问题：一是总维度较高、存在冗余；二是语义特征数量远多于统计特征，可能使融合结果被语义分支主导。所以需要把语义特征降维后再和统计特征融合。这里我们最终选择的降维方法是SupCon-AE。之前也尝试过用最简单的 PCA 算法，但 PCA 是寻找数据总体方差最大的线性方向，它不知道标签，最大方差方向不一定是最适合区分 benign、DNS 隧道和恶意软件的方向。而 SupCon-AE 可以同时加入两个目标：

1. AutoEncoder 尽可能保留原始语义信息；
2. Supervised Contrastive Learning 让同类靠近、异类远离。

### 5.1 模型结构

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

三个输出的用途分别为：

| 输出            | 用途                         |
| --------------- | ---------------------------- |
| `latent`        | 64维最终表示，输入 LightGBM  |
| `reconstructed` | 重构768维输入，计算重构损失  |
| `projection`    | 计算监督对比损失，推理时不用 |

这里使用 LayerNorm 而不是 BatchNorm，可以使较小 Batch 或最后一个不完整 Batch 的训练更加稳定。

### 5.2 重构损失和监督对比损失

AutoEncoder 的重构损失为：

$$
L_{recon}=\frac{1}{N}\sum_{i=1}^{N}\|x_i-\hat{x}_i\|_2^2
$$

```python
reconstruction_loss = F.mse_loss(reconstructed, values)
```

它要求低维表示仍然保留足够信息，使 Decoder 能够近似恢复原来的 768 维向量。

监督对比损失：

对于锚点样本 `i`，设同类别样本集合为 `P(i)`：

$$
L_i=-\frac{1}{|P(i)|}\sum_{p\in P(i)}
\log\frac{\exp(z_i\cdot z_p/\tau)}
{\sum_{a\ne i}\exp(z_i\cdot z_a/\tau)}
$$
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

最终损失为：

$$
L=\lambda_rL_{recon}+\lambda_cL_{supcon}
$$

```python
loss = (
    reconstruction_weight * reconstruction_loss
    + contrastive_weight * contrastive_loss
)
```

重构约束防止表示只追求类别分离而丢失结构信息，对比约束则使降维结果更适合后续分类。

### 5.3 最终维度

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

# 六、LightGBM 分类

### 6.1 LightGBM介绍

LightGBM 全称是 **Light Gradient Boosting Machine**，是一种基于梯度提升决策树（GBDT）的机器学习算法，尤其适合处理表格型数据。它并不是只训练一棵决策树，而是依次训练很多棵树：

```
第一棵树：先进行初步预测
    ↓
第二棵树：重点修正第一棵树的错误
    ↓
第三棵树：继续修正前两棵树的错误
    ↓
……
    ↓
组合所有树，得到最终预测
```

可以简单表示为：
$$
F_M(x)=F_0(x)+\sum_{m=1}^{M}\eta f_m(x)
$$
其中：

- \(F_0(x)\)：初始预测；
- \(f_m(x)\)：第 \(m\) 棵决策树；
- \(\eta\)：学习率；
- \(M\)：决策树数量。

每棵新树都会学习当前模型还没有处理好的部分。

与传统GBDT实现相比，LightGBM具有下面优势：

- 训练速度快；
- 内存占用相对低；
- 能处理大量样本和高维特征；
- 支持多分类；
- 支持类别特征；
- 支持并行和GPU训练。

LightGBM会把连续特征值离散到若干个区间中，称为 Histogram，从而减少寻找决策树最佳分裂点的计算量。

LightGBM如何完成分类？

假设输入一条流量的特征：

```
包数量              = 35
平均包长            = 512
IAT均值             = 0.12秒
重连次数            = 4
证书有效期          = 30天
SupCon-AE语义特征   = 64维
```

某棵决策树可能学习到类似规则：

```
IAT均值 < 0.2？
├── 是：重连次数 > 3？
│   ├── 是：更可能是恶意流量
│   └── 否：继续判断证书特征
└── 否：更可能是正常流量
```

实际上，LightGBM会组合大量决策树，而不是依靠单条固定规则。

对于这次的八分类任务，它最终输出8个类别概率：

```
benign       0.03
adware       0.04
dns2tcp      0.78
dnscat2      0.08
iodine       0.03
ransomware   0.01
scareware    0.02
smsmalware   0.01
```

最大概率对应最终预测：

```
预测类别：dns2tcp
置信度：0.78
```

为什么选择LightGBM?

SupCon-AE降维后，系统获得：

```
64维DeBERTa语义特征
+
80维流量统计特征
=
144维融合特征
```

这些数据已经属于典型的表格型数值数据，比较适合LightGBM。

它能够学习：

- 流量统计特征之间的非线性关系；
- 语义特征与统计特征之间的组合关系；
- 不同恶意类别的复杂决策边界；
- 类别不平衡情况下的加权分类。

相较于继续设计一个大型神经网络分类器，LightGBM训练更快，也更容易分析特征重要性。

### 6.2 检测前预处理

在进行分类检测前需要做的基础的数据预处理：

1. 将字符串标签映射为整数；
2. 删除不应参与训练的标识列和高基数字符串列；
3. 对低基数字符串进行编码；
5. 填补数值缺失值。

模型训练特征筛选：

```python
return [
    col
    for col in df.columns
    if col not in protected
    and pd.api.types.is_numeric_dtype(df[col])
]
```

标签、flow UID、源目的地址等 protected 字段不会进入 LightGBM。

测试集中 benign 数量明显多于各恶意类别，所以还需要类别不平衡处理。这里通过类别平衡权重训练：

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

### 6.3 LightGBM 参数

下面是我的LightGBM的参数，以供参考：

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

# 七、实验结果与指标解释

经过人工特征与DeBERTa语义特征融合后，数据集共包含40,052条双向流样本。每条样本最初由768维语义特征和80维人工数值特征表示。检测器采用分层随机划分，将28,035条样本作为训练集、4,006条作为验证集、8,011条作为测试集，比例约为70%：10%：20%。训练集用于学习模型参数，验证集用于SupCon-AE和LightGBM，测试集用于训练完成后的最终性能评估。

### 7.1 分类结果

| 类别       | Precision | Recall |     F1 | Support |
| ---------- | --------: | -----: | -----: | ------: |
| benign     |    0.9665 | 0.9749 | 0.9707 |    4946 |
| adware     |    0.8718 | 0.8313 | 0.8511 |     581 |
| dns2tcp    |    1.0000 | 0.9875 | 0.9937 |     240 |
| dnscat2    |    0.9563 | 0.9837 | 0.9698 |     245 |
| iodine     |    0.9872 | 0.9627 | 0.9748 |     241 |
| ransomware |    0.8479 | 0.8333 | 0.8406 |     582 |
| scareware  |    0.8805 | 0.8477 | 0.8638 |     591 |
| smsmalware |    0.8408 | 0.8667 | 0.8535 |     585 |

整体指标：

| 指标        |   数值 |
| ----------- | -----: |
| Accuracy    | 0.9372 |
| Macro F1    | 0.9148 |
| Weighted F1 | 0.9369 |

![image-20260810184223556](https://fastly.jsdelivr.net/gh/whyulooksad/image_bed@main/images/20260810184223855.png)

### 7.2 结果总结

模型在各类别上的预测结果主要集中在对角线位置，说明本文方法能够有效区分正常流量与多类恶意加密流量。测试集中，benign 类共有 4946 条样本，其中 4822 条被正确识别，表明模型对正常加密流量具有较强的识别能力。对于 dns2tcp、dnscat2 和 iodine 等隧道类流量，模型同样取得了较好的识别效果，正确分类样本数分别为 237、241 和 232，说明模型能够较好地捕获该类恶意加密流量在通信行为和流量特征上的差异。

从非对角线位置可以进一步观察到，少量样本在相近类别之间出现交叉预测，主要集中在 adware、ransomware、scareware 和 smsmalware 等类别中。这类流量在实际网络通信中可能具有相似的连接模式、包长分布或会话行为，因此分类边界相对更接近。尽管存在少量交叉预测，但整体误分类数量较少，模型仍能保持较稳定的多类别识别能力。

总体来看，混淆矩阵中正确分类样本占主要部分，各类别的预测结果整体较为集中，说明本文方法在多分类恶意加密流量检测任务中具有较好的整体性能和实用价值。

### 7.3 最终交付

1.加密通信流量分析工具。

2.异常行为检测模型。

3.实验环境：正常流量与攻击流量的检测对比。