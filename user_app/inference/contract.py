"""
contract.py 是训练代码、推理代码和生产模型之间共同签署的协议：
大家必须使用相同的文本长度、相同的标签编号、相同的特征含义，以及相同的特征名称和顺序。
"""

# Values that must remain identical between training and production inference.
# 规定了输入 DeBERTa 模型的文本最大长度。
MAX_LENGTH = 512

# Parser/feature contract used for every newly prepared dataset and every model.
# This is a semantic contract identifier, not the name of a second implementation.
# Archived code and models live under legacy/ and never participate at runtime.
FEATURE_SCHEMA_VERSION = "strict-tls-x509-2026-09"

# Eight-class task labels. Keep this order stable because it is written into
# classifier configs and output labels.
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

# Numeric flow-feature columns from the current 94-column
# final_multiclass_features_train/test.csv schema.
#
# Excluded on purpose:
# - identifiers / five-tuple / time: flow_uid, src_ip, src_port, dst_ip,
#   dst_port, protocol, timestamp
# - source metadata: dataset_source, subfolder, pcap_filename
# - target / raw text: label, connection_log, tls_log, x509_log
NEW_FORMAT_NUM_FEATURES = [
    # 包数量和字节数量
    "pkts_forward",    # 客户端发给服务器的包数量。
    "pkts_backward",   # 服务器发给客户端的包数量。
    "pkts_total",      # 双向所有包的总数
    "bytes_forward",   # 所有前向 IP 包长度之和。
    "bytes_backward",  # 所有后向 IP 包长度之和。
    "bytes_total",     # 双向总字节数
    "ratio_bytes_back_to_forward",  # 后向字节数 ÷ 前向字节数，这个特征能反映通信是否明显偏向某个方向。
    # 包长度统计
    "pkt_len_max",    # 最大包长度
    "pkt_len_min",    # 最小包长度
    "pkt_len_mean",   # 平均包长度
    "pkt_len_std",    # 标准差，也表示波动程度。
    "pkt_len_var",    # 方差，用于描述这些长度分散得有多厉害。
    "pkt_len_fwd_mean", # 前向包的平均长度
    "pkt_len_fwd_std",  # 前向包长度标准差
    "pkt_len_bwd_mean",  # 后向包的平均长度
    "pkt_len_bwd_std",   # 后向包长度标准差
    # 传输速率
    "flow_bytes_s",    # 每秒传输多少字节
    "flow_pkts_s",     # 每秒传输多少个包
    "fwd_pkts_s",      # 前向每秒包数。
    "bwd_pkts_s",      # 后向每秒包数。
    # 头部和载荷
    "fwd_header_len",   # 所有前向包的IP和TCP/UDP头部长度总和
    "bwd_header_len",   # 所有后向包的IP和TCP/UDP头部长度总和
    "down_up_ratio",    # 后向包数 ÷ 前向包数，可以粗略理解成下载包数和上传包数的比例。
    "transport_payload_fwd_mean",  # 前向TCP/UDP平均携带多少字节数据
    "transport_payload_bwd_mean",  # 后向TCP/UDP平均携带多少字节数据
    # 包到达时间间隔
    "iat_max", # 最大时间间隔
    "iat_min",  # 最小时间间隔
    "iat_mean",  # 平均时间间隔
    "iat_std",   # 时间间隔标准差
    "iat_fwd_max",  # 只计算前向包之间的时间间隔。
    "iat_fwd_min",
    "iat_fwd_mean",
    "iat_fwd_std",
    "iat_bwd_max",  # 只计算后向包之间的时间间隔。
    "iat_bwd_min",
    "iat_bwd_mean",
    "iat_bwd_std",
    # TCP 标志数量
    "flag_syn_count", # SYN 数量,SYN 通常用于请求建立 TCP 连接。
    "flag_fin_count", # FIN 通常用于正常关闭连接。
    "flag_rst_count", # RST 表示连接被强制重置或异常终止。
    "flag_psh_count", # PSH 通常表示希望接收方尽快把数据交给应用程序。
    "flag_ack_count", # ACK 表示确认已经收到数据。
    # 子流 (一条通信可能不是连续不断地发送,项目把相邻包间隔超过 1 秒作为新子流的分界。)
    "subflow_fwd_pkts",
    "subflow_fwd_bytes",
    "subflow_bwd_pkts",
    "subflow_bwd_bytes",
    # 活跃和空闲时间 (项目把包间隔超过 5 秒视为进入一次明显空闲。)
    "active_max", # 最长活跃段
    "active_min", # 最短活跃段
    "active_mean", # 平均活跃时长
    "active_std",  # 活跃时长波动
    "idle_max",   # 最长空闲间隔
    "idle_min",   # 最短空闲间隔
    "idle_mean",  # 平均空闲间隔
    "idle_std",   # 空闲间隔波动
    # 连接异常和时间窗口行为
    "rst_ratio", # RST包数量 ÷ TCP包数量
    "handshake_fail_rate", # 最近时间窗口中，已知 TCP 握手里失败的比例。
    "reconnect_count",   # 最近 60 秒内，同一个源与同一个目标地址和端口之间重复连接了多少次。
    "conn_count",    # 最近 60 秒窗口内出现的流总数
    "flow_interval_jitter",   # 时间间隔的抖动程度
    "flow_interval_diff_mean",  # 同一个源与同一个目标之间，最近连接时间间隔的平均值
    "tls_record_count",   # TLS 协议传输数据时使用的小容器的数量
    "reconnection_flag",  # 是否在最近窗口中观察到同一个源和同一个目标重复连接
    "unique_dst_count",   # 最近 60 秒内，同一个源连接了多少个不同的：目标IP + 目标端口
    "src_ip_abnormal_ratio", # 最近 60 秒内，同一源 IP 发起的流中，有多少比例表现为：连接被重置；连接不完整；握手未完整观察到。
    "duration_p25",   # duration 是流持续时间，duration_p25表示大约有 25% 的流时长不超过这个值。
    "duration_p50", # 第 50 百分位，也就是中位数。大约一半流比它短，一半流比它长。
    "duration_p75", # 大约有 75% 的流时长不超过这个值。
    "weighted_conn_count", # 统计最近连接，但距离当前流越近的连接权重越大。
    "weighted_avg_duration", # 最近流持续时间的加权平均。
    "abnormal_to_conn_ratio", # 最近 60 秒所有流中，异常连接占总连接的比例
    "handshake_duration", # 表示 TCP 三次握手耗时。
    # 域名和证书特征
    # （CN 是 Common Name，中文可以理解为证书中的“通用名称”。现代证书更主要依赖 SAN，但 CN 仍然可以作为分析信息。如果没有看到证书，或者证书没有 CN，这些 CN 特征会记为缺失。）
    #（SNI 是 Server Name Indication。客户端连接一个服务器时，可能先告诉服务器：我想访问 www.example.com。这个名称就是 SNI。）
    "cn_vowel_ratio", # 计算 CN 中英文字母元音的比例。
    "cn_digit_density", # CN 中数字占全部字符的比例。
    "cn_special_char_density", # CN 中非字母、非数字字符的比例。
    "cn_length", # CN 的字符长度。
    "sni_length", # 表示 SNI 主机名的字符长度。
    "cert_valid_days", # 表示证书总有效期是多少天。
    "cert_age_at_capture", # 抓包时，证书已经生效了多少天。
    "cert_remaining_days", # 抓包时距离证书过期还有多少天。
    "cert_chain_depth", # 它统计的是抓包中实际看见并解析到的证书数量，不保证是完整的全球信任链。
]
