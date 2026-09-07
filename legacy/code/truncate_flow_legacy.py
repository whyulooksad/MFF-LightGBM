"""
按双向流将 PCAP/PCAPNG 中每条连接截断为前 N 个包。
用户单包推理和开发者离线数据准备共同复用这一实现。
"""

import os
import struct
import sys
from collections import defaultdict

from user_app.inference.config import PCAP_RAW_DIR, PCAP_TRUNCATED_DIR
from user_app.inference.pcap_reader import PcapWriter, read_packets

# ==================== 配置 ====================
DEFAULT_MAX_PKTS_PER_FLOW = 200        # 每条流保留前 N 个包

# ---------- 在这里设置你的输入和输出目录 ----------
INPUT_DIR = str(PCAP_RAW_DIR)
OUTPUT_DIR = str(PCAP_TRUNCATED_DIR)
# -------------------------------------------------

# ==================== 四元组提取 ====================

def extract_four_tuple(raw_packet):
    """
    从原始报文中提取 IPv4 双向流四元组（小IP, 小端口, 大IP, 大端口）。
    返回 None 表示无法解析（非IP或异常）。
    """
    try:
        # 寻找 IP 头起始位置（以太网头部通常 14 字节，但支持 802.1Q 等）
        ip_offset = None
        for offset in [14, 16, 18, 0]:
            if len(raw_packet) >= offset + 20:
                ver = raw_packet[offset] >> 4
                if ver == 4:
                    ip_offset = offset
                    break
        if ip_offset is None:
            return None

        ip_data = raw_packet[ip_offset:]
        if len(ip_data) < 20:
            return None

        header_len = (ip_data[0] & 0x0F) * 4
        if len(ip_data) < header_len + 4:  # 至少需要协议字段和两个端口
            return None

        # IPv4 地址（网络字节序）
        sip_int = struct.unpack_from('>I', ip_data, 12)[0]
        dip_int = struct.unpack_from('>I', ip_data, 16)[0]
        # TCP/UDP 端口（紧跟 IP 头）
        sport = struct.unpack_from('>H', ip_data, header_len)[0]
        dport = struct.unpack_from('>H', ip_data, header_len + 2)[0]

        # 转换为双向键：IP 小的在前
        if sip_int < dip_int:
            return (sip_int, sport, dip_int, dport)
        else:
            return (dip_int, dport, sip_int, sport)
    except Exception:
        return None


# ==================== 核心截断函数 ====================

def truncate_flows_in_pcap(input_path, output_path, max_pkts=DEFAULT_MAX_PKTS_PER_FLOW):
    """
    读取 input_path 中的 pcap，将每条双向流截断为前 max_pkts 个包，
    写入 output_path。
    """
    flow_pkt_count = defaultdict(int)  # 每条流已写入的包数
    pkt_read = 0
    pkt_written = 0
    flow_count = 0
    linktype = 1

    print(f"[截断] 输入: {input_path}")
    print(f"       输出: {output_path}")
    print(f"       每条流保留前 {max_pkts} 个包")

    # 迭代数据包
    pkt_iter = read_packets(input_path)
    try:
        with open(output_path, 'wb') as f_out:
            writer = None  # 延迟初始化，等待第一个包获取 linktype

            for pkt_ts, raw_pkt, lt in pkt_iter:
                pkt_read += 1
                if pkt_read % 500000 == 0:
                    sys.stdout.write(f"\r  已扫描 {pkt_read} 包，写入 {pkt_written} 包，流数 {flow_count}")
                    sys.stdout.flush()

                # 初始化写入器（基于第一个包的链路类型）
                if writer is None:
                    linktype = lt
                    writer = PcapWriter(f_out, linktype)

                # 提取双向流键
                flow_key = extract_four_tuple(raw_pkt)
                if flow_key is None:
                    continue   # 非 IP 包直接跳过

                # 新流计数
                if flow_key not in flow_pkt_count:
                    flow_count += 1

                # 最多写入 max_pkts 个包
                if flow_pkt_count[flow_key] < max_pkts:
                    writer.write(raw_pkt, pkt_ts)
                    flow_pkt_count[flow_key] += 1
                    pkt_written += 1

            sys.stdout.write("\n")
    finally:
        pkt_iter.close()

    print(f"[完成] 扫描 {pkt_read} 包，写入 {pkt_written} 包，涉及 {flow_count} 条双向流")
    return pkt_read, pkt_written, flow_count


# ==================== 主程序 ====================

if __name__ == "__main__":
    # 检查输入目录
    if not os.path.isdir(INPUT_DIR):
        print(f"错误：输入目录不存在: {INPUT_DIR}")
        sys.exit(1)

    # 创建输出目录（如果不存在）
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 递归扫描 prepared/train、validation、test，便于三种数据集切分共用入口。
    pcap_files = []
    for root, _, files in os.walk(INPUT_DIR):
        for filename in files:
            if filename.lower().endswith(('.pcap', '.pcapng')):
                pcap_files.append(os.path.join(root, filename))
    if not pcap_files:
        print(f"警告：在 {INPUT_DIR} 中没有找到 .pcap/.pcapng 文件")
        sys.exit(0)

    print(f"找到 {len(pcap_files)} 个文件，开始批量处理...\n")

    total_read = 0
    total_written = 0
    total_flows = 0

    for in_file in pcap_files:
        relative_name = os.path.relpath(in_file, INPUT_DIR)
        # 输出仍保持平铺，加入 split 前缀避免不同切分中的同名文件冲突。
        out_name = relative_name.replace(os.sep, "__")
        out_file = os.path.join(OUTPUT_DIR, out_name)
        read, written, flows = truncate_flows_in_pcap(in_file, out_file)
        total_read += read
        total_written += written
        total_flows += flows
        print()  # 空行分隔

    print("======= 全部完成 =======")
    print(f"文件数: {len(pcap_files)}")
    print(f"总扫描包数: {total_read}")
    print(f"总写入包数: {total_written}")
    print(f"总流数: {total_flows}")
