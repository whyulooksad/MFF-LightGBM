"""
将 pcap/pcapng 中的每条双向流截断为前 N 个包。
直接修改下方 INPUT_DIR / OUTPUT_DIR 后运行即可，无需命令行参数。
"""

import os
import struct
import sys
import time
from collections import defaultdict

# ==================== 配置 ====================
DEFAULT_MAX_PKTS_PER_FLOW = 200        # 每条流保留前 N 个包

# ---------- 在这里设置你的输入和输出目录 ----------
INPUT_DIR = "path/to/your/input/pcaps"   # 输入目录，包含多个 pcap/pcapng 文件
OUTPUT_DIR = "path/to/your/output/pcaps" # 输出目录，结果将保存到这里（自动创建）
# -------------------------------------------------

# ==================== PCAP 引擎 ====================

def fast_pcap_iter(file_path):
    """
    纯 Python pcap/pcapng 迭代器。
    返回迭代器，每个元素为 (timestamp, raw_bytes, link_type)。
    """
    try:
        with open(file_path, 'rb') as f:
            global_hdr = f.read(24)
            if len(global_hdr) < 24:
                return

            magic = global_hdr[0:4]

            # === 标准 pcap 格式 ===
            if magic in [b'\xa1\xb2\xc3\xd4', b'\xd4\xc3\xb2\xa1']:
                endian = '>' if magic == b'\xa1\xb2\xc3\xd4' else '<'
                linktype = struct.unpack(endian + 'I', global_hdr[20:24])[0] & 0xFFFF

                while True:
                    pkt_hdr = f.read(16)
                    if len(pkt_hdr) < 16:
                        break

                    if endian == '<':
                        ts_sec = struct.unpack('<I', pkt_hdr[0:4])[0]
                        ts_usec = struct.unpack('<I', pkt_hdr[4:8])[0]
                        caplen = struct.unpack('<I', pkt_hdr[8:12])[0]
                    else:
                        ts_sec = struct.unpack('>I', pkt_hdr[0:4])[0]
                        ts_usec = struct.unpack('>I', pkt_hdr[4:8])[0]
                        caplen = struct.unpack('>I', pkt_hdr[8:12])[0]

                    if caplen <= 0 or caplen > 262144:
                        break

                    raw_packet = f.read(caplen)
                    if len(raw_packet) < caplen:
                        break

                    pkt_ts = ts_sec + ts_usec / 1_000_000.0
                    yield pkt_ts, raw_packet, linktype

            # === pcapng 格式 ===
            elif magic == b'\x0a\x0d\x0d\x0a':
                f.seek(0)  # 回到文件开头，重新解析
                linktype = 1  # 以太网默认

                while True:
                    block_hdr = f.read(8)
                    if len(block_hdr) < 8:
                        break

                    # 尝试小端解析块类型和长度
                    b_type = struct.unpack('<I', block_hdr[0:4])[0]
                    b_len = struct.unpack('<I', block_hdr[4:8])[0]

                    # 长度不合理则尝试大端
                    if b_len > 262144 or b_len < 12:
                        b_type = struct.unpack('>I', block_hdr[0:4])[0]
                        b_len = struct.unpack('>I', block_hdr[4:8])[0]

                    if b_len > 262144 or b_len < 12:
                        break

                    body_len = b_len - 12
                    body_data = f.read(body_len)
                    if len(body_data) < body_len:
                        break

                    # 读取块尾长度拷贝
                    len_copy_data = f.read(4)
                    if len(len_copy_data) < 4:
                        break
                    len_copy = struct.unpack('<I', len_copy_data)[0]
                    # 如果小端拷贝不匹配，尝试大端（保证双端校验）
                    if len_copy != b_len:
                        len_copy_be = struct.unpack('>I', len_copy_data)[0]
                        if len_copy_be != b_len:
                            break  # 畸变数据，停止

                    # 接口描述块
                    if b_type == 0x00000001 and len(body_data) >= 8:
                        linktype = struct.unpack('<H', body_data[0:2])[0] & 0xFFFF

                    # 增强包块 (EPB)
                    elif b_type == 0x00000006 and len(body_data) >= 20:
                        ts_high = struct.unpack('<I', body_data[4:8])[0]
                        ts_low = struct.unpack('<I', body_data[8:12])[0]
                        caplen = struct.unpack('<I', body_data[12:16])[0]
                        pkt_ts = ((ts_high << 32) + ts_low) / 1_000_000.0

                        if 0 < caplen <= 262144 and caplen <= (body_len - 20):
                            raw_packet = body_data[20:20 + caplen]
                            yield pkt_ts, raw_packet, linktype

                    # 简单包块 (SPB)
                    elif b_type == 0x00000003 and len(body_data) >= 4:
                        origlen = struct.unpack('<I', body_data[0:4])[0]
                        caplen = min(origlen, body_len - 4)
                        if 0 < caplen <= 262144:
                            raw_packet = body_data[4:4 + caplen]
                            yield time.time(), raw_packet, linktype
            else:
                print(f"  [错误] 无法识别的文件格式（magic: {magic!r}）")
                return
    except Exception as e:
        print(f"  [异常] PCAP 读取失败: {e}")
        return


class SimplePcapWriter:
    """纯 Python 标准 pcap 写入器"""
    def __init__(self, fileobj, linktype=1):
        self.f = fileobj
        # 写入 global header
        self.f.write(struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, linktype & 0xFFFF))

    def writepkt(self, raw_packet, pkt_ts):
        ts_sec = int(pkt_ts)
        ts_usec = int((pkt_ts - ts_sec) * 1_000_000)
        caplen = len(raw_packet)
        self.f.write(struct.pack('<IIII', ts_sec, ts_usec, caplen, caplen))
        self.f.write(raw_packet)


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
    pkt_iter = fast_pcap_iter(input_path)
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
                    writer = SimplePcapWriter(f_out, linktype)

                # 提取双向流键
                flow_key = extract_four_tuple(raw_pkt)
                if flow_key is None:
                    continue   # 非 IP 包直接跳过

                # 新流计数
                if flow_key not in flow_pkt_count:
                    flow_count += 1

                # 最多写入 max_pkts 个包
                if flow_pkt_count[flow_key] < max_pkts:
                    writer.writepkt(raw_pkt, pkt_ts)
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

    # 收集所有 pcap/pcapng 文件（仅顶层，不递归）
    pcap_files = [f for f in os.listdir(INPUT_DIR)
                  if f.lower().endswith(('.pcap', '.pcapng'))]
    if not pcap_files:
        print(f"警告：在 {INPUT_DIR} 中没有找到 .pcap/.pcapng 文件")
        sys.exit(0)

    print(f"找到 {len(pcap_files)} 个文件，开始批量处理...\n")

    total_read = 0
    total_written = 0
    total_flows = 0

    for fname in pcap_files:
        in_file = os.path.join(INPUT_DIR, fname)
        out_file = os.path.join(OUTPUT_DIR, fname)
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