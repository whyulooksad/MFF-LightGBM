import os
import sys
import gc
import time
import re
import struct
from collections import defaultdict
from tqdm import tqdm

# 开启 C 级别崩溃信号拦截器
import faulthandler

faulthandler.enable()

PCAP_DIR_2017 = r"I:\CICMalAnal2017\pcap\pcap"
PCAP_DIR_2020 = r"I:\CIC-DoHBrw-2020\PCAPs"

# 自动获取脚本自身所在文件夹的绝对路径（D:\jinxian\Pycharm\比赛\）
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_OUT_CSV = "raw_pcap_features_unlabeled.csv"
TIME_WINDOW = 60
FLUSH_THRESHOLD = 100000  # 内存积攒 10 万包后统一批量落盘一次


# ==================== 【物理基础辅助函数】 ====================

def get_label_from_filepath(pcap_path):
    path_normalized = pcap_path.replace("\\", "/").lower()
    filename_normalized = os.path.basename(pcap_path).lower()
    if "cic-dohbrw-2020" in path_normalized:
        if "dohbenign-nondoh" in path_normalized:
            return "benign"
        elif "dohmalicious" in path_normalized:
            if "dns2tcp" in path_normalized:
                return "dns2tcp"
            elif "dnscat2" in path_normalized:
                return "dnscat2"
            elif "iodine" in path_normalized:
                return "iodine"
    elif "cicmalanal2017" in path_normalized:
        if "benign" in filename_normalized: return "benign"
        for label in ["adware", "smsmalware", "ransomware", "scareware"]:
            if label in filename_normalized or f"/{label}/" in path_normalized: return label
    return None


def is_pcap_file_valid(pcap_path):
    try:
        if os.path.getsize(pcap_path) < 32: return False
        with open(pcap_path, "rb") as f:
            magic = f.read(4)
        return magic in [b'\xa1\xb2\xc3\xd4', b'\xd4\xc3\xb2\xa1', b'\x0a\x0d\x0d\x0a']
    except:
        return False


# ==================== 【终极非惰性自研物理流解析器（链路层自适应）】 ====================

def parse_pcap_safely(pcap_path):
    """
    非惰性纯 Python 报文解析器。
    读取整整 24 字节的文件头，确保物理偏移量完美对齐，数据顺畅加载。
    """
    packets = []
    try:
        with open(pcap_path, 'rb') as f:
            magic = f.read(4)
            if len(magic) < 4: return packets

            f.seek(0)

            # ---- A. 传统 PCAP 格式解析轨 ----
            if magic in [b'\xa1\xb2\xc3\xd4', b'\xd4\xc3\xb2\xa1']:
                endian = '>' if magic == b'\xa1\xb2\xc3\xd4' else '<'
                # 【核心修正点】：必须读取完整的 24 字节全局文件头
                global_hdr = f.read(24)
                if len(global_hdr) < 24: return packets

                try:
                    # 24 字节文件头中，最后 8 字节（[16:24]）是标准的 snaplen (4B) 和 linktype (4B)
                    snaplen, linktype = struct.unpack(endian + 'II', global_hdr[16:24])
                    linktype = linktype & 0xFFFF
                except:
                    return packets

                # 建立物理安全限幅
                safe_snaplen = min(snaplen, 262144) if snaplen > 0 else 65535

                while True:
                    pkt_hdr = f.read(16)
                    if len(pkt_hdr) < 16: break

                    try:
                        ts_sec, ts_usec, caplen, origlen = struct.unpack(endian + 'IIII', pkt_hdr)
                    except:
                        break

                    if caplen > safe_snaplen or caplen <= 0 or caplen > 262144:
                        break

                    raw_packet = f.read(caplen)
                    if len(raw_packet) < caplen: break

                    pkt_ts = ts_sec + ts_usec / 1000000.0
                    packets.append((pkt_ts, raw_packet, linktype))

            # ---- B. 现代 PCAPNG 格式解析轨 ----
            elif magic == b'\x0a\x0d\x0d\x0a':
                linktype = 1  # 默认链路层以太网
                f.seek(0)

                while True:
                    block_hdr = f.read(8)
                    if len(block_hdr) < 8: break

                    try:
                        b_type, b_len = struct.unpack('<II', block_hdr)
                        if b_len > 262144 or b_len < 12:
                            b_type, b_len = struct.unpack('>II', block_hdr)
                    except:
                        break

                    # 将 Block 长度上限压缩至以太网标准的 256KB
                    if b_len > 262144 or b_len < 12:
                        break

                    body_len = b_len - 12
                    body_data = f.read(body_len)
                    f.read(4)

                    if len(body_data) < body_len: break

                    # 解析 Interface Description Block (IDB)，动态提取真实的链路层类型
                    if b_type == 0x00000001:
                        if len(body_data) < 8: continue
                        try:
                            # IDB 结构: LinkType(2B) | Reserved(2B) | SnapLen(4B)
                            lt = struct.unpack('<H', body_data[:2])[0]
                            linktype = lt & 0xFFFF
                        except:
                            pass

                    # 处理 Enhanced Packet Block (EPB)
                    elif b_type == 0x00000006:
                        if len(body_data) < 20: continue
                        try:
                            interface_id, ts_high, ts_low, caplen, origlen = struct.unpack('<IIIII', body_data[:20])
                            pkt_ts = ((ts_high << 32) + ts_low) / 1000000.0
                        except:
                            continue

                        # 包长安全限制
                        if caplen <= 0 or caplen > 262144 or caplen > (body_len - 20):
                            continue

                        raw_packet = body_data[20:20 + caplen]
                        packets.append((pkt_ts, raw_packet, linktype))

                    # 处理 Simple Packet Block (SPB)
                    elif b_type == 0x00000003:
                        if len(body_data) < 4: continue
                        try:
                            origlen = struct.unpack('<I', body_data[:4])[0]
                        except:
                            continue
                        caplen = min(origlen, body_len - 4)

                        if caplen <= 0 or caplen > 262144: continue
                        raw_packet = body_data[4:4 + caplen]
                        packets.append((time.time(), raw_packet, linktype))
    except Exception:
        pass
    return packets


# ==================== 物理特征过滤类 ====================

class PcapEncryptSlicer:
    def __init__(self):
        self.buffer = defaultdict(list)

    def slice_and_filter_pcap(self, pcap_path, label):
        pcap_filename = os.path.normpath(pcap_path)
        pcap_filename = os.path.basename(pcap_filename)
        if not is_pcap_file_valid(pcap_path): return

        self.buffer.clear()

        # 物理读取
        pcap_packets = parse_pcap_safely(pcap_path)
        if not pcap_packets: return

        pkt_count = 0
        try:
            for pkt_ts, raw_packet, linktype in pcap_packets:
                pkt_count += 1

                if linktype == 1:
                    ip_offset = 14
                elif linktype == 113:
                    ip_offset = 16
                else:
                    ip_offset = 0

                if len(raw_packet) < ip_offset + 20: continue
                ip_data = raw_packet[ip_offset:]
                ip_ver = ip_data[0] >> 4
                if ip_ver == 4:
                    proto = ip_data[9]
                    ip_header_len = (ip_data[0] & 0x0F) * 4
                    trans_start = ip_header_len
                elif ip_ver == 6:
                    if len(ip_data) < 40: continue
                    proto = ip_data[6]
                    trans_start = 40
                else:
                    continue

                if proto != 6: continue
                tcp_data = ip_data[trans_start:trans_start + 4]
                if len(tcp_data) < 4: continue
                sport = int.from_bytes(tcp_data[0:2], 'big')
                dport = int.from_bytes(tcp_data[2:4], 'big')

                # 只提取端口为 443 的 TCP 加密载荷流量
                if sport != 443 and dport != 443:
                    continue

                writer_key = (label, linktype)
                self.buffer[writer_key].append((raw_packet, pkt_ts))

                if len(self.buffer[writer_key]) >= FLUSH_THRESHOLD:
                    self.flush_buffer_to_disk(writer_key)

        except Exception:
            return
        finally:
            self.flush_all_remaining_buffers()

    def flush_buffer_to_disk(self, writer_key):
        """
        【标准 PCAP 物理拼接写入引擎 - 小端字节序纠正】
        """
        label, linktype = writer_key
        packets = self.buffer[writer_key]
        if not packets: return

        clean_linktype = linktype & 0xFFFF
        link_name = "ethernet" if clean_linktype == 1 else ("cooked" if clean_linktype == 113 else "raw")
        output_name = os.path.join(WORK_DIR, f"{label}_encrypted_{link_name}.pcap")

        file_exists = os.path.exists(output_name)

        with open(output_name, 'ab') as f_out:
            # A. 若文件不存在，写入标准的 24 字节全局文件头
            if not file_exists:
                # 使用 0xa1b2c3d4 配合小端符 < 精准产生标准小端魔数 d4 c3 b2 a1
                global_hdr = struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, clean_linktype)
                f_out.write(global_hdr)

            # B. 循环追加标准的 16 字节报文头与数据载荷
            for raw_packet, pkt_ts in packets:
                ts_sec = int(pkt_ts)
                ts_usec = int((pkt_ts - ts_sec) * 1000000)
                caplen = len(raw_packet)
                origlen = caplen

                # 格式: ts_sec(4B) | ts_usec(4B) | caplen(4B) | origlen(4B)
                pkt_hdr = struct.pack('<IIII', ts_sec, ts_usec, caplen, origlen)
                f_out.write(pkt_hdr)
                f_out.write(raw_packet)

            # 向控制台实时输出落盘成果
            print(f"  -> [磁盘追加成功]: {output_name} (物理追加写入了 {len(packets)} 个 443 TCP 包)")

        # 清空当前内存缓冲区
        self.buffer[writer_key].clear()

    def flush_all_remaining_buffers(self):
        for writer_key in list(self.buffer.keys()):
            self.flush_buffer_to_disk(writer_key)
        self.buffer.clear()


def main():
    start_time = time.time()

    # 自动检索并物理清除 WORK_DIR 目录下残留的旧加密 PCAP 文件，防止旧版数据污染
    print(f"正在物理清理指定目录 ({WORK_DIR}) 下残留的旧版缓存文件...")
    for file in os.listdir(WORK_DIR):
        if file.endswith(".pcap") and "_encrypted_" in file:
            try:
                full_del_path = os.path.join(WORK_DIR, file)
                os.remove(full_del_path)
                print(f"  -> 已成功清理旧文件: {full_del_path}")
            except:
                pass

    slicer = PcapEncryptSlicer()
    pcap_tasks = []

    if os.path.exists(PCAP_DIR_2017):
        for root, _, files in os.walk(PCAP_DIR_2017):
            for file in files:
                if file.endswith(".pcap") or file.endswith(".pcap_ISCX"):
                    full_path = os.path.join(root, file)
                    lbl = get_label_from_filepath(full_path)
                    if lbl: pcap_tasks.append((full_path, lbl))

    if os.path.exists(PCAP_DIR_2020):
        for root, _, files in os.walk(PCAP_DIR_2020):
            for file in files:
                if file.endswith(".pcap") or file.endswith(".pcap_ISCX"):
                    full_path = os.path.join(root, file)
                    lbl = get_label_from_filepath(full_path)
                    if lbl: pcap_tasks.append((full_path, lbl))

    # 全局关闭 tqdm 后台监控线程
    tqdm.monitor_interval = 0

    print(f"\n共发现 {len(pcap_tasks)} 个待解析文件。启动流式‘加密包物理切片’流水线...")
    for pcap_path, label in tqdm(pcap_tasks, desc="全局切片总进度"):
        slicer.slice_and_filter_pcap(pcap_path, label)

    print(f"\n[物理切片落盘成功完成！] 物理安全防御部署成功。总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()