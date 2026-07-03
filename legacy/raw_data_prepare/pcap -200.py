import warnings

# 彻底屏蔽所有烦人的 Pandas 警告
warnings.filterwarnings("ignore")

import os
import sys
import gc
import re
import random
import time
import struct
from collections import defaultdict
from tqdm import tqdm

# 开启 C 级别崩溃信号拦截器
import faulthandler

faulthandler.enable()

# 全局关闭 tqdm 内部监控线程
tqdm.monitor_interval = 0

# 脚本会自动定位当前工作目录
WORK_DIR = os.path.dirname(os.path.abspath(__file__))

# 抽样直接指向清洗后的干净 CSV 索引根目录
PROCESSED_CSV_DIR = os.path.join(WORK_DIR, "processed_csv")

# 设定抽样阈值 (Benign 抽取 60,000 条双向流，其余恶意大类各抽取 6,000 条双向流)
LIMIT_BENIGN = 60000
LIMIT_MALICIOUS = 6000

# 【安全上限提升】最大扫描上限提升至 5000 万，满足 4230 万深度扫描需求
MAX_SCAN_PACKETS = 50000000

# 【收敛判定区间】扫描超过 200 万报文后，若连续 100 万个包无新流，才判定收敛
CONVERGENCE_INTERVAL = 1000000

# ===== 新增配置 =====
TRAIN_RATIO = 0.8  # 训练集占总配额的比例
MAX_PKTS_PER_FLOW = 200  # 每条流最多保留的数据包数量


# ==================== 【辅助工具函数】 ====================

def ip_to_int(ip_str):
    """
    将 IPv4 字符串高效转化为 32 位整型。
    """
    try:
        parts = ip_str.split('.')
        return (int(parts[0]) << 24) | (int(parts[1]) << 16) | (int(parts[2]) << 8) | int(parts[3])
    except:
        return 0


def int_to_ip(ip_int):
    """
    将 32 位整型还原为可读的 IPv4 字符串。
    """
    return f"{(ip_int >> 24) & 255}.{(ip_int >> 16) & 255}.{(ip_int >> 8) & 255}.{ip_int & 255}"


# ==================== 【自研免依赖 PCAP 物理读写引擎】 ====================

def fast_pcap_iter(file_path):
    """
    自研、纯 Python 极速物理报文读取迭代器。
    带有 PCAPNG 双端包长对齐校验，100% 预防畸形数据包失步引发的闪退。
    """
    try:
        with open(file_path, 'rb') as f:
            global_hdr = f.read(24)
            if len(global_hdr) < 24: return

            magic = global_hdr[0:4]
            if magic in [b'\xa1\xb2\xc3\xd4', b'\xd4\xc3\xb2\xa1']:
                endian = '>' if magic == b'\xa1\xb2\xc3\xd4' else '<'
                linktype = struct.unpack(endian + 'I', global_hdr[20:24])[0] & 0xFFFF

                while True:
                    pkt_hdr = f.read(16)
                    if len(pkt_hdr) < 16: break

                    if endian == '<':
                        ts_sec = pkt_hdr[0] + (pkt_hdr[1] << 8) + (pkt_hdr[2] << 16) + (pkt_hdr[3] << 24)
                        ts_usec = pkt_hdr[4] + (pkt_hdr[5] << 8) + (pkt_hdr[6] << 16) + (pkt_hdr[7] << 24)
                        caplen = pkt_hdr[8] + (pkt_hdr[9] << 8) + (pkt_hdr[10] << 16) + (pkt_hdr[11] << 24)
                    else:
                        ts_sec = (pkt_hdr[0] << 24) + (pkt_hdr[1] << 16) + (pkt_hdr[2] << 8) + pkt_hdr[3]
                        ts_usec = (pkt_hdr[4] << 24) + (pkt_hdr[5] << 16) + (pkt_hdr[6] << 8) + pkt_hdr[7]
                        caplen = (pkt_hdr[8] << 24) + (pkt_hdr[9] << 16) + (pkt_hdr[10] << 8) + pkt_hdr[11]

                    if caplen <= 0 or caplen > 262144:
                        break

                    raw_packet = f.read(caplen)
                    if len(raw_packet) < caplen: break

                    pkt_ts = ts_sec + ts_usec / 1000000.0
                    yield pkt_ts, raw_packet, linktype

            elif magic == b'\x0a\x0d\x0d\x0a':
                linktype = 1
                f.seek(0)
                while True:
                    block_hdr = f.read(8)
                    if len(block_hdr) < 8: break
                    try:
                        b_type = block_hdr[0] + (block_hdr[1] << 8) + (block_hdr[2] << 16) + (block_hdr[3] << 24)
                        b_len = block_hdr[4] + (block_hdr[5] << 8) + (block_hdr[6] << 16) + (block_hdr[7] << 24)
                        if b_len > 262144 or b_len < 12:
                            b_type = (block_hdr[0] << 24) + (block_hdr[1] << 16) + (block_hdr[2] << 8) + block_hdr[3]
                            b_len = (block_hdr[4] << 24) + (block_hdr[5] << 16) + (block_hdr[6] << 8) + block_hdr[7]
                    except:
                        break
                    if b_len > 262144 or b_len < 12: break

                    body_len = b_len - 12
                    body_data = f.read(body_len)

                    # 【核心物理同步修复】：读取块尾的 4 字节块长度拷贝
                    len_copy_data = f.read(4)
                    if len(len_copy_data) < 4: break
                    try:
                        len_copy = len_copy_data[0] + (len_copy_data[1] << 8) + (len_copy_data[2] << 16) + (
                                len_copy_data[3] << 24)
                        # 双端校验不一致，判定数据畸变/失步，立即断开，防止误将报文数据作为包头读取导致溢出
                        if len_copy != b_len and len_copy != socket_reverse_32bit(b_len):
                            break
                    except:
                        break

                    if len(body_data) < body_len: break

                    # 解析 Interface Description Block (IDB)
                    if b_type == 0x00000001:
                        if len(body_data) < 8: continue
                        try:
                            lt = body_data[0] + (body_data[1] << 8)
                            linktype = lt & 0xFFFF
                        except:
                            pass

                    # 处理 Enhanced Packet Block (EPB)
                    elif b_type == 0x00000006:
                        if len(body_data) < 20: continue
                        try:
                            ts_high = body_data[4] + (body_data[5] << 8) + (body_data[6] << 16) + (body_data[7] << 24)
                            ts_low = body_data[8] + (body_data[9] << 8) + (body_data[10] << 16) + (body_data[11] << 24)
                            caplen = body_data[12] + (body_data[13] << 8) + (body_data[14] << 16) + (
                                    body_data[15] << 24)
                            pkt_ts = ((ts_high << 32) + ts_low) / 1000000.0
                        except:
                            continue

                        if caplen <= 0 or caplen > 262144 or caplen > (body_len - 20):
                            continue

                        raw_packet = body_data[20:20 + caplen]
                        yield pkt_ts, raw_packet, linktype

                    # 处理 Simple Packet Block (SPB)
                    elif b_type == 0x00000003:
                        if len(body_data) < 4: continue
                        try:
                            origlen = body_data[0] + (body_data[1] << 8) + (body_data[2] << 16) + (body_data[3] << 24)
                        except:
                            continue
                        caplen = min(origlen, body_len - 4)

                        if caplen <= 0 or caplen > 262144: continue
                        raw_packet = body_data[4:4 + caplen]
                        yield time.time(), raw_packet, linktype
            else:
                return
    except Exception as e:
        print(f"    [迭代器系统异常] 读取 PCAP 失败: {e}")
        return


def socket_reverse_32bit(val):
    try:
        return struct.unpack("<I", struct.pack(">I", val))[0]
    except:
        return 0


class SimplePcapWriter(object):
    """
    自研、纯 Python 标准 PCAP 物理写入器。
    """

    def __init__(self, f_out, linktype=1):
        self.f = f_out
        clean_linktype = linktype & 0xFFFF
        global_hdr = struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, clean_linktype)
        self.f.write(global_hdr)

    def writepkt(self, raw_packet, pkt_ts):
        ts_sec = int(pkt_ts)
        ts_usec = int((pkt_ts - ts_sec) * 1000000)
        caplen = len(raw_packet)
        pkt_hdr = struct.pack('<IIII', ts_sec, ts_usec, caplen, caplen)
        self.f.write(pkt_hdr)
        self.f.write(raw_packet)


# ==================== 【动态加载已洗净的局部 CSV 索引】 ====================

def load_clean_local_csv_index(label, year):
    """
    自适应大小写目录的高速局部索引加载器。
    """
    local_index = defaultdict(list)

    folder_prefix = "CIC2017" if str(year) == "2017" else "DoH2020"
    base_year_dir = os.path.join(PROCESSED_CSV_DIR, folder_prefix)

    target_folder = None
    if os.path.exists(base_year_dir):
        for name in os.listdir(base_year_dir):
            if name.lower() == label.lower():
                target_folder = os.path.join(base_year_dir, name)
                break

    if not target_folder:
        for root, dirs, _ in os.walk(PROCESSED_CSV_DIR):
            for d in dirs:
                if d.lower() == label.lower():
                    if folder_prefix.lower() in root.lower():
                        target_folder = os.path.join(root, d)
                        break
            if target_folder: break

    if not target_folder or not os.path.exists(target_folder):
        print(f"  -> [提示] 未找到 {year} 数据集分类 ({label}) 的已清洗 CSV 目录。")
        return local_index

    loaded_lines = 0
    try:
        for root, _, files in os.walk(target_folder):
            for file in files:
                if file.endswith(".csv"):
                    csv_path = os.path.join(root, file)
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        header = f.readline().strip().split(',')

                        sip_idx = header.index("src_ip")
                        dip_idx = header.index("dst_ip")
                        sp_idx = header.index("src_port")
                        dp_idx = header.index("dst_port")
                        ts_idx = header.index("ts_epoch")

                        for line in f:
                            parts = line.split(',')
                            if len(parts) <= max(sip_idx, dip_idx, sp_idx, dp_idx, ts_idx): continue
                            try:
                                sip = parts[sip_idx].strip()
                                dip = parts[dip_idx].strip()

                                sp = int(float(parts[sp_idx].strip()))
                                dp = int(float(parts[dp_idx].strip()))
                                ts = int(float(parts[ts_idx].strip()))

                                if ts == 0: continue

                                sip_int = ip_to_int(sip)
                                dip_int = ip_to_int(dip)
                                if sip_int == 0 or dip_int == 0: continue

                                if sip_int < dip_int:
                                    key = (sip_int, sp, dip_int, dp)
                                else:
                                    key = (dip_int, dp, sip_int, sp)

                                local_index[key].append(ts)
                                loaded_lines += 1
                            except:
                                pass
    except Exception as e:
        print(f"  -> [读取错误] {year} 清洗缓存加载失败: {e}")

    if loaded_lines > 0:
        print(f"  -> 成功从 {folder_prefix} CSV 载入了 {loaded_lines} 条流时间记录。")
    return local_index


# ==================== 【双向会话抽样处理器】 ====================

class PcapSessionSampler:
    def __init__(self):
        pass

    def extract_four_tuple(self, raw_packet, linktype):
        """
        纯 Python 高稳定度物理报文解析器。
        彻底消除内部切片，添加严格长度判断，实现零异常、低内存开销。
        """
        try:
            ip_offset = None
            for offset in [14, 16, 18, 0]:
                if len(raw_packet) >= offset + 20:
                    ver = raw_packet[offset] >> 4
                    if ver in [4, 6]:
                        ip_offset = offset
                        break

            if ip_offset is None:
                return None

            ip_data = raw_packet[ip_offset:]
            if len(ip_data) < 20:
                return None

            ver = ip_data[0] >> 4
            if ver == 4:
                header_len = (ip_data[0] & 0x0F) * 4
                if len(ip_data) < header_len + 4:
                    return None

                sip_int = (ip_data[12] << 24) | (ip_data[13] << 16) | (ip_data[14] << 8) | ip_data[15]
                dip_int = (ip_data[16] << 24) | (ip_data[17] << 16) | (ip_data[18] << 8) | ip_data[19]
                trans_start = header_len
            else:
                return None

            sport = (ip_data[trans_start] << 8) | ip_data[trans_start + 1]
            dport = (ip_data[trans_start + 2] << 8) | ip_data[trans_start + 3]

            if sip_int < dip_int:
                return (sip_int, sport, dip_int, dport)
            else:
                return (dip_int, dport, sip_int, sport)
        except:
            return None

    def sample_pcap_sessions(self, input_pcap_path, train_limit, test_limit, label, year):
        """
        修改后的核心抽样函数：
        - train_limit: 训练集所需的最大双向流数量
        - test_limit : 测试集所需的最大双向流数量
        输出两个 PCAP 文件，每条流最多保留前 MAX_PKTS_PER_FLOW 个包。
        """
        pcap_filename = os.path.basename(input_pcap_path)
        base_name = pcap_filename.replace("_encrypted_", "_sampled_").replace(".pcap", "")
        output_train = os.path.join(WORK_DIR, f"{base_name}_train.pcap")
        output_test = os.path.join(WORK_DIR, f"{base_name}_test.pcap")

        print(f"\n[{year} 数据集双向流抽样] 扫描文件: {pcap_filename}...")
        print(f"  -> 训练限额: {train_limit} 条流，测试限额: {test_limit} 条流")

        if label != "benign":
            local_index = load_clean_local_csv_index(label, year)
            print(f"  -> 载入匹配规则共 {len(local_index)} 条")
        else:
            local_index = {}

        unique_flows = set()
        pkt_count = 0
        linktype = 1

        # 记录最后一次发现新流时的报文位置
        last_new_flow_pkt = 0
        target_total = train_limit + test_limit  # 总共需要发现的流数量

        # ==================== 1. 高速扫描阶段 ====================
        scan_gen = fast_pcap_iter(input_pcap_path)
        try:
            for pkt_ts, buf, lt in scan_gen:
                pkt_count += 1
                linktype = lt

                # 每 10 万个包打印进度
                if pkt_count % 100000 == 0:
                    sys.stdout.write(f"\r  -> 已高速扫描 {pkt_count} 报文... (当前抓取双向流: {len(unique_flows)} 条)")
                    sys.stdout.flush()

                # 每 100 万个报文强制垃圾回收工作
                if pkt_count % 1000000 == 0:
                    gc.collect()

                flow_key = self.extract_four_tuple(buf, lt)

                if flow_key:
                    is_new_flow = False

                    if label == "benign":
                        if flow_key not in unique_flows:
                            unique_flows.add(flow_key)
                            is_new_flow = True
                    elif local_index:
                        csv_times = local_index.get(flow_key)
                        if csv_times:
                            pkt_epoch = int(pkt_ts)
                            match_found = False
                            for csv_ts in csv_times:
                                if abs(pkt_epoch - csv_ts) <= 60:
                                    match_found = True
                                    break
                            if match_found:
                                if flow_key not in unique_flows:
                                    unique_flows.add(flow_key)
                                    is_new_flow = True

                    # 记录新流发现
                    if is_new_flow:
                        last_new_flow_pkt = pkt_count

                    # 满足总抽样需求，提前终止
                    if len(unique_flows) >= target_total:
                        sys.stdout.write("\n")  # 补齐换行
                        print(f" -> [提前终止] 已在文件前方凑齐经过 CSV 强认证的 {target_total} 条双向流！")
                        break

                    # 【收敛判定】：扫描超过 200 万报文后，且连续 100 万个报文无任何新会话，直接退出
                    if pkt_count > 2000000 and (pkt_count - last_new_flow_pkt) >= CONVERGENCE_INTERVAL:
                        sys.stdout.write("\n")
                        print(
                            f" -> [收敛终止] 连续 {pkt_count - last_new_flow_pkt} 个报文未发现任何新流。流已收敛，共抓取 {len(unique_flows)} 条流。")
                        break

                    # 【安全上限截断】
                    if pkt_count >= MAX_SCAN_PACKETS:
                        sys.stdout.write("\n")
                        print(f" -> [安全截断] 已扫描达 {pkt_count} 个报文的安全上限，直接进入随机抽样...")
                        break
            else:
                sys.stdout.write("\n")
        finally:
            scan_gen.close()
            del scan_gen
            gc.collect()

        total_discovered = len(unique_flows)
        print(f" -> 该 PCAP 中共探测到合格的双向流: {total_discovered} 条。")

        # ==================== 诊断打印器 (保留原逻辑) ====================
        if total_discovered == 0 and label != "benign" and len(local_index) > 0:
            print("\n[物理诊断调试信息 - 强行对比对齐偏差]")
            print(f"  A. 本地 CSV 索引前 5 个对齐 Key 样本:")
            idx_debug_count = 0
            for k, ts_list in local_index.items():
                readable_key = (int_to_ip(k[0]), k[1], int_to_ip(k[2]), k[3])
                print(f"    - {readable_key} | 包含的时间戳数组: {ts_list[:3]}")
                idx_debug_count += 1
                if idx_debug_count >= 5: break

            print(f"\n  B. 当前 PCAP 前向报文前 5 个提取 Key 样本:")
            pcap_debug_count = 0
            debug_gen = fast_pcap_iter(input_pcap_path)
            try:
                for p_ts, p_buf, p_lt in debug_gen:
                    f_key = self.extract_four_tuple(p_buf, p_lt)
                    if f_key:
                        readable_key = (int_to_ip(f_key[0]), f_key[1], int_to_ip(f_key[2]), f_key[3])
                        print(f"    - {readable_key} | 原始物理时间戳 (Epoch): {int(p_ts)} ({p_ts:.6f})")
                        pcap_debug_count += 1
                        if pcap_debug_count >= 5: break
            finally:
                debug_gen.close()
                del debug_gen
            print("==========================================================\n")

        if total_discovered == 0:
            if 'local_index' in locals():
                local_index.clear()
            return

        # ==================== 分配训练集和测试集 ====================
        # 根据实际发现的流数量，自适应确定训练/测试集大小
        if total_discovered >= target_total:
            # 足够多：严格按限额分配
            train_actual = train_limit
            test_actual = test_limit
        else:
            # 不足：按比例缩放，但不超过各自限额
            train_actual = min(train_limit, int(total_discovered * TRAIN_RATIO))
            test_actual = min(test_limit, total_discovered - train_actual)
            # 防止 test_actual 因取整导致负值
            if test_actual < 0:
                test_actual = 0
                train_actual = total_discovered

        print(f" -> 实际分配训练流: {train_actual}，测试流: {test_actual}")

        # 随机选取流（固定种子保证可复现）
        random.seed(42)
        all_selected = random.sample(list(unique_flows), train_actual + test_actual)
        train_flows = set(all_selected[:train_actual])
        test_flows = set(all_selected[train_actual:])

        unique_flows.clear()
        if local_index:
            local_index.clear()
        gc.collect()

        # ==================== 2. 二次物理写入阶段（前 MAX_PKTS_PER_FLOW 个包） ====================
        print(f"  -> 开始写入抽样报文至训练/测试文件...")
        flow_pkt_count = defaultdict(int)  # 记录每条流已遇到的报文序号（从1开始）
        written_train = 0
        written_test = 0

        write_gen = fast_pcap_iter(input_pcap_path)
        try:
            with open(output_train, 'wb') as f_train, open(output_test, 'wb') as f_test:
                writer_train = SimplePcapWriter(f_train, linktype=linktype)
                writer_test = SimplePcapWriter(f_test, linktype=linktype)

                for pkt_ts, buf, lt_write in write_gen:
                    flow_key = self.extract_four_tuple(buf, lt_write)
                    if flow_key is None:
                        continue

                    # 更新流内包序号
                    flow_pkt_count[flow_key] += 1
                    pkt_seq = flow_pkt_count[flow_key]

                    if pkt_seq > MAX_PKTS_PER_FLOW:
                        # 跳过该流超出的包
                        continue

                    if flow_key in train_flows:
                        writer_train.writepkt(buf, pkt_ts)
                        written_train += 1
                    elif flow_key in test_flows:
                        writer_test.writepkt(buf, pkt_ts)
                        written_test += 1
        finally:
            write_gen.close()
            del write_gen
            gc.collect()

        print(f"  -> [物理写入成功]: 训练集 {output_train} (保留 {written_train} 个报文)")
        print(f"  -> [物理写入成功]: 测试集 {output_test} (保留 {written_test} 个报文)")


def main():
    start_time = time.time()

    sampler = PcapSessionSampler()

    target_files = []
    for file in os.listdir(WORK_DIR):
        if file.endswith(".pcap") and "_encrypted_" in file:
            label = file.split("_encrypted_")[0]

            if "cooked" in file:
                year = "2020"
            elif "ethernet" in file:
                year = "2017"
            else:
                year = "2017"

            target_files.append((os.path.join(WORK_DIR, file), label, year))

    if not target_files:
        print("未在当前目录下发现 2017/2020 相关的中间文件（即含有 '_encrypted_' 的 pcap 文件）。")
        return

    # 排序：将大体积的 benign 任务平滑扫尾
    target_files.sort(key=lambda x: x[1] == "benign")

    print(f"共发现 {len(target_files)} 个待抽样文件。智能准备中...")

    skipped_count = 0
    run_files = []

    # 【核心续跑逻辑升级】：训练和测试文件均已存在才跳过
    for pcap_path, label, year in target_files:
        pcap_filename = os.path.basename(pcap_path)
        base_name = pcap_filename.replace("_encrypted_", "_sampled_").replace(".pcap", "")
        out_train = os.path.join(WORK_DIR, f"{base_name}_train.pcap")
        out_test = os.path.join(WORK_DIR, f"{base_name}_test.pcap")

        train_ok = os.path.exists(out_train) and os.path.getsize(out_train) > 24
        test_ok = os.path.exists(out_test) and os.path.getsize(out_test) > 24

        if train_ok and test_ok:
            skipped_count += 1
        else:
            run_files.append((pcap_path, label, year))

    if skipped_count > 0:
        print(
            f" -> [断点智能续跑已启用] 检测到 {skipped_count} 个文件的训练/测试集先前已抽样完成，自动跳过，继续运行剩余 {len(run_files)} 个任务。")

    if not run_files:
        print("\n[全部成果已在本地！] 所有抽样文件已在历史执行中成功生成。")
        return

    print(f"\n开始抽样剩余的 {len(run_files)} 个文件...")
    for pcap_path, label, year in tqdm(run_files, desc="全局抽样总进度"):
        # 计算该类型对应的训练/测试限额
        if label == "benign":
            total = LIMIT_BENIGN
        else:
            total = LIMIT_MALICIOUS

        train_limit = int(total * TRAIN_RATIO)
        test_limit = total - train_limit

        sampler.sample_pcap_sessions(
            pcap_path,
            train_limit=train_limit,
            test_limit=test_limit,
            label=label,
            year=year
        )

    print(f"\n[物理双向会话抽样全部完成！] 智能协议栈解析完成。总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
