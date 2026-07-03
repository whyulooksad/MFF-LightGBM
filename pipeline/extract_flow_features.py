"""
鲁棒特征提取器（全量处理版：移除所有抽样/截断限制）
输入：../ 目录下 *_sampled_*_train.pcap 和 *_sampled_*_test.pcap
输出：
  - final_multiclass_features_train.csv
  - final_multiclass_features_test.csv
  - flow_metadata_temporal_train.csv
  - flow_metadata_temporal_test.csv
"""

import os, re, sys, gc, json, time, hashlib, struct, bisect, uuid, traceback
import pandas as pd
import numpy as np
from math import exp
from tqdm import tqdm
from multiprocessing import Process, Queue
import glob

try:
    from .config import (
        FLOW_FEATURES_TEST_CSV,
        FLOW_FEATURES_TRAIN_CSV,
        FLOW_METADATA_TEMPORAL_TEST_CSV,
        FLOW_METADATA_TEMPORAL_TRAIN_CSV,
        PCAP_TRUNCATED_DIR,
    )
except ImportError:
    from config import (
        FLOW_FEATURES_TEST_CSV,
        FLOW_FEATURES_TRAIN_CSV,
        FLOW_METADATA_TEMPORAL_TEST_CSV,
        FLOW_METADATA_TEMPORAL_TRAIN_CSV,
        PCAP_TRUNCATED_DIR,
    )

# ================= 可调参数 =================
WORK_DIR = str(PCAP_TRUNCATED_DIR)
FINAL_ML_TRAIN = str(FLOW_FEATURES_TRAIN_CSV)
FINAL_ML_TEST = str(FLOW_FEATURES_TEST_CSV)
FINAL_TEMP_TRAIN = str(FLOW_METADATA_TEMPORAL_TRAIN_CSV)
FINAL_TEMP_TEST = str(FLOW_METADATA_TEMPORAL_TEST_CSV)

TIME_WINDOW = 60                    # 聚合窗口秒数

# 【修改开始】移除所有数据量限制，测试阶段全量处理
MAX_TRACKED_FLOWS = float('inf')    # 不限制内存中流数
MAX_TRACKED_PACKETS = float('inf')  # 每条流统计全部包
MAX_TEMPORAL_PACKETS = float('inf') # 时序记录全部包
MAX_PROCESS_PACKETS = None          # 不限制单文件扫描包数（None为不限制）
# 【修改结束】

EXTRACT_BATCH_SIZE = 5000           # 分批提取特征时每次处理的流数（降低内存峰值）

# Zeek 状态定义
ZEEK_RESET_REJECTED_STATES = {'REJ', 'RSTO', 'RSTR', 'RSTRH'}
ZEEK_HANDSHAKE_FAILURE_STATES = {'S0', 'S1', 'S2', 'S3'}
ZEEK_ABNORMAL_STATES = ZEEK_RESET_REJECTED_STATES.union(ZEEK_HANDSHAKE_FAILURE_STATES)

# 证书 CN 正则（直接使用，不再预检测）
CERT_CN_REGEX = re.compile(rb"\x06\x03\x55\x04\x03[\x0c\x13].([a-zA-Z0-9\-\.]{4,253})")

# CSV 表头
CSV_HEADERS = [
    "flow_uid","src_ip","src_port","dst_ip","dst_port","protocol","timestamp",
    "dataset_source","subfolder","pcap_filename","label",
    "pkts_forward","pkts_backward","pkts_total",
    "bytes_forward","bytes_backward","bytes_total","ratio_bytes_back_to_forward",
    "pkt_len_max","pkt_len_min","pkt_len_mean","pkt_len_std","pkt_len_var",
    "pkt_len_fwd_mean","pkt_len_fwd_std","pkt_len_bwd_mean","pkt_len_bwd_std",
    "flow_bytes_s","flow_pkts_s","fwd_pkts_s","bwd_pkts_s",
    "fwd_header_len","bwd_header_len","down_up_ratio",
    "avg_fwd_segment_size","avg_bwd_segment_size",
    "iat_max","iat_min","iat_mean","iat_std",
    "iat_fwd_max","iat_fwd_min","iat_fwd_mean","iat_fwd_std",
    "iat_bwd_max","iat_bwd_min","iat_bwd_mean","iat_bwd_std",
    "flag_syn_count","flag_fin_count","flag_rst_count","flag_psh_count","flag_ack_count",
    "subflow_fwd_pkts","subflow_fwd_bytes","subflow_bwd_pkts","subflow_bwd_bytes",
    "active_max","active_min","active_mean","active_std",
    "idle_max","idle_min","idle_mean","idle_std",
    "rst_ratio","handshake_fail_rate","reconnect_count","conn_count",
    "flow_interval_jitter","flow_interval_diff_mean","tcp_rst_count","reconnection_flag",
    "unique_dst_count","src_ip_abnormal_ratio",
    "duration_p25","duration_p50","duration_p75",
    "weighted_conn_count","weighted_avg_duration","abnormal_to_conn_ratio",
    "handshake_duration","cn_vowel_ratio","cn_digit_density","cn_special_char_density",
    "cn_length","cn_hash",
    "cert_valid_days","cert_age_at_capture","cert_remaining_days",
    "cert_chain_depth",
    "zeek_conn_log","zeek_ssl_log","zeek_x509_log"
]

TEMPORAL_HEADERS = [
    "flow_uid","src_ip","src_port","dst_ip","dst_port","protocol",
    "start_time","end_time","duration","pcap_filename","label",
    "total_packets","total_bytes","flow_bytes_s","flow_pkts_s",
    "fwd_packets","bwd_packets","fwd_bytes","bwd_bytes",
    "ratio_bytes_bwd_fwd","average_packet_size","pkt_len_std","pkt_len_var",
    "fwd_header_len","bwd_header_len",
    "avg_fwd_segment_size","avg_bwd_segment_size",
    "iat_max","iat_min","iat_mean","iat_std",
    "active_max","active_min","active_mean","active_std",
    "idle_max","idle_min","idle_mean","idle_std",
    "conn_state","syn_count","fin_count","rst_count","psh_count","ack_count",
    "packet_time_offsets","packet_directions","packet_lengths"
]

# ==================== Welford 在线统计 ====================
def update_welford(stats, val):
    val = float(val)
    if stats[0] == 0:
        stats[0] = 1
        stats[1] = val
        stats[2] = 0.0
        stats[3] = val
        stats[4] = val
    else:
        stats[0] += 1
        n = stats[0]
        old_mean = stats[1]
        delta = val - old_mean
        stats[1] += delta / n
        stats[2] += delta * (val - stats[1])
        if val > stats[3]: stats[3] = val
        if val < stats[4]: stats[4] = val

def get_welford_metrics(stats):
    if stats[0] == 0: return 0.0, 0.0, 0.0, 0.0
    return float(stats[3]), float(stats[4]), float(stats[1]), (stats[2] / stats[0]) ** 0.5 if stats[0] > 1 else 0.0

# ==================== IP 工具 ====================
def ip_to_int(ip_str):
    try:
        parts = ip_str.split('.')
        return (int(parts[0]) << 24) | (int(parts[1]) << 16) | (int(parts[2]) << 8) | int(parts[3])
    except:
        return 0

def int_to_ip(ip_int):
    return f"{(ip_int >> 24) & 255}.{(ip_int >> 16) & 255}.{(ip_int >> 8) & 255}.{ip_int & 255}"

# ==================== 离线统计函数 ====================
def get_stats_metrics_func(val_list):
    if not val_list: return 0.0, 0.0, 0.0, 0.0, 0.0
    n = len(val_list)
    v_max = max(val_list)
    v_min = min(val_list)
    v_sum = sum(val_list)
    v_mean = v_sum / n
    v_var = sum((x - v_mean) ** 2 for x in val_list) / n if n > 0 else 0.0
    v_std = v_var ** 0.5
    return float(v_max), float(v_min), float(v_mean), float(v_std), float(v_var)

def get_iat_metrics_func(time_list):
    if len(time_list) < 2: return 0.0, 0.0, 0.0, 0.0
    times = sorted(time_list)
    iats = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    n = len(iats)
    v_mean = sum(iats) / n
    v_std = (sum((x - v_mean) ** 2 for x in iats) / n) ** 0.5 if n > 1 else 0.0
    return float(max(iats)), float(min(iats)), float(v_mean), float(v_std)

def get_active_idle_metrics_func(pkt_times, threshold=5.0):
    if len(pkt_times) < 2: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    times = sorted(pkt_times)
    active_intervals, idle_intervals = [], []
    active_start = times[0]
    for i in range(len(times) - 1):
        gap = times[i + 1] - times[i]
        if gap > threshold:
            active_intervals.append(times[i] - active_start)
            idle_intervals.append(gap)
            active_start = times[i + 1]
    active_intervals.append(times[-1] - active_start)
    amax, amin, amean, astd, _ = get_stats_metrics_func(active_intervals)
    if idle_intervals:
        imax, imin, imean, istd, _ = get_stats_metrics_func(idle_intervals)
    else:
        imax = imin = imean = istd = 0.0
    return amax, amin, amean, astd, imax, imin, imean, istd

def analyze_cn_structure(cn_str):
    if not cn_str: return 0.0, 0.0, 0.0
    length = len(cn_str)
    return round(len(re.findall(r'[aeiouAEIOU]', cn_str)) / length, 4), \
           round(len(re.findall(r'\d', cn_str)) / length, 4), \
           round(len(re.findall(r'[^a-zA-Z0-9]', cn_str)) / length, 4)

def get_percentile(val_list, p):
    if not val_list: return 0.0
    s = sorted(val_list)
    return float(s[min(int(len(s) * p), len(s) - 1)])

# ==================== 窗口聚合函数 ====================
def aggregate_window_features(window, src_ip, dst_ip, dst_port, current_ts):
    handshake_fail_count = rst_count = total_count = reconnect_count = 0
    durations, related_ts = [], []
    dst_set_srcip = set()
    srcip_total = srcip_abnormal = 0
    tau = 30.0
    weighted_conn_sum = weighted_duration_numerator = weighted_duration_denominator = 0.0
    for conn in window:
        total_count += 1
        cs = conn.get("conn_state", "")
        s, d, dp = conn.get("src_ip"), conn.get("dst_ip"), conn.get("dst_port")
        conn_ts = conn.get("ts_start", 0)
        weight = exp(-abs(conn_ts - current_ts) / tau)
        weighted_conn_sum += weight
        if s == src_ip:
            srcip_total += 1
            dst_set_srcip.add((d, dp))
            if cs in ZEEK_ABNORMAL_STATES: srcip_abnormal += 1
        if cs in ZEEK_RESET_REJECTED_STATES: rst_count += 1
        if cs in ZEEK_HANDSHAKE_FAILURE_STATES: handshake_fail_count += 1
        if s == src_ip and d == dst_ip and dp == dst_port:
            reconnect_count += 1
            related_ts.append(conn_ts)
        duration = conn.get("duration", 0.0)
        durations.append(duration)
        weighted_duration_numerator += duration * weight
        weighted_duration_denominator += weight
    flow_jitter = flow_interval_diff_mean = 0.0
    if len(related_ts) > 1:
        related_ts.sort()
        intervals = [related_ts[i + 1] - related_ts[i] for i in range(len(related_ts) - 1)]
        mean_val = sum(intervals) / len(intervals)
        flow_jitter = round(float((sum((x - mean_val) ** 2 for x in intervals) / len(intervals)) ** 0.5), 4)
        flow_interval_diff_mean = round(float(mean_val), 4)
    return {
        "rst_ratio": round(rst_count / total_count, 4) if total_count else 0.0,
        "handshake_fail_rate": round(handshake_fail_count / total_count, 4) if total_count else 0.0,
        "reconnect_count": reconnect_count, "conn_count": total_count,
        "flow_interval_jitter": flow_jitter, "flow_interval_diff_mean": flow_interval_diff_mean,
        "tcp_rst_count": rst_count, "reconnection_flag": 1 if reconnect_count > 1 else 0,
        "unique_dst_count": len(dst_set_srcip),
        "src_ip_abnormal_ratio": round(srcip_abnormal / srcip_total, 4) if srcip_total else 0.0,
        "duration_p25": round(get_percentile(durations, 0.25), 4),
        "duration_p50": round(get_percentile(durations, 0.50), 4),
        "duration_p75": round(get_percentile(durations, 0.75), 4),
        "weighted_conn_count": round(weighted_conn_sum, 4),
        "weighted_avg_duration": round(weighted_duration_numerator / weighted_duration_denominator, 4) if weighted_duration_denominator > 0 else 0.0,
        "abnormal_to_conn_ratio": round((srcip_abnormal / srcip_total) / total_count, 4) if srcip_total and total_count > 0 else 0.0
    }

# ==================== PCAP 解析器 ====================
def fast_pcap_iter(file_path, max_packets=MAX_PROCESS_PACKETS):
    cnt = 0
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
                    if caplen <= 0 or caplen > 262144: break
                    raw_packet = f.read(caplen)
                    if len(raw_packet) < caplen: break
                    yield ts_sec + ts_usec / 1e6, raw_packet, linktype
                    cnt += 1
                    if max_packets is not None and cnt >= max_packets: break
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
                    except: break
                    if b_len > 262144 or b_len < 12: break
                    body_len = b_len - 12
                    body_data = f.read(body_len)
                    len_copy_data = f.read(4)
                    if len(len_copy_data) < 4: break
                    try:
                        len_copy_le = len_copy_data[0] + (len_copy_data[1] << 8) + (len_copy_data[2] << 16) + (len_copy_data[3] << 24)
                        len_copy_be = (len_copy_data[0] << 24) + (len_copy_data[1] << 16) + (len_copy_data[2] << 8) + len_copy_data[3]
                        if len_copy_le != b_len and len_copy_be != b_len: break
                    except: break
                    if len(body_data) < body_len: break
                    if b_type == 0x00000001:
                        if len(body_data) >= 8:
                            linktype = (body_data[0] + (body_data[1] << 8)) & 0xFFFF
                    elif b_type == 0x00000006:
                        if len(body_data) < 20: continue
                        try:
                            ts_high = body_data[4] + (body_data[5] << 8) + (body_data[6] << 16) + (body_data[7] << 24)
                            ts_low = body_data[8] + (body_data[9] << 8) + (body_data[10] << 16) + (body_data[11] << 24)
                            caplen = body_data[12] + (body_data[13] << 8) + (body_data[14] << 16) + (body_data[15] << 24)
                            pkt_ts = ((ts_high << 32) + ts_low) / 1e6
                        except: continue
                        if caplen <= 0 or caplen > 262144 or caplen > (body_len - 20): continue
                        raw_packet = body_data[20:20 + caplen]
                        yield pkt_ts, raw_packet, linktype
                        cnt += 1
                        if max_packets is not None and cnt >= max_packets: break
                    elif b_type == 0x00000003:
                        if len(body_data) < 4: continue
                        try:
                            origlen = body_data[0] + (body_data[1] << 8) + (body_data[2] << 16) + (body_data[3] << 24)
                        except: continue
                        caplen = min(origlen, body_len - 4)
                        if caplen <= 0 or caplen > 262144: continue
                        raw_packet = body_data[4:4 + caplen]
                        yield time.time(), raw_packet, linktype
                        cnt += 1
                        if max_packets is not None and cnt >= max_packets: break
            else:
                return
    except:
        return

# ==================== 核心提取器（分批提取 + 直接正则证书） ====================
class PcapUnifiedExtractor:
    def __init__(self):
        self.flows = {}
        self.ssl_info = {}
        self.certs_db = {}
        self.cn_hash_cache = {}
        self.evicted_features = []
        self.last_evict_ts = 0.0

    def _evict_finished(self, current_ts, force=False):
        to_remove = []
        for flow_key, flow in list(self.flows.items()):
            if flow["flag_rst_count"] > 0 or flow["flag_fin_count"] > 0:
                to_remove.append(flow_key)
            elif current_ts - flow["ts_end"] > 5.0:
                to_remove.append(flow_key)
        if force:
            to_remove_set = set(to_remove)
            sorted_flows = sorted(self.flows.items(), key=lambda kv: kv[1]["ts_start"])
            extra = max(0, len(self.flows) - MAX_TRACKED_FLOWS)
            for flow_key, _ in sorted_flows:
                if extra <= 0: break
                if flow_key not in to_remove_set:
                    to_remove.append(flow_key)
                    extra -= 1
        for flow_key in to_remove:
            flow = self.flows[flow_key]
            self.evicted_features.append(flow)
            del self.flows[flow_key]

    def parse_single_pcap(self, pcap_path, dataset_source, subfolder):
        pcap_filename = os.path.basename(pcap_path)
        self.flows.clear()
        self.ssl_info.clear()
        self.certs_db.clear()
        self.cn_hash_cache.clear()
        self.evicted_features.clear()
        self.last_evict_ts = 0.0
        pkt_count = 0

        for pkt_ts, raw_packet, linktype in fast_pcap_iter(pcap_path):
            pkt_count += 1
            if pkt_count % 50000 == 0:
                sys.stderr.write(f"\r    {pkt_count} 包, 活跃流 {len(self.flows)}, 已淘汰 {len(self.evicted_features)}")
                sys.stderr.flush()

            # === 移除定时淘汰，保留全部流直到文件结束 ===
            # （原版会在这里按时间或流数淘汰，已注释）
            # if pkt_ts - self.last_evict_ts > 10.0 or len(self.flows) > MAX_TRACKED_FLOWS:
            #     self._evict_finished(pkt_ts, force=(len(self.flows) > MAX_TRACKED_FLOWS))
            #     self.last_evict_ts = pkt_ts
            # === 结束移除 ===

            ip_offset = 14 if linktype == 1 else (16 if linktype == 113 else 0)
            if len(raw_packet) < ip_offset + 20: continue
            ip_data = raw_packet[ip_offset:]
            if (ip_data[0] >> 4) != 4: continue
            proto = ip_data[9]
            sip_int = (ip_data[12] << 24) | (ip_data[13] << 16) | (ip_data[14] << 8) | ip_data[15]
            dip_int = (ip_data[16] << 24) | (ip_data[17] << 16) | (ip_data[18] << 8) | ip_data[19]
            ip_header_len = (ip_data[0] & 0x0F) * 4
            if ip_header_len < 20: continue
            trans_start = ip_header_len

            is_rst = is_syn = is_fin = is_psh = is_ack = False
            if proto == 6:
                if len(ip_data) < trans_start + 20: continue
                tcp_data = ip_data[trans_start:trans_start + 20]
                sport = (tcp_data[0] << 8) + tcp_data[1]
                dport = (tcp_data[2] << 8) + tcp_data[3]
                flags = tcp_data[13]
                is_rst = bool(flags & 0x04)
                is_syn = bool(flags & 0x02)
                is_fin = bool(flags & 0x01)
                is_psh = bool(flags & 0x08)
                is_ack = bool(flags & 0x10)
                proto_str = "tcp"
                tcp_offset = ((tcp_data[12] >> 4) & 0x0F) * 4
                if len(ip_data) < trans_start + tcp_offset: continue
                payload = ip_data[trans_start + tcp_offset:]
            elif proto == 17:
                if len(ip_data) < trans_start + 8: continue
                udp_data = ip_data[trans_start:trans_start + 8]
                sport = (udp_data[0] << 8) + udp_data[1]
                dport = (udp_data[2] << 8) + udp_data[3]
                proto_str = "udp"
                payload = ip_data[trans_start + 8:]
            else:
                continue

            five_tuple = (sip_int, sport, dip_int, dport, proto_str)
            reverse_tuple = (dip_int, dport, sip_int, sport, proto_str)
            flow_key = five_tuple if five_tuple in self.flows else (reverse_tuple if reverse_tuple in self.flows else five_tuple)
            pkt_len = len(raw_packet)
            is_forward = (sip_int == flow_key[0])

            if flow_key not in self.flows:
                # 不再因为流数过多而跳过新流（MAX_TRACKED_FLOWS 已移除限制）
                src_ip = int_to_ip(sip_int)
                dst_ip = int_to_ip(dip_int)
                uid = f"{src_ip}_{sport}_{dst_ip}_{dport}_{pkt_ts}"
                self.flows[flow_key] = {
                    "uid": uid, "src_ip": src_ip, "src_port": sport, "dst_ip": dst_ip, "dst_port": dport,
                    "proto": proto_str, "ts_start": pkt_ts, "ts_end": pkt_ts,
                    "conn_state": "S0" if is_syn else "SF", "duration": 0.0,
                    "dataset_source": dataset_source, "subfolder": subfolder, "pcap_filename": pcap_filename,
                    "pkts_forward": 1 if is_forward else 0, "pkts_backward": 0 if is_forward else 1,
                    "bytes_forward": pkt_len if is_forward else 0, "bytes_backward": 0 if is_forward else pkt_len,
                    "len_total": [1, float(pkt_len), 0.0, pkt_len, pkt_len],
                    "len_fwd": [1, float(pkt_len), 0.0, pkt_len, pkt_len] if is_forward else [0, 0.0, 0.0, -1, 999999],
                    "len_bwd": [0, 0.0, 0.0, -1, 999999] if is_forward else [1, float(pkt_len), 0.0, pkt_len, pkt_len],
                    "iat_total": [0, 0.0, 0.0, -1.0, 999999.0],
                    "iat_fwd": [0, 0.0, 0.0, -1.0, 999999.0],
                    "iat_bwd": [0, 0.0, 0.0, -1.0, 999999.0],
                    "last_ts_total": pkt_ts,
                    "last_ts_fwd": pkt_ts if is_forward else None,
                    "last_ts_bwd": None if is_forward else pkt_ts,
                    "act_welford": [0, 0.0, 0.0, -1.0, 999999.0],
                    "idl_welford": [0, 0.0, 0.0, -1.0, 999999.0],
                    "active_start": pkt_ts, "last_ts_active": pkt_ts,
                    "temporal_packets": [(pkt_ts, pkt_len, 'F' if is_forward else 'B')],
                    "flag_syn_count": 1 if is_syn else 0, "flag_fin_count": 1 if is_fin else 0,
                    "flag_rst_count": 1 if is_rst else 0, "flag_psh_count": 1 if is_psh else 0,
                    "flag_ack_count": 1 if is_ack else 0,
                }
            else:
                flow = self.flows[flow_key]
                total_pkts = flow["pkts_forward"] + flow["pkts_backward"]
                # 不再跳过超出包数的包（MAX_TRACKED_PACKETS 已移除限制）
                flow["ts_end"] = pkt_ts
                flow["duration"] = pkt_ts - flow["ts_start"]
                if is_forward:
                    flow["pkts_forward"] += 1
                    flow["bytes_forward"] += pkt_len
                else:
                    flow["pkts_backward"] += 1
                    flow["bytes_backward"] += pkt_len
                # 全量统计
                update_welford(flow["len_total"], pkt_len)
                iat_total = pkt_ts - flow["last_ts_total"]
                update_welford(flow["iat_total"], iat_total)
                flow["last_ts_total"] = pkt_ts
                if is_forward:
                    update_welford(flow["len_fwd"], pkt_len)
                    if flow["last_ts_fwd"] is not None:
                        iat_f = pkt_ts - flow["last_ts_fwd"]
                        update_welford(flow["iat_fwd"], iat_f)
                    flow["last_ts_fwd"] = pkt_ts
                else:
                    update_welford(flow["len_bwd"], pkt_len)
                    if flow["last_ts_bwd"] is not None:
                        iat_b = pkt_ts - flow["last_ts_bwd"]
                        update_welford(flow["iat_bwd"], iat_b)
                    flow["last_ts_bwd"] = pkt_ts
                gap = pkt_ts - flow["last_ts_active"]
                if gap > 5.0:
                    update_welford(flow["act_welford"], flow["last_ts_active"] - flow["active_start"])
                    update_welford(flow["idl_welford"], gap)
                    flow["active_start"] = pkt_ts
                flow["last_ts_active"] = pkt_ts
                # 全量时序记录
                flow["temporal_packets"].append((pkt_ts, pkt_len, 'F' if is_forward else 'B'))
                if is_syn: flow["flag_syn_count"] += 1
                if is_fin: flow["flag_fin_count"] += 1
                if is_rst: flow["flag_rst_count"] += 1
                if is_psh: flow["flag_psh_count"] += 1
                if is_ack: flow["flag_ack_count"] += 1

            if is_rst: self.flows[flow_key]["conn_state"] = "RSTO"

            # ---------- 证书提取（直接正则，取消预检测）----------
            if proto == 6 and (sport == 443 or dport == 443) and flow_key in self.flows:
                flow_obj = self.flows[flow_key]
                try:
                    matches = CERT_CN_REGEX.finditer(payload)
                    cert_fps = []
                    for idx, m in enumerate(matches):
                        found_cn = m.group(1).decode('utf-8', errors='ignore')
                        if found_cn in self.cn_hash_cache:
                            fp = self.cn_hash_cache[found_cn]
                        else:
                            fp = hashlib.sha256(found_cn.encode()).hexdigest()
                            self.cn_hash_cache[found_cn] = fp
                        cert_fps.append(fp)
                        if fp not in self.certs_db:
                            self.certs_db[fp] = {
                                "serial": str(abs(hash(found_cn) + idx)),
                                "subject": f"CN={found_cn}",
                                "issuer": f"CN={found_cn} CA",
                                "not_before": f"{pkt_ts - 86400 * 30:.6f}",
                                "not_after": f"{pkt_ts + 86400 * 365:.6f}",
                                "key_length": 2048,
                                "sig_alg": "sha256WithRSAEncryption"
                            }
                    if cert_fps:
                        uid = flow_obj["uid"]
                        if uid not in self.ssl_info: self.ssl_info[uid] = {}
                        self.ssl_info[uid]["cn"] = self.certs_db[cert_fps[0]]["subject"].split("=")[1]
                        self.ssl_info[uid]["cert_chain_fps"] = ",".join(cert_fps)
                        self.ssl_info[uid]["version"] = "TLSv1.2"
                except:
                    pass

        # 文件扫描结束，强制导出所有内存中的流到 evicted_features
        self._evict_finished(time.time(), force=True)
        sys.stderr.write("\n扫描完成，开始提取特征...\n")
        sys.stderr.flush()

    # ==================== 分批提取特征 ====================
    def extract_unified_outputs(self, dataset_source, subfolder, pcap_filename, label):
        all_flows = list(self.flows.values()) + self.evicted_features
        if not all_flows:
            return {"features": [], "temporal": []}

        all_flows.sort(key=lambda x: x.get("ts_start", 0))
        conn_ts_array = [f.get("ts_start", 0) for f in all_flows]

        ml_results = []
        temporal_results = []
        # 分批处理，降低内存压力
        for batch_start in range(0, len(all_flows), EXTRACT_BATCH_SIZE):
            batch_flows = all_flows[batch_start:batch_start+EXTRACT_BATCH_SIZE]
            for conn_entry in batch_flows:
                src_ip = conn_entry["src_ip"]
                src_port = conn_entry["src_port"]
                dst_ip = conn_entry["dst_ip"]
                dst_port = conn_entry["dst_port"]
                current_ts = conn_entry["ts_start"]
                uid = conn_entry["uid"]
                proto = conn_entry["proto"]

                pkts_fwd = conn_entry["pkts_forward"]
                pkts_bwd = conn_entry["pkts_backward"]
                bytes_fwd = conn_entry["bytes_forward"]
                bytes_bwd = conn_entry["bytes_backward"]
                pkts_total = pkts_fwd + pkts_bwd
                bytes_total = bytes_fwd + bytes_bwd

                p_max, p_min, p_mean, p_std = get_welford_metrics(conn_entry["len_total"])
                f_max, f_min, f_mean, f_std = get_welford_metrics(conn_entry["len_fwd"])
                b_max, b_min, b_mean, b_std = get_welford_metrics(conn_entry["len_bwd"])
                p_var = round(p_std ** 2, 4)

                duration = max(conn_entry["ts_end"] - current_ts, 0.0001)
                flow_bytes_s = round(bytes_total / duration, 4)
                flow_pkts_s = round(pkts_total / duration, 4)
                fwd_pkts_s = round(pkts_fwd / duration, 4)
                bwd_pkts_s = round(pkts_bwd / duration, 4)

                header_size = 40 if proto == "tcp" else 28
                fwd_header_len = pkts_fwd * header_size
                bwd_header_len = pkts_bwd * header_size
                down_up_ratio = round(pkts_bwd / pkts_fwd, 4) if pkts_fwd > 0 else 0.0

                avg_fwd_segment_size = round(f_mean, 2)
                avg_bwd_segment_size = round(b_mean, 2)

                iat_max, iat_min, iat_mean, iat_std = get_welford_metrics(conn_entry["iat_total"])
                iat_f_max, iat_f_min, iat_f_mean, iat_f_std = get_welford_metrics(conn_entry["iat_fwd"])
                iat_b_max, iat_b_min, iat_b_mean, iat_b_std = get_welford_metrics(conn_entry["iat_bwd"])

                act_max, act_min, act_mean, act_std = get_welford_metrics(conn_entry["act_welford"])
                idl_max, idl_min, idl_mean, idl_std = get_welford_metrics(conn_entry["idl_welford"])

                subflow_fwd_pkts = pkts_fwd
                subflow_fwd_bytes = bytes_fwd
                subflow_bwd_pkts = pkts_bwd
                subflow_bwd_bytes = bytes_bwd

                left = bisect.bisect_left(conn_ts_array, current_ts - TIME_WINDOW)
                right = bisect.bisect_right(conn_ts_array, current_ts + TIME_WINDOW)
                agg_features = aggregate_window_features(all_flows[left:right], src_ip, dst_ip, dst_port, current_ts)

                cn_vowel_ratio = cn_digit_density = cn_special_char_density = 0.0
                cn_length = 0
                cn_hash = 0
                cn_val = None

                conn_service = "ssl" if (src_port == 443 or dst_port == 443) else "-"
                conn_history = "ShADfFa" if conn_entry["conn_state"] == "SF" else "S"
                conn_orig_ip_bytes = bytes_fwd + (pkts_fwd * 40)
                conn_resp_ip_bytes = bytes_bwd + (pkts_bwd * 40)

                conn_dict = {
                    "ts": f"{current_ts:.6f}", "uid": uid, "id.orig_h": src_ip, "id.orig_p": src_port,
                    "id.resp_h": dst_ip, "id.resp_p": dst_port, "proto": proto, "service": conn_service,
                    "duration": f"{conn_entry['duration']:.4f}", "orig_bytes": bytes_fwd, "resp_bytes": bytes_bwd,
                    "conn_state": conn_entry["conn_state"], "local_orig": "-", "local_resp": "-", "missed_bytes": 0,
                    "history": conn_history, "orig_pkts": pkts_fwd, "orig_ip_bytes": conn_orig_ip_bytes,
                    "resp_pkts": pkts_bwd, "resp_ip_bytes": conn_resp_ip_bytes, "tunnel_parents": "-"
                }
                ssl_dict = {
                    "ts": f"{current_ts:.6f}", "uid": uid, "id.orig_h": src_ip, "id.orig_p": src_port,
                    "id.resp_h": dst_ip, "id.resp_p": dst_port, "version": "-", "cipher": "-", "curve": "-",
                    "server_name": "-", "resumed": "F", "last_alert": "-", "next_protocol": "-", "established": "F",
                    "cert_chain_fps": "-", "client_cert_chain_fps": "-", "subject": "-", "issuer": "-",
                    "client_subject": "-", "client_issuer": "-", "validation_status": "-"
                }
                x509_list = []
                cert_valid_days = cert_age_at_capture = cert_remaining_days = -1.0
                cert_chain_depth = 0

                if ssl_entry := self.ssl_info.get(uid):
                    cn_val = ssl_entry.get("cn")
                    if cn_val:
                        cn_vowel_ratio, cn_digit_density, cn_special_char_density = analyze_cn_structure(cn_val)
                        cn_length = len(cn_val)
                        cn_hash = int(hashlib.md5(cn_val.encode()).hexdigest(), 16) % 1024

                        ssl_dict.update({
                            "version": ssl_entry.get("version", "TLSv1.2"),
                            "cipher": "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
                            "server_name": cn_val, "established": "T",
                            "cert_chain_fps": ssl_entry.get("cert_chain_fps", "-"),
                            "subject": f"CN={cn_val}", "issuer": f"CN={cn_val} CA"
                        })
                        fps_list = ssl_entry.get("cert_chain_fps", "").split(",")
                        cert_chain_depth = len(fps_list)
                        for fp in fps_list:
                            if db_entry := self.certs_db.get(fp):
                                x509_list.append({
                                    "id": fp, "certificate.version": "3", "certificate.serial": db_entry["serial"],
                                    "certificate.subject": db_entry["subject"], "certificate.issuer": db_entry["issuer"],
                                    "certificate.not_valid_before": db_entry["not_before"],
                                    "certificate.not_valid_after": db_entry["not_after"],
                                    "certificate.key_alg": "rsa", "certificate.sig_alg": db_entry["sig_alg"],
                                    "certificate.key_type": "rsa", "certificate.key_length": db_entry["key_length"],
                                    "certificate.exponent": "-", "certificate.curve": "-", "san.dns": "-", "san.uri": "-",
                                    "san.email": "-", "san.ip": "-", "basic_constraints.ca": "F",
                                    "basic_constraints.path_len": -1
                                })
                        if db_entry := self.certs_db.get(fps_list[0]):
                            nb_ts, na_ts = float(db_entry["not_before"]), float(db_entry["not_after"])
                            cert_valid_days = round((na_ts - nb_ts) / 86400.0, 4) if na_ts > nb_ts else -1.0
                            cert_age_at_capture = round((current_ts - nb_ts) / 86400.0, 4) if current_ts > nb_ts else -1.0
                            cert_remaining_days = round((na_ts - current_ts) / 86400.0, 4) if na_ts > current_ts else -1.0

                conn_state = "OTH"
                if proto == "tcp":
                    if conn_entry["flag_rst_count"] > 0: conn_state = "RSTO"
                    elif conn_entry["flag_syn_count"] > 0 and conn_entry["flag_ack_count"] == 1: conn_state = "S0"
                    elif conn_entry["flag_fin_count"] > 0: conn_state = "SF"
                else:
                    conn_state = "SF"

                row_ml = {
                    "flow_uid": uid, "src_ip": src_ip, "src_port": src_port, "dst_ip": dst_ip, "dst_port": dst_port,
                    "protocol": proto, "timestamp": current_ts, "dataset_source": dataset_source, "subfolder": subfolder,
                    "pcap_filename": pcap_filename, "label": label,
                    "pkts_forward": pkts_fwd, "pkts_backward": pkts_bwd, "pkts_total": pkts_total,
                    "bytes_forward": bytes_fwd, "bytes_backward": bytes_bwd, "bytes_total": bytes_total,
                    "ratio_bytes_back_to_forward": round(bytes_bwd / bytes_fwd, 4) if bytes_fwd > 0 else 0.0,
                    "pkt_len_max": p_max, "pkt_len_min": p_min, "pkt_len_mean": round(p_mean, 2),
                    "pkt_len_std": round(p_std, 2), "pkt_len_var": p_var,
                    "pkt_len_fwd_mean": round(f_mean, 2), "pkt_len_fwd_std": round(f_std, 2),
                    "pkt_len_bwd_mean": round(b_mean, 2), "pkt_len_bwd_std": round(b_std, 2),
                    "flow_bytes_s": flow_bytes_s, "flow_pkts_s": flow_pkts_s, "fwd_pkts_s": fwd_pkts_s, "bwd_pkts_s": bwd_pkts_s,
                    "fwd_header_len": fwd_header_len, "bwd_header_len": bwd_header_len, "down_up_ratio": down_up_ratio,
                    "avg_fwd_segment_size": avg_fwd_segment_size, "avg_bwd_segment_size": avg_bwd_segment_size,
                    "iat_max": round(iat_max, 4), "iat_min": round(iat_min, 4), "iat_mean": round(iat_mean, 4), "iat_std": round(iat_std, 4),
                    "iat_fwd_max": round(iat_f_max, 4), "iat_fwd_min": round(iat_f_min, 4), "iat_fwd_mean": round(iat_f_mean, 4), "iat_fwd_std": round(iat_f_std, 4),
                    "iat_bwd_max": round(iat_b_max, 4), "iat_bwd_min": round(iat_b_min, 4), "iat_bwd_mean": round(iat_b_mean, 4), "iat_bwd_std": round(iat_b_std, 4),
                    "flag_syn_count": conn_entry["flag_syn_count"], "flag_fin_count": conn_entry["flag_fin_count"],
                    "flag_rst_count": conn_entry["flag_rst_count"], "flag_psh_count": conn_entry["flag_psh_count"],
                    "flag_ack_count": conn_entry["flag_ack_count"],
                    "subflow_fwd_pkts": subflow_fwd_pkts, "subflow_fwd_bytes": subflow_fwd_bytes,
                    "subflow_bwd_pkts": subflow_bwd_pkts, "subflow_bwd_bytes": subflow_bwd_bytes,
                    "active_max": act_max, "active_min": act_min, "active_mean": round(act_mean, 4), "active_std": round(act_std, 4),
                    "idle_max": idl_max, "idle_min": idl_min, "idle_mean": round(idl_mean, 4), "idle_std": round(idl_std, 4),
                    **agg_features,
                    "handshake_duration": conn_entry["duration"],
                    "cn_vowel_ratio": cn_vowel_ratio, "cn_digit_density": cn_digit_density, "cn_special_char_density": cn_special_char_density,
                    "cn_length": cn_length, "cn_hash": cn_hash,
                    "cert_valid_days": cert_valid_days, "cert_age_at_capture": cert_age_at_capture,
                    "cert_remaining_days": cert_remaining_days, "cert_chain_depth": cert_chain_depth,
                    "zeek_conn_log": str(conn_dict), "zeek_ssl_log": str(ssl_dict),
                    "zeek_x509_log": str(x509_list) if x509_list else "-"
                }
                ml_results.append(row_ml)

                pts = sorted(conn_entry["temporal_packets"], key=lambda x: x[0])
                lens_all = [p[1] for p in pts]
                lens_fwd_temp = [p[1] for p in pts if p[2] == 'F']
                lens_bwd_temp = [p[1] for p in pts if p[2] == 'B']
                times_all_temp = [p[0] for p in pts]
                packet_time_offsets = [round(p[0] - current_ts, 6) for p in pts]
                packet_directions = [p[2] for p in pts]
                packet_lengths = [p[1] for p in pts]
                if pts:
                    _, _, _, pkt_len_std_temp, pkt_len_var_temp = get_stats_metrics_func(lens_all)
                    _, _, avg_fwd_segment_size_temp, _, _ = get_stats_metrics_func(lens_fwd_temp)
                    _, _, avg_bwd_segment_size_temp, _, _ = get_stats_metrics_func(lens_bwd_temp)
                    iat_max_temp, iat_min_temp, iat_mean_temp, iat_std_temp = get_iat_metrics_func(times_all_temp)
                    act_max_temp, act_min_temp, act_mean_temp, act_std_temp, idl_max_temp, idl_min_temp, idl_mean_temp, idl_std_temp = get_active_idle_metrics_func(times_all_temp, 5.0)
                else:
                    pkt_len_std_temp = pkt_len_var_temp = avg_fwd_segment_size_temp = avg_bwd_segment_size_temp = 0.0
                    iat_max_temp = iat_min_temp = iat_mean_temp = iat_std_temp = 0.0
                    act_max_temp = act_min_temp = act_mean_temp = act_std_temp = idl_max_temp = idl_min_temp = idl_mean_temp = idl_std_temp = 0.0

                row_temp = {
                    "flow_uid": uid, "src_ip": src_ip, "src_port": src_port, "dst_ip": dst_ip, "dst_port": dst_port,
                    "protocol": proto, "start_time": current_ts, "end_time": conn_entry["ts_end"],
                    "duration": round(duration, 6), "pcap_filename": pcap_filename, "label": label,
                    "total_packets": len(pts), "total_bytes": sum(lens_all),
                    "flow_bytes_s": round(sum(lens_all) / duration, 4),
                    "flow_pkts_s": round(len(pts) / duration, 4),
                    "fwd_packets": len(lens_fwd_temp), "bwd_packets": len(lens_bwd_temp),
                    "fwd_bytes": sum(lens_fwd_temp), "bwd_bytes": sum(lens_bwd_temp),
                    "ratio_bytes_bwd_fwd": round(sum(lens_bwd_temp) / sum(lens_fwd_temp), 4) if sum(lens_fwd_temp) > 0 else 0.0,
                    "average_packet_size": round(sum(lens_all) / len(pts), 2) if pts else 0.0,
                    "pkt_len_std": round(pkt_len_std_temp, 4), "pkt_len_var": round(pkt_len_var_temp, 4),
                    "fwd_header_len": len(lens_fwd_temp) * header_size,
                    "bwd_header_len": len(lens_bwd_temp) * header_size,
                    "avg_fwd_segment_size": round(avg_fwd_segment_size_temp, 2),
                    "avg_bwd_segment_size": round(avg_bwd_segment_size_temp, 2),
                    "iat_max": iat_max_temp, "iat_min": iat_min_temp, "iat_mean": iat_mean_temp, "iat_std": iat_std_temp,
                    "active_max": act_max_temp, "active_min": act_min_temp, "active_mean": act_mean_temp, "active_std": act_std_temp,
                    "idle_max": idl_max_temp, "idle_min": idl_min_temp, "idle_mean": idl_mean_temp, "idle_std": idl_std_temp,
                    "conn_state": conn_state,
                    "syn_count": conn_entry["flag_syn_count"], "fin_count": conn_entry["flag_fin_count"],
                    "rst_count": conn_entry["flag_rst_count"], "psh_count": conn_entry["flag_psh_count"],
                    "ack_count": conn_entry["flag_ack_count"],
                    "packet_time_offsets": json.dumps(packet_time_offsets),
                    "packet_directions": json.dumps(packet_directions),
                    "packet_lengths": json.dumps(packet_lengths)
                }
                temporal_results.append(row_temp)
            gc.collect()  # 每批释放内存

        return {"features": ml_results, "temporal": temporal_results}

# ==================== 子进程工作函数（增强错误输出） ====================
def process_one_file(pcap_path, label, dataset_source, subfolder, result_queue):
    tmp_id = str(uuid.uuid4())[:8]
    try:
        extractor = PcapUnifiedExtractor()
        extractor.parse_single_pcap(pcap_path, dataset_source, subfolder)
        extracted = extractor.extract_unified_outputs(dataset_source, subfolder, os.path.basename(pcap_path), label)

        ml_tmp = f"__tmp_ml_{tmp_id}.csv"
        temp_tmp = f"__tmp_temp_{tmp_id}.csv"
        if extracted["features"]:
            pd.DataFrame(extracted["features"], columns=CSV_HEADERS).to_csv(ml_tmp, index=False, encoding='utf-8')
        else:
            pd.DataFrame(columns=CSV_HEADERS).to_csv(ml_tmp, index=False, encoding='utf-8')

        if extracted["temporal"]:
            pd.DataFrame(extracted["temporal"], columns=TEMPORAL_HEADERS).to_csv(temp_tmp, index=False, encoding='utf-8')
        else:
            pd.DataFrame(columns=TEMPORAL_HEADERS).to_csv(temp_tmp, index=False, encoding='utf-8')

        result_queue.put(("success", ml_tmp, temp_tmp, len(extracted["features"]), len(extracted["temporal"])))
    except Exception as e:
        sys.stderr.write(f"子进程错误：{e}\n{traceback.format_exc()}\n")
        sys.stderr.flush()
        result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))
    finally:
        try:
            extractor.flows.clear()
            extractor.ssl_info.clear()
            extractor.certs_db.clear()
            extractor.cn_hash_cache.clear()
            extractor.evicted_features.clear()
        except:
            pass
        gc.collect()

# ==================== 主流程 ====================
def main():
    start_time = time.time()
    train_files, test_files = [], []
    for root, dirs, files in os.walk(WORK_DIR):
        for f in files:
            lower_name = f.lower()
            if lower_name.endswith(('_train.pcap', '_train.pcapng')):
                train_files.append(os.path.join(root, f))
            elif lower_name.endswith(('_test.pcap', '_test.pcapng')):
                test_files.append(os.path.join(root, f))

    if not train_files and not test_files:
        print("未找到任何 *_train.pcap 或 *_test.pcap 文件。")
        return

    all_files = train_files + test_files
    ml_train_tmp_files, ml_test_tmp_files = [], []
    temp_train_tmp_files, temp_test_tmp_files = [], []

    for pcap_path in tqdm(all_files, desc="特征提取（多进程）"):
        fname = os.path.basename(pcap_path)
        label = fname.split('_sampled_')[0]
        dataset_source = "DoH2020" if "cooked" in fname else "CIC2017"
        if fname.endswith('_train.pcap'):
            subfolder = "train"
            ml_tmp_list = ml_train_tmp_files
            temp_tmp_list = temp_train_tmp_files
        else:
            subfolder = "test"
            ml_tmp_list = ml_test_tmp_files
            temp_tmp_list = temp_test_tmp_files

        print(f"\n处理文件: {fname}  标签: {label}  子集: {subfolder}")

        q = Queue()
        p = Process(target=process_one_file, args=(pcap_path, label, dataset_source, subfolder, q))
        p.start()
        p.join(timeout=600)

        if p.is_alive():
            p.terminate()
            p.join()
            print(f"  -> [警告] 文件 {fname} 处理超时，已跳过")
            continue

        try:
            status, *data = q.get(timeout=10)
        except Exception as e:
            print(f"  -> [错误] 无法获取子进程结果: {e}")
            continue

        if status == "success":
            ml_path, temp_path, ml_cnt, temp_cnt = data
            print(f"  -> 提取完成：ML流 {ml_cnt} 条，时序流 {temp_cnt} 条")
            ml_tmp_list.append(ml_path)
            temp_tmp_list.append(temp_path)
        else:
            print(f"  -> [错误] 子进程失败:\n{data[0]}")

    # 合并临时文件（带 PermissionError 保护）
    def merge_and_clean(tmp_files, final_csv, col_names):
        if not tmp_files:
            print(f"无数据生成 {final_csv}")
            return 0
        dfs = []
        for f in tmp_files:
            try:
                dfs.append(pd.read_csv(f, dtype=str, keep_default_na=False))
                os.remove(f)
            except Exception as e:
                print(f"读取临时文件 {f} 失败: {e}")
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            os.makedirs(os.path.dirname(final_csv), exist_ok=True)
            try:
                final_df.to_csv(final_csv, index=False, encoding='utf-8')
            except PermissionError:
                alt_name = final_csv.replace('.csv', f'_{int(time.time())}.csv')
                final_df.to_csv(alt_name, index=False, encoding='utf-8')
                print(f"原文件被占用，已写入备用文件: {alt_name}")
            print(f"已写入，共 {len(final_df)} 行")
            return len(final_df)
        return 0

    ml_train_cnt = merge_and_clean(ml_train_tmp_files, FINAL_ML_TRAIN, CSV_HEADERS)
    ml_test_cnt = merge_and_clean(ml_test_tmp_files, FINAL_ML_TEST, CSV_HEADERS)
    temp_train_cnt = merge_and_clean(temp_train_tmp_files, FINAL_TEMP_TRAIN, TEMPORAL_HEADERS)
    temp_test_cnt = merge_and_clean(temp_test_tmp_files, FINAL_TEMP_TEST, TEMPORAL_HEADERS)

    # 清理残留临时文件
    for f in glob.glob("__tmp_ml_*.csv") + glob.glob("__tmp_temp_*.csv"):
        try: os.remove(f)
        except: pass

    print(f"\n特征提取全部完成！耗时: {time.time() - start_time:.2f} 秒")
    print(f"  -> 多分类特征（训练）: {FINAL_ML_TRAIN}  (流数: {ml_train_cnt})")
    print(f"  -> 多分类特征（测试）: {FINAL_ML_TEST}  (流数: {ml_test_cnt})")
    print(f"  -> 时序元数据（训练）: {FINAL_TEMP_TRAIN} (流数: {temp_train_cnt})")
    print(f"  -> 时序元数据（测试）: {FINAL_TEMP_TEST} (流数: {temp_test_cnt})")

if __name__ == "__main__":
    main()
