"""Canonical PCAP -> bidirectional-flow feature extractor.

This is shared by offline dataset preparation and production inference.  It
uses Scapy for capture/link/IP decoding, explicit sequence-aware TCP stream
reassembly, bounded TLS parsing, and cryptography for X.509.  It never creates
plausible-looking values for data absent from the capture.

pcap_reader.py 负责从 PCAP 里读出一个个 TCP/UDP 包；
tcp_reassembly.py 负责恢复连续 TCP 字节；
tls_parser.py 负责解析 TLS；
x509_parser.py 负责解析证书；
extract_flow_features.py 把这些结果组合起来，生成一条流对应的 80 个数值特征和日志。

    输入PCAP
        ↓
    iter_transport_packets()
    读取一个个TCP/UDP包
        ↓
    _canonical_key()
    让两个方向得到相同的流键
        ↓
    collect_flows()
    按五元组、120秒超时、新SYN划分双向流
        ↓
    Flow.add()
    保存包方向、更新时间、收集两个方向TCP数据
        ↓
    TCPStream.contiguous_chunks()
    恢复连续TCP字节
        ↓
    parse_tls_streams()
    解析TLS、ClientHello、ServerHello和证书
        ↓
    _flow_row()
    计算当前流的包数、字节数、速率、IAT、TCP标志、
    握手时间、Active/Idle、TLS和证书特征
        ↓
    _window_features()
    计算最近60秒的重连、目标数量、异常比例、
    时长百分位和时间衰减特征
        ↓
    返回两类结果
    ├── flow_features：模型特征和日志
    └── flow_temporal：每包时间、方向和长度
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from user_app.inference.contract import FEATURE_SCHEMA_VERSION, NEW_FORMAT_NUM_FEATURES
from user_app.inference.pcap_reader import TransportPacket, iter_transport_packets
from user_app.inference.tcp_reassembly import TCPStream
from user_app.inference.tls_parser import parse_tls_streams

FLOW_TIMEOUT_SECONDS = 120.0
ACTIVE_IDLE_THRESHOLD_SECONDS = 5.0

METADATA_HEADERS = [
    "feature_schema_version", "flow_uid", "src_ip", "src_port", "dst_ip", "dst_port",
    "protocol", "timestamp", "dataset_source", "subfolder", "pcap_filename", "label",
    "group_id", "split",
]
LOG_HEADERS = ["connection_log", "tls_log", "x509_log"]
CSV_HEADERS = METADATA_HEADERS + list(NEW_FORMAT_NUM_FEATURES) + LOG_HEADERS
TEMPORAL_HEADERS = [
    "feature_schema_version", "flow_uid", "src_ip", "src_port", "dst_ip", "dst_port",
    "protocol", "start_time", "end_time", "duration", "pcap_filename", "label",
    "group_id", "split", "total_packets", "total_bytes", "packet_time_offsets",
    "packet_directions", "packet_lengths",
]


@dataclass
class WelfordStats:
    """Numerically stable one-pass population statistics.
    当数字很多、很大或很接近时，简单公式可能产生浮点精度问题。
    Welford 算法可以：
        每来一个值就更新统计结果
        不需要反复扫描
        数值稳定性更好
    """

    count: int = 0  # 目前加入了多少个值。
    mean: float = 0.0  # 当前平均值。
    m2: float = 0.0  # 用于计算方差的中间量。
    minimum: float = math.inf #最小值
    maximum: float = -math.inf #最大值

    def add(self, value: float) -> None:
        value = float(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def result(self) -> tuple[float, float, float, float, float]:
        if not self.count:
            return math.nan, math.nan, math.nan, math.nan, math.nan
        variance = self.m2 / self.count
        return self.maximum, self.minimum, self.mean, math.sqrt(max(0.0, variance)), variance


def _stats(values: Iterable[float]) -> tuple[float, float, float, float, float]:
    stats = WelfordStats()
    for value in values:
        stats.add(value)
    return stats.result()


def _iat(times: list[float]) -> tuple[float, float, float, float]:
    if len(times) < 2:
        return math.nan, math.nan, math.nan, math.nan
    ordered = sorted(times)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    maximum, minimum, mean, std, _ = _stats(gaps)
    return maximum, minimum, mean, std


def _active_idle(times: list[float]) -> tuple[float, ...]:
    if not times:
        return (math.nan,) * 8
    ordered = sorted(times)
    active: list[float] = []
    idle: list[float] = []
    start = ordered[0]
    for previous, current in zip(ordered, ordered[1:]):
        gap = current - previous
        if gap > ACTIVE_IDLE_THRESHOLD_SECONDS:
            active.append(previous - start)
            idle.append(gap)
            start = current
    active.append(ordered[-1] - start)  # settle the final active segment
    amax, amin, amean, astd, _ = _stats(active)
    imax, imin, imean, istd, _ = _stats(idle)
    return amax, amin, amean, astd, imax, imin, imean, istd


def _endpoint(packet: TransportPacket, source: bool) -> tuple[str, int]:
    return (packet.src_ip, packet.src_port) if source else (packet.dst_ip, packet.dst_port)


def _canonical_key(packet: TransportPacket) -> tuple[str, tuple[str, int], tuple[str, int]]:
    a, b = _endpoint(packet, True), _endpoint(packet, False)
    return (packet.protocol, a, b) if a <= b else (packet.protocol, b, a)


@dataclass
class Flow:
    client: tuple[str, int]
    server: tuple[str, int]
    protocol: str
    start: float
    end: float
    orientation_source: str
    packets: list[tuple[TransportPacket, int]] = field(default_factory=list)
    streams: tuple[TCPStream, TCPStream] = field(default_factory=lambda: (TCPStream(), TCPStream()))
    closed: bool = False

    def add(self, packet: TransportPacket) -> None:
        bare_syn = packet.protocol == "tcp" and bool(packet.tcp_flags & 0x02) and not bool(packet.tcp_flags & 0x10)
        if bare_syn and self.orientation_source != "tcp_syn":
            if _endpoint(packet, True) != self.client:
                self.client, self.server = self.server, self.client
                self.packets = [(old_packet, -direction) for old_packet, direction in self.packets]
                self.streams = (self.streams[1], self.streams[0])
            self.orientation_source = "tcp_syn"
        direction = 1 if _endpoint(packet, True) == self.client else -1
        self.packets.append((packet, direction))
        self.start = min(self.start, packet.timestamp)
        self.end = max(self.end, packet.timestamp)
        if packet.protocol == "tcp":
            flags = packet.tcp_flags
            self.streams[0 if direction == 1 else 1].add(
                packet.tcp_seq or 0, packet.payload, syn=bool(flags & 0x02)
            )
            if flags & 0x05:  # FIN or RST
                self.closed = True


def _new_flow(packet: TransportPacket) -> Flow:
    source, destination = _endpoint(packet, True), _endpoint(packet, False)
    syn = packet.protocol == "tcp" and bool(packet.tcp_flags & 0x02)
    ack = packet.protocol == "tcp" and bool(packet.tcp_flags & 0x10)
    # A bare SYN is the only packet-level evidence that identifies the client.
    orientation = "tcp_syn" if syn and not ack else "first_observed_packet"
    flow = Flow(source, destination, packet.protocol, packet.timestamp, packet.timestamp, orientation)
    flow.add(packet)
    return flow


def collect_flows(path: str | Path) -> list[Flow]:
    active: dict[tuple, Flow] = {}
    completed: list[Flow] = []
    for packet in iter_transport_packets(path):
        key = _canonical_key(packet)
        flow = active.get(key)
        bare_syn = packet.protocol == "tcp" and bool(packet.tcp_flags & 0x02) and not bool(packet.tcp_flags & 0x10)
        if flow is not None and (packet.timestamp - flow.end > FLOW_TIMEOUT_SECONDS or (flow.closed and bare_syn)):
            completed.append(flow)
            flow = None
        if flow is None:
            flow = active[key] = _new_flow(packet)
        else:
            flow.add(packet)
    completed.extend(active.values())
    return sorted(completed, key=lambda item: item.start)


def _cn_features(cn: str | None) -> tuple[float, float, float, float]:
    if not cn:
        return (math.nan,) * 4
    length = len(cn)
    vowels = sum(char.lower() in "aeiou" for char in cn) / length
    digits = sum(char.isdigit() for char in cn) / length
    specials = sum(not char.isalnum() for char in cn) / length
    return vowels, digits, specials, float(length)


def _iso_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _flow_row(flow: Flow, index: int, pcap: Path, metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    packets = sorted(flow.packets, key=lambda item: item[0].timestamp)
    forward = [packet for packet, direction in packets if direction == 1]
    backward = [packet for packet, direction in packets if direction == -1]
    all_packets = [packet for packet, _ in packets]
    times = [packet.timestamp for packet in all_packets]
    fwd_times = [packet.timestamp for packet in forward]
    bwd_times = [packet.timestamp for packet in backward]
    lengths = [float(packet.packet_length) for packet in all_packets]
    fwd_lengths = [float(packet.packet_length) for packet in forward]
    bwd_lengths = [float(packet.packet_length) for packet in backward]
    duration = max(0.0, flow.end - flow.start)
    duration_divisor = duration if duration > 0 else math.nan
    total_bytes = sum(lengths)
    fwd_bytes, bwd_bytes = sum(fwd_lengths), sum(bwd_lengths)
    flags = [packet.tcp_flags for packet in all_packets if packet.protocol == "tcp"]
    syn_count = sum(bool(value & 0x02) for value in flags)
    fin_count = sum(bool(value & 0x01) for value in flags)
    rst_count = sum(bool(value & 0x04) for value in flags)
    psh_count = sum(bool(value & 0x08) for value in flags)
    ack_count = sum(bool(value & 0x10) for value in flags)

    handshake_duration = math.nan
    syn_time = synack_time = None
    handshake_complete = False
    history: list[str] = []
    for packet, direction in packets:
        if packet.protocol != "tcp":
            continue
        flag = packet.tcp_flags
        history.append(("C" if direction == 1 else "S") + f":0x{flag:02x}")
        if direction == 1 and flag & 0x02 and not flag & 0x10 and syn_time is None:
            syn_time = packet.timestamp
        elif direction == -1 and flag & 0x12 == 0x12 and syn_time is not None and synack_time is None:
            synack_time = packet.timestamp
        elif direction == 1 and flag & 0x10 and not flag & 0x02 and synack_time is not None:
            handshake_duration = packet.timestamp - syn_time
            handshake_complete = True
            break
    handshake_status = (
        "complete" if handshake_complete else "failed" if syn_time is not None else "not_observed"
    )
    if flow.protocol == "udp":
        state = "datagram"
    elif rst_count:
        state = "reset"
    elif fin_count >= 2:
        state = "closed"
    elif handshake_complete:
        state = "established"
    else:
        state = "incomplete_capture_or_handshake"

    tls = parse_tls_streams(flow.streams[0].contiguous_chunks(), flow.streams[1].contiguous_chunks())
    certs = tls.certificates
    leaf = certs[0] if certs and certs[0].get("parse_status") == "ok" else None
    cn = leaf.get("certificate.subject_cn") if leaf else None
    cn_vowel, cn_digit, cn_special, cn_length = _cn_features(cn)
    cert_valid = cert_age = cert_remaining = math.nan
    if leaf:
        not_before = _iso_timestamp(leaf.get("certificate.not_valid_before"))
        not_after = _iso_timestamp(leaf.get("certificate.not_valid_after"))
        if not_before is not None and not_after is not None:
            cert_valid = (not_after - not_before) / 86400
            cert_age = (flow.start - not_before) / 86400
            cert_remaining = (not_after - flow.start) / 86400

    maximum, minimum, mean, std, variance = _stats(lengths)
    _, _, fwd_mean, fwd_std, _ = _stats(fwd_lengths)
    _, _, bwd_mean, bwd_std, _ = _stats(bwd_lengths)
    _, _, fwd_payload_mean, _, _ = _stats([float(len(packet.payload)) for packet in forward])
    _, _, bwd_payload_mean, _, _ = _stats([float(len(packet.payload)) for packet in backward])
    iat_max, iat_min, iat_mean, iat_std = _iat(times)
    fi_max, fi_min, fi_mean, fi_std = _iat(fwd_times)
    bi_max, bi_min, bi_mean, bi_std = _iat(bwd_times)
    active = _active_idle(times)
    subflow_count = 1 + sum((current - previous) > 1.0 for previous, current in zip(times, times[1:]))
    uid_seed = f"{pcap.resolve()}|{flow.client}|{flow.server}|{flow.protocol}|{flow.start:.9f}|{index}"
    flow_uid = hashlib.sha256(uid_seed.encode()).hexdigest()[:24]
    connection_log = {
        "proto": flow.protocol,
        "service": "tls" if tls.detected else None,
        "duration": duration,
        "orig_bytes": fwd_bytes,
        "resp_bytes": bwd_bytes,
        "conn_state": state,
        "history": history,
        "orig_pkts": len(forward),
        "resp_pkts": len(backward),
        "orientation_source": flow.orientation_source,
        "byte_semantics": "IP packet length",
        "header_semantics": "IP plus transport header length",
    }
    row: dict[str, Any] = {name: math.nan for name in NEW_FORMAT_NUM_FEATURES}
    row.update(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "flow_uid": flow_uid, "src_ip": flow.client[0], "src_port": flow.client[1],
            "dst_ip": flow.server[0], "dst_port": flow.server[1], "protocol": flow.protocol,
            "timestamp": flow.start, "dataset_source": metadata.get("dataset_source", ""),
            "subfolder": metadata.get("subfolder", ""), "pcap_filename": pcap.name,
            "label": metadata.get("label", ""), "group_id": metadata.get("group_id", ""),
            "split": metadata.get("split", ""),
            "pkts_forward": len(forward), "pkts_backward": len(backward), "pkts_total": len(all_packets),
            "bytes_forward": fwd_bytes, "bytes_backward": bwd_bytes, "bytes_total": total_bytes,
            "ratio_bytes_back_to_forward": bwd_bytes / fwd_bytes if fwd_bytes else math.nan,
            "pkt_len_max": maximum, "pkt_len_min": minimum, "pkt_len_mean": mean,
            "pkt_len_std": std, "pkt_len_var": variance, "pkt_len_fwd_mean": fwd_mean,
            "pkt_len_fwd_std": fwd_std, "pkt_len_bwd_mean": bwd_mean, "pkt_len_bwd_std": bwd_std,
            "flow_bytes_s": total_bytes / duration_divisor, "flow_pkts_s": len(all_packets) / duration_divisor,
            "fwd_pkts_s": len(forward) / duration_divisor, "bwd_pkts_s": len(backward) / duration_divisor,
            "fwd_header_len": sum(packet.header_length for packet in forward),
            "bwd_header_len": sum(packet.header_length for packet in backward),
            "down_up_ratio": len(backward) / len(forward) if forward else math.nan,
            "transport_payload_fwd_mean": fwd_payload_mean,
            "transport_payload_bwd_mean": bwd_payload_mean,
            "iat_max": iat_max, "iat_min": iat_min, "iat_mean": iat_mean, "iat_std": iat_std,
            "iat_fwd_max": fi_max, "iat_fwd_min": fi_min, "iat_fwd_mean": fi_mean, "iat_fwd_std": fi_std,
            "iat_bwd_max": bi_max, "iat_bwd_min": bi_min, "iat_bwd_mean": bi_mean, "iat_bwd_std": bi_std,
            "flag_syn_count": syn_count, "flag_fin_count": fin_count, "flag_rst_count": rst_count,
            "flag_psh_count": psh_count, "flag_ack_count": ack_count,
            # Mean counts/bytes per activity subflow; a new subflow begins
            # after a one-second inter-packet gap.
            "subflow_fwd_pkts": len(forward) / subflow_count,
            "subflow_fwd_bytes": fwd_bytes / subflow_count,
            "subflow_bwd_pkts": len(backward) / subflow_count,
            "subflow_bwd_bytes": bwd_bytes / subflow_count,
            "active_max": active[0], "active_min": active[1], "active_mean": active[2], "active_std": active[3],
            "idle_max": active[4], "idle_min": active[5], "idle_mean": active[6], "idle_std": active[7],
            "rst_ratio": rst_count / len(flags) if flags else math.nan,
            "handshake_fail_rate": math.nan,
            "tls_record_count": tls.records, "handshake_duration": handshake_duration,
            "cn_vowel_ratio": cn_vowel, "cn_digit_density": cn_digit,
            "cn_special_char_density": cn_special, "cn_length": cn_length,
            "sni_length": len(tls.client_hello.get("server_name")) if tls.client_hello and tls.client_hello.get("server_name") else math.nan,
            "cert_valid_days": cert_valid, "cert_age_at_capture": cert_age,
            "cert_remaining_days": cert_remaining,
            "cert_chain_depth": len(certs) if tls.certificate_visibility == "plaintext_certificate_observed" else math.nan,
            "connection_log": json.dumps(connection_log, ensure_ascii=False, separators=(",", ":")),
            "tls_log": json.dumps(tls.as_log(), ensure_ascii=False, separators=(",", ":")),
            "x509_log": json.dumps(certs, ensure_ascii=False, separators=(",", ":")),
        }
    )
    temporal = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "flow_uid": flow_uid,
        "src_ip": flow.client[0], "src_port": flow.client[1], "dst_ip": flow.server[0],
        "dst_port": flow.server[1], "protocol": flow.protocol, "start_time": flow.start,
        "end_time": flow.end, "duration": duration, "pcap_filename": pcap.name,
        "label": metadata.get("label", ""), "group_id": metadata.get("group_id", ""),
        "split": metadata.get("split", ""), "total_packets": len(all_packets),
        "total_bytes": total_bytes,
        "packet_time_offsets": json.dumps([round(value - flow.start, 9) for value in times]),
        "packet_directions": json.dumps([direction for _, direction in packets]),
        "packet_lengths": json.dumps([int(value) for value in lengths]),
    }
    row["_duration"] = duration
    row["_state"] = state
    row["_handshake_status"] = handshake_status
    return row, temporal


def _window_features(rows: list[dict[str, Any]]) -> None:
    """Compute source/destination aggregates without pretending to be Zeek."""
    for current in rows:
        start = float(current["timestamp"])
        window = [row for row in rows if 0 <= start - float(row["timestamp"]) <= 60]
        same_src = [row for row in window if row["src_ip"] == current["src_ip"]]
        same_peer = [row for row in same_src if (row["dst_ip"], row["dst_port"]) == (current["dst_ip"], current["dst_port"])]
        durations = [float(row["_duration"]) for row in window]
        peer_times = sorted(float(row["timestamp"]) for row in same_peer)
        gaps = [b - a for a, b in zip(peer_times, peer_times[1:])]
        abnormal = [row for row in same_src if row["_state"] in {"reset", "incomplete_capture_or_handshake"}]
        weights = [math.exp(-abs(start - float(row["timestamp"])) / 30.0) for row in window]
        known_handshakes = [row for row in window if row["_handshake_status"] in {"complete", "failed"}]
        current.update(
            {
                "reconnect_count": max(0, len(same_peer) - 1), "conn_count": len(window),
                "flow_interval_jitter": _stats(gaps)[3] if len(gaps) > 1 else (0.0 if gaps else math.nan),
                "flow_interval_diff_mean": _stats(gaps)[2] if gaps else math.nan,
                "reconnection_flag": int(len(same_peer) > 1),
                "unique_dst_count": len({(row["dst_ip"], row["dst_port"]) for row in same_src}),
                "src_ip_abnormal_ratio": len(abnormal) / len(same_src) if same_src else math.nan,
                "duration_p25": _percentile(durations, 0.25), "duration_p50": _percentile(durations, 0.50),
                "duration_p75": _percentile(durations, 0.75), "weighted_conn_count": sum(weights),
                "weighted_avg_duration": sum(d * w for d, w in zip(durations, weights)) / sum(weights) if weights else math.nan,
                "abnormal_to_conn_ratio": len([r for r in window if r["_state"] in {"reset", "incomplete_capture_or_handshake"}]) / len(window) if window else math.nan,
                "handshake_fail_rate": sum(row["_handshake_status"] == "failed" for row in known_handshakes) / len(known_handshakes) if known_handshakes else math.nan,
            }
        )
    for row in rows:
        row.pop("_duration", None)
        row.pop("_state", None)
        row.pop("_handshake_status", None)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def extract_pcap(path: str | Path, metadata: dict[str, Any] | None = None) -> tuple[list[dict], list[dict]]:
    pcap = Path(path)
    meta = dict(metadata or {})
    meta.setdefault("group_id", pcap.stem)
    rows, temporal = [], []
    for index, flow in enumerate(collect_flows(pcap)):
        row, time_row = _flow_row(flow, index, pcap, meta)
        rows.append(row)
        temporal.append(time_row)
    _window_features(rows)
    return rows, temporal


def write_csv(path: str | Path, rows: list[dict], headers: list[str]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process_one_file(pcap_path, label, dataset_source, subfolder, result_queue, metadata=None):
    """Multiprocessing-compatible adapter retained as the single public entry."""
    try:
        meta = dict(metadata or {})
        meta.update({"label": label, "dataset_source": dataset_source, "subfolder": subfolder})
        rows, temporal = extract_pcap(pcap_path, meta)
        token = uuid.uuid4().hex
        root = Path(tempfile.gettempdir())
        ml_path = root / f"flow-features-{token}.csv"
        temporal_path = root / f"flow-features-time-{token}.csv"
        write_csv(ml_path, rows, CSV_HEADERS)
        write_csv(temporal_path, temporal, TEMPORAL_HEADERS)
        result_queue.put(("success", str(ml_path), str(temporal_path), len(rows), len(temporal)))
    except Exception:
        result_queue.put(("error", traceback.format_exc()))
