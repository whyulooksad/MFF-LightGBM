"""
打开一个记录了大量网络包的 PCAP 文件，借助 Scapy 拆开每个包的包装，找出
“什么时候发送、谁发给谁、使用TCP还是 UDP、从哪个端口发往哪个端口、携带了哪些原始数据 ”，
然后把这些内容填写进项目统一的 TransportPacket 表格，交给后面的模块继续处理。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class TransportPacket:
    timestamp: float
    ip_version: int
    src_ip: str
    dst_ip: str
    protocol: str
    src_port: int
    dst_port: int
    payload: bytes
    packet_length: int
    header_length: int
    tcp_seq: int | None = None
    tcp_flags: int = 0
    fragmented: bool = False


def _decode_transport(packet, *, fragmented: bool = False) -> TransportPacket | None:
    "接收 Scapy 解析出来的复杂网络包，返回一个简单的 TransportPacket"
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6

    if IP in packet:
        network = packet[IP]
        version = 4
        packet_length = int(network.len or len(bytes(network)))
        ip_header_length = int(network.ihl or 5) * 4
    elif IPv6 in packet:
        network = packet[IPv6]
        version = 6
        packet_length = 40 + int(network.plen or max(0, len(bytes(network)) - 40))
        # IPv6 extension headers are included below by measuring from the
        # network layer to the transport layer rather than assuming 40 bytes.
        ip_header_length = 40
    else:
        return None

    if TCP in packet:
        transport = packet[TCP]
        ip_header_length = max(ip_header_length, len(bytes(network)) - len(bytes(transport)))
        tcp_header_length = int(transport.dataofs or 5) * 4
        return TransportPacket(
            float(packet.time), version, str(network.src), str(network.dst), "tcp",
            int(transport.sport), int(transport.dport), bytes(transport.payload),
            packet_length, ip_header_length + tcp_header_length,
            int(transport.seq), int(transport.flags), fragmented,
        )
    if UDP in packet:
        transport = packet[UDP]
        ip_header_length = max(ip_header_length, len(bytes(network)) - len(bytes(transport)))
        return TransportPacket(
            float(packet.time), version, str(network.src), str(network.dst), "udp",
            int(transport.sport), int(transport.dport), bytes(transport.payload),
            packet_length, ip_header_length + 8, None, 0, fragmented,
        )
    return None


def iter_transport_packets(path: str | Path, max_packets: int | None = None) -> Iterator[TransportPacket]:
    """Yield decoded TCP/UDP IP datagrams from PCAP or PCAPNG.

    Scapy's readers handle PCAP byte order, micro/nanosecond timestamps,
    PCAPNG interface resolution, Ethernet, VLAN, SLL and SLL2. IPv4 fragments
    are reassembled by ``IPSession``. IPv6 fragments are buffered and passed to
    Scapy's IPv6 defragmenter at EOF.
    """
    from scapy.layers.inet import IP
    from scapy.layers.inet6 import IPv6, IPv6ExtHdrFragment, defragment6
    from scapy.sessions import IPSession
    from scapy.utils import PcapReader

    ipv4_session = IPSession()
    ipv6_fragments: dict[tuple, list] = {}
    seen = 0
    with PcapReader(str(path)) as reader:
        for packet in reader:
            if max_packets is not None and seen >= max_packets:
                break
            seen += 1
            if IPv6 in packet and IPv6ExtHdrFragment in packet:
                frag = packet[IPv6ExtHdrFragment]
                key = (str(packet[IPv6].src), str(packet[IPv6].dst), int(frag.id), int(frag.nh))
                ipv6_fragments.setdefault(key, []).append(packet)
                continue
            processed = ipv4_session.process(packet) if IP in packet else packet
            if processed is None:
                continue
            decoded = _decode_transport(processed, fragmented=bool(IP in packet and (packet[IP].frag or packet[IP].flags.MF)))
            if decoded is not None:
                yield decoded

    for packets in ipv6_fragments.values():
        try:
            reassembled = defragment6(packets)
            reassembled.time = min(packet.time for packet in packets)
            decoded = _decode_transport(reassembled, fragmented=True)
            if decoded is not None:
                yield decoded
        except Exception:
            # Incomplete capture: do not manufacture transport bytes across a
            # missing fragment. The omission is counted by extraction diagnostics.
            continue

