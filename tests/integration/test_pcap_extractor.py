import json
import pytest

from scapy.layers.inet import IP, TCP, UDP, fragment
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import CookedLinux, CookedLinuxV2, Dot1Q, Ether
from scapy.packet import Raw
from scapy.utils import PcapNgWriter, PcapWriter, wrpcap

from user_app.inference.extract_flow_features import WelfordStats, extract_pcap
from user_app.inference.pcap_reader import iter_transport_packets


def _tls_client_hello() -> bytes:
    name = b"nonstandard.example"
    sni = b"\x00\x00" + (5 + len(name)).to_bytes(2, "big") + (3 + len(name)).to_bytes(2, "big") + b"\x00" + len(name).to_bytes(2, "big") + name
    body = b"\x03\x03" + bytes(32) + b"\x00" + b"\x00\x02\x13\x01" + b"\x01\x00" + len(sni).to_bytes(2, "big") + sni
    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + len(handshake).to_bytes(2, "big") + handshake


def test_welford_population_statistics_match_expected_values():
    stats = WelfordStats()
    for value in (1.0, 2.0, 3.0, 4.0):
        stats.add(value)
    maximum, minimum, mean, std, variance = stats.result()
    assert (maximum, minimum, mean) == (4.0, 1.0, 2.5)
    assert variance == pytest.approx(1.25)
    assert std == pytest.approx(1.25 ** 0.5)


def test_pcap_vlan_ipv4_fragment_and_pcapng_ipv6_are_decoded(tmp_path):
    fragmented = fragment(IP(src="10.0.0.1", dst="10.0.0.2") / UDP(sport=1234, dport=4321) / Raw(b"x" * 64), fragsize=24)
    pcap = tmp_path / "formats.pcap"
    wrpcap(str(pcap), [Ether() / Dot1Q(vlan=7) / packet for packet in fragmented])
    decoded = list(iter_transport_packets(pcap))
    assert len(decoded) == 1
    assert decoded[0].fragmented
    assert decoded[0].payload == b"x" * 64

    pcapng = tmp_path / "ipv6.pcapng"
    writer = PcapNgWriter(str(pcapng))
    writer.write(Ether() / IPv6(src="2001:db8::1", dst="2001:db8::2") / UDP(sport=9, dport=10) / Raw(b"hello"))
    writer.close()
    decoded6 = list(iter_transport_packets(pcapng))
    assert len(decoded6) == 1
    assert decoded6[0].ip_version == 6
    assert decoded6[0].payload == b"hello"


def test_big_endian_nanosecond_pcap_and_linux_cooked_links(tmp_path):
    packet = Ether() / IP(src="192.0.2.1", dst="192.0.2.2") / TCP(sport=12, dport=34) / Raw(b"nano")
    packet.time = 1234.123456789
    big_nano = tmp_path / "big-nano.pcap"
    writer = PcapWriter(str(big_nano), endianness=">", nano=True, sync=True)
    writer.write(packet)
    writer.close()
    decoded = list(iter_transport_packets(big_nano))
    assert decoded[0].payload == b"nano"
    assert decoded[0].timestamp == pytest.approx(1234.123456789, abs=1e-8)

    for name, linktype, layer in (
        ("sll.pcap", 113, CookedLinux(proto=0x0800)),
        ("sll2.pcap", 276, CookedLinuxV2(proto=0x0800)),
    ):
        path = tmp_path / name
        writer = PcapWriter(str(path), linktype=linktype, sync=True)
        writer.write(layer / IP(src="198.51.100.1", dst="198.51.100.2") / UDP(sport=1, dport=2) / Raw(b"cooked"))
        writer.close()
        cooked = list(iter_transport_packets(path))
        assert len(cooked) == 1
        assert cooked[0].payload == b"cooked"


def test_full_extractor_reassembles_out_of_order_tls_on_non_443_port(tmp_path):
    data = _tls_client_hello()
    first, second = data[:19], data[19:]
    base = 1001
    packets = [
        Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=51000, dport=8443, flags="S", seq=1000),
        Ether() / IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=8443, dport=51000, flags="SA", seq=5000, ack=1001),
        Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=51000, dport=8443, flags="A", seq=base, ack=5001),
        Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=51000, dport=8443, flags="PA", seq=base + len(first)) / Raw(second),
        Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=51000, dport=8443, flags="PA", seq=base) / Raw(first),
    ]
    for index, packet in enumerate(packets):
        packet.time = 1000 + index * 0.01
    pcap = tmp_path / "tls.pcap"
    wrpcap(str(pcap), packets)
    rows, _ = extract_pcap(pcap, {"label": "benign", "group_id": "capture-1", "split": "test"})
    assert len(rows) == 1
    assert rows[0]["dst_port"] == 8443
    assert json.loads(rows[0]["tls_log"])["server_name"] == "nonstandard.example"
    assert rows[0]["handshake_duration"] == pytest.approx(0.02)
