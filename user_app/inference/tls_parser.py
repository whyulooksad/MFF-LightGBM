"""
它先从连续 TCP 字节中拆出一箱箱 TLS Record，再从握手类型的箱子里拆出 ClientHello、ServerHello 和证书；
遇到长度不对、数据不完整或中间缺失时，它宁可停止当前解析，也不伪造不存在的信息。
最后得到：
    是否检测到TLS
    TLS Record数量
    异常Record数量
    客户端支持的版本
    服务器选择的版本
    SNI域名
    ALPN协议
    加密套件
    证书列表
    证书为什么可见或不可见
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from user_app.inference.x509_parser import parse_der_certificate

TLS_VERSIONS = {
    0x0300: "SSLv3",
    0x0301: "TLSv1.0",
    0x0302: "TLSv1.1",
    0x0303: "TLSv1.2",
    0x0304: "TLSv1.3",
}


def _u16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "big")


def _u24(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 3], "big")


def _extensions(data: bytes, offset: int) -> tuple[dict[int, list[bytes]], int]:
    if offset + 2 > len(data):
        return {}, offset
    total = _u16(data, offset)
    offset += 2
    end = min(len(data), offset + total)
    out: dict[int, list[bytes]] = {}
    while offset + 4 <= end:
        kind, size = _u16(data, offset), _u16(data, offset + 2)
        offset += 4
        if offset + size > end:
            break
        out.setdefault(kind, []).append(data[offset : offset + size])
        offset += size
    return out, offset


def _parse_client_hello(body: bytes) -> dict[str, Any] | None:
    if len(body) < 35:
        return None
    pos = 34
    sid_len = body[pos]
    pos += 1 + sid_len
    if pos + 2 > len(body):
        return None
    cipher_len = _u16(body, pos)
    pos += 2
    if cipher_len % 2 or pos + cipher_len > len(body):
        return None
    ciphers = [_u16(body, i) for i in range(pos, pos + cipher_len, 2)]
    pos += cipher_len
    if pos >= len(body):
        return None
    comp_len = body[pos]
    pos += 1 + comp_len
    exts, _ = _extensions(body, pos)
    sni = None
    if 0 in exts and len(exts[0][0]) >= 5:
        block = exts[0][0]
        name_len = _u16(block, 3)
        if block[2] == 0 and 5 + name_len <= len(block):
            # SNI host_name is an ASCII A-label on the wire. Preserve malformed
            # bytes visibly instead of allowing codec failure to abort a PCAP.
            sni = block[5 : 5 + name_len].decode("ascii", errors="replace")
    alpn: list[str] = []
    if 16 in exts and len(exts[16][0]) >= 2:
        block, p = exts[16][0], 2
        while p < len(block):
            size = block[p]
            p += 1
            if p + size > len(block):
                break
            alpn.append(block[p : p + size].decode("ascii", errors="replace"))
            p += size
    versions: list[int] = []
    if 43 in exts and exts[43][0]:
        block = exts[43][0]
        if len(block) >= 1:
            size = block[0]
            versions = [_u16(block, i) for i in range(1, min(1 + size, len(block)), 2) if i + 2 <= len(block)]
    return {
        "legacy_version": TLS_VERSIONS.get(_u16(body, 0), hex(_u16(body, 0))),
        "server_name": sni,
        "alpn_offered": alpn,
        "supported_versions": [TLS_VERSIONS.get(v, hex(v)) for v in versions],
        "cipher_suites": [f"0x{value:04x}" for value in ciphers],
        "extensions": sorted(exts),
    }


def _parse_server_hello(body: bytes) -> dict[str, Any] | None:
    if len(body) < 38:
        return None
    pos = 34
    sid_len = body[pos]
    pos += 1 + sid_len
    if pos + 3 > len(body):
        return None
    cipher = _u16(body, pos)
    pos += 3
    exts, _ = _extensions(body, pos)
    selected = _u16(body, 0)
    if 43 in exts and len(exts[43][0]) == 2:
        selected = _u16(exts[43][0], 0)
    alpn = None
    if 16 in exts and len(exts[16][0]) >= 3:
        size = exts[16][0][2]
        alpn = exts[16][0][3 : 3 + size].decode("ascii", errors="replace")
    return {
        "version": TLS_VERSIONS.get(selected, hex(selected)),
        "cipher": f"0x{cipher:04x}",
        "next_protocol": alpn,
        "extensions": sorted(exts),
    }


def _certificates(body: bytes, tls13: bool) -> list[bytes]:
    pos = 0
    if tls13:
        if not body:
            return []
        pos = 1 + body[0]
    if pos + 3 > len(body):
        return []
    end = min(len(body), pos + 3 + _u24(body, pos))
    pos += 3
    certs: list[bytes] = []
    while pos + 3 <= end:
        size = _u24(body, pos)
        pos += 3
        if pos + size > end:
            break
        certs.append(body[pos : pos + size])
        pos += size
        if tls13:
            if pos + 2 > end:
                break
            ext_len = _u16(body, pos)
            pos += 2 + ext_len
    return certs


@dataclass
class TLSResult:
    detected: bool = False
    records: int = 0
    malformed_records: int = 0
    client_hello: dict[str, Any] | None = None
    server_hello: dict[str, Any] | None = None
    certificates: list[dict[str, Any]] = field(default_factory=list)
    certificate_visibility: str = "not_observed"

    def as_log(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "detected": self.detected,
            "records": self.records,
            "malformed_records": self.malformed_records,
            "certificate_visibility": self.certificate_visibility,
        }
        if self.client_hello:
            result.update(self.client_hello)
        if self.server_hello:
            result.update(self.server_hello)
        result["established"] = bool(self.server_hello)
        return result


def parse_tls_streams(client_chunks: list[bytes], server_chunks: list[bytes]) -> TLSResult:
    result = TLSResult()
    handshake_buffers = {"client": bytearray(), "server": bytearray()}
    messages: list[tuple[str, int, bytes]] = []
    for direction, chunks in (("client", client_chunks), ("server", server_chunks)):
        for chunk in chunks:
            if handshake_buffers[direction]:
                # A separate TCP chunk means capture bytes were missing. Do
                # not bridge a handshake message across that unknown gap.
                handshake_buffers[direction].clear()
                result.malformed_records += 1
            pos = 0
            while pos + 5 <= len(chunk):
                content_type, version, size = chunk[pos], _u16(chunk, pos + 1), _u16(chunk, pos + 3)
                if content_type not in {20, 21, 22, 23, 24} or version not in TLS_VERSIONS or size > 18432:
                    pos += 1
                    continue
                if pos + 5 + size > len(chunk):
                    result.malformed_records += 1
                    break
                result.detected = True
                result.records += 1
                payload = chunk[pos + 5 : pos + 5 + size]
                pos += 5 + size
                if content_type != 22:
                    continue
                buf = handshake_buffers[direction]
                buf.extend(payload)
                while len(buf) >= 4:
                    msg_size = _u24(buf, 1)
                    if msg_size > 16 * 1024 * 1024:
                        buf.clear()
                        break
                    if len(buf) < 4 + msg_size:
                        break
                    messages.append((direction, buf[0], bytes(buf[4 : 4 + msg_size])))
                    del buf[: 4 + msg_size]

    cert_der: list[bytes] = []
    for direction, kind, body in messages:
        if kind == 1 and result.client_hello is None:
            result.client_hello = _parse_client_hello(body)
        elif kind == 2 and result.server_hello is None:
            result.server_hello = _parse_server_hello(body)
        elif kind == 11:
            tls13 = bool(result.server_hello and result.server_hello.get("version") == "TLSv1.3")
            cert_der.extend(_certificates(body, tls13))
    result.certificates = [parse_der_certificate(value) for value in cert_der]
    if result.certificates:
        result.certificate_visibility = "plaintext_certificate_observed"
    elif result.server_hello and result.server_hello.get("version") == "TLSv1.3":
        result.certificate_visibility = "encrypted_or_not_captured_tls13"
    elif result.detected:
        result.certificate_visibility = "not_captured_or_session_resumed"
    return result
