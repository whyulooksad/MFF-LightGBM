"""Strict X.509 DER decoding for the canonical feature schema.

No value in this module is guessed.  A malformed or unavailable certificate is
represented by ``None`` plus an explicit parse status.
X.509 证书解析器。

pcap_reader.py
    从PCAP读取一个个TCP包
        ↓
tcp_reassembly.py
    按TCP序列号恢复连续字节
        ↓
tls_parser.py
    从字节流中找到TLS Record
    再找到Certificate握手消息
    从中切出一张张DER证书
        ↓
x509_parser.py
    解析每张DER证书中的具体字段
"""

from __future__ import annotations

import hashlib
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import dsa, ec, ed25519, ed448, rsa
from cryptography.x509.oid import NameOID


def _public_key_info(key: Any) -> tuple[str | None, int | None, str | None]:
    if isinstance(key, rsa.RSAPublicKey):
        return "RSA", key.key_size, None
    if isinstance(key, dsa.DSAPublicKey):
        return "DSA", key.key_size, None
    if isinstance(key, ec.EllipticCurvePublicKey):
        return "EC", key.key_size, key.curve.name
    if isinstance(key, ed25519.Ed25519PublicKey):
        return "Ed25519", 256, None
    if isinstance(key, ed448.Ed448PublicKey):
        return "Ed448", 456, None
    return type(key).__name__, getattr(key, "key_size", None), None


def parse_der_certificate(der: bytes) -> dict[str, Any]:
    base: dict[str, Any] = {
        "parse_status": "error",
        "parse_error": None,
        "der_sha256": hashlib.sha256(der).hexdigest() if der else None,
    }
    if not der:
        base["parse_error"] = "empty DER certificate"
        return base
    try:
        cert = x509.load_der_x509_certificate(der)
        pub = cert.public_key()
        key_type, key_length, curve = _public_key_info(pub)
        cn_values = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        try:
            san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            san_dns = san_ext.get_values_for_type(x509.DNSName)
            san_ip = [str(value) for value in san_ext.get_values_for_type(x509.IPAddress)]
            san_uri = san_ext.get_values_for_type(x509.UniformResourceIdentifier)
        except x509.ExtensionNotFound:
            san_dns, san_ip, san_uri = [], [], []
        try:
            ca = cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
        except x509.ExtensionNotFound:
            ca = None
        base.update(
            {
                "parse_status": "ok",
                "parse_error": None,
                "certificate.version": cert.version.name,
                "certificate.serial_number": format(cert.serial_number, "x"),
                "certificate.subject": cert.subject.rfc4514_string(),
                "certificate.subject_cn": cn_values[0].value if cn_values else None,
                "certificate.issuer": cert.issuer.rfc4514_string(),
                "certificate.not_valid_before": cert.not_valid_before_utc.isoformat(),
                "certificate.not_valid_after": cert.not_valid_after_utc.isoformat(),
                "certificate.sig_alg": cert.signature_algorithm_oid.dotted_string,
                "certificate.sig_alg_name": getattr(cert.signature_algorithm_oid, "_name", None),
                "certificate.key_type": key_type,
                "certificate.key_length": key_length,
                "certificate.curve": curve,
                "san.dns": san_dns,
                "san.ip": san_ip,
                "san.uri": san_uri,
                "basic_constraints.ca": ca,
            }
        )
    except Exception as exc:  # malformed third-party input must not kill the PCAP
        base["parse_error"] = f"{type(exc).__name__}: {exc}"
    return base
