from user_app.inference.tls_parser import parse_tls_streams


def _record(payload: bytes) -> bytes:
    return b"\x16\x03\x01" + len(payload).to_bytes(2, "big") + payload


def test_client_hello_fields_are_binary_parsed():
    sni = b"example.com"
    sni_ext = b"\x00\x00" + (5 + len(sni)).to_bytes(2, "big") + (3 + len(sni)).to_bytes(2, "big") + b"\x00" + len(sni).to_bytes(2, "big") + sni
    alpn_data = b"\x00\x03\x02h2"
    alpn_ext = b"\x00\x10" + len(alpn_data).to_bytes(2, "big") + alpn_data
    versions_data = b"\x04\x03\x04\x03\x03"
    versions_ext = b"\x00\x2b" + len(versions_data).to_bytes(2, "big") + versions_data
    exts = sni_ext + alpn_ext + versions_ext
    body = b"\x03\x03" + bytes(32) + b"\x00" + b"\x00\x02\x13\x01" + b"\x01\x00" + len(exts).to_bytes(2, "big") + exts
    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    result = parse_tls_streams([_record(handshake)], [])
    assert result.detected
    assert result.client_hello["server_name"] == "example.com"
    assert result.client_hello["alpn_offered"] == ["h2"]
    assert result.client_hello["supported_versions"] == ["TLSv1.3", "TLSv1.2"]
    assert result.client_hello["cipher_suites"] == ["0x1301"]


def test_truncated_record_is_not_accepted_as_tls():
    result = parse_tls_streams([b"\x16\x03\x03\x00\x10abc"], [])
    assert not result.detected
    assert result.malformed_records == 1
