# app/tests/test_tls_log_parser.py
import json
from user_app.backend.tls_log_parser import parse_tls_logs

def test_parse_empty_row():
    assert parse_tls_logs({}) == {}

def test_parse_row_with_ssl_and_x509():
    row = {
        "connection_log": json.dumps({"service": "tls"}),
        "tls_log": json.dumps({"server_name": "example.com", "version": "TLSv1.2",
                                "cipher": "0x1302", "next_protocol": "h2"}),
        "x509_log": json.dumps([{"certificate.issuer": "CN=Let's Encrypt",
                                  "certificate.subject": "CN=example.com",
                                  "certificate.not_valid_before": "2026-01-01",
                                  "certificate.not_valid_after": "2027-01-01"}]),
    }
    out = parse_tls_logs(row)
    assert out["sni"] == "example.com"
    assert out["tls_version"] == "TLSv1.2"
    assert out["cipher_suite"] == "0x1302"
    assert out["alpn"] == "h2"
    assert out["cert_issuer"] == "CN=Let's Encrypt"
    assert out["cert_chain_depth"] == 1

def test_parse_malformed_json_returns_empty():
    row = {"tls_log": "{not valid json"}
    out = parse_tls_logs(row)
    # 不抛异常，缺失字段返回空 dict 或默认值
    assert "sni" not in out or out.get("sni") in (None, "")
