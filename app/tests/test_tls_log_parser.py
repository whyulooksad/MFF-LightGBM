# app/tests/test_tls_log_parser.py
import json
from tls_log_parser import parse_tls_logs

def test_parse_empty_row():
    assert parse_tls_logs({}) == {}

def test_parse_row_with_ssl_and_x509():
    row = {
        "zeek_conn_log": json.dumps({"c": {"id_orig_h": "10.0.0.1", "service": "ssl"}}),
        "zeek_ssl_log": json.dumps({"s": {"server_name": "example.com",
                                           "version": "TLSv12",
                                           "cipher": "TLS_AES_256_GCM_SHA384",
                                           "alpn": "h2"}}),
        "zeek_x509_log": json.dumps({"x": {"certificate_issuer": "CN=Let's Encrypt",
                                            "certificate_subject": "CN=example.com",
                                            "certificate_not_valid_before": "2026-01-01",
                                            "certificate_not_valid_after": "2027-01-01",
                                            "certificate_chain_depth": 0}}),
    }
    out = parse_tls_logs(row)
    assert out["sni"] == "example.com"
    assert out["tls_version"] == "TLSv12"
    assert out["cipher_suite"] == "TLS_AES_256_GCM_SHA384"
    assert out["alpn"] == "h2"
    assert out["cert_issuer"] == "CN=Let's Encrypt"
    assert out["cert_chain_depth"] == 0

def test_parse_malformed_json_returns_empty():
    row = {"zeek_ssl_log": "{not valid json"}
    out = parse_tls_logs(row)
    # 不抛异常，缺失字段返回空 dict 或默认值
    assert "sni" not in out or out.get("sni") in (None, "")
