# app/tls_log_parser.py
import ast
import json
from typing import Any


def _safe_json(s):
    if not s or not isinstance(s, str):
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, (dict, list)) else {}
    except (json.JSONDecodeError, ValueError):
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, (dict, list)) else {}
        except (SyntaxError, ValueError):
            return {}


def _unwrap(log):
    """zeek log 形如 {"c": {...}} / {"s": {...}} / {"x": {...}}，取内层。"""
    if isinstance(log, list):
        return log[0] if log and isinstance(log[0], dict) else {}
    if not isinstance(log, dict) or not log:
        return {}
    # Compact logs use {"s": {...}}; canonical flow-feature CSV stores the
    # Zeek-like dict directly.
    if len(log) == 1:
        inner = next(iter(log.values()))
        if isinstance(inner, dict):
            return inner
    return log


def parse_tls_logs(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    conn = _unwrap(_safe_json(row.get("zeek_conn_log")))
    ssl = _unwrap(_safe_json(row.get("zeek_ssl_log")))
    x509 = _unwrap(_safe_json(row.get("zeek_x509_log")))

    if ssl:
        for k_src, k_dst in [("server_name", "sni"), ("version", "tls_version"),
                             ("cipher", "cipher_suite"), ("alpn", "alpn")]:
            v = ssl.get(k_src)
            if v not in (None, ""):
                out[k_dst] = v
    if x509:
        for k_src, k_dst in [("certificate.issuer", "cert_issuer"),
                             ("certificate.subject", "cert_subject"),
                             ("certificate.not_valid_before", "cert_valid_from"),
                             ("certificate.not_valid_after", "cert_valid_to"),
                             ("certificate_issuer", "cert_issuer"),
                             ("certificate_subject", "cert_subject"),
                             ("certificate_not_valid_before", "cert_valid_from"),
                             ("certificate_not_valid_after", "cert_valid_to")]:
            v = x509.get(k_src)
            if v not in (None, ""):
                out[k_dst] = v
        if "certificate_chain_depth" in x509:
            out["cert_chain_depth"] = x509["certificate_chain_depth"]
    return out
