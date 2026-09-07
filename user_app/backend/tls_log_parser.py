import json
from typing import Any


def _safe_json(s):
    if not s or not isinstance(s, str):
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, (dict, list)) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _unwrap(log):
    """For certificate chains the UI shows the leaf certificate."""
    if isinstance(log, list):
        return log[0] if log and isinstance(log[0], dict) else {}
    if not isinstance(log, dict) or not log:
        return {}
    return log


def parse_tls_logs(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    conn = _unwrap(_safe_json(row.get("connection_log")))
    ssl = _unwrap(_safe_json(row.get("tls_log")))
    x509 = _unwrap(_safe_json(row.get("x509_log")))

    if ssl:
        for k_src, k_dst in [("server_name", "sni"), ("version", "tls_version"),
                             ("cipher", "cipher_suite"), ("next_protocol", "alpn")]:
            v = ssl.get(k_src)
            if v not in (None, ""):
                out[k_dst] = v
    if x509:
        for k_src, k_dst in [("certificate.issuer", "cert_issuer"),
                             ("certificate.subject", "cert_subject"),
                             ("certificate.not_valid_before", "cert_valid_from"),
                             ("certificate.not_valid_after", "cert_valid_to")]:
            v = x509.get(k_src)
            if v not in (None, ""):
                out[k_dst] = v
        out["cert_chain_depth"] = len(_safe_json(row.get("x509_log")))
    return out
