from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from user_app.inference.x509_parser import parse_der_certificate


def test_real_der_certificate_fields_are_extracted():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "example.test")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(123).not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("example.test")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    parsed = parse_der_certificate(cert.public_bytes(serialization.Encoding.DER))
    assert parsed["parse_status"] == "ok"
    assert parsed["certificate.subject_cn"] == "example.test"
    assert parsed["san.dns"] == ["example.test"]
    assert parsed["certificate.key_type"] == "RSA"
    assert parsed["certificate.key_length"] == 2048


def test_malformed_der_is_explicitly_missing_not_fabricated():
    parsed = parse_der_certificate(b"not-a-certificate")
    assert parsed["parse_status"] == "error"
    assert parsed["parse_error"]
    assert "certificate.subject" not in parsed
