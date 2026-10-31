import runpy
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from inference.core.interfaces.http.uvicorn_config import build_ssl_uvicorn_kwargs

ROOT = Path(__file__).resolve().parents[6]


def certificate(tmp_path, name, issuer=None):
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.now(timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer[0].subject if issuer else subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.BasicConstraints(ca=issuer is None, path_length=None), critical=True
        )
    )
    cert = builder.sign(issuer[1] if issuer else key, hashes.SHA256())
    cert_path, key_path = tmp_path / (name + ".pem"), tmp_path / (name + ".key")
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert, key, cert_path, key_path


def handshake_without_network(server_context, client_context):
    server_in, server_out, client_in, client_out = [ssl.MemoryBIO() for _ in range(4)]
    server = server_context.wrap_bio(server_in, server_out, server_side=True)
    client = client_context.wrap_bio(
        client_in, client_out, server_side=False, server_hostname="test-server"
    )
    server_done = client_done = False
    for _ in range(20):
        try:
            client.do_handshake()
            client_done = True
        except ssl.SSLWantReadError:
            pass
        server_in.write(client_out.read())
        try:
            server.do_handshake()
            server_done = True
        except ssl.SSLWantReadError:
            pass
        client_in.write(server_out.read())
        if server_done and client_done:
            return
    raise AssertionError("TLS handshake did not finish")


@pytest.mark.parametrize("client_identity", ["none", "untrusted", "trusted"])
def test_actual_uvicorn_tls_context_requires_trusted_client_without_network(
    tmp_path, client_identity
):
    ca = certificate(tmp_path, "test-ca")
    server = certificate(tmp_path, "test-server", ca)
    kwargs = build_ssl_uvicorn_kwargs(
        True, str(server[2]), str(server[3]), ssl_ca_certs=str(ca[2])
    )
    config = uvicorn.Config(app=lambda scope, receive, send: None, **kwargs)
    config.load()
    assert config.ssl.verify_mode == ssl.CERT_REQUIRED
    client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client_context.check_hostname = False
    client_context.load_verify_locations(cafile=str(ca[2]))
    if client_identity != "none":
        client_ca = (
            ca if client_identity == "trusted" else certificate(tmp_path, "other-ca")
        )
        client = certificate(tmp_path, "test-client", client_ca)
        client_context.load_cert_chain(str(client[2]), str(client[3]))
    if client_identity == "trusted":
        handshake_without_network(config.ssl, client_context)
    else:
        with pytest.raises(ssl.SSLError):
            handshake_without_network(config.ssl, client_context)


def test_parallel_launcher_requires_certificates_without_launching_services(
    monkeypatch,
):
    from inference.core import env

    for name, value in [
        ("ENABLE_HTTPS", True),
        ("SSL_CERTFILE", "test-cert.pem"),
        ("SSL_KEYFILE", "test-key.pem"),
        ("SSL_CA_CERTS", "test-ca.pem"),
    ]:
        monkeypatch.setattr(env, name, value)
    with patch("os.system") as launch:
        module = runpy.run_path(
            str(ROOT / "inference/enterprise/parallel/entrypoint.py")
        )
    assert "--cert-reqs=2" in module["_gunicorn_ssl_flags"]()
    assert "--cert-reqs=2" in launch.call_args.args[0]


def test_parallel_launcher_rejects_unsupported_encrypted_key_before_any_service_launch(
    monkeypatch,
):
    from inference.core import env

    monkeypatch.setattr(env, "ENABLE_HTTPS", True)
    monkeypatch.setattr(env, "SSL_CERTFILE", "test-cert.pem")
    monkeypatch.setattr(env, "SSL_KEYFILE", "test-key.pem")
    monkeypatch.setattr(env, "SSL_KEYFILE_PASSWORD", "test-only-password")
    with patch("os.system") as launch:
        with pytest.raises(
            RuntimeError,
            match="Gunicorn launcher does not support SSL_KEYFILE_PASSWORD",
        ):
            runpy.run_path(str(ROOT / "inference/enterprise/parallel/entrypoint.py"))
    launch.assert_not_called()
