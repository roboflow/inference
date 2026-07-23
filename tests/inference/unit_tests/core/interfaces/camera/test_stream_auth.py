import base64
import json
import secrets
from urllib.parse import urlparse

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from inference.core.interfaces.camera.stream_auth import (
    decrypt_stream_credentials,
    resolve_operational_video_reference,
)

_HKDF_INFO = b"roboflow-stream-credentials-v1"


def _encrypt_stream_credentials(creds: dict, api_key: str) -> str:
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(12)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=_HKDF_INFO,
    )
    key = hkdf.derive(api_key.encode("utf-8"))
    plaintext = json.dumps(creds).encode("utf-8")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    blob = bytes([1]) + salt + nonce + ciphertext
    return base64.b64encode(blob).decode("ascii")


def test_round_trip_encrypt_decrypt():
    api_key = "test-device-api-key"
    creds = {"username": "camera-user", "password": "camera-pass"}
    blob = _encrypt_stream_credentials(creds, api_key)

    assert decrypt_stream_credentials(blob, api_key) == creds


def test_resolve_without_blob_returns_unchanged():
    video_reference = "rtsp://cam.local/live/main"

    result = resolve_operational_video_reference(video_reference, None, "api-key")

    assert result == video_reference


def test_resolve_with_blob_embeds_credentials():
    api_key = "test-device-api-key"
    video_reference = "rtsp://cam.local/live/main"
    blob = _encrypt_stream_credentials(
        {"username": "camera-user", "password": "camera-pass"}, api_key
    )

    result = resolve_operational_video_reference(video_reference, blob, api_key)
    parsed = urlparse(result)

    assert parsed.scheme == "rtsp"
    assert parsed.username == "camera-user"
    assert parsed.hostname == "cam.local"
    assert parsed.password == "camera-pass"
