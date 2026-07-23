import base64
import json
import logging
from typing import Any, Optional
from urllib.parse import quote, urlparse, urlunparse

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)

_STREAM_CREDENTIALS_INFO = b"roboflow-stream-credentials-v1"
_BLOB_VERSION = 1
_SALT_LEN = 16
_NONCE_LEN = 12


def _derive_key(device_api_key: str, salt: bytes) -> bytes:
    return HKDF(
        algorithm=SHA256(),
        length=32,
        salt=salt,
        info=_STREAM_CREDENTIALS_INFO,
    ).derive(device_api_key.encode("utf-8"))


def _decode_blob(stream_credentials: str) -> tuple[bytes, bytes, bytes]:
    raw = base64.b64decode(stream_credentials, validate=True)
    min_len = 1 + _SALT_LEN + _NONCE_LEN + 16
    if len(raw) < min_len:
        raise ValueError("stream_credentials blob too short")
    version = raw[0]
    if version != _BLOB_VERSION:
        raise ValueError(f"unsupported stream_credentials version: {version}")
    salt = raw[1 : 1 + _SALT_LEN]
    nonce = raw[1 + _SALT_LEN : 1 + _SALT_LEN + _NONCE_LEN]
    ciphertext = raw[1 + _SALT_LEN + _NONCE_LEN :]
    return salt, nonce, ciphertext


def decrypt_stream_credentials(stream_credentials: str, api_key: str) -> dict:
    salt, nonce, ciphertext = _decode_blob(stream_credentials)
    key = _derive_key(api_key, salt)
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    return json.loads(plaintext.decode("utf-8"))


def _embed_credentials(video_reference: str, username: str, password: str) -> str:
    parsed = urlparse(video_reference)
    if not parsed.scheme or not parsed.hostname:
        return video_reference
    userinfo = f"{quote(username, safe='')}:{quote(password, safe='')}"
    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    netloc = f"{userinfo}@{host}"
    return urlunparse(parsed._replace(netloc=netloc))


def resolve_operational_video_reference(
    video_reference: Any,
    stream_credentials: Optional[str],
    api_key: Optional[str],
) -> Any:
    if not stream_credentials or not api_key or not video_reference:
        return video_reference
    if not isinstance(video_reference, str):
        return video_reference
    try:
        creds = decrypt_stream_credentials(stream_credentials, api_key)
        username = creds.get("username", "")
        password = creds.get("password", "")
        if not username and not password:
            return video_reference
        return _embed_credentials(video_reference, username, password)
    except Exception:
        logger.warning(
            "Failed to decrypt stream_credentials for video reference %r",
            video_reference,
        )
        return video_reference
