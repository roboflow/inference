"""Helpers for translating SSL env vars into uvicorn configuration."""

from typing import Any, Dict, List, Optional


class HTTPSConfigurationError(ValueError):
    """Raised when HTTPS is enabled but the SSL configuration is incomplete."""


def build_ssl_uvicorn_kwargs(
    enable_https: bool,
    ssl_certfile: Optional[str],
    ssl_keyfile: Optional[str],
    ssl_keyfile_password: Optional[str] = None,
    ssl_ca_certs: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a dict of SSL kwargs suitable for ``uvicorn.run``.

    Returns an empty dict when ``enable_https`` is falsy. When enabled, both
    ``ssl_certfile`` and ``ssl_keyfile`` are required.
    """
    if not enable_https:
        return {}
    if not ssl_certfile or not ssl_keyfile:
        raise HTTPSConfigurationError(
            "ENABLE_HTTPS is set but SSL_CERTFILE and SSL_KEYFILE must both be "
            "configured to serve HTTPS."
        )
    kwargs: Dict[str, Any] = {
        "ssl_certfile": ssl_certfile,
        "ssl_keyfile": ssl_keyfile,
    }
    if ssl_keyfile_password:
        kwargs["ssl_keyfile_password"] = ssl_keyfile_password
    if ssl_ca_certs:
        kwargs["ssl_ca_certs"] = ssl_ca_certs
    return kwargs


def build_ssl_uvicorn_cli_args(
    enable_https: bool,
    ssl_certfile: Optional[str],
    ssl_keyfile: Optional[str],
    ssl_keyfile_password: Optional[str] = None,
    ssl_ca_certs: Optional[str] = None,
) -> List[str]:
    """Return a list of CLI flags for the uvicorn binary.

    Mirrors :func:`build_ssl_uvicorn_kwargs` for callers that shell out to
    ``uvicorn`` rather than calling ``uvicorn.run`` directly.
    """
    kwargs = build_ssl_uvicorn_kwargs(
        enable_https=enable_https,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_ca_certs=ssl_ca_certs,
    )
    flag_map = {
        "ssl_certfile": "--ssl-certfile",
        "ssl_keyfile": "--ssl-keyfile",
        "ssl_keyfile_password": "--ssl-keyfile-password",
        "ssl_ca_certs": "--ssl-ca-certs",
    }
    args: List[str] = []
    for key, flag in flag_map.items():
        if key in kwargs:
            args.extend([flag, str(kwargs[key])])
    return args
