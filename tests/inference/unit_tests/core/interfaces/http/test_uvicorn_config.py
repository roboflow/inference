import pytest

from inference.core.interfaces.http.uvicorn_config import (
    HTTPSConfigurationError,
    build_ssl_uvicorn_cli_args,
    build_ssl_uvicorn_kwargs,
)


def test_build_ssl_uvicorn_kwargs_returns_empty_when_disabled() -> None:
    assert (
        build_ssl_uvicorn_kwargs(
            enable_https=False,
            ssl_certfile="/tmp/cert.pem",
            ssl_keyfile="/tmp/key.pem",
        )
        == {}
    )


def test_build_ssl_uvicorn_kwargs_minimal_pair() -> None:
    kwargs = build_ssl_uvicorn_kwargs(
        enable_https=True,
        ssl_certfile="/tmp/cert.pem",
        ssl_keyfile="/tmp/key.pem",
    )
    assert kwargs == {
        "ssl_certfile": "/tmp/cert.pem",
        "ssl_keyfile": "/tmp/key.pem",
    }


def test_build_ssl_uvicorn_kwargs_full() -> None:
    kwargs = build_ssl_uvicorn_kwargs(
        enable_https=True,
        ssl_certfile="/tmp/cert.pem",
        ssl_keyfile="/tmp/key.pem",
        ssl_keyfile_password="hunter2",
        ssl_ca_certs="/tmp/ca.pem",
    )
    assert kwargs == {
        "ssl_certfile": "/tmp/cert.pem",
        "ssl_keyfile": "/tmp/key.pem",
        "ssl_keyfile_password": "hunter2",
        "ssl_ca_certs": "/tmp/ca.pem",
    }


def test_build_ssl_uvicorn_kwargs_skips_blank_optional_values() -> None:
    kwargs = build_ssl_uvicorn_kwargs(
        enable_https=True,
        ssl_certfile="/tmp/cert.pem",
        ssl_keyfile="/tmp/key.pem",
        ssl_keyfile_password="",
        ssl_ca_certs=None,
    )
    assert "ssl_keyfile_password" not in kwargs
    assert "ssl_ca_certs" not in kwargs


@pytest.mark.parametrize(
    "certfile,keyfile",
    [(None, "/tmp/key.pem"), ("/tmp/cert.pem", None), ("", ""), (None, None)],
)
def test_build_ssl_uvicorn_kwargs_raises_when_pair_incomplete(
    certfile, keyfile
) -> None:
    with pytest.raises(HTTPSConfigurationError):
        build_ssl_uvicorn_kwargs(
            enable_https=True,
            ssl_certfile=certfile,
            ssl_keyfile=keyfile,
        )


def test_build_ssl_uvicorn_cli_args_disabled_returns_empty_list() -> None:
    assert (
        build_ssl_uvicorn_cli_args(
            enable_https=False,
            ssl_certfile="/tmp/cert.pem",
            ssl_keyfile="/tmp/key.pem",
        )
        == []
    )


def test_build_ssl_uvicorn_cli_args_full() -> None:
    args = build_ssl_uvicorn_cli_args(
        enable_https=True,
        ssl_certfile="/tmp/cert.pem",
        ssl_keyfile="/tmp/key.pem",
        ssl_keyfile_password="hunter2",
        ssl_ca_certs="/tmp/ca.pem",
    )
    assert args == [
        "--ssl-certfile",
        "/tmp/cert.pem",
        "--ssl-keyfile",
        "/tmp/key.pem",
        "--ssl-keyfile-password",
        "hunter2",
        "--ssl-ca-certs",
        "/tmp/ca.pem",
    ]
