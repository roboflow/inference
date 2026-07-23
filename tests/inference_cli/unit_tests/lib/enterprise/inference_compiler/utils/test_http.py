"""Regression tests for `upload_file_to_cloud` routing through SECURE_GATEWAY.

`upload_file_to_cloud` PUTs compiled TRT engine / timing cache artefacts to a
signed GCS URL returned by the Roboflow API. On edge deployments locked down
to only allow egress to api.roboflow.com/repo.roboflow.com, an unwrapped PUT
straight to storage.googleapis.com hangs until it times out. These tests pin
that the URL is routed through `roboflow_secure_gateway_proxy_url_builder()`
(SECURE_GATEWAY proxy) when configured, and passed through unchanged otherwise.
"""

from unittest.mock import MagicMock

from inference_cli.lib.enterprise.inference_compiler.utils import http as http_module

RAW_UPLOAD_URL = "https://storage.googleapis.com/bucket/object?X-Goog-Signature=abc"


def _fake_response(status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


def test_upload_wraps_url_when_secure_gateway_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(
        http_module,
        "roboflow_secure_gateway_proxy_url_builder",
        lambda url, query: f"http://gateway/proxy?url={url}",
    )
    put = MagicMock(return_value=_fake_response())
    monkeypatch.setattr(http_module.requests, "put", put)
    file_path = tmp_path / "engine.plan"
    file_path.write_bytes(b"trt-engine-bytes")

    http_module.upload_file_to_cloud(
        file_path=str(file_path), url=RAW_UPLOAD_URL, headers={"content-type": "application/octet-stream"}
    )

    assert put.call_count == 1
    called_url = put.call_args.args[0]
    assert called_url == f"http://gateway/proxy?url={RAW_UPLOAD_URL}"
    assert called_url != RAW_UPLOAD_URL


def test_upload_leaves_url_untouched_when_no_secure_gateway(monkeypatch, tmp_path):
    monkeypatch.setattr(
        http_module,
        "roboflow_secure_gateway_proxy_url_builder",
        lambda url, query: url,
    )
    put = MagicMock(return_value=_fake_response())
    monkeypatch.setattr(http_module.requests, "put", put)
    file_path = tmp_path / "engine.plan"
    file_path.write_bytes(b"trt-engine-bytes")

    http_module.upload_file_to_cloud(file_path=str(file_path), url=RAW_UPLOAD_URL, headers={})

    called_url = put.call_args.args[0]
    assert called_url == RAW_UPLOAD_URL


def test_upload_sets_an_explicit_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        http_module,
        "roboflow_secure_gateway_proxy_url_builder",
        lambda url, query: url,
    )
    put = MagicMock(return_value=_fake_response())
    monkeypatch.setattr(http_module.requests, "put", put)
    file_path = tmp_path / "engine.plan"
    file_path.write_bytes(b"trt-engine-bytes")

    http_module.upload_file_to_cloud(file_path=str(file_path), url=RAW_UPLOAD_URL, headers={})

    assert put.call_args.kwargs["timeout"] == http_module.UPLOAD_TIMEOUT_SECONDS
