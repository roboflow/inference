import pytest

from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    _as_ws_endpoint_url,
    _coerce_http_endpoint_to_ws_endpoint,
)


@pytest.mark.parametrize(
    "http_url,expected",
    [
        (
            "https://roboflow--webexec-executor-execute-block.modal.run",
            "https://roboflow--webexec-executor-wsapp.modal.run",
        ),
        (
            "https://roboflow--webexec-roboflow-staging-executor-execute-block.modal.run",
            "https://roboflow--webexec-roboflow-staging-executor-wsapp.modal.run",
        ),
        (
            "https://roboflow-eu--webexec-executor-execute-block.eu-west.modal.run",
            "https://roboflow-eu--webexec-executor-wsapp.eu-west.modal.run",
        ),
        (
            "https://roboflow--webexec-executor-execute-block.modal.run/",
            "https://roboflow--webexec-executor-wsapp.modal.run",
        ),
    ],
)
def test_coerce_http_endpoint_to_ws_endpoint(http_url: str, expected: str) -> None:
    assert _coerce_http_endpoint_to_ws_endpoint(http_url) == expected


def test_coerce_leaves_non_execute_block_url_untouched() -> None:
    url = "https://roboflow--webexec-executor-wsapp.modal.run"
    assert _coerce_http_endpoint_to_ws_endpoint(url) == url


def test_as_ws_endpoint_url() -> None:
    result = _as_ws_endpoint_url(
        "https://roboflow--webexec-executor-wsapp.eu-west.modal.run",
        workspace_id="my-workspace",
    )
    assert result == (
        "wss://roboflow--webexec-executor-wsapp.eu-west.modal.run/ws"
        "?workspace_id=my-workspace"
    )
