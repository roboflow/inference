from unittest import mock
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from inference_cli import main
from inference_sdk.http.errors import HTTPCallErrorError, HTTPClientError

runner = CliRunner()


@mock.patch.object(main.inference_cli.lib, "infer")
def test_infer_explains_what_to_do_when_the_server_is_unreachable(
    infer_mock: MagicMock,
) -> None:
    # given — the SDK raises the base client error when the connection itself fails
    infer_mock.side_effect = HTTPClientError(
        "Error with server connection: HTTPConnectionPool(host='localhost', port=9001): "
        "Max retries exceeded with url: /some/model (Caused by NewConnectionError(...))"
    )

    # when
    result = runner.invoke(
        main.app,
        ["infer", "--input", "image.jpg", "--model_id", "some/1", "--api-key", "key"],
    )

    # then
    assert result.exit_code == 1
    assert "Could not reach an inference server at http://localhost:9001" in result.output
    # the three ways out are all offered, because the reporter did not know a server was needed
    assert "inference server start" in result.output
    assert "--host" in result.output
    assert "https://detect.roboflow.com" in result.output


@mock.patch.object(main.inference_cli.lib, "infer")
def test_infer_keeps_the_plain_message_when_the_server_answered(
    infer_mock: MagicMock,
) -> None:
    # given — the server replied, so reachability is not the problem and the hint would mislead
    infer_mock.side_effect = HTTPCallErrorError(
        description="404 Client Error",
        status_code=404,
        api_message="Model not found",
    )

    # when
    result = runner.invoke(
        main.app,
        ["infer", "--input", "image.jpg", "--model_id", "missing/1", "--api-key", "key"],
    )

    # then
    assert result.exit_code == 1
    assert "Command failed. Cause:" in result.output
    assert "Could not reach an inference server" not in result.output


@mock.patch.object(main.inference_cli.lib, "infer")
def test_infer_reports_the_host_that_was_actually_used(infer_mock: MagicMock) -> None:
    # given
    infer_mock.side_effect = HTTPClientError("Error with server connection: refused")

    # when
    result = runner.invoke(
        main.app,
        [
            "infer",
            "--input", "image.jpg",
            "--model_id", "some/1",
            "--api-key", "key",
            "--host", "http://192.168.1.10:9001",
        ],
    )

    # then
    assert result.exit_code == 1
    assert "http://192.168.1.10:9001" in result.output
