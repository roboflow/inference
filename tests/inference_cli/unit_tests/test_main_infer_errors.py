from unittest import mock
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from inference_cli import main
from inference_sdk.http.errors import (
    APIKeyNotProvided,
    HTTPCallErrorError,
    HTTPClientError,
    HTTPServerConnectionError,
)

runner = CliRunner()


@mock.patch.object(main.inference_cli.lib, "infer")
def test_infer_explains_what_to_do_when_the_server_is_unreachable(
    infer_mock: MagicMock,
) -> None:
    # given
    infer_mock.side_effect = HTTPServerConnectionError(
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
    # the original SDK message is kept
    assert "Max retries exceeded" in result.output
    assert (
        "Could not reach an inference server at http://localhost:9001" in result.output
    )
    assert "inference server start" in result.output
    assert "--host" in result.output
    assert "https://serverless.roboflow.com" in result.output


@mock.patch.object(main.inference_cli.lib, "infer")
def test_infer_keeps_the_plain_message_when_the_server_answered(
    infer_mock: MagicMock,
) -> None:
    # given
    infer_mock.side_effect = HTTPCallErrorError(
        description="404 Client Error",
        status_code=404,
        api_message="Model not found",
    )

    # when
    result = runner.invoke(
        main.app,
        [
            "infer",
            "--input",
            "image.jpg",
            "--model_id",
            "missing/1",
            "--api-key",
            "key",
        ],
    )

    # then
    assert result.exit_code == 1
    assert "Command failed. Cause:" in result.output
    assert "404" in result.output
    assert "Model not found" in result.output
    assert "Could not reach an inference server" not in result.output
    assert "inference server start" not in result.output


@pytest.mark.parametrize(
    "error",
    [
        APIKeyNotProvided("API key must be provided"),
        HTTPClientError("Could not load image from URL. Details: Connection refused"),
    ],
)
@mock.patch.object(main.inference_cli.lib, "infer")
def test_infer_does_not_blame_reachability_for_other_client_errors(
    infer_mock: MagicMock,
    error: HTTPClientError,
) -> None:
    # given
    infer_mock.side_effect = error

    # when
    result = runner.invoke(
        main.app,
        ["infer", "--input", "image.jpg", "--model_id", "some/1"],
    )

    # then
    assert result.exit_code == 1
    assert str(error) in result.output
    assert "Could not reach an inference server" not in result.output
    assert "inference server start" not in result.output


@mock.patch.object(main.inference_cli.lib, "infer")
def test_infer_reports_the_host_that_was_actually_used(infer_mock: MagicMock) -> None:
    # given
    infer_mock.side_effect = HTTPServerConnectionError(
        "Error with server connection: refused"
    )

    # when
    result = runner.invoke(
        main.app,
        [
            "infer",
            "--input",
            "image.jpg",
            "--model_id",
            "some/1",
            "--api-key",
            "key",
            "--host",
            "http://192.168.1.10:9001",
        ],
    )

    # then
    assert result.exit_code == 1
    assert (
        "Could not reach an inference server at http://192.168.1.10:9001"
        in result.output
    )
    assert infer_mock.call_args.kwargs["host"] == "http://192.168.1.10:9001"
