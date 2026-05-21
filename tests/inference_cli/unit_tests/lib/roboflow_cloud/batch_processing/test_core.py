"""Tests for the --custom-python-block-timeout CLI flag plumbing."""

from unittest import mock

from typer.testing import CliRunner

from inference_cli.lib.roboflow_cloud.batch_processing import core as core_module
from inference_cli.lib.roboflow_cloud.batch_processing.core import batch_processing_app


def _baseline_args(command: str) -> list:
    args = [
        command,
        "--batch-id",
        "b-1",
        "--workflow-id",
        "wf-1",
        "--api-key",
        "test-key",
    ]
    return args


def test_images_command_passes_timeout_when_flag_supplied() -> None:
    runner = CliRunner()
    with mock.patch.object(
        core_module, "trigger_job_with_workflows_images_processing", return_value="job-xyz"
    ) as mock_trigger:
        result = runner.invoke(
            batch_processing_app,
            _baseline_args("process-images-with-workflow")
            + ["--custom-python-block-timeout", "45"],
        )

    assert result.exit_code == 0, result.stdout
    _, kwargs = mock_trigger.call_args
    assert kwargs["custom_python_block_timeout_seconds"] == 45


def test_videos_command_passes_timeout_when_flag_supplied() -> None:
    runner = CliRunner()
    with mock.patch.object(
        core_module, "trigger_job_with_workflows_videos_processing", return_value="job-xyz"
    ) as mock_trigger:
        result = runner.invoke(
            batch_processing_app,
            _baseline_args("process-videos-with-workflow")
            + ["--custom-python-block-timeout", "90"],
        )

    assert result.exit_code == 0, result.stdout
    _, kwargs = mock_trigger.call_args
    assert kwargs["custom_python_block_timeout_seconds"] == 90


def test_images_command_omits_timeout_when_flag_absent() -> None:
    runner = CliRunner()
    with mock.patch.object(
        core_module, "trigger_job_with_workflows_images_processing", return_value="job-xyz"
    ) as mock_trigger:
        result = runner.invoke(
            batch_processing_app, _baseline_args("process-images-with-workflow")
        )

    assert result.exit_code == 0, result.stdout
    _, kwargs = mock_trigger.call_args
    # None is the contract for "field absent from payload" — the api_operations
    # layer constructs WorkflowsProcessingSpecificationV1 with this value and
    # serialises with exclude_none=True at the wire boundary.
    assert kwargs["custom_python_block_timeout_seconds"] is None


def test_images_command_rejects_out_of_range_timeout() -> None:
    runner = CliRunner()
    with mock.patch.object(
        core_module, "trigger_job_with_workflows_images_processing"
    ) as mock_trigger:
        result = runner.invoke(
            batch_processing_app,
            _baseline_args("process-images-with-workflow")
            + ["--custom-python-block-timeout", "200"],
        )

    assert result.exit_code != 0
    mock_trigger.assert_not_called()
