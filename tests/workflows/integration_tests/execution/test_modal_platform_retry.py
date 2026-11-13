import subprocess
import traceback
from types import SimpleNamespace
from typing import List
from unittest import mock

import pytest
import requests
from roboflow_workflows.errors import DynamicBlockError
from roboflow_workflows.execution_engine.v1.dynamic_blocks import modal_executor
from roboflow_workflows.execution_engine.v1.dynamic_blocks.entities import PythonCode

from tests.workflows.integration_tests.execution import modal_platform_retry

TRACEBACK_TAIL = "roboflow_workflows.errors.DynamicBlockError: "

# Errors that failed Modal tests in CI, as the subprocess's last stderr line
# prints them after the exception name.
MODAL_PLATFORM_ERRORS = [
    "Modal endpoint returned status 502: ",
    "Modal endpoint returned status 500: modal-http: internal error: status "
    "InternalFailure: Server has lost track of input",
    "Modal endpoint returned status 408: Missing request, possibly due to "
    "expiry or cancellation",
    "Failed to connect to Modal endpoint: HTTPSConnectionPool(host='roboflow--"
    "webexec-roboflow-staging-executor-execute-block.modal.run', port=443): "
    "Read timed out. (read timeout=30)",
]

OTHER_FAILURES = [
    # The block's own exception, which test_error_handling_in_modal expects.
    "roboflow_workflows.errors.DynamicBlockCodeError: Error in line 4, in run: "
    "ValueError: Value must be non-negative",
    # The webexec app itself crashed, not Modal's platform.
    TRACEBACK_TAIL + "Modal endpoint returned status 500: Internal Server Error",
    TRACEBACK_TAIL + "Modal endpoint returned status 401: modal-http: invalid "
    "credentials",
    "AssertionError",
]


class FakeRuns:
    """Stands in for subprocess.run: each run fails with the next queued stderr."""

    def __init__(self) -> None:
        self.stderr_queue: List[str] = []
        self.count = 0

    def run(self, args: List[str], **kwargs) -> subprocess.CompletedProcess:
        stderr = self.stderr_queue.pop(0) if self.stderr_queue else ""
        self.count += 1
        return subprocess.CompletedProcess(
            args=args, returncode=1 if stderr else 0, stdout="", stderr=stderr
        )


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> FakeRuns:
    fake_runs = FakeRuns()
    monkeypatch.setattr(
        modal_platform_retry, "subprocess", SimpleNamespace(run=fake_runs.run)
    )
    monkeypatch.setattr(modal_platform_retry, "time", mock.MagicMock())
    monkeypatch.setattr(modal_platform_retry, "_modal_is_down", False)
    return fake_runs


@pytest.mark.parametrize("error", MODAL_PLATFORM_ERRORS)
def test_run_failed_by_a_modal_platform_error_is_run_again(
    runs: FakeRuns, error: str
) -> None:
    runs.stderr_queue.append("Traceback ...\n" + TRACEBACK_TAIL + error + "\n")

    with pytest.warns(UserWarning, match="attempt 1 of 3"):
        result = modal_platform_retry.run_with_modal_platform_retries(["python"])

    assert result.returncode == 0
    assert runs.count == 2
    modal_platform_retry.time.sleep.assert_called_once_with(2.0)


@pytest.mark.parametrize("stderr", OTHER_FAILURES)
def test_other_failures_are_not_run_again(runs: FakeRuns, stderr: str) -> None:
    runs.stderr_queue.append("Traceback ...\n" + stderr + "\n")

    result = modal_platform_retry.run_with_modal_platform_retries(["python"])

    assert result.returncode == 1
    assert runs.count == 1
    modal_platform_retry.time.sleep.assert_not_called()


def test_runs_stop_retrying_after_one_fails_every_attempt(runs: FakeRuns) -> None:
    runs.stderr_queue.extend([TRACEBACK_TAIL + MODAL_PLATFORM_ERRORS[0]] * 4)

    with pytest.warns(UserWarning):
        first = modal_platform_retry.run_with_modal_platform_retries(["python"])
    second = modal_platform_retry.run_with_modal_platform_retries(["python"])

    assert first.returncode == 1
    assert second.returncode == 1
    assert runs.count == 4
    assert modal_platform_retry.time.sleep.call_args_list == [
        mock.call(2.0),
        mock.call(4.0),
    ]


def _error_raised_by_modal_executor(response=None, exception=None) -> str:
    session = mock.MagicMock()
    session.post.return_value = response
    session.post.side_effect = exception
    executor = modal_executor.ModalExecutor(workspace_id="my-workspace")
    executor._session = session
    python_code = PythonCode(
        type="PythonCode", run_function_code="def run(self):\n    return {}\n"
    )

    with mock.patch.object(modal_executor, "MODAL_AVAILABLE", True):
        with pytest.raises(DynamicBlockError) as error:
            executor.execute_remote(
                block_type_name="Block", python_code=python_code, inputs={}
            )

    stderr_tail = "".join(traceback.format_exception_only(error.type, error.value))
    return stderr_tail


@pytest.mark.parametrize(
    "status_code, body, is_platform_error",
    [
        (408, "Missing request, possibly due to expiry or cancellation", True),
        (500, "modal-http: internal error: status InternalFailure", True),
        (502, "", True),
        (503, "", True),
        (504, "", True),
        (500, "Internal Server Error", False),
        (401, "modal-http: invalid credentials", False),
    ],
)
def test_platform_errors_match_what_modal_executor_raises(
    status_code: int, body: str, is_platform_error: bool
) -> None:
    response = mock.MagicMock(status_code=status_code, text=body)

    stderr_tail = _error_raised_by_modal_executor(response=response)

    found = modal_platform_retry.find_modal_platform_error(stderr_tail)
    assert (found is not None) is is_platform_error


def test_connection_errors_match_what_modal_executor_raises() -> None:
    stderr_tail = _error_raised_by_modal_executor(
        exception=requests.exceptions.ReadTimeout("Read timed out.")
    )

    found = modal_platform_retry.find_modal_platform_error(stderr_tail)
    assert found == "Failed to connect to Modal endpoint: Read timed out."
