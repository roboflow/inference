"""Unit tests for ModalExecutor timeout handling and exception mapping."""

from unittest import mock

import pytest
import requests

from inference.core.workflows.errors import (
    DynamicBlockCodeError,
    DynamicBlockTimeoutError,
    WorkflowCompilerError,
    WorkflowExecutionEngineError,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.constants import (
    MODAL_TIMEOUT_ERROR_TYPE,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    modal_executor as modal_executor_module,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    DEFAULT_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS,
    ModalExecutor,
)


@pytest.fixture
def trivial_python_code() -> PythonCode:
    return PythonCode(
        type="PythonCode",
        imports=[],
        run_function_code="def run(self):\n    return {'ok': True}\n",
        run_function_name="run",
        init_function_code=None,
        init_function_name="init",
    )


# ---------------------------------------------------------------------------
# Timeout precedence (Task 8.2)
# ---------------------------------------------------------------------------


def test_timeout_resolution_default_when_arg_and_env_unset(monkeypatch) -> None:
    monkeypatch.setattr(
        modal_executor_module, "CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS", None, raising=True
    )
    executor = ModalExecutor()
    assert executor._timeout_seconds == DEFAULT_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS


def test_timeout_resolution_env_wins_over_default(monkeypatch) -> None:
    monkeypatch.setattr(
        modal_executor_module, "CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS", 75, raising=True
    )
    executor = ModalExecutor()
    assert executor._timeout_seconds == 75


def test_timeout_resolution_constructor_arg_wins_over_env(monkeypatch) -> None:
    monkeypatch.setattr(
        modal_executor_module, "CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS", 75, raising=True
    )
    executor = ModalExecutor(custom_python_block_timeout_seconds=15)
    assert executor._timeout_seconds == 15


def test_timeout_resolution_out_of_range_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setattr(
        modal_executor_module, "CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS", None, raising=True
    )
    executor = ModalExecutor(custom_python_block_timeout_seconds=500)
    assert executor._timeout_seconds == DEFAULT_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS


def test_timeout_resolution_non_int_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setattr(
        modal_executor_module, "CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS", None, raising=True
    )
    executor = ModalExecutor(custom_python_block_timeout_seconds=True)  # bool is invalid
    assert executor._timeout_seconds == DEFAULT_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Response/exception mapping (Task 8.3)
# ---------------------------------------------------------------------------


def _patch_modal_available(monkeypatch) -> None:
    monkeypatch.setattr(modal_executor_module, "MODAL_AVAILABLE", True, raising=True)
    monkeypatch.setattr(
        modal_executor_module, "MODAL_TOKEN_ID", "test-token-id", raising=True
    )
    monkeypatch.setattr(
        modal_executor_module, "MODAL_TOKEN_SECRET", "test-token-secret", raising=True
    )


def _build_response(payload: dict, status_code: int = 200) -> mock.MagicMock:
    resp = mock.MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = ""
    return resp


def test_server_timeout_response_raises_typed_exception(monkeypatch, trivial_python_code) -> None:
    _patch_modal_available(monkeypatch)
    # The check is `from inference.core.env import MODAL_ALLOW_ANONYMOUS_EXECUTION`
    # INSIDE execute_remote, so we must patch the source module, not the importer.
    import inference.core.env as env_module

    monkeypatch.setattr(
        env_module, "MODAL_ALLOW_ANONYMOUS_EXECUTION", True, raising=False
    )

    payload = {
        "success": False,
        "error": "Custom Python Block exceeded the configured timeout of 5s",
        "error_type": MODAL_TIMEOUT_ERROR_TYPE,
        "stdout": "about to hang\n",
        "stderr": None,
    }
    with mock.patch.object(
        modal_executor_module.requests, "post", return_value=_build_response(payload)
    ):
        executor = ModalExecutor(workspace_id="anonymous", custom_python_block_timeout_seconds=5)
        with pytest.raises(DynamicBlockTimeoutError) as exc:
            executor.execute_remote(
                block_type_name="test-block",
                python_code=trivial_python_code,
                inputs={},
                workspace_id="anonymous",
            )

    assert "5s" in exc.value.public_message
    assert exc.value.stdout == "about to hang\n"
    assert isinstance(exc.value, DynamicBlockCodeError)


def test_client_read_timeout_raises_typed_exception(monkeypatch, trivial_python_code) -> None:
    _patch_modal_available(monkeypatch)
    # The check is `from inference.core.env import MODAL_ALLOW_ANONYMOUS_EXECUTION`
    # INSIDE execute_remote, so we must patch the source module, not the importer.
    import inference.core.env as env_module

    monkeypatch.setattr(
        env_module, "MODAL_ALLOW_ANONYMOUS_EXECUTION", True, raising=False
    )

    with mock.patch.object(
        modal_executor_module.requests,
        "post",
        side_effect=requests.exceptions.ReadTimeout("read timed out"),
    ):
        executor = ModalExecutor(workspace_id="anonymous", custom_python_block_timeout_seconds=30)
        with pytest.raises(DynamicBlockTimeoutError) as exc:
            executor.execute_remote(
                block_type_name="test-block",
                python_code=trivial_python_code,
                inputs={},
                workspace_id="anonymous",
            )
    assert "client read timeout" in exc.value.public_message


def test_non_timeout_http_failure_raises_execution_engine_error(monkeypatch, trivial_python_code) -> None:
    _patch_modal_available(monkeypatch)
    # The check is `from inference.core.env import MODAL_ALLOW_ANONYMOUS_EXECUTION`
    # INSIDE execute_remote, so we must patch the source module, not the importer.
    import inference.core.env as env_module

    monkeypatch.setattr(
        env_module, "MODAL_ALLOW_ANONYMOUS_EXECUTION", True, raising=False
    )

    with mock.patch.object(
        modal_executor_module.requests,
        "post",
        side_effect=requests.exceptions.ConnectionError("connection refused"),
    ):
        executor = ModalExecutor(workspace_id="anonymous")
        with pytest.raises(WorkflowExecutionEngineError) as exc:
            executor.execute_remote(
                block_type_name="test-block",
                python_code=trivial_python_code,
                inputs={},
                workspace_id="anonymous",
            )

    # Critical: must NOT be a compiler-side error (the pre-existing
    # categorisation bug we're fixing).
    assert not isinstance(exc.value, WorkflowCompilerError)
    assert not isinstance(exc.value, DynamicBlockTimeoutError)


def test_payload_includes_timeout_and_client_uses_n_plus_headroom(monkeypatch, trivial_python_code) -> None:
    _patch_modal_available(monkeypatch)
    # The check is `from inference.core.env import MODAL_ALLOW_ANONYMOUS_EXECUTION`
    # INSIDE execute_remote, so we must patch the source module, not the importer.
    import inference.core.env as env_module

    monkeypatch.setattr(
        env_module, "MODAL_ALLOW_ANONYMOUS_EXECUTION", True, raising=False
    )

    payload = {"success": True, "result": "{}", "stdout": None, "stderr": None}
    with mock.patch.object(
        modal_executor_module.requests, "post", return_value=_build_response(payload)
    ) as mock_post:
        executor = ModalExecutor(workspace_id="anonymous", custom_python_block_timeout_seconds=45)
        executor.execute_remote(
            block_type_name="test-block",
            python_code=trivial_python_code,
            inputs={},
            workspace_id="anonymous",
        )

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["timeout_seconds"] == 45
    assert kwargs["timeout"] == 55  # 45 + 10s headroom


def test_validation_uses_fixed_timeout_not_env_var(monkeypatch, trivial_python_code) -> None:
    """validate_code_in_modal must pass `timeout_seconds=VALIDATION_TIMEOUT_SECONDS`
    regardless of the env-var-configured per-frame budget."""
    from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
        VALIDATION_TIMEOUT_SECONDS,
        validate_code_in_modal,
    )

    _patch_modal_available(monkeypatch)
    monkeypatch.setattr(
        modal_executor_module, "CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS", 120, raising=True
    )
    # The check is `from inference.core.env import MODAL_ALLOW_ANONYMOUS_EXECUTION`
    # INSIDE execute_remote, so we must patch the source module, not the importer.
    import inference.core.env as env_module

    monkeypatch.setattr(
        env_module, "MODAL_ALLOW_ANONYMOUS_EXECUTION", True, raising=False
    )

    payload = {"success": True, "result": '{"valid": true}', "stdout": None, "stderr": None}
    with mock.patch.object(
        modal_executor_module.requests, "post", return_value=_build_response(payload)
    ) as mock_post:
        validate_code_in_modal(trivial_python_code, workspace_id="anonymous")

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["timeout_seconds"] == VALIDATION_TIMEOUT_SECONDS
    # Client timeout should match the fixed validation budget, not the env var (120s).
    assert kwargs["timeout"] == VALIDATION_TIMEOUT_SECONDS + 10
