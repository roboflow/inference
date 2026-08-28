"""The sandbox reports how long user code ran, and the client bills that.

Two halves of one contract: ``modal_app`` measures the user function and puts
``execution_time_seconds`` on its response, and the executor republishes that
number for the usage collector instead of the client's wall clock.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

from inference.core.workflows.execution_engine.v1.dynamic_blocks import modal_executor
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    ModalExecutor,
)
from inference.usage_tracking.block_execution import (
    BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
    clear_measured_block_execution,
    consume_measured_block_execution,
)


class _FakeModalImage:
    @classmethod
    def debian_slim(cls, *args, **kwargs):
        return cls()

    @classmethod
    def from_registry(cls, *args, **kwargs):
        return cls()

    def apt_install(self, *args, **kwargs):
        return self

    def pip_install(self, *args, **kwargs):
        return self

    def entrypoint(self, *args, **kwargs):
        return self


class _FakeModalApp:
    def __init__(self, name: str):
        self.name = name

    def cls(self, *args, **kwargs):
        return lambda cls: cls


def _identity_decorator(*args, **kwargs):
    return lambda obj: obj


@pytest.fixture()
def modal_app_with_fake_modal(monkeypatch):
    fake_modal = ModuleType("modal")
    fake_modal.App = _FakeModalApp
    fake_modal.Image = _FakeModalImage
    fake_modal.parameter = lambda *args, **kwargs: None
    fake_modal.enter = _identity_decorator
    fake_modal.fastapi_endpoint = _identity_decorator
    fake_modal.asgi_app = _identity_decorator
    fake_modal.concurrent = _identity_decorator
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    modal_app_path = Path(__file__).resolve().parents[5] / "modal" / "modal_app.py"
    spec = importlib.util.spec_from_file_location(
        "modal_app_execution_timing_test", modal_app_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cleared_remote_duration():
    clear_measured_block_execution()
    yield
    clear_measured_block_execution()


def _run_user_code_ws(modal_app, code: str, run_function_name: str, inputs: dict):
    executor = modal_app.Executor.__new__(modal_app.Executor)
    executor._code_namespaces = {}
    executor._namespace_lock = None

    return modal_app.Executor._run_user_code_ws(
        executor,
        code,
        [],
        run_function_name,
        inputs,
    )


def test_sandbox_reports_how_long_the_user_function_ran(
    modal_app_with_fake_modal, monkeypatch
):
    # given
    code = "def run(a):\n    return {'value': a}\n"
    clock = iter([10.0, 10.25])
    monkeypatch.setattr(
        modal_app_with_fake_modal.time, "perf_counter", lambda: next(clock)
    )

    # when
    response = _run_user_code_ws(modal_app_with_fake_modal, code, "run", {"a": 1})

    # then
    assert response["success"] is True
    assert response["execution_time_seconds"] == pytest.approx(0.25)


def test_sandbox_reports_runtime_for_a_failing_user_function(modal_app_with_fake_modal):
    # given
    code = "def run(a):\n    raise ValueError('boom')\n"

    # when
    response = _run_user_code_ws(modal_app_with_fake_modal, code, "run", {"a": 1})

    # then
    assert response["success"] is False
    assert response["execution_time_seconds"] >= 0


def test_executor_publishes_the_runtime_the_sandbox_reported():
    # given
    executor = ModalExecutor(workspace_id="test-workspace")
    python_code = PythonCode(
        type="PythonCode",
        run_function_code="def run():\n    return {}\n",
        run_function_name="run",
        imports=[],
    )

    # when
    with mock.patch.object(modal_executor, "MODAL_AVAILABLE", True), mock.patch.object(
        ModalExecutor, "_get_endpoint_url", return_value="https://example.invalid"
    ), mock.patch.object(
        ModalExecutor,
        "_post_execute",
        return_value={
            "success": True,
            "result": "{}",
            "execution_time_seconds": 0.25,
        },
    ):
        executor.execute_remote(
            block_type_name="MeteredBlock",
            python_code=python_code,
            inputs={},
            workspace_id="test-workspace",
        )

    # then
    measured = consume_measured_block_execution()
    assert measured.duration == 0.25
    assert measured.source == BLOCK_DURATION_SOURCE_REMOTE_RUNTIME


def test_executor_publishes_nothing_when_the_sandbox_reports_no_runtime():
    # given - a sandbox deployment predating the reported runtime
    executor = ModalExecutor(workspace_id="test-workspace")
    python_code = PythonCode(
        type="PythonCode",
        run_function_code="def run():\n    return {}\n",
        run_function_name="run",
        imports=[],
    )

    # when
    with mock.patch.object(modal_executor, "MODAL_AVAILABLE", True), mock.patch.object(
        ModalExecutor, "_get_endpoint_url", return_value="https://example.invalid"
    ), mock.patch.object(
        ModalExecutor,
        "_post_execute",
        return_value={"success": True, "result": "{}"},
    ):
        executor.execute_remote(
            block_type_name="MeteredBlock",
            python_code=python_code,
            inputs={},
            workspace_id="test-workspace",
        )

    # then
    assert consume_measured_block_execution() is None
