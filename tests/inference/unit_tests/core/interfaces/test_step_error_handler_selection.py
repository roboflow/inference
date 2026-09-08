import ast
from pathlib import Path

import pytest

from inference.core.exceptions import FeatureDeprecatedError
from inference.core.interfaces.workflows_step_error_handlers import (
    extended_roboflow_errors_handler,
    legacy_step_error_handler,
    resolve_step_error_handler,
)
from inference.core.workflows.errors import (
    ClientCausedStepExecutionError,
    WorkflowEnvironmentConfigurationError,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.v1 import core as ee_core

# tests/inference/unit_tests/core/interfaces/<this file> -> five levels up is the repo root
REPO_ROOT = Path(__file__).resolve().parents[5]
COMPOSITION_ROOTS = [
    "inference/core/interfaces/http/http_api.py",
    "inference/core/interfaces/stream/inference_pipeline.py",
    "inference_cli/lib/workflows/local_image_adapter.py",
]


def test_server_default_is_the_extended_handler(monkeypatch) -> None:
    monkeypatch.delenv("DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER", raising=False)
    assert resolve_step_error_handler() is extended_roboflow_errors_handler


def test_explicit_legacy_keeps_the_server_legacy_behaviour(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER", "legacy")
    assert resolve_step_error_handler() is legacy_step_error_handler


def test_unknown_name_is_passed_through_and_the_engine_still_rejects_it(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER", "nope")
    assert resolve_step_error_handler() == "nope"
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        ExecutionEngine.init(
            workflow_definition={}, step_error_handler=resolve_step_error_handler()
        )


def test_direct_engine_default_maps_nothing_while_server_handler_maps_deprecation(
    monkeypatch,
) -> None:
    # The Roboflow mapping left workflows, so the two sides now differ and that
    # difference is authorised: a direct `ExecutionEngine.init(...)` caller that
    # does not pass `step_error_handler` gets the mapping-free workflows
    # default, while every server composition root passes the resolver below
    # and keeps the mapping. Pinned here so the divergence cannot drift.
    monkeypatch.delenv("DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER", raising=False)
    error = FeatureDeprecatedError(feature="example/feature@v1", reason="gone")

    direct_engine_handler = ee_core.REGISTERED_STEP_ERROR_HANDLERS[
        ee_core.DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER
    ]
    assert direct_engine_handler("step-id", error) is None

    with pytest.raises(ClientCausedStepExecutionError) as captured:
        resolve_step_error_handler()("step-id", error)
    assert captured.value.status_code == 410
    assert captured.value.inner_error is error


def test_every_server_composition_root_passes_a_handler() -> None:
    # The default argument is bound at definition time inside workflows, so
    # the server must pass its choice explicitly at every ExecutionEngine.init.
    for relative in COMPOSITION_ROOTS:
        tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "init"
            and getattr(node.func.value, "id", None) == "ExecutionEngine"
        ]
        assert calls, relative
        for call in calls:
            assert any(k.arg == "step_error_handler" for k in call.keywords), (
                relative,
                call.lineno,
            )
