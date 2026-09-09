"""Every composition root must hand the platform bindings to ExecutionEngine.init.

A root that forgets one leaves blocks on the standalone default, which refuses
to talk to the Roboflow API - and there is no failing test at the block level
for that, because the default only raises when the step runs. So the values are
observed where they actually arrive.
"""

import ast
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from inference.core.cache import cache as server_cache
from inference.core.interfaces.roboflow_platform_client import (
    SERVER_PLATFORM_CLIENT,
    SERVER_WORKSPACE_RESOLVER,
    default_inner_workflow_spec_resolver,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]

REQUIRED = {
    "workflows_core.cache": server_cache,
    "workflows_core.platform_client": SERVER_PLATFORM_CLIENT,
    "workflows_core.workspace_resolver": SERVER_WORKSPACE_RESOLVER,
    "workflows_core.inner_workflow_spec_resolver": default_inner_workflow_spec_resolver,
}

# One entry per file: the number of ExecutionEngine.init call sites the runtime
# cases below cover. A new root shows up here as a mismatch.
ROOT_SITES = {
    "inference/core/interfaces/http/http_api.py": 2,
    "inference/core/interfaces/stream/inference_pipeline.py": 1,
    "inference_cli/lib/workflows/local_image_adapter.py": 1,
}

TRIVIAL_WORKFLOW = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}


class _Captured(Exception):
    def __init__(self, init_parameters):
        super().__init__("captured")
        self.init_parameters = init_parameters


def _capturing_init(**kwargs):
    raise _Captured(kwargs.get("init_parameters"))


def _assert_bindings(init_parameters: dict, site: str) -> None:
    assert init_parameters is not None, f"{site} passed no init_parameters"
    for key, expected in REQUIRED.items():
        assert key in init_parameters, f"{site} did not pass {key}"
        assert (
            init_parameters[key] is expected
        ), f"{site} passed a different object for {key}"


def test_ast_site_count_matches_the_runtime_cases() -> None:
    for relative_path, expected_sites in ROOT_SITES.items():
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        sites = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "init"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ExecutionEngine"
        ]
        assert len(sites) == expected_sites, (
            f"{relative_path}: {len(sites)} ExecutionEngine.init call sites, "
            f"expected {expected_sites} - add a runtime case for the new root"
        )


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


def _build_test_client(monkeypatch) -> TestClient:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    return TestClient(http_api.HttpInterface(model_manager=model_manager).app)


def _http_case(monkeypatch, path, json_body, site):
    import inference.core.interfaces.http.http_api as http_api

    captured = {}

    class _Engine:
        @staticmethod
        def init(**kwargs):
            captured["init_parameters"] = kwargs.get("init_parameters")
            raise _Captured(kwargs.get("init_parameters"))

    monkeypatch.setattr(http_api, "ExecutionEngine", _Engine)
    _build_test_client(monkeypatch).post(path, json=json_body)
    _assert_bindings(captured.get("init_parameters"), site)


def test_http_workflow_run_root_passes_the_bindings(monkeypatch) -> None:
    _http_case(
        monkeypatch,
        "/workflows/run",
        {"specification": TRIVIAL_WORKFLOW, "inputs": {}},
        "http_api /workflows/run",
    )


def test_http_workflow_validate_root_passes_the_bindings(monkeypatch) -> None:
    _http_case(
        monkeypatch,
        "/workflows/validate",
        TRIVIAL_WORKFLOW,
        "http_api /workflows/validate",
    )


def test_cli_root_passes_the_bindings(monkeypatch) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import inference_cli.lib.workflows.local_image_adapter as adapter

    monkeypatch.setattr(adapter.ExecutionEngine, "init", staticmethod(_capturing_init))
    with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(_Captured) as error:
        adapter._run_workflow_for_single_image_with_inference(
            model_manager=MagicMock(),
            image_path="unused.jpg",
            workflow_specification=TRIVIAL_WORKFLOW,
            workflow_id=None,
            image_input_name="image",
            workflow_parameters=None,
            api_key="k",
            thread_pool_executor=pool,
            max_concurrent_workflows_steps=1,
        )
    _assert_bindings(error.value.init_parameters, "inference_cli local_image_adapter")


def test_pipeline_root_passes_the_bindings(monkeypatch) -> None:
    import inference.core.workflows.execution_engine.core as engine_module
    from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

    monkeypatch.setattr(
        engine_module.ExecutionEngine, "init", staticmethod(_capturing_init)
    )
    with pytest.raises(_Captured) as error:
        InferencePipeline.init_with_workflow(
            video_reference="unused.mp4",
            workflow_specification=TRIVIAL_WORKFLOW,
            api_key="k",
            model_manager=MagicMock(),
        )
    _assert_bindings(
        error.value.init_parameters, "InferencePipeline.init_with_workflow"
    )


def test_pipeline_does_not_overwrite_a_caller_supplied_resolver(monkeypatch) -> None:
    import inference.core.workflows.execution_engine.core as engine_module
    from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

    def caller_resolver(*args, **kwargs):
        return {}

    monkeypatch.setattr(
        engine_module.ExecutionEngine, "init", staticmethod(_capturing_init)
    )
    with pytest.raises(_Captured) as error:
        InferencePipeline.init_with_workflow(
            video_reference="unused.mp4",
            workflow_specification=TRIVIAL_WORKFLOW,
            api_key="k",
            model_manager=MagicMock(),
            workflow_init_parameters={
                "workflows_core.inner_workflow_spec_resolver": caller_resolver
            },
        )
    parameters = error.value.init_parameters
    assert parameters["workflows_core.inner_workflow_spec_resolver"] is caller_resolver
    assert parameters["workflows_core.platform_client"] is SERVER_PLATFORM_CLIENT


def test_cli_does_not_overwrite_caller_supplied_engine_init_params(monkeypatch) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import inference_cli.lib.workflows.local_image_adapter as adapter

    def caller_resolver(*args, **kwargs):
        return {}

    monkeypatch.setattr(adapter.ExecutionEngine, "init", staticmethod(_capturing_init))
    with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(_Captured) as error:
        adapter._run_workflow_for_single_image_with_inference(
            model_manager=MagicMock(),
            image_path="unused.jpg",
            workflow_specification=TRIVIAL_WORKFLOW,
            workflow_id=None,
            image_input_name="image",
            workflow_parameters=None,
            api_key="k",
            thread_pool_executor=pool,
            max_concurrent_workflows_steps=1,
            workflows_execution_engine_init_params={
                "workflows_core.inner_workflow_spec_resolver": caller_resolver
            },
        )
    assert (
        error.value.init_parameters["workflows_core.inner_workflow_spec_resolver"]
        is caller_resolver
    )
