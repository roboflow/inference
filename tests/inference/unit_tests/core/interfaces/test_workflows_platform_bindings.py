"""A caller-supplied resolver must survive to ExecutionEngine.init.

`install_workflows_platform_bindings` uses `setdefault`, so the roots that
forward a caller's dictionary must not overwrite what the caller put there -
while still installing the rest of the server's platform objects.

The default bindings each root installs are asserted against a real engine in
`test_image_codec_binding.py` (the four server/CLI roots) and
`test_direct_caller_bindings.py` (the two scripts); the inventory of roots is
`test_workflows_composition_roots.py`.
"""

from unittest.mock import MagicMock

import pytest

from inference.core.interfaces.roboflow_platform_client import SERVER_PLATFORM_CLIENT

TRIVIAL_WORKFLOW = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}


class _Captured(Exception):
    def __init__(self, init_parameters):
        super().__init__("captured")
        self.init_parameters = init_parameters


def _capturing_init(**kwargs):
    raise _Captured(kwargs.get("init_parameters"))


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
