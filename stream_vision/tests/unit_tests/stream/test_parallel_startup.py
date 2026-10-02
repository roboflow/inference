"""Exercise the standalone constructor and provider contract without a host."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Event

import pytest
from roboflow_workflows.execution_engine.core import ExecutionEngine
from streamvision.stream import pipeline as pipeline_module
from streamvision.stream.pipeline import InferencePipeline


class Source:
    def __init__(self):
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def terminate(self, **kwargs):
        self.stops += 1


class Provider(dict):
    def add_model(self, model_id, api_key, **kwargs):
        self[model_id] = (api_key, kwargs)


@pytest.mark.parametrize("model_limit", [0, 1])
def test_standalone_provider_and_other_bindings_survive_startup(
    monkeypatch, model_limit
):
    source, provider = Source(), Provider()
    marker = object()
    bindings = {
        "workflows_core.model_manager": provider,
        "plugin.binding": marker,
        "workflows_core.api_key": lambda: "inert",
    }
    original = dict(bindings)
    monkeypatch.setattr(pipeline_module, "prepare_video_sources", lambda **kw: [source])
    workflow = {
        "version": "1.0",
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "detection",
                "images": "$inputs.image",
                "model_id": "inert/1",
            }
        ],
        "outputs": [],
    }

    def forbidden_run(*args, **kwargs):
        raise AssertionError("startup must not execute a workflow")

    monkeypatch.setattr(ExecutionEngine, "run", forbidden_run)
    instance = InferencePipeline.init_with_workflow(
        "inert",
        workflow_specification=workflow,
        workflow_init_parameters=bindings,
        step_error_handler=None,
        parallel_startup=True,
        startup_model_limit=model_limit,
    )
    assert bindings == original
    assert bindings["plugin.binding"] is marker
    assert source.starts == 1
    assert ("inert/1" in provider) == bool(model_limit)
    if model_limit:
        assert provider["inert/1"] == ("inert", {})
    instance.terminate()
    instance.join()
    assert source.stops == 1
    assert ("inert/1" in provider) == bool(model_limit)  # caller-owned


def test_zero_model_budget_needs_no_provider_and_cancelled_result_cannot_start(
    monkeypatch,
):
    source, cancelled = Source(), Event()
    monkeypatch.setattr(pipeline_module, "prepare_video_sources", lambda **kw: [source])
    instance = InferencePipeline.init_with_workflow(
        "inert",
        workflow_specification={
            "version": "1.0",
            "inputs": [],
            "steps": [],
            "outputs": [],
        },
        workflow_init_parameters={},
        step_error_handler=None,
        parallel_startup=True,
        startup_model_limit=0,
        startup_cancel_event=cancelled,
    )
    cancelled.set()
    with pytest.raises(CancelledError):
        instance.start()
    assert source.starts == source.stops == 1


@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_invalid_budget_rejected_before_starting_sources(monkeypatch, invalid):
    def forbidden(**kwargs):
        raise AssertionError("source constructed before option validation")

    monkeypatch.setattr(pipeline_module, "prepare_video_sources", forbidden)
    with pytest.raises(ValueError, match="non-negative integer"):
        InferencePipeline.init_with_workflow(
            "inert",
            workflow_specification={},
            workflow_init_parameters={},
            step_error_handler=None,
            parallel_startup=True,
            startup_model_limit=invalid,
        )


def test_terminate_and_join_serialize_source_cleanup(monkeypatch):
    source = Source()
    entered, release = Event(), Event()
    original_terminate = source.terminate

    def terminate(**kwargs):
        entered.set()
        assert release.wait(5)
        original_terminate(**kwargs)

    source.terminate = terminate
    monkeypatch.setattr(pipeline_module, "prepare_video_sources", lambda **kw: [source])
    instance = InferencePipeline.init_with_workflow(
        "inert",
        workflow_specification={
            "version": "1.0",
            "inputs": [],
            "steps": [],
            "outputs": [],
        },
        workflow_init_parameters={},
        step_error_handler=None,
        parallel_startup=True,
        startup_model_limit=0,
    )
    with ThreadPoolExecutor(2) as executor:
        stopping = executor.submit(instance.terminate)
        try:
            assert entered.wait(5)
            joining = executor.submit(instance.join)
        finally:
            release.set()
        stopping.result(5)
        joining.result(5)
    assert source.stops == 1


def test_cancel_during_run_keeps_provider_available_to_handler_close(monkeypatch):
    source, provider, cancelled = Source(), Provider(), Event()
    provider["resident"] = object()
    monkeypatch.setattr(pipeline_module, "prepare_video_sources", lambda **kw: [source])
    monkeypatch.setattr(
        pipeline_module, "_rfdetr_stream_pipeline_enabled", lambda: True
    )
    instance = InferencePipeline.init_with_workflow(
        "inert",
        workflow_specification={
            "version": "1.0",
            "inputs": [],
            "steps": [],
            "outputs": [],
        },
        workflow_init_parameters={"workflows_core.model_manager": provider},
        step_error_handler=None,
        parallel_startup=True,
        startup_model_limit=0,
        startup_cancel_event=cancelled,
    )
    closed = Event()

    class Handler:
        def close(self):
            assert "resident" in instance._parallel_startup.gated_manager
            closed.set()

    def frames(**kwargs):
        cancelled.set()
        return iter([])

    instance._on_video_frame = Handler()
    monkeypatch.setattr(pipeline_module, "multiplex_videos", frames)
    instance.start(use_main_thread=False)
    instance.join()
    assert closed.is_set()
    assert source.stops == 1
