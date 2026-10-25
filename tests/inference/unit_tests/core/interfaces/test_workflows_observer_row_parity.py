"""The adapter records exactly what the decorator used to record.

Before this phase, `run_workflow`, the dynamic-block entrypoint and the SAM
video `run` were decorated directly. Now the server wraps them from outside.
The claim that "the extraction logic is called, not duplicated" is only worth
anything if the rows are identical, so each test below defines the *legacy*
decorated entry point - same signature, same category - runs both under a
frozen clock, and compares the kwargs the collector was handed.

`GCP_SERVERLESS` is pinned False so `_apply_duration_floor` does not lift a
frozen 0.25s to the serverless minimum in one arm and not the other.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from inference.core.interfaces.workflows_execution_observer import (
    UsageTrackingExecutionObserver,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks import block_duration
from inference.usage_tracking import collector as collector_module
from inference.usage_tracking.collector import usage_collector

# Two readings per decorated call: t1 before, t2 after.
FROZEN_CLOCK = [100.0, 100.25]


@pytest.fixture(autouse=True)
def _deterministic_collector():
    block_duration.clear_block_duration()
    with mock.patch.object(collector_module, "GCP_SERVERLESS", False):
        yield
    block_duration.clear_block_duration()


def _record_once(call) -> dict:
    """Run `call` with a frozen clock and return the kwargs the collector got."""
    with mock.patch.object(
        collector_module.time, "time", side_effect=list(FROZEN_CLOCK)
    ), mock.patch.object(usage_collector, "record_usage") as record_usage:
        call()
    return record_usage.call_args.kwargs


def _compiled_workflow() -> SimpleNamespace:
    return SimpleNamespace(
        init_parameters={"workflows_core.api_key": "parity-key"},
        workflow_json={"steps": [{"type": "SomeBlock", "name": "a_step"}]},
    )


# --- workflows category ----------------------------------------------------


@usage_collector("workflows")
def _legacy_run_workflow(
    workflow,
    runtime_parameters,
    max_concurrent_steps,
    kinds_serializers,
    serialize_results=False,
    profiler=None,
    executor=None,
    step_error_handler=None,
    defer_stream_pipeline_flush=False,
    resolve_output_futures=True,
):
    """`run_workflow`'s pre-phase signature, decorated the way it used to be."""
    return [{"out": 1}]


def test_workflow_row_is_identical_to_the_decorated_entry_point() -> None:
    # given
    workflow = _compiled_workflow()
    runtime_parameters = {"image": [SimpleNamespace(_image_reference="s3://x")]}
    observer = UsageTrackingExecutionObserver()

    # when
    legacy = _record_once(
        lambda: _legacy_run_workflow(
            workflow=workflow,
            runtime_parameters=runtime_parameters,
            max_concurrent_steps=1,
            kinds_serializers=None,
            usage_fps=12.5,
            usage_workflow_id="wf-internal-id",
            usage_workflow_preview=True,
        )
    )
    adapted = _record_once(
        lambda: observer.observe_workflow_run(
            workflow=workflow,
            runtime_parameters=runtime_parameters,
            workflow_id="wf-internal-id",
            fps=12.5,
            is_preview=True,
            run=lambda: [{"out": 1}],
        )
    )

    # then
    assert adapted == legacy


# --- workflow_block category ----------------------------------------------


@usage_collector("workflow_block")
def _legacy_metered_run(self, block_args, block_kwargs):
    """`block_scaffolding._metered_run`'s pre-phase shape."""
    return self._run_dynamic_block(*block_args, **block_kwargs)


class _FakeBlock:
    _usage_resource_id = "custom_python/abc123"
    _api_key = "parity-key"
    _usage_block_kind = "custom_python"
    _usage_block_type = "MeteredBlock"
    _workflow_step_type = "MeteredBlock"
    _workflow_step_name = "my_step"

    def _run_dynamic_block(self, *args, **kwargs):
        return {"result": 8}


def test_block_row_is_identical_to_the_decorated_entry_point() -> None:
    # given - neither arm publishes a measured duration, so both fall back to
    # the decorator's (frozen) wall clock
    block = _FakeBlock()
    observer = UsageTrackingExecutionObserver()

    # when
    legacy = _record_once(
        lambda: _legacy_metered_run(block, block_args=(), block_kwargs={"a": 1})
    )
    adapted = _record_once(
        lambda: observer.observe_block_run(
            block=block,
            block_args=(),
            block_kwargs={"a": 1},
            run=lambda: block._run_dynamic_block(a=1),
        )
    )

    # then
    assert adapted == legacy
    assert adapted["execution_duration"] == pytest.approx(0.25)


def test_block_row_is_identical_when_a_duration_was_measured() -> None:
    # given - the engine's channel in the adapted arm, the host's channel in
    # the legacy arm: the relay must make them indistinguishable
    from inference.usage_tracking import block_execution as server_channel

    block = _FakeBlock()
    observer = UsageTrackingExecutionObserver()

    def legacy_call():
        server_channel.record_measured_block_execution(
            duration=0.5, source=server_channel.BLOCK_DURATION_SOURCE_LOCAL_RUNTIME
        )
        return _legacy_metered_run(block, block_args=(), block_kwargs={})

    def adapted_call():
        def run():
            block_duration.record_block_duration(
                duration=0.5,
                source=block_duration.BLOCK_DURATION_SOURCE_LOCAL_RUNTIME,
            )
            return {"result": 8}

        return observer.observe_block_run(
            block=block, block_args=(), block_kwargs={}, run=run
        )

    # when
    legacy = _record_once(legacy_call)
    adapted = _record_once(adapted_call)

    # then
    assert adapted == legacy
    assert adapted["resource_details"]["duration_source"] == "local_runtime"
    assert adapted["execution_duration"] == pytest.approx(0.5)


# --- model category --------------------------------------------------------


class _FakeSamBlock:
    _api_key = "parity-key"

    @usage_collector("model")
    def legacy_run(
        self, images, boxes, model_id, prompt_mode, prompt_interval, threshold
    ):
        """SAM2's pre-phase decorated `run`."""
        return [{"masks": None}]

    def body(self, images, model_id):
        return [{"masks": None}]


def test_model_row_is_identical_to_the_decorated_entry_point() -> None:
    # given
    block = _FakeSamBlock()
    observer = UsageTrackingExecutionObserver()
    images = [object(), object()]

    # when
    legacy = _record_once(
        lambda: block.legacy_run(
            images=images,
            boxes=None,
            model_id="sam2video/small",
            prompt_mode="first_frame",
            prompt_interval=30,
            threshold=0.0,
        )
    )
    adapted = _record_once(
        lambda: observer.observe_model_run(
            block=block,
            model_id="sam2video/small",
            images=images,
            run=lambda: block.body(images, "sam2video/small"),
        )
    )

    # then
    assert adapted == legacy
    assert adapted["frames"] == 2
