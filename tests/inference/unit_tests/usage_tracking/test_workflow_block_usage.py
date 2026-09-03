"""Usage rows emitted for the ``workflow_block`` category."""

import json

import pytest

from inference.core.workflows.execution_engine.entities.base import Batch
from inference.usage_tracking.block_execution import (
    BLOCK_DURATION_SOURCE_DECORATOR_WALL_CLOCK,
    BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
    BLOCK_EXECUTION_MODE_REMOTE,
    clear_measured_block_execution,
    record_measured_block_execution,
)
from inference.usage_tracking.decorator_helpers import (
    get_workflow_block_api_key_from_kwargs,
    get_workflow_block_frames_from_kwargs,
    get_workflow_block_resource_details_from_kwargs,
    get_workflow_block_resource_id_from_kwargs,
    usage_billing_suppressed,
    usage_source_tags,
    usage_workflow_is_preview,
)


class _FakeDynamicBlock:
    """Stands in for an assembled custom Python block instance."""

    _api_key = "block-api-key"
    _usage_resource_id = "custom_python/0123456789abcdef"
    _usage_block_kind = "custom_python"
    _usage_block_type = "MyCustomBlock"
    _workflow_step_name = "my_step"
    _workflow_step_type = "MyCustomBlock"

    def run(self, *args, **kwargs):
        return None


def _block_run(self, *args, **kwargs):
    """Mirrors the signature of the decorated dynamic-block entrypoint."""
    return None


@pytest.fixture(autouse=True)
def cleared_block_execution():
    clear_measured_block_execution()
    yield
    clear_measured_block_execution()


def test_resource_id_comes_from_the_identity_the_block_published():
    resource_id = get_workflow_block_resource_id_from_kwargs(
        {"self": _FakeDynamicBlock()}
    )

    assert resource_id == "custom_python/0123456789abcdef"


def test_resource_id_is_absent_for_a_block_that_published_no_identity():
    assert get_workflow_block_resource_id_from_kwargs({"self": object()}) is None
    assert get_workflow_block_resource_id_from_kwargs({}) is None


def test_api_key_comes_from_the_block_init_parameter():
    api_key = get_workflow_block_api_key_from_kwargs({"self": _FakeDynamicBlock()})

    assert api_key == "block-api-key"


def test_resource_details_describe_the_block():
    resource_details = get_workflow_block_resource_details_from_kwargs(
        {"self": _FakeDynamicBlock()}
    )

    assert resource_details == {
        "block_kind": "custom_python",
        "block_type": "MyCustomBlock",
        "step_name": "my_step",
    }


def test_frames_default_to_one_for_a_scalar_invocation():
    frames = get_workflow_block_frames_from_kwargs({"block_kwargs": {"a": 1, "b": 2}})

    assert frames == 1


def test_frames_count_batch_elements_for_a_batch_oriented_block():
    batch = Batch.init(content=[1, 2, 3], indices=[(0,), (1,), (2,)])

    frames = get_workflow_block_frames_from_kwargs(
        {"block_kwargs": {"images": batch, "threshold": 0.5}}
    )

    assert frames == 3


def test_frames_count_leaf_elements_for_a_nested_batch():
    inner = [
        Batch.init(content=[1, 2, 3, 4], indices=[(0, i) for i in range(4)]),
        Batch.init(content=[5, 6], indices=[(1, i) for i in range(2)]),
    ]
    nested = Batch.init(content=inner, indices=[(0,), (1,)])

    frames = get_workflow_block_frames_from_kwargs({"block_kwargs": {"crops": nested}})

    assert frames == 6, "Outer length (2) under-reports the 6 items handed to run()"


def _extract_workflow_block_params(collector, execution_duration=1.0, **block_kwargs):
    return collector._extract_usage_params_from_func_kwargs(
        usage_fps=0,
        usage_api_key="",
        usage_workflow_id="",
        usage_workflow_preview=False,
        usage_inference_test_run=False,
        usage_billable=True,
        execution_duration=execution_duration,
        func=_block_run,
        category="workflow_block",
        error_details=None,
        args=(_FakeDynamicBlock(),),
        kwargs=block_kwargs,
    )


def test_extract_usage_params_builds_a_workflow_block_row(
    usage_collector_with_mocked_threads,
):
    usage_params = _extract_workflow_block_params(
        usage_collector_with_mocked_threads,
        a=1,
        b=2,
    )

    assert usage_params["category"] == "workflow_block"
    assert usage_params["resource_id"] == "custom_python/0123456789abcdef"
    assert usage_params["api_key"] == "block-api-key"
    assert usage_params["frames"] == 1
    assert usage_params["resource_details"]["block_kind"] == "custom_python"
    assert usage_params["resource_details"]["block_type"] == "MyCustomBlock"
    assert usage_params["resource_details"]["step_name"] == "my_step"
    assert usage_params["resource_details"]["billable"] is True
    assert usage_params["resource_details"]["is_preview"] is False


def test_extract_usage_params_prefers_the_duration_the_block_measured(
    usage_collector_with_mocked_threads,
):
    # given - a remote executor reported what the sandbox actually spent, which
    # is less than the decorator's wall clock around the round trip
    record_measured_block_execution(
        duration=0.25,
        source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
    )

    # when
    usage_params = _extract_workflow_block_params(
        usage_collector_with_mocked_threads,
        execution_duration=4.0,
    )

    # then
    assert usage_params["execution_duration"] == 0.25
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_REMOTE_RUNTIME
    )
    assert (
        usage_params["resource_details"]["execution_mode"]
        == BLOCK_EXECUTION_MODE_REMOTE
    )


def test_extract_usage_params_falls_back_to_the_decorator_wall_clock(
    usage_collector_with_mocked_threads,
):
    usage_params = _extract_workflow_block_params(
        usage_collector_with_mocked_threads,
        execution_duration=4.0,
    )

    assert usage_params["execution_duration"] == 4.0
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_DECORATOR_WALL_CLOCK
    )
    assert "execution_mode" not in usage_params["resource_details"]


def test_measured_duration_is_not_reused_by_the_next_invocation(
    usage_collector_with_mocked_threads,
):
    # given - one invocation publishes a measurement and consumes it
    record_measured_block_execution(
        duration=0.25,
        source=BLOCK_DURATION_SOURCE_REMOTE_RUNTIME,
    )
    _extract_workflow_block_params(usage_collector_with_mocked_threads)

    # when - a later invocation publishes nothing
    usage_params = _extract_workflow_block_params(
        usage_collector_with_mocked_threads,
        execution_duration=4.0,
    )

    # then - it is billed for its own wall clock, not the earlier measurement
    assert usage_params["execution_duration"] == 4.0
    assert (
        usage_params["resource_details"]["duration_source"]
        == BLOCK_DURATION_SOURCE_DECORATOR_WALL_CLOCK
    )


def test_workflow_block_rows_aggregate_per_block_identity(
    usage_collector_with_mocked_threads,
):
    # given
    collector = usage_collector_with_mocked_threads

    # when - the same block runs twice
    for duration in (0.25, 0.75):
        collector.record_usage(
            source="",
            category="workflow_block",
            api_key="block-api-key",
            resource_id="custom_python/0123456789abcdef",
            resource_details={"billable": True, "block_kind": "custom_python"},
            execution_duration=duration,
        )

    # then - one row, durations summed
    api_key_usage = collector._usage["block-api-key"]
    usage_key = (
        "workflow_block:custom_python/0123456789abcdef:billable=true:outcome=success"
    )
    assert list(api_key_usage.keys()) == [usage_key]
    assert api_key_usage[usage_key]["execution_duration"] == pytest.approx(1.0)


def test_extract_usage_params_inherits_preview_from_request_context(
    usage_collector_with_mocked_threads,
):
    token = usage_workflow_is_preview.set(True)
    try:
        usage_params = _extract_workflow_block_params(
            usage_collector_with_mocked_threads
        )
    finally:
        usage_workflow_is_preview.reset(token)

    assert usage_params["resource_details"]["is_preview"] is True
    assert usage_params["resource_details"]["billable"] is True


def test_extract_usage_params_inherits_billing_suppression(
    usage_collector_with_mocked_threads,
):
    token = usage_billing_suppressed.set(True)
    try:
        usage_params = _extract_workflow_block_params(
            usage_collector_with_mocked_threads
        )
    finally:
        usage_billing_suppressed.reset(token)

    assert usage_params["resource_details"]["billable"] is False
    assert usage_params["resource_details"]["is_preview"] is False


def test_extract_usage_params_inherits_source_tag(
    usage_collector_with_mocked_threads,
):
    token = usage_source_tags.set({"source": "workflow-editor"})
    try:
        usage_params = _extract_workflow_block_params(
            usage_collector_with_mocked_threads
        )
    finally:
        usage_source_tags.reset(token)

    assert usage_params["resource_details"]["source"] == "workflow-editor"


def test_preview_decorator_kwarg_reaches_nested_workflow_block_row(
    usage_collector_with_mocked_threads,
):
    usage_collector = usage_collector_with_mocked_threads

    class Block(_FakeDynamicBlock):
        @usage_collector(category="workflow_block")
        def run(self, *args, **kwargs):
            return "ok"

    @usage_collector(category="workflows")
    def run_workflow(workflow, api_key="block-api-key"):
        return Block().run()

    run_workflow(None, usage_workflow_id="workflow-1", usage_workflow_preview=True)

    api_key_usage = usage_collector._usage["block-api-key"]
    block_rows = [
        row for key, row in api_key_usage.items() if key.startswith("workflow_block:")
    ]
    workflow_rows = [
        row for key, row in api_key_usage.items() if key.startswith("workflows:")
    ]

    assert len(block_rows) == 1
    assert json.loads(block_rows[0]["resource_details"])["is_preview"] is True
    assert json.loads(workflow_rows[0]["resource_details"])["is_preview"] is True


def test_authenticated_opt_out_reaches_nested_workflow_block_row(
    usage_collector_with_mocked_threads,
    configured_service_secret,
):
    usage_collector = usage_collector_with_mocked_threads

    class Block(_FakeDynamicBlock):
        @usage_collector(category="workflow_block")
        def run(self, *args, **kwargs):
            return "ok"

    @usage_collector(category="request")
    def handler(
        workflow_request,
        countinference=None,
        service_secret=None,
        api_key="block-api-key",
    ):
        return Block().run()

    handler(
        object(),
        countinference=False,
        service_secret=configured_service_secret,
    )

    api_key_usage = usage_collector._usage["block-api-key"]
    block_rows = [
        row for key, row in api_key_usage.items() if key.startswith("workflow_block:")
    ]

    assert len(block_rows) == 1
    assert json.loads(block_rows[0]["resource_details"])["billable"] is False
