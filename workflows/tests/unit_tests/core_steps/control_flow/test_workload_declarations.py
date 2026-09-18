"""Workload declarations of the flow-control blocks.

Two things matter here: a block that evaluates a UQL condition declares the
evaluation as work of its own, and the inner-workflow dispatcher declares its
child as OPAQUE instead of pretending to know what the child does.
"""

from roboflow_workflows.core_steps.flow_control.delta_filter.v1 import (
    DeltaFilterManifest,
)
from roboflow_workflows.core_steps.flow_control.inner_workflow.v1 import (
    BlockManifest as InnerWorkflowManifest,
)
from roboflow_workflows.core_steps.flow_control.rate_limiter.v1 import (
    RateLimiterManifest,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import COOLDOWN_HTTP_SOFT_PORTABLE_RESTRICTION


def test_rate_limiter_declares_flow_control_and_the_cooldown_caveat() -> None:
    manifest = RateLimiterManifest(
        type="roboflow_core/rate_limiter@v1",
        name="limiter",
        depends_on="$steps.model.predictions",
        next_steps=["$steps.sink"],
    )
    assert manifest.discover_work_operations() == [WorkOperation.FLOW_CONTROL]
    assert manifest.discover_portable_restrictions() == [
        COOLDOWN_HTTP_SOFT_PORTABLE_RESTRICTION
    ]


def test_delta_filter_declares_the_state_it_keeps_between_frames() -> None:
    manifest = DeltaFilterManifest(
        type="roboflow_core/delta_filter@v1",
        name="delta",
        image="$inputs.image",
        value="$steps.counter.count_in",
        next_steps=["$steps.sink"],
    )
    assert manifest.discover_work_operations() == [
        WorkOperation.FLOW_CONTROL,
        WorkOperation.TEMPORAL_BUFFERING,
    ]
    assert [
        restriction.code for restriction in manifest.discover_portable_restrictions()
    ] == [
        "stateful_video_state_resets_on_stateless_http",
        "temporal_block_no_benefit_on_still_image",
    ]


def test_inner_workflow_declares_its_child_as_opaque() -> None:
    manifest = InnerWorkflowManifest(
        type="roboflow_core/inner_workflow@v1",
        name="child",
        parameter_bindings={"image": "$inputs.image"},
        workflow_workspace_id="workspace",
        workflow_id="child-workflow",
    )
    operations = manifest.discover_work_operations()
    assert isinstance(operations, Discovery)
    assert operations.complete is False
    assert operations.items == [WorkOperation.EXTERNAL_REQUEST]
    assert operations.unknown_reasons == ["remote_dispatch_child_opaque:$steps.child"]
    restrictions = manifest.discover_portable_restrictions()
    assert isinstance(restrictions, Discovery)
    assert restrictions.complete is False
    assert restrictions.items == []
    assert restrictions.unknown_reasons == ["remote_dispatch_child_opaque:$steps.child"]


def test_inner_workflow_reason_carries_the_step_it_belongs_to() -> None:
    """The reason is a stable code plus step context, never a traceback."""
    manifest = InnerWorkflowManifest(
        type="roboflow_core/inner_workflow@v1",
        name="other_step",
        parameter_bindings={},
        workflow_workspace_id="workspace",
        workflow_id="child-workflow",
    )
    assert manifest.discover_work_operations().unknown_reasons == [
        "remote_dispatch_child_opaque:$steps.other_step"
    ]
