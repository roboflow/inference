"""Legacy vs actual state-loss restrictions.

The editor-facing ``get_restrictions()`` keeps its historic REMOTE
step-execution scope and its serialized shape. ``get_actual_restrictions()``
drops only that scope, because block state is lost whether the model runs
locally or remotely. Severity, runtimes and input modes are the same on both
sides.

Blocks are resolved through the block loader, so the numpy or the tensor
variant is exercised according to the active test mode.
"""

from typing import List, Optional, Type, get_args

import pytest
from roboflow_workflows.core_steps.models.workload_presets import (
    REQUIRES_GPU_FOR_LOCAL_EXECUTION,
    hosted_endpoint_disabled_by_flag,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Runtime,
    RuntimeInputMode,
    RuntimeRestriction,
    Severity,
    StepExecutionMode,
)
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    load_core_workflow_blocks,
)
from roboflow_workflows.prototypes.block import (
    COOLDOWN_HTTP_SOFT_RESTRICTION,
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    WorkflowBlockManifest,
)

from tests.unit_tests.workload_declaration_helpers import declared_restrictions

STATEFUL_VIDEO_CODE = "stateful_video_state_resets_on_stateless_http"
COOLDOWN_CODE = "cooldown_timer_resets_on_stateless_http"
S3_APPEND_CODE = "s3_append_buffer_resets_on_stateless_http"
HOSTED_RUNTIMES = {Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT}


def _manifest_class(block_type: str) -> Type[WorkflowBlockManifest]:
    for block in load_core_workflow_blocks():
        type_annotation = block.manifest_class.model_fields["type"].annotation
        if block_type in get_args(type_annotation):
            return block.manifest_class
    raise AssertionError(f"{block_type} is not a loaded core block")


def _instance(block_type: str, **fields: str) -> WorkflowBlockManifest:
    manifest_class = _manifest_class(block_type)
    values = {"type": block_type, "name": "step", **fields}
    for field_name, field in manifest_class.model_fields.items():
        if field_name not in values and field.is_required():
            values[field_name] = None

    return manifest_class.model_construct(**values)


def _state_caveat(restrictions: List[RuntimeRestriction]) -> RuntimeRestriction:
    # the still-image caveat some of these blocks also declare is not about state
    caveats = [
        restriction
        for restriction in restrictions
        if restriction.applies_to_input_modes != [RuntimeInputMode.IMAGE]
    ]
    assert len(caveats) == 1, caveats

    return caveats[0]


STATE_LOSS_CASES = [
    # block type, instance fields, actual code, input modes, shared legacy object
    (
        "roboflow_core/byte_tracker@v3",
        {},
        STATEFUL_VIDEO_CODE,
        [RuntimeInputMode.VIDEO],
        STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    ),
    (
        "roboflow_core/image_stack@v1",
        {},
        STATEFUL_VIDEO_CODE,
        [RuntimeInputMode.VIDEO],
        None,
    ),
    (
        "roboflow_core/heatmap_visualization@v1",
        {},
        STATEFUL_VIDEO_CODE,
        [RuntimeInputMode.VIDEO],
        None,
    ),
    (
        "roboflow_core/trace_visualization@v1",
        {},
        STATEFUL_VIDEO_CODE,
        [RuntimeInputMode.VIDEO],
        None,
    ),
    (
        "roboflow_core/webhook_sink@v1",
        {},
        COOLDOWN_CODE,
        None,
        COOLDOWN_HTTP_SOFT_RESTRICTION,
    ),
    (
        "roboflow_core/rate_limiter@v1",
        {},
        COOLDOWN_CODE,
        None,
        COOLDOWN_HTTP_SOFT_RESTRICTION,
    ),
    (
        "roboflow_core/s3_sink@v1",
        {"output_mode": "append_log"},
        S3_APPEND_CODE,
        None,
        None,
    ),
]


@pytest.mark.parametrize(
    "block_type, fields, code, input_modes, legacy_object",
    STATE_LOSS_CASES,
    ids=[case[0] for case in STATE_LOSS_CASES],
)
def test_actual_state_caveat_drops_only_the_legacy_remote_scope(
    block_type: str,
    fields: dict,
    code: str,
    input_modes: Optional[List[RuntimeInputMode]],
    legacy_object: Optional[RuntimeRestriction],
) -> None:
    legacy = _state_caveat(_manifest_class(block_type).get_restrictions())
    actual = _state_caveat(declared_restrictions(_instance(block_type, **fields)))

    # the editor view is untouched: same object where it is shared, same shape
    if legacy_object is not None:
        assert legacy is legacy_object
    expected_legacy_payload = {
        "severity": "soft",
        "note": legacy.note,
        "applies_to_runtimes": ["hosted_serverless", "dedicated_deployment"],
        "applies_to_step_execution_modes": ["remote"],
    }
    if input_modes is not None:
        expected_legacy_payload["applies_to_input_modes"] = ["video"]
    assert legacy.to_dict() == expected_legacy_payload
    assert "remote step execution" in legacy.note

    # the actual view: no step-execution-mode filter, every other condition kept
    assert actual.code == code
    assert actual.applies_to_step_execution_modes is None
    assert actual.note != legacy.note
    for caveat in (legacy, actual):
        assert caveat.severity is Severity.SOFT
        assert set(caveat.applies_to_runtimes) == HOSTED_RUNTIMES
        assert caveat.applies_to_input_modes == input_modes


def test_s3_separate_files_declares_no_actual_append_caveat() -> None:
    legacy = _manifest_class("roboflow_core/s3_sink@v1").get_restrictions()
    actual = declared_restrictions(
        _instance("roboflow_core/s3_sink@v1", output_mode="separate_files")
    )

    assert len(legacy) == 1
    assert actual == []


def test_genuine_model_placement_restrictions_keep_their_mode_filters() -> None:
    endpoint_disabled = hosted_endpoint_disabled_by_flag("MOONDREAM2_ENABLED")

    assert REQUIRES_GPU_FOR_LOCAL_EXECUTION.applies_to_runtimes == [
        Runtime.SELF_HOSTED_CPU
    ]
    assert REQUIRES_GPU_FOR_LOCAL_EXECUTION.applies_to_step_execution_modes == [
        StepExecutionMode.LOCAL
    ]
    assert endpoint_disabled.applies_to_runtimes == [Runtime.HOSTED_SERVERLESS]
    assert endpoint_disabled.applies_to_step_execution_modes == [
        StepExecutionMode.REMOTE
    ]
