"""The REMOTE step-execution prohibition of the stateful streaming-video blocks.

SAM2 Video, SAM3 Video and Action Recognition keep per-video state in their own
block instance. Their `run()` raises `NotImplementedError` before loading a
model whenever the step execution mode is REMOTE - on every runtime, a
self-hosted GPU included. These tests pin that the declarations say so:

* every numpy / tensor variant declares a HARD restriction whose ONLY
  condition is the REMOTE step execution mode, in both discovery views;
* the condition prohibits REMOTE on every runtime and input mode, and does not
  prohibit LOCAL;
* this host's flags neither drop nor narrow it;
* the legacy `get_restrictions()` carries the same caveat;
* public workload introspection reports it;
* the runtime rejection really happens before any model is loaded.

Nothing here loads a model, needs a GPU or opens a network connection.
"""

from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Type

import numpy as np
import pytest
from roboflow_workflows import environment
from roboflow_workflows.core_steps.models.foundation.segment_anything2_video import (
    v1 as sam2_video_v1,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything2_video import (
    v1_tensor as sam2_video_v1_tensor,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3_video import (
    v1 as sam3_video_v1,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3_video import (
    v1_tensor as sam3_video_v1_tensor,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything_common.streaming_video import (
    SAM3_CONCEPT_VIDEO_MODEL_ID,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition import (
    v1 as action_recognition_v1,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition import (
    v1_tensor as action_recognition_v1_tensor,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.workload import (
    RestrictionCondition,
    RestrictionMetadata,
    Runtime,
    RuntimeInputMode,
    Severity,
    StepExecutionMode,
)
from roboflow_workflows.execution_engine.introspection.restriction_environment import (
    EVALUABLE_CONFIGURATION_KEYS,
    ConfigurationMatch,
    evaluate_configuration_condition,
)
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)
from roboflow_workflows.prototypes.block import WorkflowBlock, WorkflowBlockManifest

from tests.unit_tests.prototypes.platform_client_double import RecordingPlatformClient
from tests.unit_tests.workload_declaration_helpers import (
    declared_restrictions,
    portable_restrictions,
    portable_restrictions_discovery,
)

REMOTE_CODE = "remote_step_execution_not_supported"
SAM2_VIDEO = "roboflow_core/segment_anything_2_video@v1"
SAM3_VIDEO = "roboflow_core/sam3_video@v1"
ACTION_RECOGNITION = "roboflow_core/roboflow_action_recognition_model@v1"

# Every code these blocks declare, in the canonical discovery order. The three
# pre-existing caveats must survive next to the new one.
EXPECTED_CODES = [
    REMOTE_CODE,
    "requires_gpu_for_local_execution",
    "stateful_video_state_resets_on_stateless_http",
    "temporal_block_no_benefit_on_still_image",
]

# The legacy editor list keeps its historic order; the new caveat is appended.
EXPECTED_LEGACY_CODES = [
    "stateful_video_state_resets_on_stateless_http",
    "requires_gpu_for_local_execution",
    "temporal_block_no_benefit_on_still_image",
    REMOTE_CODE,
]

EXPECTED_REMOTE_RESTRICTION = RestrictionMetadata(
    code=REMOTE_CODE,
    severity=Severity.HARD,
    when=RestrictionCondition(step_execution_modes=[StepExecutionMode.REMOTE]),
)


class _Variant(NamedTuple):
    """One numpy or tensor implementation of a streaming-video block."""

    block_type: str
    manifest_class: Type[WorkflowBlockManifest]
    block_class: Type[WorkflowBlock]
    manifest_fields: Dict[str, Any]
    run_arguments: Dict[str, Any]
    takes_platform_client: bool


_SAM2_RUN_ARGUMENTS = {
    "boxes": None,
    "model_id": "sam2video/small",
    "prompt_mode": "first_frame",
    "prompt_interval": 30,
    "threshold": 0.0,
}
_SAM3_RUN_ARGUMENTS = {
    "class_names": ["person"],
    "model_id": SAM3_CONCEPT_VIDEO_MODEL_ID,
    "threshold": 0.5,
}
_ACTION_RECOGNITION_RUN_ARGUMENTS = {"model_id": "my-actions/2"}

VARIANTS = {
    "sam2_video_numpy": _Variant(
        block_type=SAM2_VIDEO,
        manifest_class=sam2_video_v1.BlockManifest,
        block_class=sam2_video_v1.SegmentAnything2VideoBlockV1,
        manifest_fields={},
        run_arguments=_SAM2_RUN_ARGUMENTS,
        takes_platform_client=True,
    ),
    "sam2_video_tensor": _Variant(
        block_type=SAM2_VIDEO,
        manifest_class=sam2_video_v1_tensor.BlockManifest,
        block_class=sam2_video_v1_tensor.SegmentAnything2VideoBlockV1,
        manifest_fields={},
        run_arguments=_SAM2_RUN_ARGUMENTS,
        takes_platform_client=True,
    ),
    "sam3_video_numpy": _Variant(
        block_type=SAM3_VIDEO,
        manifest_class=sam3_video_v1.BlockManifest,
        block_class=sam3_video_v1.SegmentAnything3VideoBlockV1,
        manifest_fields={"class_names": ["person"]},
        run_arguments=_SAM3_RUN_ARGUMENTS,
        takes_platform_client=True,
    ),
    "sam3_video_tensor": _Variant(
        block_type=SAM3_VIDEO,
        manifest_class=sam3_video_v1_tensor.BlockManifest,
        block_class=sam3_video_v1_tensor.SegmentAnything3VideoBlockV1,
        manifest_fields={"class_names": ["person"]},
        run_arguments=_SAM3_RUN_ARGUMENTS,
        takes_platform_client=True,
    ),
    "action_recognition_numpy": _Variant(
        block_type=ACTION_RECOGNITION,
        manifest_class=action_recognition_v1.BlockManifest,
        block_class=action_recognition_v1.ActionRecognitionModelBlockV1,
        manifest_fields={"model_id": "my-actions/2"},
        run_arguments=_ACTION_RECOGNITION_RUN_ARGUMENTS,
        takes_platform_client=False,
    ),
    "action_recognition_tensor": _Variant(
        block_type=ACTION_RECOGNITION,
        manifest_class=action_recognition_v1_tensor.BlockManifest,
        block_class=action_recognition_v1_tensor.ActionRecognitionModelBlockV1,
        manifest_fields={"model_id": "my-actions/2"},
        run_arguments=_ACTION_RECOGNITION_RUN_ARGUMENTS,
        takes_platform_client=False,
    ),
}

variant_parameters = pytest.mark.parametrize(
    "variant", list(VARIANTS.values()), ids=list(VARIANTS)
)


class _UntouchableModelsProvider:
    """A models provider that fails the test on any use.

    Loading a model needs the provider (the artifact cache for SAM2 / SAM3,
    `load_action_recognition_model()` for action recognition). Every attribute
    request is recorded and raises, so a model load attempt cannot pass as the
    expected `NotImplementedError`.
    """

    def __init__(self) -> None:
        self.requested: List[str] = []

    def __getattr__(self, name: str) -> Any:
        self.requested.append(name)
        raise AssertionError(f"the models provider was asked for `{name}`")


def _manifest(variant: _Variant) -> WorkflowBlockManifest:
    manifest = variant.manifest_class.model_validate(
        {
            "type": variant.block_type,
            "name": "tracker",
            "images": "$inputs.image",
            **variant.manifest_fields,
        }
    )

    return manifest


def _frame() -> WorkflowImageData:
    frame = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="stream-0:0"),
        numpy_image=np.zeros((48, 64, 3), dtype=np.uint8),
        video_metadata=VideoMetadata(
            video_identifier="stream-0",
            frame_number=0,
            fps=30,
            frame_timestamp=datetime(2024, 1, 1),
        ),
    )

    return frame


def _condition_holds(
    condition: RestrictionCondition,
    *,
    runtime: Runtime,
    step_execution_mode: StepExecutionMode,
    input_mode: RuntimeInputMode,
) -> bool:
    """Apply the documented `RestrictionCondition` semantics to one target.

    AND across the axes, OR within an axis, and an unset axis restricts
    nothing. Configuration predicates are judged by the production evaluator
    in the host-flag test, not here.
    """
    axes = (
        (condition.runtimes, runtime),
        (condition.step_execution_modes, step_execution_mode),
        (condition.input_modes, input_mode),
    )
    holds = all(allowed is None or target in allowed for allowed, target in axes)

    return holds


def _prohibiting_codes(
    restrictions: List[RestrictionMetadata],
    *,
    runtime: Runtime,
    step_execution_mode: StepExecutionMode,
    input_mode: RuntimeInputMode,
) -> List[str]:
    """The codes of the HARD restrictions that apply to one target."""
    codes = [
        restriction.code
        for restriction in restrictions
        if restriction.severity is Severity.HARD
        and _condition_holds(
            restriction.when,
            runtime=runtime,
            step_execution_mode=step_execution_mode,
            input_mode=input_mode,
        )
    ]

    return codes


def _only_remote_restriction(
    restrictions: List[RestrictionMetadata],
) -> RestrictionMetadata:
    matching = [item for item in restrictions if item.code == REMOTE_CODE]
    assert len(matching) == 1, restrictions

    return matching[0]


# ---------------------------------------------------------------------------
# declarations
# ---------------------------------------------------------------------------


@variant_parameters
@pytest.mark.parametrize("ignore_environment_restrictions", [True, False])
def test_both_discovery_views_declare_the_remote_prohibition_completely(
    variant: _Variant, ignore_environment_restrictions: bool
) -> None:
    # given
    manifest = _manifest(variant)

    # when
    discovery = manifest.get_actual_restrictions(
        ignore_environment_restrictions=ignore_environment_restrictions
    )

    # then - a complete answer: the new caveat plus every pre-existing one
    assert discovery.complete is True
    assert discovery.unknown_reasons == []
    assert [item.code for item in discovery.items] == EXPECTED_CODES


@variant_parameters
def test_the_remote_prohibition_is_conditioned_on_the_remote_mode_only(
    variant: _Variant,
) -> None:
    # given
    manifest = _manifest(variant)

    # when
    restriction = _only_remote_restriction(portable_restrictions(manifest))

    # then - no runtime, input-mode or configuration axis narrows it
    assert restriction == EXPECTED_REMOTE_RESTRICTION
    assert restriction.when.runtimes is None
    assert restriction.when.input_modes is None
    assert restriction.when.configuration_equals == {}


@variant_parameters
def test_a_self_hosted_gpu_prohibits_remote_but_not_local_execution(
    variant: _Variant,
) -> None:
    # given
    restrictions = portable_restrictions(_manifest(variant))

    # when
    remote = _prohibiting_codes(
        restrictions,
        runtime=Runtime.SELF_HOSTED_GPU,
        step_execution_mode=StepExecutionMode.REMOTE,
        input_mode=RuntimeInputMode.VIDEO,
    )
    local = _prohibiting_codes(
        restrictions,
        runtime=Runtime.SELF_HOSTED_GPU,
        step_execution_mode=StepExecutionMode.LOCAL,
        input_mode=RuntimeInputMode.VIDEO,
    )

    # then - a GPU does not help REMOTE, and LOCAL video on a GPU is allowed
    assert remote == [REMOTE_CODE]
    assert local == []


@variant_parameters
@pytest.mark.parametrize("runtime", list(Runtime))
@pytest.mark.parametrize("input_mode", list(RuntimeInputMode))
def test_the_remote_prohibition_holds_on_every_runtime_and_input_mode(
    variant: _Variant, runtime: Runtime, input_mode: RuntimeInputMode
) -> None:
    # given
    restriction = _only_remote_restriction(portable_restrictions(_manifest(variant)))

    # when
    applies_remotely = _condition_holds(
        restriction.when,
        runtime=runtime,
        step_execution_mode=StepExecutionMode.REMOTE,
        input_mode=input_mode,
    )
    applies_locally = _condition_holds(
        restriction.when,
        runtime=runtime,
        step_execution_mode=StepExecutionMode.LOCAL,
        input_mode=input_mode,
    )

    # then
    assert applies_remotely is True
    assert applies_locally is False


@variant_parameters
@pytest.mark.parametrize("flag_value", [True, False])
@pytest.mark.parametrize("host_step_execution_mode", ["local", "remote"])
def test_host_configuration_does_not_drop_the_remote_prohibition(
    variant: _Variant,
    flag_value: bool,
    host_step_execution_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given - every boolean flag the host-view evaluator can read, plus the
    # host's own step execution mode, set one way
    manifest = _manifest(variant)
    flipped = []
    for key in sorted(EVALUABLE_CONFIGURATION_KEYS):
        if isinstance(getattr(environment, key, None), bool):
            monkeypatch.setattr(environment, key, flag_value)
            flipped.append(key)
    monkeypatch.setattr(
        environment, "WORKFLOWS_STEP_EXECUTION_MODE", host_step_execution_mode
    )
    assert "ENABLE_TENSOR_DATA_REPRESENTATION" in flipped

    # when
    portable_view = portable_restrictions_discovery(manifest)
    host_view = manifest.get_actual_restrictions(ignore_environment_restrictions=False)
    host_remote = [item for item in host_view.items if item.code == REMOTE_CODE]

    # then - kept in both views, complete, not narrowed, and a definite
    # configuration match rather than an unknown
    assert portable_view.complete is True
    assert _only_remote_restriction(list(portable_view.items)) == (
        EXPECTED_REMOTE_RESTRICTION
    )
    assert host_view.complete is True
    assert host_view.unknown_reasons == []
    assert [item.code for item in host_view.items] == EXPECTED_CODES
    assert len(host_remote) == 1
    assert evaluate_configuration_condition(restriction=host_remote[0]) == (
        ConfigurationMatch.MATCHES,
        [],
    )


@variant_parameters
def test_the_legacy_declaration_carries_the_same_remote_prohibition(
    variant: _Variant,
) -> None:
    # given
    manifest = _manifest(variant)
    authored = declared_restrictions(manifest)

    # when
    legacy = variant.manifest_class.get_restrictions()

    # then - same code and axes on both sides; editor payload unchanged in shape
    assert [item.code for item in legacy] == EXPECTED_LEGACY_CODES
    legacy_remote = legacy[-1]
    assert [item for item in authored if item.code == REMOTE_CODE] == [legacy_remote]
    assert legacy_remote.to_dict() == {
        "severity": "hard",
        "note": legacy_remote.note,
        "applies_to_step_execution_modes": ["remote"],
    }
    assert "NotImplementedError" in legacy_remote.note


# ---------------------------------------------------------------------------
# public workload introspection
# ---------------------------------------------------------------------------


def test_workload_introspection_reports_the_remote_prohibition() -> None:
    # given - the three families, as registered in this process
    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {"type": SAM2_VIDEO, "name": "sam2", "images": "$inputs.image"},
            {
                "type": SAM3_VIDEO,
                "name": "sam3",
                "images": "$inputs.image",
                "class_names": ["person"],
            },
            {
                "type": ACTION_RECOGNITION,
                "name": "actions",
                "images": "$inputs.image",
                "model_id": "my-actions/2",
            },
        ],
        "outputs": [],
    }

    # when
    introspection = describe_workflow_workload(definition)

    # then
    steps = {step.node_id: step for step in introspection.steps}
    for node_id in ("$steps.sam2", "$steps.sam3", "$steps.actions"):
        restrictions = steps[node_id].restrictions
        assert restrictions.complete is True, node_id
        assert [item.code for item in restrictions.items] == EXPECTED_CODES, node_id
        assert _only_remote_restriction(list(restrictions.items)) == (
            EXPECTED_REMOTE_RESTRICTION
        )


# ---------------------------------------------------------------------------
# the runtime behaviour the declaration describes
# ---------------------------------------------------------------------------


@variant_parameters
def test_remote_execution_is_rejected_before_any_model_is_loaded(
    variant: _Variant,
) -> None:
    # given
    models_provider = _UntouchableModelsProvider()
    platform_client = RecordingPlatformClient()
    init_arguments: Dict[str, Any] = {
        "model_manager": models_provider,
        "api_key": None,
        "step_execution_mode": StepExecutionMode.REMOTE,
    }
    if variant.takes_platform_client:
        init_arguments["platform_client"] = platform_client
    block = variant.block_class(**init_arguments)

    # when
    with pytest.raises(NotImplementedError, match="only supports LOCAL workflow"):
        block.run(images=[_frame()], **variant.run_arguments)

    # then - no model, no weights request, no provider access
    assert block._model is None
    assert models_provider.requested == []
    assert platform_client.weights_calls == []
