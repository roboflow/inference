"""Exact per-family assertions for the model-block workload declarations.

`discover_work_operations()` and `discover_portable_restrictions()` are the
portable, compile-time counterparts of what a model block does and of the
caveats `get_restrictions()` already publishes. These tests pin the exact
values for one representative block of every operation family and of every
restriction shape, so a silent re-classification fails here.
"""

import inspect
from typing import List, Type

import pytest
from roboflow_workflows.core_steps.models.foundation.anthropic_claude.v5 import (
    BlockManifest as AnthropicClaudeV5Manifest,
)
from roboflow_workflows.core_steps.models.foundation.cog_vlm.v1 import (
    BlockManifest as CogVLMV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.florence2.v1 import (
    BlockManifest as Florence2V1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.florence2.v2 import (
    V2BlockManifest as Florence2V2Manifest,
)
from roboflow_workflows.core_steps.models.foundation.gaze.v1 import (
    BlockManifest as GazeV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.lmm.v1 import (
    BlockManifest as LMMV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.lmm_classifier.v1 import (
    BlockManifest as LMMClassifierV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.moondream2.v1 import (
    BlockManifest as Moondream2V1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.openai.v7 import (
    BlockManifest as OpenAIV7Manifest,
)
from roboflow_workflows.core_steps.models.foundation.qwen_vlm.v1 import (
    BlockManifest as QwenVLMV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.seg_preview.v1 import (
    BlockManifest as SegPreviewV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3.v1 import (
    BlockManifest as SAM3V1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3_video.v1 import (
    BlockManifest as SAM3VideoV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.stability_ai.inpainting.v1 import (
    BlockManifest as StabilityInpaintingV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.yolo_world.v1 import (
    BlockManifest as YoloWorldV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.yolo_world.v1_tensor import (
    BlockManifest as YoloWorldV1TensorManifest,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    BlockManifest as ActionRecognitionV1Manifest,
)
from roboflow_workflows.core_steps.models.roboflow.object_detection.v1 import (
    BlockManifest as ObjectDetectionV1Manifest,
)
from roboflow_workflows.core_steps.models.roboflow.object_detection.v2 import (
    BlockManifest as ObjectDetectionV2Manifest,
)
from roboflow_workflows.core_steps.models.third_party.barcode_detection.v1 import (
    BlockManifest as BarcodeDetectorV1Manifest,
)
from roboflow_workflows.core_steps.models.third_party.qr_code_detection.v1 import (
    BlockManifest as QRCodeDetectorV1Manifest,
)
from roboflow_workflows.core_steps.models.workload_presets import (
    DEPRECATED_BLOCK_ALWAYS_RAISES,
    REQUIRES_GPU_FOR_LOCAL_EXECUTION,
    ROBOFLOW_INTERNAL_ENDPOINT_ONLY,
    UNSUPPORTED_IN_TENSOR_REPRESENTATION,
    hosted_endpoint_disabled_by_flag,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RestrictionCondition,
    RestrictionMetadata,
    Runtime,
    RuntimeInputMode,
    Severity,
    StepExecutionMode,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION,
)

MODEL_INFERENCE_ONLY = [WorkOperation.MODEL_INFERENCE]
VENDOR_API_OPERATIONS = [
    WorkOperation.MODEL_INFERENCE,
    WorkOperation.EXTERNAL_REQUEST,
    WorkOperation.IMAGE_ENCODING,
]


def _build(manifest_class: Type, **kwargs):
    """Instantiate a manifest with the minimum every model block requires."""
    payload = {"name": "step"}
    if "images" in manifest_class.model_fields:
        payload["images"] = "$inputs.image"
    payload.update(kwargs)
    return manifest_class(type=_first_type_literal(manifest_class), **payload)


def _first_type_literal(manifest_class: Type) -> str:
    annotation = manifest_class.model_fields["type"].annotation
    return annotation.__args__[0]


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------


def test_roboflow_object_detection_v1_declares_model_inference_only() -> None:
    # given
    manifest = _build(ObjectDetectionV1Manifest, model_id="yolov8n-640")

    # when
    operations = manifest.discover_work_operations()

    # then - remote dispatch is environment-defined transport, so the block
    # must NOT claim EXTERNAL_REQUEST here.
    assert operations == MODEL_INFERENCE_ONLY
    assert manifest.discover_portable_restrictions() == []


def test_roboflow_object_detection_v2_declares_model_inference_only() -> None:
    # given
    manifest = _build(ObjectDetectionV2Manifest, model_id="yolov8n-640")

    # when
    operations = manifest.discover_work_operations()

    # then
    assert operations == MODEL_INFERENCE_ONLY
    assert manifest.discover_portable_restrictions() == []


def test_action_recognition_declares_temporal_buffering() -> None:
    # given
    manifest = _build(ActionRecognitionV1Manifest, model_id="my-action-model/1")

    # when
    operations = manifest.discover_work_operations()

    # then - the block keeps a sliding window of frames per video stream.
    assert operations == [
        WorkOperation.MODEL_INFERENCE,
        WorkOperation.TEMPORAL_BUFFERING,
    ]


def test_sam3_video_declares_tracking() -> None:
    # given
    manifest = _build(SAM3VideoV1Manifest, class_names=["person"])

    # when
    operations = manifest.discover_work_operations()

    # then - the streaming SAM3 block propagates masks and emits tracker ids.
    assert operations == [WorkOperation.MODEL_INFERENCE, WorkOperation.TRACKING]


def test_third_party_analyser_declares_image_analysis_without_model_inference() -> None:
    # given
    manifest = _build(BarcodeDetectorV1Manifest)

    # when
    operations = manifest.discover_work_operations()

    # then - pyzbar is not a model.
    assert operations == [WorkOperation.IMAGE_ANALYSIS]
    assert manifest.discover_portable_restrictions() == []


def test_vendor_api_block_declares_external_request_and_encoding() -> None:
    # given
    manifest = _build(OpenAIV7Manifest, prompt="describe the image")

    # when
    operations = manifest.discover_work_operations()

    # then
    assert operations == VENDOR_API_OPERATIONS
    assert manifest.discover_portable_restrictions() == []


def test_anthropic_claude_declares_external_request_and_encoding() -> None:
    # given
    manifest = _build(
        AnthropicClaudeV5Manifest, api_key="secret", prompt="describe the image"
    )

    # when / then
    assert manifest.discover_work_operations() == VENDOR_API_OPERATIONS


def test_stability_inpainting_declares_mask_composition() -> None:
    # given
    manifest = _build(
        StabilityInpaintingV1Manifest,
        image="$inputs.image",
        segmentation_mask="$steps.model.predictions",
        prompt="replace the object",
        api_key="secret",
    )

    # when
    operations = manifest.discover_work_operations()

    # then - the block rasterises the segmentation mask itself before upload.
    assert operations == [
        WorkOperation.MODEL_INFERENCE,
        WorkOperation.EXTERNAL_REQUEST,
        WorkOperation.IMAGE_ENCODING,
        WorkOperation.IMAGE_COMPOSITION,
    ]


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("native", MODEL_INFERENCE_ONLY),
        ("openrouter", VENDOR_API_OPERATIONS),
    ],
)
def test_qwen_vlm_operations_follow_the_selected_backend(
    backend: str, expected: List[WorkOperation]
) -> None:
    # given - a literal manifest setting selects the work the block performs
    manifest = _build(QwenVLMV1Manifest, backend=backend, prompt="describe")

    # when
    operations = manifest.discover_work_operations()

    # then
    assert operations == expected
    assert manifest.discover_portable_restrictions() == []


# ---------------------------------------------------------------------------
# portable restrictions
# ---------------------------------------------------------------------------


def test_gpu_and_flag_gated_block_declares_both_branches() -> None:
    # given
    manifest = _build(Moondream2V1Manifest)

    # when
    restrictions = manifest.discover_portable_restrictions()

    # then - the hosted-endpoint branch is declared unconditionally, carrying
    # the flag value it applies to instead of being filtered out here.
    assert restrictions == [
        RestrictionMetadata(
            code="requires_gpu_for_local_execution",
            severity=Severity.HARD,
            when=RestrictionCondition(
                runtimes=[Runtime.SELF_HOSTED_CPU],
                step_execution_modes=[StepExecutionMode.LOCAL],
            ),
        ),
        RestrictionMetadata(
            code="hosted_endpoint_disabled_by_flag",
            severity=Severity.HARD,
            when=RestrictionCondition(
                runtimes=[Runtime.HOSTED_SERVERLESS],
                step_execution_modes=[StepExecutionMode.REMOTE],
                configuration_equals={"MOONDREAM2_ENABLED": False},
            ),
        ),
    ]


def test_sam3_declares_gpu_and_sam3_flag() -> None:
    # given
    manifest = _build(SAM3V1Manifest)

    # when
    restrictions = manifest.discover_portable_restrictions()

    # then
    assert restrictions == [
        REQUIRES_GPU_FOR_LOCAL_EXECUTION,
        hosted_endpoint_disabled_by_flag("CORE_MODEL_SAM3_ENABLED"),
    ]


@pytest.mark.parametrize("manifest_class", [LMMV1Manifest, LMMClassifierV1Manifest])
def test_lmm_blocks_declare_the_flag_branch_without_a_gpu_caveat(
    manifest_class: Type,
) -> None:
    # given - legacy get_restrictions() starts from an EMPTY list here
    manifest = _build(
        manifest_class,
        prompt="describe",
        lmm_type="gpt_4v",
        classes=["cat", "dog"],
    )

    # when
    restrictions = manifest.discover_portable_restrictions()

    # then
    assert restrictions == [hosted_endpoint_disabled_by_flag("LMM_ENABLED")]


@pytest.mark.parametrize("manifest_class", [Florence2V1Manifest, Florence2V2Manifest])
def test_both_florence2_versions_declare_the_same_caveats(
    manifest_class: Type,
) -> None:
    # given - V2BlockManifest inherits the legacy declaration from BaseManifest
    manifest = _build(manifest_class, classes=["cat", "dog"])

    # when
    restrictions = manifest.discover_portable_restrictions()

    # then
    assert restrictions == [
        REQUIRES_GPU_FOR_LOCAL_EXECUTION,
        hosted_endpoint_disabled_by_flag("FLORENCE2_ENABLED"),
    ]
    assert manifest.discover_work_operations() == MODEL_INFERENCE_ONLY


def test_seg_preview_declares_the_roboflow_internal_endpoint_caveat() -> None:
    # given
    manifest = _build(SegPreviewV1Manifest)

    # when
    restrictions = manifest.discover_portable_restrictions()

    # then - mirrors the legacy axes: three self-hosted runtimes, no mode axis
    assert restrictions == [ROBOFLOW_INTERNAL_ENDPOINT_ONLY]
    assert restrictions[0].when == RestrictionCondition(
        runtimes=[
            Runtime.SELF_HOSTED_CPU,
            Runtime.SELF_HOSTED_GPU,
            Runtime.INFERENCE_PIPELINE,
        ],
    )
    assert restrictions[0].when.step_execution_modes is None


def test_streaming_video_block_declares_the_shared_presets_and_gpu() -> None:
    # given
    manifest = _build(SAM3VideoV1Manifest, class_names=["person"])

    # when
    restrictions = manifest.discover_portable_restrictions()

    # then
    assert restrictions == [
        STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION,
        REQUIRES_GPU_FOR_LOCAL_EXECUTION,
        STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION,
    ]
    assert restrictions[0].severity is Severity.SOFT
    assert restrictions[0].when.input_modes == [RuntimeInputMode.VIDEO]
    assert restrictions[2].when.input_modes == [RuntimeInputMode.IMAGE]


def test_action_recognition_declares_the_same_restriction_shape() -> None:
    # given
    manifest = _build(ActionRecognitionV1Manifest, model_id="my-action-model/1")

    # when
    restrictions = manifest.discover_portable_restrictions()

    # then
    assert restrictions == [
        STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION,
        REQUIRES_GPU_FOR_LOCAL_EXECUTION,
        STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION,
    ]


def test_presets_carry_no_note_field() -> None:
    # then - the portable entity intentionally drops the human note
    assert "note" not in RestrictionMetadata.model_fields
    with pytest.raises(Exception):
        RestrictionMetadata(
            code="requires_gpu_for_local_execution",
            severity=Severity.HARD,
            note="anything",
        )


# ---------------------------------------------------------------------------
# deprecated blocks, representation gating, backend-dependent operations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manifest_class, flag, extra",
    [
        (CogVLMV1Manifest, "LMM_ENABLED", {"prompt": "describe"}),
        (GazeV1Manifest, "CORE_MODEL_GAZE_ENABLED", {}),
    ],
)
def test_a_permanently_deprecated_block_declares_no_work(
    manifest_class: Type, flag: str, extra: dict
) -> None:
    # given - run() raises FeatureDeprecatedError before touching a model
    manifest = _build(manifest_class, **extra)

    # when
    operations = manifest.discover_work_operations()
    restrictions = manifest.discover_portable_restrictions()

    # then - a complete, truthful "no operation", plus the hard caveat that
    # the step can never produce a result; the legacy caveats are kept.
    assert operations == []
    assert restrictions == [
        DEPRECATED_BLOCK_ALWAYS_RAISES,
        REQUIRES_GPU_FOR_LOCAL_EXECUTION,
        hosted_endpoint_disabled_by_flag(flag),
    ]
    assert restrictions[0].severity is Severity.HARD
    assert restrictions[0].when == RestrictionCondition()


@pytest.mark.parametrize(
    "manifest_class", [YoloWorldV1Manifest, YoloWorldV1TensorManifest]
)
def test_yolo_world_declares_the_tensor_representation_caveat(
    manifest_class: Type,
) -> None:
    # given
    manifest = _build(manifest_class, class_names=["dog"])

    # when
    restrictions = manifest.discover_portable_restrictions()

    # then - the tensor sibling raises, so the block only works on a numpy
    # server; the condition names the True branch of the representation flag.
    assert restrictions == [UNSUPPORTED_IN_TENSOR_REPRESENTATION]
    assert restrictions[0].severity is Severity.HARD
    assert restrictions[0].when.configuration_equals == {
        "ENABLE_TENSOR_DATA_REPRESENTATION": True
    }
    assert manifest.discover_work_operations() == MODEL_INFERENCE_ONLY


def test_yolo_world_declares_the_same_text_in_both_representations() -> None:
    for hook in ("discover_work_operations", "discover_portable_restrictions"):
        numpy_source = inspect.getsource(getattr(YoloWorldV1Manifest, hook))
        tensor_source = inspect.getsource(getattr(YoloWorldV1TensorManifest, hook))

        assert numpy_source == tensor_source


@pytest.mark.parametrize(
    "manifest_class, extra",
    [
        (LMMV1Manifest, {"prompt": "describe"}),
        (LMMClassifierV1Manifest, {"classes": ["cat", "dog"]}),
    ],
)
def test_lmm_operations_follow_the_literal_lmm_type(
    manifest_class: Type, extra: dict
) -> None:
    # given
    manifest = _build(manifest_class, lmm_type="gpt_4v", **extra)

    # when
    operations = manifest.discover_work_operations()

    # then - gpt_4v reaches the OpenAI API on both execution paths
    assert operations == VENDOR_API_OPERATIONS


@pytest.mark.parametrize(
    "manifest_class, extra",
    [
        (LMMV1Manifest, {"prompt": "describe"}),
        (LMMClassifierV1Manifest, {"classes": ["cat", "dog"]}),
    ],
)
def test_lmm_operations_are_incomplete_when_lmm_type_is_a_selector(
    manifest_class: Type, extra: dict
) -> None:
    # given
    manifest = _build(manifest_class, lmm_type="$inputs.lmm_type", **extra)

    # when
    operations = manifest.discover_work_operations()

    # then - never a guess: an explicit incomplete discovery with a reason
    assert isinstance(operations, Discovery)
    assert list(operations.items) == [WorkOperation.MODEL_INFERENCE]
    assert operations.complete is False
    assert operations.unknown_reasons == ["lmm_type_selector_unresolved:$steps.step"]


# ---------------------------------------------------------------------------
# dependent resources (D019): a proven absence, not an unknown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manifest_class", [BarcodeDetectorV1Manifest, QRCodeDetectorV1Manifest]
)
def test_non_model_analysers_declare_no_dependent_resource(
    manifest_class: Type,
) -> None:
    # given
    manifest = _build(manifest_class)

    # when
    resources = manifest.discover_dependent_resources()

    # then - [] means "provably none", which is not the base None ("unknown")
    assert resources == []
    assert resources is not None


@pytest.mark.parametrize(
    "manifest_class, extra",
    [
        (CogVLMV1Manifest, {"prompt": "describe"}),
        (GazeV1Manifest, {}),
    ],
)
def test_a_permanently_deprecated_block_fetches_no_resource(
    manifest_class: Type, extra: dict
) -> None:
    # given - run() raises before a model is ever fetched
    manifest = _build(manifest_class, **extra)

    # when
    resources = manifest.discover_dependent_resources()

    # then
    assert resources == []


def test_a_vendor_api_without_a_model_identity_stays_unknown() -> None:
    # given - the Stability AI edit endpoint is fixed and the manifest has no
    # field naming a model, so there is nothing truthful to declare
    manifest = _build(
        StabilityInpaintingV1Manifest,
        image="$inputs.image",
        segmentation_mask="$steps.model.predictions",
        prompt="replace the object",
        api_key="secret",
    )

    # when
    resources = manifest.discover_dependent_resources()

    # then - an audited None, written in the class body rather than inherited
    assert resources is None
    assert "discover_dependent_resources" in vars(StabilityInpaintingV1Manifest)


def test_a_dedicated_loader_path_stays_unknown() -> None:
    # given - the block loads with model_manager.load_action_recognition_model(),
    # not the add_model() registration the dependency pre-loader performs
    manifest = _build(ActionRecognitionV1Manifest, model_id="my-action-model/1")

    # when
    resources = manifest.discover_dependent_resources()

    # then
    assert resources is None
    assert "discover_dependent_resources" in vars(ActionRecognitionV1Manifest)
