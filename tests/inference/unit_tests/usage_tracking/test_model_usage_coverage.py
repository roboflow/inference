"""Guards that model families still report a ``model``-category usage row.

``@usage_collector("model")`` normally rides along on ``BaseInference.infer``.
Families that override ``infer()`` without calling ``super()``, or that serve
production traffic through ``infer_from_request`` / a workflow ``run()`` that
never calls ``infer()``, must decorate those entrypoints themselves. Without
that they emit no model-category row and disappear from per-model telemetry.
"""

import inspect

import pytest

# (module, class) pairs whose ``infer()`` bypasses ``BaseInference.infer``.
MODELS_OVERRIDING_INFER = [
    ("inference.models.doctr.doctr_model", "DocTR"),
    (
        "inference.models.doctr.doctr_model_inference_models",
        "InferenceModelsDocTRAdapter",
    ),
    ("inference.models.grounding_dino.grounding_dino", "GroundingDINO"),
    (
        "inference.models.grounding_dino.grounding_dino_inference_models",
        "InferenceModelsGroundingDINOAdapter",
    ),
    ("inference.models.yolo_world.yolo_world", "YOLOWorld"),
    ("inference.models.owlv2.owlv2", "OwlV2"),
    ("inference.models.owlv2.owlv2", "SerializedOwlV2"),
    ("inference.models.owlv2.owlv2_inference_models", "InferenceModelsOwlV2Adapter"),
    (
        "inference.models.owlv2.rf_instant_inference_models",
        "InferenceModelsRFInstantModelAdapter",
    ),
]

# Production traffic for these families is ``infer_from_request``, which does
# not call ``infer()``. The decorator has to live on that method.
MODELS_DECORATING_INFER_FROM_REQUEST = [
    ("inference.models.clip.clip_model", "Clip"),
    (
        "inference.models.clip.clip_inference_models",
        "InferenceModelsClipAdapter",
    ),
    (
        "inference.models.perception_encoder.perception_encoder",
        "PerceptionEncoder",
    ),
    (
        "inference.models.perception_encoder.perception_encoder_inference_models",
        "InferenceModelsPerceptionEncoderAdapter",
    ),
    ("inference.models.sam.segment_anything", "SegmentAnything"),
    (
        "inference.models.sam.segment_anything_inference_models",
        "InferenceModelsSAMAdapter",
    ),
    (
        "inference.models.sam2.segment_anything2",
        "SegmentAnything2",
    ),
    (
        "inference.models.sam2.segment_anything2_inference_models",
        "InferenceModelsSAM2Adapter",
    ),
    (
        "inference.models.sam3.segment_anything3",
        "SegmentAnything3",
    ),
    (
        "inference.models.sam3.segment_anything3_inference_models",
        "InferenceModelsSAM3Adapter",
    ),
    (
        "inference.models.sam3.visual_segmentation",
        "Sam3ForInteractiveImageSegmentation",
    ),
    (
        "inference.models.sam3.visual_segmentation_inference_models",
        "InferenceModelsSAM3InteractiveAdapter",
    ),
    (
        "inference.models.sam3_3d.segment_anything_3d",
        "SegmentAnything3_3D_Objects",
    ),
]

# Video trackers load ``AutoModel`` in the block and never go through
# ModelManager, so the block's tracked entrypoint must emit the model-category
# row. SAM3 tracks ``_tracked_run`` so that ``run()`` can first swap in the
# visual model id the decorator should attribute usage to.
BLOCKS_DECORATING_RUN = [
    (
        "inference.core.workflows.core_steps.models.foundation.segment_anything2_video.v1",
        "SegmentAnything2VideoBlockV1",
        "run",
    ),
    (
        "inference.core.workflows.core_steps.models.foundation.segment_anything3_video.v1",
        "SegmentAnything3VideoBlockV1",
        "_tracked_run",
    ),
]


def _is_usage_collected(func) -> bool:
    # functools.wraps hides the wrapper, so read the unfollowed signature and
    # look for the keyword-only arguments the decorator injects.
    parameters = inspect.signature(func, follow_wrapped=False).parameters
    return "usage_api_key" in parameters and "usage_billable" in parameters


@pytest.mark.parametrize("module_path, class_name", MODELS_OVERRIDING_INFER)
def test_model_overriding_infer_is_usage_collected(module_path, class_name):
    module = pytest.importorskip(module_path)
    model_class = getattr(module, class_name)

    assert _is_usage_collected(
        model_class.infer
    ), f"{class_name}.infer() must be decorated with @usage_collector('model')"


@pytest.mark.parametrize(
    "module_path, class_name", MODELS_DECORATING_INFER_FROM_REQUEST
)
def test_model_infer_from_request_is_usage_collected(module_path, class_name):
    module = pytest.importorskip(module_path)
    model_class = getattr(module, class_name)

    assert _is_usage_collected(model_class.infer_from_request), (
        f"{class_name}.infer_from_request() must be decorated with "
        "@usage_collector('model')"
    )


@pytest.mark.parametrize("module_path, class_name, method_name", BLOCKS_DECORATING_RUN)
def test_video_block_run_is_usage_collected(module_path, class_name, method_name):
    module = pytest.importorskip(module_path)
    block_class = getattr(module, class_name)

    assert _is_usage_collected(getattr(block_class, method_name)), (
        f"{class_name}.{method_name}() must be decorated with "
        "@usage_collector('model')"
    )


def test_detection_helper_rejects_undecorated_function():
    # Without this the parametrized test above would pass vacuously.
    def infer(self, image, **kwargs):
        return None

    assert not _is_usage_collected(infer)
