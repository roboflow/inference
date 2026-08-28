"""Guards that model families still report a ``model``-category usage row.

``@usage_collector("model")`` normally rides along on ``BaseInference.infer``.
Families that override ``infer()`` without calling ``super()``, or that serve
production traffic through ``infer_from_request`` / a workflow ``run()`` that
never calls ``infer()``, must decorate those entrypoints themselves. Without
that they emit no model-category row and disappear from per-model telemetry.

These checks parse the module source with ``ast`` so they run on the stock
unit-test image. Importing the modules would skip CLIP/SAM/SAM2/SAM3 under
``pytest.importorskip`` because those extras are not installed in CI.
"""

import ast
from pathlib import Path

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
# Tensor-native siblings (v1_tensor.py) are not decorated; they currently
# emit no model-category row under ENABLE_TENSOR_DATA_REPRESENTATION.
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

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _module_source_path(module_path: str) -> Path:
    return _REPO_ROOT / Path(*module_path.split(".")).with_suffix(".py")


def _decorator_callable_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_callable_name(node.func)
    return ""


def _decorator_category(call: ast.Call) -> str:
    if call.args and isinstance(call.args[0], ast.Constant):
        value = call.args[0].value
        if isinstance(value, str):
            return value
    for keyword in call.keywords:
        if keyword.arg == "category" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                return value
    return ""


def _has_usage_collector_model_decorator(func: ast.FunctionDef) -> bool:
    for decorator in func.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if _decorator_callable_name(decorator) != "usage_collector":
            continue
        if _decorator_category(decorator) == "model":
            return True
    return False


def _class_method_has_usage_collector(
    source: str, class_name: str, method_name: str
) -> bool:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                return _has_usage_collector_model_decorator(item)
    return False


def _assert_method_is_usage_collected(module_path, class_name, method_name):
    source_path = _module_source_path(module_path)
    source = source_path.read_text()

    assert _class_method_has_usage_collector(source, class_name, method_name), (
        f"{class_name}.{method_name}() in {source_path} must be decorated "
        "with @usage_collector('model')"
    )


@pytest.mark.parametrize("module_path, class_name", MODELS_OVERRIDING_INFER)
def test_model_overriding_infer_is_usage_collected(module_path, class_name):
    _assert_method_is_usage_collected(module_path, class_name, "infer")


@pytest.mark.parametrize(
    "module_path, class_name", MODELS_DECORATING_INFER_FROM_REQUEST
)
def test_model_infer_from_request_is_usage_collected(module_path, class_name):
    _assert_method_is_usage_collected(module_path, class_name, "infer_from_request")


@pytest.mark.parametrize("module_path, class_name, method_name", BLOCKS_DECORATING_RUN)
def test_video_block_run_is_usage_collected(module_path, class_name, method_name):
    _assert_method_is_usage_collected(module_path, class_name, method_name)


def test_detection_helper_rejects_undecorated_function():
    source = (
        "class Fake:\n"
        "    def infer(self, image, **kwargs):\n"
        "        return None\n"
    )

    assert not _class_method_has_usage_collector(source, "Fake", "infer")
