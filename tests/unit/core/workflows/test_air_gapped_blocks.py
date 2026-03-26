"""
Tests verifying that internet-requiring blocks declare air-gapped unavailability
and Roboflow model blocks declare compatible task types.
"""

import pytest


# --- Part 1: Air-gapped availability tests ---

AIR_GAPPED_BLOCK_MANIFESTS = []


def _import_manifest(module_path):
    """Import BlockManifest from a module path."""
    import importlib

    mod = importlib.import_module(module_path)
    return mod.BlockManifest


# Foundation model blocks
_FOUNDATION_BASE = "inference.core.workflows.core_steps.models.foundation"
_FOUNDATION_MODULES = [
    f"{_FOUNDATION_BASE}.anthropic_claude.v1",
    f"{_FOUNDATION_BASE}.anthropic_claude.v2",
    f"{_FOUNDATION_BASE}.anthropic_claude.v3",
    f"{_FOUNDATION_BASE}.openai.v1",
    f"{_FOUNDATION_BASE}.openai.v2",
    f"{_FOUNDATION_BASE}.openai.v3",
    f"{_FOUNDATION_BASE}.openai.v4",
    f"{_FOUNDATION_BASE}.google_gemini.v1",
    f"{_FOUNDATION_BASE}.google_gemini.v2",
    f"{_FOUNDATION_BASE}.google_gemini.v3",
    f"{_FOUNDATION_BASE}.google_vision_ocr.v1",
    f"{_FOUNDATION_BASE}.stability_ai.image_gen.v1",
    f"{_FOUNDATION_BASE}.stability_ai.inpainting.v1",
    f"{_FOUNDATION_BASE}.stability_ai.outpainting.v1",
    f"{_FOUNDATION_BASE}.lmm.v1",
    f"{_FOUNDATION_BASE}.lmm_classifier.v1",
    f"{_FOUNDATION_BASE}.llama_vision.v1",
    f"{_FOUNDATION_BASE}.segment_anything3_3d.v1",
]

# Sink blocks
_SINKS_BASE = "inference.core.workflows.core_steps.sinks"
_SINK_MODULES = [
    f"{_SINKS_BASE}.twilio.sms.v1",
    f"{_SINKS_BASE}.twilio.sms.v2",
    f"{_SINKS_BASE}.slack.notification.v1",
    f"{_SINKS_BASE}.webhook.v1",
    f"{_SINKS_BASE}.roboflow.dataset_upload.v1",
    f"{_SINKS_BASE}.roboflow.dataset_upload.v2",
    f"{_SINKS_BASE}.roboflow.custom_metadata.v1",
    f"{_SINKS_BASE}.roboflow.model_monitoring_inference_aggregator.v1",
]

ALL_AIR_GAPPED_MODULES = _FOUNDATION_MODULES + _SINK_MODULES


@pytest.mark.parametrize("module_path", ALL_AIR_GAPPED_MODULES)
def test_air_gapped_availability_exists(module_path):
    manifest = _import_manifest(module_path)
    assert hasattr(manifest, "get_air_gapped_availability"), (
        f"{module_path}.BlockManifest missing get_air_gapped_availability classmethod"
    )


@pytest.mark.parametrize("module_path", ALL_AIR_GAPPED_MODULES)
def test_air_gapped_availability_returns_correct_value(module_path):
    manifest = _import_manifest(module_path)
    result = manifest.get_air_gapped_availability()
    assert isinstance(result, dict)
    assert result["available"] is False
    assert result["reason"] == "requires_internet"


# --- Part 2: Compatible task types tests ---

_ROBOFLOW_BASE = "inference.core.workflows.core_steps.models.roboflow"

TASK_TYPE_TEST_CASES = [
    (f"{_ROBOFLOW_BASE}.object_detection.v1", ["object-detection"]),
    (f"{_ROBOFLOW_BASE}.object_detection.v2", ["object-detection"]),
    (f"{_ROBOFLOW_BASE}.instance_segmentation.v1", ["instance-segmentation"]),
    (f"{_ROBOFLOW_BASE}.instance_segmentation.v2", ["instance-segmentation"]),
    (f"{_ROBOFLOW_BASE}.keypoint_detection.v1", ["keypoint-detection"]),
    (f"{_ROBOFLOW_BASE}.keypoint_detection.v2", ["keypoint-detection"]),
    (f"{_ROBOFLOW_BASE}.multi_class_classification.v1", ["classification"]),
    (f"{_ROBOFLOW_BASE}.multi_class_classification.v2", ["classification"]),
    (f"{_ROBOFLOW_BASE}.multi_label_classification.v1", ["multi-label-classification"]),
    (f"{_ROBOFLOW_BASE}.multi_label_classification.v2", ["multi-label-classification"]),
]


@pytest.mark.parametrize(
    "module_path,expected_types",
    TASK_TYPE_TEST_CASES,
    ids=[t[0].split(".")[-2] + "_" + t[0].split(".")[-1] for t in TASK_TYPE_TEST_CASES],
)
def test_compatible_task_types_exists(module_path, expected_types):
    manifest = _import_manifest(module_path)
    assert hasattr(manifest, "get_compatible_task_types"), (
        f"{module_path}.BlockManifest missing get_compatible_task_types classmethod"
    )


@pytest.mark.parametrize(
    "module_path,expected_types",
    TASK_TYPE_TEST_CASES,
    ids=[t[0].split(".")[-2] + "_" + t[0].split(".")[-1] for t in TASK_TYPE_TEST_CASES],
)
def test_compatible_task_types_returns_correct_value(module_path, expected_types):
    manifest = _import_manifest(module_path)
    result = manifest.get_compatible_task_types()
    assert isinstance(result, list)
    assert result == expected_types
