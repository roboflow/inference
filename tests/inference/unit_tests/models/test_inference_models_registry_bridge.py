import pytest

from inference.core.env import USE_INFERENCE_MODELS
from inference.models.utils import ROBOFLOW_MODEL_TYPES


@pytest.mark.skipif(
    not USE_INFERENCE_MODELS,
    reason="Registry bridge only applies when USE_INFERENCE_MODELS is enabled.",
)
def test_inference_models_registry_pairs_are_registered_in_outer_lookup() -> None:
    expected_pairs = {
        ("classification", "resnet"),
        ("multi-label-classification", "dinov3_probe"),
        ("multi-label-classification", "resnet"),
        ("multi-label-classification", "vit"),
        ("instance-segmentation", "segment-anything-2-rt"),
        ("semantic-segmentation", "deep-lab-v3-plus"),
    }

    missing_pairs = expected_pairs.difference(ROBOFLOW_MODEL_TYPES)

    assert missing_pairs == set()


@pytest.mark.skipif(
    not USE_INFERENCE_MODELS,
    reason="Registry bridge only applies when USE_INFERENCE_MODELS is enabled.",
)
def test_inference_models_registry_bridge_preserves_legacy_outer_aliases() -> None:
    expected_legacy_pairs = {
        ("classification", "resnet50"),
        ("classification", "vit"),
        ("object-detection", "rfdetr-nano"),
        ("semantic-segmentation", "deeplabv3plus"),
    }

    missing_pairs = expected_legacy_pairs.difference(ROBOFLOW_MODEL_TYPES)

    assert missing_pairs == set()
