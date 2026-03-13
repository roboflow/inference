import base64
import io

import numpy as np
import pytest
import supervision as sv
from PIL import Image
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v1 import (
    BlockManifest,
    RoboflowSemanticSegmentationModelBlockV1,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_semantic_segmentation_model_validation_when_minimalistic_config_is_provided(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_semantic_segmentation_model@v1",
        "name": "some",
        images_field_alias: "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/roboflow_semantic_segmentation_model@v1",
        name="some",
        images="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "images", "model_id"])
def test_semantic_segmentation_model_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_semantic_segmentation_model@v1",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_semantic_segmentation_model_validation_when_invalid_type_provided() -> None:
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_semantic_segmentation_model_validation_when_model_id_has_invalid_type() -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_semantic_segmentation_model@v1",
        "name": "some",
        "images": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_semantic_segmentation_model_validation_when_images_selector_has_invalid_type() -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_semantic_segmentation_model@v1",
        "name": "some",
        "images": "some",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


# --- _convert_to_sv_detections tests ---


def _encode_mask_as_base64_png(mask_array: np.ndarray) -> str:
    pil_img = Image.fromarray(mask_array.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_convert_to_sv_detections_produces_per_class_detections() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    seg[10:40, 10:40] = 1
    seg[60:90, 60:90] = 2

    result = RoboflowSemanticSegmentationModelBlockV1._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "class_map": {"1": "cat", "2": "dog"},
        }
    )

    assert isinstance(result, sv.Detections)
    assert len(result) == 2
    assert set(result.class_id.tolist()) == {1, 2}
    assert result.mask is not None
    assert result.mask.shape == (2, 100, 100)
    assert set(result.data["class_name"].tolist()) == {"cat", "dog"}
    # no confidence mask → defaults to 1.0
    assert result.confidence is not None
    assert (result.confidence == 1.0).all()


def test_convert_to_sv_detections_derives_confidence_from_mask() -> None:
    seg = np.zeros((50, 50), dtype=np.uint8)
    seg[10:40, 10:40] = 1
    conf = np.full((50, 50), 200, dtype=np.uint8)  # 200/255 ≈ 0.784

    result = RoboflowSemanticSegmentationModelBlockV1._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "confidence_mask": _encode_mask_as_base64_png(conf),
            "class_map": {"1": "cat"},
        }
    )

    assert "confidence_mask" in result.data
    assert result.data["confidence_mask"].shape == (50, 50)
    assert result.confidence is not None
    assert abs(float(result.confidence[0]) - 200 / 255.0) < 0.01


def test_convert_to_sv_detections_empty_when_all_background() -> None:
    seg = np.zeros((50, 50), dtype=np.uint8)

    result = RoboflowSemanticSegmentationModelBlockV1._convert_to_sv_detections(
        {"segmentation_mask": _encode_mask_as_base64_png(seg), "class_map": {}}
    )

    assert len(result) == 0
