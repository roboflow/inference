import base64
import io
import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from inference.core.entities.requests.inference import (
    ObjectDetectionInferenceRequest,
    SemanticSegmentationInferenceRequest,
    ensure_wire_safe_mask_format,
)
from inference.core.entities.responses.inference import (
    SemanticSegmentationPrediction,
)


def _decode_b64_png(value: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))))


def _encode_b64_png(mask: np.ndarray) -> str:
    img = Image.fromarray(mask)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


def _random_masks():
    rng = np.random.default_rng(11)
    seg = rng.choice(np.array([0, 3, 7], dtype=np.uint8), size=(48, 64))
    conf = rng.integers(0, 256, size=(48, 64), dtype=np.uint8)
    return seg, conf


def test_request_defaults_to_base64_png_mask_format() -> None:
    request = SemanticSegmentationInferenceRequest(image=[], model_id="some/1")

    assert request.response_mask_format == "base64_png"


def test_request_rejects_unknown_mask_format() -> None:
    with pytest.raises(ValidationError):
        SemanticSegmentationInferenceRequest(
            image=[], model_id="some/1", response_mask_format="tiff"
        )


def test_prediction_python_dump_passes_numpy_masks_through() -> None:
    seg, conf = _random_masks()
    prediction = SemanticSegmentationPrediction(
        segmentation_mask=seg,
        confidence_mask=conf,
        class_map={"3": "cat", "7": "dog"},
    )

    dumped = prediction.model_dump(by_alias=True, exclude_none=True)

    assert dumped["segmentation_mask"] is seg
    assert dumped["confidence_mask"] is conf


def test_prediction_json_dump_lazily_encodes_numpy_masks_to_base64_png() -> None:
    seg, conf = _random_masks()
    prediction = SemanticSegmentationPrediction(
        segmentation_mask=seg,
        confidence_mask=conf,
        class_map={"3": "cat", "7": "dog"},
    )

    serialized = json.loads(prediction.model_dump_json())

    assert isinstance(serialized["segmentation_mask"], str)
    assert np.array_equal(_decode_b64_png(serialized["segmentation_mask"]), seg)
    assert np.array_equal(_decode_b64_png(serialized["confidence_mask"]), conf)


def test_prediction_json_dump_passes_string_masks_through_unchanged() -> None:
    seg, conf = _random_masks()
    seg_b64, conf_b64 = _encode_b64_png(seg), _encode_b64_png(conf)
    prediction = SemanticSegmentationPrediction(
        segmentation_mask=seg_b64,
        confidence_mask=conf_b64,
        class_map={},
    )

    serialized = json.loads(prediction.model_dump_json())

    assert serialized["segmentation_mask"] == seg_b64
    assert serialized["confidence_mask"] == conf_b64


def test_ensure_wire_safe_mask_format_coerces_numpy() -> None:
    request = SemanticSegmentationInferenceRequest(
        image=[], model_id="some/1", response_mask_format="numpy"
    )

    ensure_wire_safe_mask_format(request)

    assert request.response_mask_format == "base64_png"


def test_ensure_wire_safe_mask_format_keeps_default_untouched() -> None:
    request = SemanticSegmentationInferenceRequest(image=[], model_id="some/1")

    ensure_wire_safe_mask_format(request)

    assert request.response_mask_format == "base64_png"


def test_ensure_wire_safe_mask_format_is_a_noop_for_other_request_types() -> None:
    request = ObjectDetectionInferenceRequest(image=[], model_id="some/1")

    ensure_wire_safe_mask_format(request)  # must not raise

    assert not hasattr(request, "response_mask_format")
