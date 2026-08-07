import base64
import io

import numpy as np
import torch
from PIL import Image

from inference.core.models.semantic_segmentation_base import (
    SemanticSegmentationBaseOnnxRoboflowInferenceModel,
)


def _decode_b64_png(value: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))))


def _model_shell() -> SemanticSegmentationBaseOnnxRoboflowInferenceModel:
    model = SemanticSegmentationBaseOnnxRoboflowInferenceModel.__new__(
        SemanticSegmentationBaseOnnxRoboflowInferenceModel
    )
    model.class_names = ["background", "cat", "dog"]
    return model


def _synthetic_logits(h: int = 8, w: int = 10) -> np.ndarray:
    # (N=1, C=3, H, W) logits with a deterministic argmax layout:
    # class 1 in the top-left quadrant, class 2 bottom-right, else background
    logits = np.zeros((1, 3, h, w), dtype=np.float32)
    logits[0, 1, : h // 2, : w // 2] = 5.0
    logits[0, 2, h // 2 :, w // 2 :] = 5.0
    return logits


def test_legacy_make_response_masks_decode_to_resized_label_map() -> None:
    model = _model_shell()
    img_dim = (16, 20)  # (height, width) - larger than the 8x10 logits

    responses = model.make_response(_synthetic_logits(), [img_dim])

    assert len(responses) == 1
    prediction = responses[0].predictions
    seg = _decode_b64_png(prediction.segmentation_mask)
    conf = _decode_b64_png(prediction.confidence_mask)
    assert seg.shape == img_dim
    assert conf.shape == img_dim
    # nearest-neighbour resize preserves the quadrant layout exactly
    assert set(np.unique(seg).tolist()) == {0, 1, 2}
    assert seg[0, 0] == 1
    assert seg[-1, -1] == 2
    assert responses[0].image.width == img_dim[1]
    assert responses[0].image.height == img_dim[0]


def test_legacy_make_response_present_class_ids_matches_decoded_mask() -> None:
    model = _model_shell()
    img_dim = (16, 20)

    responses = model.make_response(_synthetic_logits(), [img_dim])

    prediction = responses[0].predictions
    seg = _decode_b64_png(prediction.segmentation_mask)
    assert prediction.present_class_ids == np.unique(seg).tolist()
