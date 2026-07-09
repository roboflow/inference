import cv2
import numpy as np
import pytest
import torch

# The PP-OCRv6 model classes require onnxruntime (the onnx-* extra). Import them
# lazily inside each test — like the other `*_predictions_onnx.py` integration
# tests — so this module still collects in the torch-only jobs (e.g.
# `torch_models`), where these `onnx_extras`-marked tests are deselected.

TEXT_LINES = ["hello world", "goodbye 123"]


def _render_text_image() -> np.ndarray:
    image = np.full((200, 640, 3), 255, dtype=np.uint8)
    for index, line in enumerate(TEXT_LINES):
        cv2.putText(
            image,
            line,
            (40, 70 + index * 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
    return image


def _crops_in_reading_order(image: np.ndarray, detections) -> list:
    ordered = sorted(detections.xyxy.tolist(), key=lambda box: (box[1], box[0]))
    crops = []
    for x_min, y_min, x_max, y_max in ordered:
        crops.append(image[int(y_min) : int(y_max), int(x_min) : int(x_max)])
    return crops


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_pp_ocrv6_detection_finds_text_lines(
    pp_ocrv6_tiny_det_onnx_package: str,
) -> None:
    from inference_models.models.pp_ocrv6.pp_ocrv6_detection_onnx import (
        PPOCRv6DetectionOnnx,
    )

    model = PPOCRv6DetectionOnnx.from_pretrained(
        pp_ocrv6_tiny_det_onnx_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )
    image = _render_text_image()

    detections = model(image)[0]

    assert len(detections.xyxy) == len(TEXT_LINES)
    assert detections.confidence.min().item() > 0.5
    assert all("polygon" in meta for meta in detections.bboxes_metadata)


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_pp_ocrv6_recognition_reads_text_line(
    pp_ocrv6_tiny_rec_onnx_package: str,
) -> None:
    from inference_models.models.pp_ocrv6.pp_ocrv6_recognition_onnx import (
        PPOCRv6RecognitionOnnx,
    )

    model = PPOCRv6RecognitionOnnx.from_pretrained(
        pp_ocrv6_tiny_rec_onnx_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )
    image = _render_text_image()

    texts = model(image[30:90, 20:400])

    assert texts == ["hello world"]


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_pp_ocrv6_recognition_float_tensor_matches_uint8(
    pp_ocrv6_tiny_rec_onnx_package: str,
) -> None:
    # Package-wide input contract: float images are assumed to already be on
    # the [0, 255] scale, so a float tensor must match its uint8 counterpart.
    from inference_models.models.pp_ocrv6.pp_ocrv6_recognition_onnx import (
        PPOCRv6RecognitionOnnx,
    )

    model = PPOCRv6RecognitionOnnx.from_pretrained(
        pp_ocrv6_tiny_rec_onnx_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )
    crop_bgr = _render_text_image()[30:90, 20:400]
    crop_tensor = (
        torch.from_numpy(np.ascontiguousarray(crop_bgr[:, :, ::-1]))
        .permute(2, 0, 1)
        .float()
    )

    assert model(crop_tensor) == model(crop_bgr) == ["hello world"]


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_pp_ocrv6_two_stage_pipeline_transcribes_all_lines(
    pp_ocrv6_tiny_det_onnx_package: str,
    pp_ocrv6_tiny_rec_onnx_package: str,
) -> None:
    from inference_models.models.pp_ocrv6.pp_ocrv6_detection_onnx import (
        PPOCRv6DetectionOnnx,
    )
    from inference_models.models.pp_ocrv6.pp_ocrv6_recognition_onnx import (
        PPOCRv6RecognitionOnnx,
    )

    detector = PPOCRv6DetectionOnnx.from_pretrained(
        pp_ocrv6_tiny_det_onnx_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )
    recognizer = PPOCRv6RecognitionOnnx.from_pretrained(
        pp_ocrv6_tiny_rec_onnx_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )
    image = _render_text_image()

    detections = detector(image)[0]
    crops = _crops_in_reading_order(image=image, detections=detections)
    texts = recognizer(crops)

    assert texts == TEXT_LINES
