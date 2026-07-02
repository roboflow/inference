from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

try:
    import onnxruntime  # noqa: F401

    _ONNXRUNTIME_AVAILABLE = True
except ImportError:
    _ONNXRUNTIME_AVAILABLE = False

if _ONNXRUNTIME_AVAILABLE:
    from inference_models.models.pp_ocrv6.pp_ocrv6_detection_onnx import (
        PPOCRv6DetectionOnnx,
    )
from inference_models.models.pp_ocrv6.pp_ocrv6_detection_utils import (
    DBNetConfig,
    boxes_from_probability_map,
    load_detection_config,
    normalize_detection_image,
    resize_for_detection,
)

# The stub-session model tests below import the ONNX model class, which requires
# ``onnxruntime`` (the ``onnx-*`` extra). The unit-test CI job installs only
# ``[test]``, so those tests are skipped there while the backend-free utility
# tests in this module still run. Model coverage runs in the onnx-enabled jobs.
requires_onnxruntime = pytest.mark.skipif(
    not _ONNXRUNTIME_AVAILABLE,
    reason="onnxruntime is not installed (requires the onnx-* extra)",
)


class _StubOrtInput:
    name = "x"
    type = "tensor(float)"


class _StubDetectionSession:
    """Session stub emitting a probability map with one high-confidence region."""

    def get_inputs(self):
        return [_StubOrtInput()]

    def run(self, output_names, inputs):
        _, _, height, width = inputs["x"].shape
        probability_map = np.zeros((1, 1, height, width), dtype=np.float32)
        probability_map[:, :, height // 4 : height // 2, width // 8 : -width // 8] = 1.0
        return [probability_map]


def _stub_detection_model() -> PPOCRv6DetectionOnnx:
    return PPOCRv6DetectionOnnx(
        session=_StubDetectionSession(),
        input_name="x",
        config=_config(limit_side_len=64),
        device=torch.device("cpu"),
    )


def _config(**overrides) -> DBNetConfig:
    defaults = dict(
        limit_side_len=736,
        limit_type="min",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        scale=1.0 / 255.0,
        to_rgb=False,
        thresh=0.2,
        box_thresh=0.45,
        unclip_ratio=1.4,
        max_candidates=3000,
    )
    defaults.update(overrides)
    return DBNetConfig(**defaults)


def test_load_detection_config_parses_preprocess_and_postprocess(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "inference.yml"
    config_path.write_text(
        "\n".join(
            [
                "PostProcess:",
                "  box_thresh: 0.45",
                "  max_candidates: 3000",
                "  name: DBPostProcess",
                "  thresh: 0.2",
                "  unclip_ratio: 1.4",
                "PreProcess:",
                "  transform_ops:",
                "  - DecodeImage:",
                "      img_mode: BGR",
                "  - DetResizeForTest: null",
                "  - NormalizeImage:",
                "      mean:",
                "      - 0.485",
                "      - 0.456",
                "      - 0.406",
                "      order: hwc",
                "      scale: 1./255.",
                "      std:",
                "      - 0.229",
                "      - 0.224",
                "      - 0.225",
            ]
        ),
        encoding="utf-8",
    )

    config = load_detection_config(str(config_path))

    assert config.thresh == pytest.approx(0.2)
    assert config.box_thresh == pytest.approx(0.45)
    assert config.unclip_ratio == pytest.approx(1.4)
    assert config.max_candidates == 3000
    assert config.mean == (0.485, 0.456, 0.406)
    assert config.std == (0.229, 0.224, 0.225)
    assert config.scale == pytest.approx(1.0 / 255.0)
    assert config.to_rgb is False
    # DetResizeForTest is null -> defaults are used
    assert config.limit_side_len == 736
    assert config.limit_type == "min"


def test_resize_for_detection_rounds_to_multiple_of_32() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    resized, ratio_h, ratio_w = resize_for_detection(
        image=image, limit_side_len=736, limit_type="min"
    )

    assert resized.shape[0] % 32 == 0
    assert resized.shape[1] % 32 == 0
    # min side (480) is scaled up to 736
    assert resized.shape[0] == 736
    assert ratio_h == pytest.approx(736 / 480)
    assert ratio_w == pytest.approx(resized.shape[1] / 640)


def test_normalize_detection_image_produces_nchw_float() -> None:
    image = np.full((64, 96, 3), 255, dtype=np.uint8)

    result = normalize_detection_image(image_bgr=image, config=_config())

    assert result.shape == (1, 3, 64, 96)
    assert result.dtype == np.float32


def test_boxes_from_probability_map_recovers_rectangle_in_source_coords() -> None:
    probability_map = np.zeros((100, 200), dtype=np.float32)
    probability_map[30:60, 40:160] = 1.0

    boxes = boxes_from_probability_map(
        probability_map=probability_map,
        source_height=200,
        source_width=400,
        config=_config(),
    )

    assert len(boxes) == 1
    quad, score = boxes[0]
    assert quad.shape == (4, 2)
    assert score > 0.9
    # source is 2x the probability-map resolution on both axes
    assert quad[:, 0].max() <= 400
    assert quad[:, 1].max() <= 200
    assert quad[:, 0].max() > 200
    assert quad[:, 1].max() > 100


def test_boxes_from_probability_map_returns_empty_for_blank_map() -> None:
    probability_map = np.zeros((100, 200), dtype=np.float32)

    boxes = boxes_from_probability_map(
        probability_map=probability_map,
        source_height=100,
        source_width=200,
        config=_config(),
    )

    assert boxes == []


def test_boxes_from_probability_map_drops_sub_line_sized_boxes() -> None:
    # A blob that clears the bitmap-coordinate size checks but scales down to a
    # few source pixels must be dropped, matching PaddleOCR's filter_tag_det_res
    # (width/height <= 3px in the original image). Otherwise sub-character blobs
    # surface as spurious non-text detections.
    probability_map = np.zeros((100, 100), dtype=np.float32)
    probability_map[30:70, 30:70] = 1.0

    kept = boxes_from_probability_map(
        probability_map=probability_map,
        source_height=200,
        source_width=200,
        config=_config(),
    )
    dropped = boxes_from_probability_map(
        probability_map=probability_map,
        source_height=4,
        source_width=4,
        config=_config(),
    )

    assert len(kept) == 1  # large source: real box survives
    assert dropped == []  # tiny source: box (<=3px) filtered as sub-line-sized


@requires_onnxruntime
def test_detection_model_infer_runs_end_to_end_on_uint8_numpy() -> None:
    model = _stub_detection_model()
    image = np.full((64, 128, 3), 255, dtype=np.uint8)

    detections = model(image)

    assert len(detections) == 1
    assert len(detections[0].xyxy) == 1
    assert model.class_names == ["text"]
    assert detections[0].confidence[0].item() > 0.9
    assert "polygon" in detections[0].bboxes_metadata[0]


@requires_onnxruntime
def test_detection_model_infer_handles_unit_range_float_tensor() -> None:
    # Regression: float [0, 1] tensors were previously scaled by 1/255 twice and
    # the detector saw a near-black image, returning garbage with no error.
    model = _stub_detection_model()
    image_uint8 = np.full((64, 128, 3), 255, dtype=np.uint8)
    image_tensor = torch.ones((3, 64, 128), dtype=torch.float32)

    pre_processed_uint8, _ = model.pre_process(image_uint8)
    pre_processed_float, _ = model.pre_process(image_tensor)
    detections = model(image_tensor)

    assert torch.allclose(pre_processed_uint8[0], pre_processed_float[0], atol=0.02)
    assert len(detections[0].xyxy) == 1


@requires_onnxruntime
def test_detection_model_post_process_handles_empty_maps() -> None:
    model = _stub_detection_model()
    empty_map = torch.zeros((64, 64), dtype=torch.float32)

    detections = model.post_process(
        model_results=[empty_map],
        pre_processing_meta=[{"source_height": 64, "source_width": 64}],
    )

    assert len(detections) == 1
    assert len(detections[0].xyxy) == 0
    assert detections[0].bboxes_metadata == []
