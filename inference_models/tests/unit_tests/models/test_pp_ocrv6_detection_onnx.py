from pathlib import Path

import numpy as np
import pytest

from inference_models.models.pp_ocrv6.pp_ocrv6_detection_utils import (
    DBNetConfig,
    boxes_from_probability_map,
    load_detection_config,
    normalize_detection_image,
    resize_for_detection,
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
