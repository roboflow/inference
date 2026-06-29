from pathlib import Path

import numpy as np

from inference_models.models.pp_ocrv6.pp_ocrv6_recognition_utils import (
    ctc_decode,
    load_inference_config,
    resize_and_pad_text_line,
)


def test_load_inference_config_parses_shape_and_characters(tmp_path: Path) -> None:
    config_path = tmp_path / "inference.yml"
    config_path.write_text(
        "\n".join(
            [
                "PreProcess:",
                "  transform_ops:",
                "  - RecResizeImg:",
                "      image_shape:",
                "      - 3",
                "      - 48",
                "      - 320",
                "PostProcess:",
                "  name: CTCLabelDecode",
                "  character_dict:",
                "  - A",
                "  - '\"'",
                "  - ''''",
            ]
        )
    )

    image_shape, characters = load_inference_config(str(config_path))

    assert image_shape == (3, 48, 320)
    assert characters == ["A", '"', "'"]


def test_resize_and_pad_text_line_returns_normalized_nchw_image() -> None:
    image = np.full((24, 80, 3), 255, dtype=np.uint8)

    result = resize_and_pad_text_line(
        image=image,
        target_height=48,
        target_width=320,
    )

    assert result.shape == (3, 48, 320)
    assert result.dtype == np.float32
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_ctc_decode_removes_blanks_and_collapses_repeated_tokens() -> None:
    predictions = np.zeros((1, 6, 4), dtype=np.float32)
    for step, token in enumerate([0, 1, 1, 0, 2, 3]):
        predictions[0, step, token] = 1.0

    result = ctc_decode(predictions=predictions, characters=["A", "B"])

    assert result == [("AB ", 1.0)]
