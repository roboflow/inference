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
    from inference_models.models.pp_ocrv6.pp_ocrv6_recognition_onnx import (
        PPOCRv6RecognitionOnnx,
    )
from inference_models.models.pp_ocrv6.pp_ocrv6_recognition_utils import (
    ctc_decode,
    load_inference_config,
    preprocess_text_lines,
    resize_and_pad_text_line,
)

# The stub-session model tests need onnxruntime (the onnx-* extra); the unit-test
# CI job installs only ``[test]`` and skips them, while the backend-free utility
# tests in this module still run. Model coverage runs in the onnx-enabled jobs.
requires_onnxruntime = pytest.mark.skipif(
    not _ONNXRUNTIME_AVAILABLE,
    reason="onnxruntime is not installed (requires the onnx-* extra)",
)


class _StubOrtInput:
    name = "x"
    type = "tensor(float)"


class _StubRecognitionSession:
    """Session stub emitting logits that CTC-decode to "hi" for every image."""

    def get_inputs(self):
        return [_StubOrtInput()]

    def run(self, output_names, inputs):
        batch_size = inputs["x"].shape[0]
        # characters ["h", "i"] -> index map ["", "h", "i", " "]
        logits = np.zeros((batch_size, 4, 4), dtype=np.float32)
        for step, token_idx in enumerate([1, 0, 2, 2]):
            logits[:, step, token_idx] = 1.0
        return [logits]


def _stub_recognition_model() -> PPOCRv6RecognitionOnnx:
    return PPOCRv6RecognitionOnnx(
        session=_StubRecognitionSession(),
        input_name="x",
        image_shape=(3, 48, 320),
        characters=["h", "i"],
        device=torch.device("cpu"),
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


def test_load_inference_config_preserves_unicode_whitespace_characters(
    tmp_path: Path,
) -> None:
    # Regression: the ideographic space (U+3000) and other Unicode whitespace must
    # survive parsing. A naive line-based parser that calls str.strip() drops it and
    # silently truncates the character dictionary, corrupting every downstream decode.
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
                "  character_dict:",
                "  - A",
                '  - "　"',
                "  - B",
            ]
        ),
        encoding="utf-8",
    )

    _, characters = load_inference_config(str(config_path))

    assert characters == ["A", "　", "B"]


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


def test_preprocess_text_lines_grows_width_with_aspect_ratio() -> None:
    # Regression: a wide line must not be squashed into the config min width. The ONNX
    # graph accepts a dynamic width, and squashing to 320 distorts the glyphs.
    wide = np.full((48, 960, 3), 255, dtype=np.uint8)

    batch = preprocess_text_lines(images=[wide], target_height=48, min_width=320)

    assert batch.shape == (1, 3, 48, 960)
    assert batch.dtype == np.float32


def test_preprocess_text_lines_pads_with_zeros_after_normalization() -> None:
    # Padding must be applied after normalization (value 0.0), matching PaddleOCR.
    # Padding before normalization would map the padded region to -1.0.
    narrow = np.full((48, 96, 3), 255, dtype=np.uint8)

    batch = preprocess_text_lines(images=[narrow], target_height=48, min_width=320)

    assert batch.shape == (1, 3, 48, 320)
    assert np.allclose(batch[0, :, :, :96], 1.0)
    assert np.allclose(batch[0, :, :, 96:], 0.0)


def test_ctc_decode_removes_blanks_and_collapses_repeated_tokens() -> None:
    predictions = np.zeros((1, 6, 4), dtype=np.float32)
    for step, token in enumerate([0, 1, 1, 0, 2, 3]):
        predictions[0, step, token] = 1.0

    result = ctc_decode(predictions=predictions, characters=["A", "B"])

    assert result == [("AB ", 1.0)]


@requires_onnxruntime
def test_recognition_model_infer_runs_end_to_end_on_uint8_numpy() -> None:
    model = _stub_recognition_model()
    image = np.full((24, 96, 3), 255, dtype=np.uint8)

    result = model(image)

    assert result == ["hi"]


@requires_onnxruntime
def test_recognition_model_reads_float_tensor_on_255_scale() -> None:
    # Float tensors are read on the [0, 255] scale (matching sibling ONNX
    # models): a white (255.0) input normalizes to the +1.0 content level.
    model = _stub_recognition_model()
    image = torch.full((3, 24, 96), 255.0, dtype=torch.float32)

    pre_processed = model.pre_process(image)
    result = model(image)

    assert isinstance(pre_processed, torch.Tensor)
    assert pre_processed.max().item() > 0.99
    assert result == ["hi"]


@requires_onnxruntime
def test_recognition_model_pre_process_batches_multiple_images() -> None:
    model = _stub_recognition_model()
    images = [
        np.full((24, 96, 3), 255, dtype=np.uint8),
        np.full((48, 48, 3), 0, dtype=np.uint8),
    ]

    pre_processed = model.pre_process(images)

    assert pre_processed.shape[0] == 2
    assert pre_processed.shape[1] == 3
    assert pre_processed.shape[2] == 48
    assert model(images) == ["hi", "hi"]
