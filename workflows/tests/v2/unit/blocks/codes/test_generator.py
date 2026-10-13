"""QR code generator: readable codes, V1 sizes and colors, clear errors."""

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.codes.generator import QRCodeGeneratorBlock, qr_color
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND

DEFAULTS = {
    "error_correct": "Medium (~15% word recovery)",
    "border": 4,
    "fill_color": "BLACK",
    "back_color": "WHITE",
}


def _generate(text: str, **overrides) -> ImageData:
    image = QRCodeGeneratorBlock().run(text=text, **{**DEFAULTS, **overrides})[
        "qr_code"
    ]

    return image


def _decode(image: ImageData) -> str:
    hwc = np.ascontiguousarray(image.tensor_image.permute(1, 2, 0).numpy())
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(hwc)

    return text


@pytest.mark.parametrize(
    "text", ["https://roboflow.com", "Hello World", "zażółć 🙂", "x" * 300]
)
def test_generated_code_decodes_to_its_text(text) -> None:
    # when
    image = _generate(text)

    # then
    IMAGE_KIND.check(image)
    assert _decode(image) == text


@pytest.mark.parametrize("border", [0, 4, 10])
def test_image_is_ten_pixels_per_module_plus_border_on_the_cpu(border) -> None:
    # when
    image = _generate("A", border=border)

    # then: "A" fits version 1, which has 21 modules per side.
    side = (21 + 2 * border) * 10
    assert image.tensor_image.shape == (3, side, side)
    assert image.tensor_image.dtype == torch.uint8
    assert image.device == torch.device("cpu")
    assert image.parent.frame_id == image.root.frame_id == image.image_id


def test_higher_error_correction_holds_the_same_text_in_a_larger_code() -> None:
    # given
    text = "https://roboflow.com/workflows/qr-code-generator"

    # when
    low = _generate(
        text, error_correct="Low (~7% word recovery / highest data capacity)"
    )
    high = _generate(
        text, error_correct="High (~30% word recovery / lowest data capacity)"
    )

    # then
    assert high.width > low.width
    assert _decode(low) == _decode(high) == text


@pytest.mark.parametrize(
    "fill_color, back_color, fill_rgb, back_rgb",
    [
        ("#FF0000", "rgb(255, 255, 0)", (255, 0, 0), (255, 255, 0)),
        ("bgr(255, 0, 0)", "WHITE", (0, 0, 255), (255, 255, 255)),
        ("GREEN", "white", (0, 255, 0), (255, 255, 255)),
        ("navy", "ivory", (0, 0, 128), (255, 255, 240)),
    ],
)
def test_colors_use_the_v1_grammar_then_css_names(
    fill_color, back_color, fill_rgb, back_rgb
) -> None:
    # when
    image = _generate("colors", fill_color=fill_color, back_color=back_color)

    # then: the corner is quiet zone, the finder pattern starts at module 4.
    pixels = image.tensor_image
    assert tuple(pixels[:, 0, 0].tolist()) == back_rgb
    assert tuple(pixels[:, 45, 45].tolist()) == fill_rgb


def test_each_call_returns_a_new_image_with_its_own_pixels() -> None:
    # when
    first, second = _generate("same"), _generate("same")

    # then
    assert first.image_id != second.image_id
    assert torch.equal(first.tensor_image, second.tensor_image)
    first.tensor_image.zero_()
    assert int(second.tensor_image.max()) == 255


@pytest.mark.parametrize("border", [-1, 101, True, 2.5])
def test_border_outside_range_or_not_integer_is_rejected(border) -> None:
    # when / then
    with pytest.raises(ValueError, match="border must be"):
        _generate("A", border=border)


def test_unknown_color_names_both_grammars() -> None:
    # when / then
    with pytest.raises(ValueError, match="fill_color: invalid color 'blurple'.*CSS"):
        _generate("A", fill_color="blurple")


def test_too_much_text_names_the_limit() -> None:
    # when / then
    with pytest.raises(ValueError, match="3000 characters does not fit"):
        _generate(
            "x" * 3000,
            error_correct="High (~30% word recovery / lowest data capacity)",
        )


def test_qr_color_reads_short_hex_like_v1() -> None:
    # when
    color = qr_color("#0f0", field="fill_color")

    # then
    assert color == (0, 255, 0)
