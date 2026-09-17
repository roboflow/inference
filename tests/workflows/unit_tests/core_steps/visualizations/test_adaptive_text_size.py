import pytest

from inference.core.workflows.core_steps.visualizations.common.label_text import (
    REFERENCE_LABEL_TEXT_SCALE,
    REFERENCE_MIN_DIMENSION_PX,
    REFERENCE_RICH_FONT_SIZE_PT,
    TEXT_SIZE_MODE_AUTOMATIC,
    TEXT_SIZE_MODE_MANUAL,
    compute_adaptive_label_text_scale,
    compute_adaptive_rich_font_size,
)


@pytest.mark.parametrize(
    "height,width,manual,expected",
    [
        (1080, 1920, 28, 28),
        (540, 960, 14, 14),
    ],
)
def test_compute_adaptive_rich_font_size_manual_mode(
    height: int, width: int, manual: int, expected: int
) -> None:
    assert (
        compute_adaptive_rich_font_size(
            height, width, manual_font_size=manual, text_size_mode=TEXT_SIZE_MODE_MANUAL
        )
        == expected
    )


def test_compute_adaptive_rich_font_size_automatic_mode() -> None:
    # at the reference resolution, the manual value acts as a multiplier base
    assert (
        compute_adaptive_rich_font_size(
            REFERENCE_MIN_DIMENSION_PX,
            1920,
            manual_font_size=REFERENCE_RICH_FONT_SIZE_PT,
            text_size_mode=TEXT_SIZE_MODE_AUTOMATIC,
        )
        == REFERENCE_RICH_FONT_SIZE_PT
    )
    assert (
        compute_adaptive_rich_font_size(
            REFERENCE_MIN_DIMENSION_PX,
            1920,
            manual_font_size=REFERENCE_RICH_FONT_SIZE_PT * 2,
            text_size_mode=TEXT_SIZE_MODE_AUTOMATIC,
        )
        == REFERENCE_RICH_FONT_SIZE_PT * 2
    )


@pytest.mark.parametrize(
    "height,width,manual,expected",
    [
        (1080, 1920, 0.5, 0.5),
        (540, 960, 1.0, 1.0),
    ],
)
def test_compute_adaptive_label_text_scale_manual_mode(
    height: int, width: int, manual: float, expected: float
) -> None:
    assert (
        compute_adaptive_label_text_scale(
            height,
            width,
            manual_text_scale=manual,
            text_size_mode=TEXT_SIZE_MODE_MANUAL,
        )
        == expected
    )


def test_compute_adaptive_label_text_scale_automatic_at_reference_resolution() -> None:
    assert compute_adaptive_label_text_scale(
        REFERENCE_MIN_DIMENSION_PX,
        1920,
        manual_text_scale=REFERENCE_LABEL_TEXT_SCALE,
        text_size_mode=TEXT_SIZE_MODE_AUTOMATIC,
    ) == pytest.approx(REFERENCE_LABEL_TEXT_SCALE)
