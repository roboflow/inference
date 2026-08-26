import supervision as sv

from inference.core.workflows.core_steps.visualizations.bounding_box.v1 import (
    BoundingBoxVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.common.utils import (
    UNKNOWN_CLASS_COLOR,
    NegativeSafeColorPalette,
    wrap_color_palette,
)


def test_negative_index_returns_gray() -> None:
    palette = NegativeSafeColorPalette(
        colors=[sv.Color.RED, sv.Color.GREEN, sv.Color.BLUE]
    )

    assert palette.by_idx(-1) == UNKNOWN_CLASS_COLOR
    assert palette.by_idx(-2) == UNKNOWN_CLASS_COLOR


def test_non_negative_index_uses_palette_wrap() -> None:
    palette = NegativeSafeColorPalette(
        colors=[sv.Color.RED, sv.Color.GREEN, sv.Color.BLUE]
    )

    assert palette.by_idx(0) == sv.Color.RED
    assert palette.by_idx(1) == sv.Color.GREEN
    assert palette.by_idx(2) == sv.Color.BLUE
    assert palette.by_idx(3) == sv.Color.RED


def test_wrap_color_palette_preserves_named_palette_colors() -> None:
    wrapped = wrap_color_palette(sv.ColorPalette.DEFAULT)

    assert isinstance(wrapped, NegativeSafeColorPalette)
    assert wrapped.by_idx(0) == sv.ColorPalette.DEFAULT.by_idx(0)
    assert wrapped.by_idx(-1) == UNKNOWN_CLASS_COLOR


def test_wrap_color_palette_is_idempotent() -> None:
    palette = NegativeSafeColorPalette(colors=[sv.Color.RED])

    assert wrap_color_palette(palette) is palette


def test_get_palette_returns_gray_for_negative_index_on_every_branch() -> None:
    default = BoundingBoxVisualizationBlockV1.getPalette("DEFAULT", 10, [])
    custom = BoundingBoxVisualizationBlockV1.getPalette(
        "CUSTOM", 10, ["#FF0000", "#00FF00"]
    )
    matplotlib_palette = BoundingBoxVisualizationBlockV1.getPalette(
        "Matplotlib Viridis", 8, []
    )

    for palette in (default, custom, matplotlib_palette):
        assert isinstance(palette, NegativeSafeColorPalette)
        assert palette.by_idx(-1) == UNKNOWN_CLASS_COLOR
        assert palette.by_idx(0) != UNKNOWN_CLASS_COLOR
