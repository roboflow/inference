import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.visualizations.bounding_box.v1 import (
    BoundingBoxVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.common.utils import (
    UNKNOWN_CLASS_COLOR,
    NegativeSafeColorPalette,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

_PALETTE = [sv.Color.RED, sv.Color.GREEN, sv.Color.BLUE]


def test_by_idx_paints_negative_gray_and_keeps_positive_wrap() -> None:
    palette = NegativeSafeColorPalette(colors=list(_PALETTE))

    assert palette.by_idx(-1) == UNKNOWN_CLASS_COLOR
    assert palette.by_idx(-2) == UNKNOWN_CLASS_COLOR
    assert palette.by_idx(0) == sv.Color.RED
    assert palette.by_idx(3) == sv.Color.RED


def test_by_idx_raises_on_empty_palette_for_known_class() -> None:
    palette = NegativeSafeColorPalette(colors=[])

    assert palette.by_idx(-1) == UNKNOWN_CLASS_COLOR
    with pytest.raises(ValueError, match="at least one color"):
        palette.by_idx(0)


def test_get_palette_wraps_every_construction_branch() -> None:
    default = BoundingBoxVisualizationBlockV1.getPalette("DEFAULT", 10, [])
    custom = BoundingBoxVisualizationBlockV1.getPalette("CUSTOM", 10, ["#FF0000"])
    matplotlib_palette = BoundingBoxVisualizationBlockV1.getPalette(
        "Matplotlib Viridis", 8, []
    )

    assert default.by_idx(-1) == UNKNOWN_CLASS_COLOR
    assert default.by_idx(0) == sv.ColorPalette.DEFAULT.by_idx(0)
    assert custom.by_idx(-1) == UNKNOWN_CLASS_COLOR
    assert custom.by_idx(0) == sv.Color.from_hex("#FF0000")
    assert matplotlib_palette.by_idx(-1) == UNKNOWN_CLASS_COLOR
    assert matplotlib_palette.by_idx(0) == sv.ColorPalette.from_matplotlib(
        "viridis", 8
    ).by_idx(0)


def test_bounding_box_keeps_known_class_color_and_paints_unknown_gray() -> None:
    default = sv.ColorPalette.DEFAULT
    known_bgr = default.by_idx(0).as_bgr()
    wrapped_bgr = default.by_idx(len(default.colors) - 1).as_bgr()
    gray_bgr = UNKNOWN_CLASS_COLOR.as_bgr()

    output = BoundingBoxVisualizationBlockV1().run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((48, 80, 3), dtype=np.uint8),
        ),
        predictions=sv.Detections(
            xyxy=np.array([[2, 2, 22, 22], [50, 2, 70, 22]], dtype=np.float64),
            class_id=np.array([0, -1]),
        ),
        copy_image=True,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=[],
        color_axis="CLASS",
        thickness=1,
        roundness=0,
    )

    annotated = output["image"].numpy_image
    known_region = annotated[2:23, 2:23]
    unknown_region = annotated[2:23, 50:71]

    assert np.any(np.all(known_region == known_bgr, axis=-1))
    assert np.any(np.all(unknown_region == gray_bgr, axis=-1))
    assert not np.any(np.all(unknown_region == known_bgr, axis=-1))
    assert not np.any(np.all(unknown_region == wrapped_bgr, axis=-1))
    assert not np.any(np.all(known_region == gray_bgr, axis=-1))
