"""The four decorative blocks in compiled V2 workflows fed by a real model block.

A fake shapes model (``scene.py``) stands in for loaded weights; the instance
segmentation, crop and gate blocks are the real V2 blocks::

    instance seg ─▶ background color ─▶ color ─▶ icon (per object) ─▶ icon (watermark)
    input, background, color ─▶ grid                  one canvas per image
    instance seg ─▶ dynamic crop ─▶ background on each crop (parent rows)
    instance seg, icon, icon_alpha (tensor input) ─▶ transparent icon per object
    one logo + alpha parameter ─▶ icon (static) at each image's own corner

Pixel exactness against V1 is the unit suites' job. These tests check what a
workflow builder relies on: blocks chain, ragged batches keep their rows,
selected values behave like literals, gates skip images, inputs stay
unchanged, and parent predictions are clipped to a crop.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.visualization.annotations.background_color_visualization.v1 import BackgroundColorVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.color_visualization.v1 import ColorVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.grid_visualization.v1 import GridVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.icon_visualization.v1 import IconVisualizationBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.integration.blocks.visualization.annotations.scene import (
    ALLEY,
    ALLEY_SIZE,
    OBJECT_COLORS,
    STREET,
    STREET_SIZE,
    SceneLoader,
    alley_image,
    annotation_catalogue,
    changed,
    pixels_of,
    shape_mask,
    street_image,
)

MODES = ("run", "phases")
DECORATIVE_BLOCKS = (
    BackgroundColorVisualizationBlock,
    ColorVisualizationBlock,
    GridVisualizationBlock,
    IconVisualizationBlock,
)
SEG = {
    "type": "roboflow_core/roboflow_instance_segmentation_model@v4",
    "name": "seg",
    "image": "$inputs.image",
    "model_id": "shapes-seg",
}
CROP = {
    "type": "roboflow_core/dynamic_crop@v1",
    "name": "crop",
    "image": "$inputs.image",
    "predictions": "$steps.seg.predictions",
}
GREEN, MAGENTA = (0, 255, 0), (255, 0, 255)
ICON_COLOR = (250, 200, 0)


def run_rows(
    steps: List[Dict[str, Any]],
    outputs: Dict[str, str],
    images: Sequence[ImageData],
    *,
    inputs: Sequence[Dict[str, Any]] = (),
    values: Optional[Dict[str, Any]] = None,
    mode: str = "run",
) -> List[Dict[str, Any]]:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}, *inputs],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    catalogue = annotation_catalogue().with_blocks(DECORATIVE_BLOCKS)
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session({"models": ModelCache(SceneLoader())})
    rows = session.run({"image": list(images), **(values or {})}).rows()

    return rows


def union_of(shapes, size_hw) -> np.ndarray:
    union = np.zeros(size_hw, dtype=bool)
    for shape in shapes:
        union |= shape_mask(shape, size_hw)

    return union


def solid_icon(count: int) -> List[ImageData]:
    pixels = np.empty((4, 4, 3), dtype=np.uint8)
    pixels[:] = ICON_COLOR

    return [
        ImageData.from_numpy_rgb(pixels, image_id=f"icon-{k}") for k in range(count)
    ]


# --- chained drawings over a ragged batch --------------------------------------


DECORATION = [
    SEG,
    {
        "type": "roboflow_core/background_color_visualization@v1",
        "name": "background",
        "image": "$inputs.image",
        "predictions": "$steps.seg.segmentation",
        "color": "WHITE",
        "opacity": 1.0,
    },
    {
        "type": "roboflow_core/color_visualization@v1",
        "name": "color",
        "image": "$steps.background.image",
        "predictions": "$steps.seg.segmentation",
        "color_palette": "CUSTOM",
        "custom_colors": ["#00FF00", "#FF00FF"],
        "opacity": 1.0,
    },
    {
        "type": "roboflow_core/icon_visualization@v1",
        "name": "marks",
        "image": "$steps.color.image",
        "icon": "$inputs.icon",
        "predictions": "$steps.seg.segmentation",
        "position": "CENTER",
        "icon_width": 4,
        "icon_height": 4,
    },
    {
        "type": "roboflow_core/icon_visualization@v1",
        "name": "watermark",
        "image": "$steps.marks.image",
        "icon": "$inputs.icon",
        "mode": "static",
        "icon_width": 4,
        "icon_height": 4,
        "x_position": -2,
        "y_position": -2,
    },
]
DECORATION_OUTPUTS = {
    "background": "$steps.background.image",
    "color": "$steps.color.image",
    "marks": "$steps.marks.image",
    "watermark": "$steps.watermark.image",
}


@pytest.mark.parametrize("mode", MODES)
def test_background_color_and_icons_chain_over_a_ragged_batch(mode: str) -> None:
    images = [street_image(), alley_image()]

    rows = run_rows(
        DECORATION,
        DECORATION_OUTPUTS,
        images,
        inputs=[{"type": "WorkflowImage", "name": "icon"}],
        values={"icon": solid_icon(2)},
        mode=mode,
    )

    for image, row, shapes, size_hw in zip(
        images, rows, (STREET, ALLEY), (STREET_SIZE, ALLEY_SIZE)
    ):
        height, width = size_hw
        # Background: the objects keep their pixels, the rest turns white.
        kept = union_of(shapes, size_hw)
        background = pixels_of(row["background"])
        assert np.array_equal(background[kept], pixels_of(image)[kept])
        assert (background[~kept] == 255).all()
        # Color: each object's box is filled with its class color.
        color = pixels_of(row["color"])
        for shape in shapes:
            x0, y0, x1, y1 = shape.xyxy
            class_id, _ = OBJECT_COLORS[shape.color]
            class_color = (GREEN, MAGENTA)[class_id]
            assert (color[y0:y1, x0:x1] == class_color).all()
        # Marks: one icon centred on each box.
        marks = pixels_of(row["marks"])
        for shape in shapes:
            x0, y0, x1, y1 = shape.xyxy
            centre_x, centre_y = (x0 + x1) // 2, (y0 + y1) // 2
            assert (
                marks[centre_y - 2 : centre_y + 2, centre_x - 2 : centre_x + 2]
                == ICON_COLOR
            ).all()
        # Watermark: 2 px from the bottom-right corner.
        watermark = pixels_of(row["watermark"])
        assert (
            watermark[height - 6 : height - 2, width - 6 : width - 2] == ICON_COLOR
        ).all()
        assert row["watermark"].image_id == image.image_id


def test_selected_style_values_draw_like_the_same_literals() -> None:
    literal = {
        "type": "roboflow_core/color_visualization@v1",
        "name": "fill",
        "image": "$inputs.image",
        "predictions": "$steps.seg.segmentation",
        "color_axis": "INDEX",
        "opacity": 0.5,
    }
    selected = {**literal, "color_axis": "$inputs.axis", "opacity": "$inputs.opacity"}
    parameters = [
        {"type": "WorkflowParameter", "name": "axis"},
        {"type": "WorkflowParameter", "name": "opacity"},
    ]

    (by_literal,) = run_rows(
        [SEG, literal], {"fill": "$steps.fill.image"}, [street_image()]
    )
    (by_selector,) = run_rows(
        [SEG, selected],
        {"fill": "$steps.fill.image"},
        [street_image()],
        inputs=parameters,
        values={"axis": "INDEX", "opacity": 0.5},
    )

    assert np.array_equal(pixels_of(by_literal["fill"]), pixels_of(by_selector["fill"]))
    assert changed(street_image(), by_literal["fill"]).any()


def test_rows_pick_colors_by_class_or_by_position() -> None:
    fill = {
        "type": "roboflow_core/color_visualization@v1",
        "name": "fill",
        "image": "$inputs.image",
        "predictions": "$steps.seg.segmentation",
        "color_palette": "CUSTOM",
        "custom_colors": ["#00FF00", "#FF00FF"],
        "color_axis": "$inputs.axis",
        "opacity": 1.0,
    }

    def centre_colors(axis: str) -> List[tuple]:
        (row,) = run_rows(
            [SEG, fill],
            {"fill": "$steps.fill.image"},
            [street_image()],
            inputs=[{"type": "WorkflowParameter", "name": "axis"}],
            values={"axis": axis},
        )
        pixels = pixels_of(row["fill"])
        return [
            tuple(pixels[(s.xyxy[1] + s.xyxy[3]) // 2, (s.xyxy[0] + s.xyxy[2]) // 2])
            for s in STREET
        ]

    # Classes: red is class 0, both cars class 1. Positions: rows 0, 1, 2.
    assert centre_colors("CLASS") == [GREEN, MAGENTA, MAGENTA]
    assert centre_colors("INDEX") == [GREEN, MAGENTA, GREEN]


def test_a_selected_invalid_opacity_fails_the_row_with_the_literal_text() -> None:
    step = {
        "type": "roboflow_core/background_color_visualization@v1",
        "name": "background",
        "image": "$inputs.image",
        "predictions": "$steps.seg.segmentation",
        "opacity": "$inputs.opacity",
    }

    with pytest.raises(Exception) as error:
        run_rows(
            [SEG, step],
            {"background": "$steps.background.image"},
            [street_image()],
            inputs=[{"type": "WorkflowParameter", "name": "opacity"}],
            values={"opacity": 2.0},
        )

    assert "opacity must be a number in [0.0, 1.0], got 2.0" in str(error.value)


# --- icon transparency -------------------------------------------------------------


# Per icon column: transparent, transparent, half, opaque.
ICON_ALPHA_COLUMNS = (0, 0, 128, 255)
ALPHA_STEPS = [
    SEG,
    {
        "type": "roboflow_core/icon_visualization@v1",
        "name": "marks",
        "image": "$inputs.image",
        "icon": "$inputs.icon",
        "icon_alpha": "$inputs.icon_alpha",
        "predictions": "$steps.seg.segmentation",
        "position": "CENTER",
        "icon_width": 4,
        "icon_height": 4,
    },
]
ALPHA_INPUTS = [
    {"type": "WorkflowImage", "name": "icon"},
    {"type": "WorkflowBatchInput", "name": "icon_alpha", "kind": ["tensor"]},
]


def icon_alphas() -> List[torch.Tensor]:
    """The same transparency as uint8 for the first image, float for the second."""
    as_uint8 = torch.tensor([ICON_ALPHA_COLUMNS] * 4, dtype=torch.uint8)

    return [as_uint8, as_uint8.to(torch.float32) / 255]


@pytest.mark.parametrize("mode", MODES)
def test_selected_icon_alpha_keeps_blends_and_covers_per_icon_column(
    mode: str,
) -> None:
    images = [street_image(), alley_image()]

    rows = run_rows(
        ALPHA_STEPS,
        {"marks": "$steps.marks.image"},
        images,
        inputs=ALPHA_INPUTS,
        values={"icon": solid_icon(2), "icon_alpha": icon_alphas()},
        mode=mode,
    )

    for image, row, shapes in zip(images, rows, (STREET, ALLEY)):
        below, marks = pixels_of(image).astype(int), pixels_of(row["marks"])
        for shape in shapes:
            x0, y0, x1, y1 = shape.xyxy
            left, top = (x0 + x1) // 2 - 2, (y0 + y1) // 2 - 2
            for column, alpha in enumerate(ICON_ALPHA_COLUMNS):
                x = left + column
                expected = (
                    below[top : top + 4, x] * (255 - alpha)
                    + np.array(ICON_COLOR) * alpha
                ) // 255
                assert np.array_equal(marks[top : top + 4, x], expected)


def test_an_icon_alpha_of_another_size_fails_with_both_shapes() -> None:
    with pytest.raises(Exception) as error:
        run_rows(
            ALPHA_STEPS,
            {"marks": "$steps.marks.image"},
            [street_image()],
            inputs=ALPHA_INPUTS,
            values={
                "icon": solid_icon(1),
                "icon_alpha": [torch.zeros((3, 3), dtype=torch.uint8)],
            },
        )

    assert (
        "icon_alpha must have the icon's (height, width) (4, 4), got shape (3, 3)"
        in (str(error.value))
    )


# --- one logo for every image ----------------------------------------------------


# The README recipe: one raw logo and one alpha shared by every image.
LOGO_INPUTS = [
    {"type": "WorkflowParameter", "name": "logo", "kind": ["image"]},
    {"type": "WorkflowParameter", "name": "logo_alpha", "kind": ["tensor"]},
]
LOGO_POSITIONS = {"watermark": (-2, -2), "clipped": (-62, -46)}
LOGO_STEPS = [
    {
        "type": "roboflow_core/icon_visualization@v1",
        "name": name,
        "image": "$inputs.image",
        "icon": "$inputs.logo",
        "icon_alpha": "$inputs.logo_alpha",
        "mode": "static",
        "icon_width": 4,
        "icon_height": 4,
        "x_position": x_position,
        "y_position": y_position,
    }
    for name, (x_position, y_position) in LOGO_POSITIONS.items()
]
# Top-left corner (x, y) of the 4x4 logo, from each image's own right/bottom edge:
#   street 112x72: watermark (112-2-4, 72-2-4), clipped (112-62-4, 72-46-4)
#   alley   64x48: watermark ( 64-2-4, 48-2-4), clipped (-2, -2): only logo
#                  rows and columns 2..3 (alpha 128, 255) land on the image
LOGO_CORNERS = {
    "street": {"watermark": (106, 66), "clipped": (46, 22)},
    "alley": {"watermark": (58, 42), "clipped": (-2, -2)},
}


def logo_alpha(form: str) -> Any:
    """ICON_ALPHA_COLUMNS as a uint8 tensor, or as the 0..1 nested list of JSON."""
    if form == "uint8_tensor":
        alpha = torch.tensor([ICON_ALPHA_COLUMNS] * 4, dtype=torch.uint8)
    else:
        alpha = [[value / 255 for value in ICON_ALPHA_COLUMNS] for _ in range(4)]

    return alpha


def stamped(below: np.ndarray, *, corner: Tuple[int, int]) -> np.ndarray:
    """Expected pixels: the 4x4 logo blended per column at ``corner``, clipped."""
    expected = below.astype(int)
    height, width = below.shape[:2]
    left, top = corner
    for row in range(4):
        for column, alpha in enumerate(ICON_ALPHA_COLUMNS):
            y, x = top + row, left + column
            if 0 <= y < height and 0 <= x < width:
                expected[y, x] = (
                    expected[y, x] * (255 - alpha) + np.array(ICON_COLOR) * alpha
                ) // 255

    return expected


@pytest.mark.parametrize("alpha_form", ["uint8_tensor", "json_list"])
@pytest.mark.parametrize("mode", MODES)
def test_one_parameter_logo_and_alpha_mark_each_image_at_its_own_corners(
    mode: str, alpha_form: str
) -> None:
    images = [street_image(), alley_image()]
    before = [pixels_of(image).copy() for image in images]
    logo = np.full((4, 4, 3), ICON_COLOR, dtype=np.uint8)
    alpha = logo_alpha(alpha_form)

    rows = run_rows(
        LOGO_STEPS,
        {name: f"$steps.{name}.image" for name in LOGO_POSITIONS},
        images,
        inputs=LOGO_INPUTS,
        values={"logo": logo, "logo_alpha": alpha},
        mode=mode,
    )

    assert len(rows) == len(images)
    for image, pixels, row in zip(images, before, rows):
        for name, corner in LOGO_CORNERS[image.image_id].items():
            assert np.array_equal(pixels_of(row[name]), stamped(pixels, corner=corner))
            assert row[name].image_id == image.image_id
        assert np.array_equal(pixels_of(image), pixels)
    assert (logo == ICON_COLOR).all()
    assert torch.equal(torch.as_tensor(alpha), torch.as_tensor(logo_alpha(alpha_form)))


# --- grid ------------------------------------------------------------------------


GRID_STEPS = DECORATION[:3] + [
    {
        "type": "roboflow_core/grid_visualization@v1",
        "name": "grid",
        "images": ["$inputs.image", "$steps.background.image", "$steps.color.image"],
        "width": 224,
        "height": 144,
    }
]


@pytest.mark.parametrize("mode", MODES)
def test_grid_shows_an_image_and_its_drawings_on_one_canvas_per_image(
    mode: str,
) -> None:
    images = [street_image(), alley_image()]

    rows = run_rows(
        GRID_STEPS,
        {
            "grid": "$steps.grid.image",
            "background": "$steps.background.image",
            "color": "$steps.color.image",
        },
        images,
        mode=mode,
    )

    for image, row in zip(images, rows):
        canvas = row["grid"]
        assert canvas.size_hw == (144, 224)
        assert canvas.is_composite and canvas.image_id != image.image_id
        assert [source.image_id for source in canvas.composite_sources] == [
            image.image_id
        ] * 3
    # The 112 x 72 street fills its 112 x 72 cells unscaled: tiles are exact copies.
    street, street_row = images[0], rows[0]
    tiles = {
        source.index: source.canvas_xyxy
        for source in street_row["grid"].composite_sources
    }
    assert tiles == {
        (0,): (0, 0, 112, 72),
        (1,): (112, 0, 224, 72),
        (2,): (0, 72, 112, 144),
    }
    canvas = pixels_of(street_row["grid"])
    assert np.array_equal(canvas[0:72, 0:112], pixels_of(street))
    assert np.array_equal(canvas[0:72, 112:224], pixels_of(street_row["background"]))
    assert np.array_equal(canvas[72:144, 0:112], pixels_of(street_row["color"]))
    assert not canvas[72:144, 112:224].any()


def test_a_gated_image_gets_no_grid_and_the_others_do() -> None:
    steps = [
        {
            "type": "roboflow_core/continue_if_boolean@v1",
            "name": "keep",
            "condition": "$inputs.keep",
            "next_steps": ["$steps.grid"],
        },
        {
            "type": "roboflow_core/grid_visualization@v1",
            "name": "grid",
            "images": ["$inputs.image"],
            "width": 56,
            "height": 36,
        },
    ]

    rows = run_rows(
        steps,
        {"grid": "$steps.grid.image"},
        [street_image(), alley_image()],
        inputs=[{"type": "WorkflowBatchInput", "name": "keep"}],
        values={"keep": [False, True]},
    )

    assert rows[0]["grid"] is None
    assert rows[1]["grid"].size_hw == (36, 56)


# --- frames ------------------------------------------------------------------------


def test_parent_rows_drawn_on_crops_keep_each_crops_own_region() -> None:
    steps = [
        SEG,
        CROP,
        {
            "type": "roboflow_core/background_color_visualization@v1",
            "name": "on_crop",
            "image": "$steps.crop.crops",
            "predictions": "$steps.seg.segmentation",
            "opacity": 1.0,
        },
        {
            "type": "roboflow_core/background_color_visualization@v1",
            "name": "direct",
            "image": "$inputs.image",
            "predictions": "$steps.seg.segmentation",
            "opacity": 1.0,
        },
    ]
    outputs = {
        "on_crop": "$steps.on_crop.image",
        "crops": "$steps.crop.crops",
        "direct": "$steps.direct.image",
    }

    (row,) = run_rows(steps, outputs, [street_image()])

    direct = pixels_of(row["direct"])
    for drawing, crop, shape in zip(row["on_crop"], row["crops"], STREET):
        x0, y0, x1, y1 = shape.xyxy
        assert drawing.image_id == crop.image_id
        assert np.array_equal(pixels_of(drawing), direct[y0:y1, x0:x1])


def test_the_inputs_stay_unchanged() -> None:
    images = [street_image(), alley_image()]
    before = [pixels_of(image).copy() for image in images]
    icons = solid_icon(2)

    run_rows(
        GRID_STEPS + DECORATION[3:],
        {"grid": "$steps.grid.image", "watermark": "$steps.watermark.image"},
        images,
        inputs=[{"type": "WorkflowImage", "name": "icon"}],
        values={"icon": icons},
    )

    for image, pixels in zip(images, before):
        assert np.array_equal(pixels_of(image), pixels)
    assert all((pixels_of(icon) == ICON_COLOR).all() for icon in icons)
