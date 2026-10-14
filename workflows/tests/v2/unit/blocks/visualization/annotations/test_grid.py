"""``roboflow_core/grid_visualization@v1``: images tiled on one composite canvas.

Expected pixels come from hand-computed layouts of solid images and from the
latest V1 block on the same images (an exact oracle: both resize with
OpenCV INTER_AREA rules). The output's geometry is checked through its
``composite_sources``.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pytest
from roboflow_workflows.core_steps.visualizations.grid.v1 import (
    GridVisualizationBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.blocks.visualization.annotations.grid_visualization.v1 import GridVisualizationBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from tests.v2.unit.blocks.visualization.annotations.support import pixels_of, scene

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


def solid(height: int, width: int, color: Tuple[int, int, int], name: str) -> ImageData:
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = color

    return ImageData.from_numpy_rgb(pixels, image_id=name)


def tile(images: List[Any], **params: Any) -> ImageData:
    """Validate ``params`` as the engine does, run the block, return the canvas."""
    selectors = [f"$inputs.image_{index}" for index in range(len(images))]
    validated = spec_of(GridVisualizationBlock).validate_params(
        {"images": selectors, **params}
    )
    arguments = {**dict(validated), "images": images}
    result = GridVisualizationBlock().run(**arguments)

    return result["image"]


def test_two_images_fit_their_cells_keeping_aspect_ratio_and_centred() -> None:
    wide, tall = solid(4, 8, RED, "wide"), solid(10, 4, GREEN, "tall")

    canvas = pixels_of(tile([wide, tall], width=20, height=10))

    # 2 x 2 grid of 10 x 5 cells: 8 x 4 grows to 10 x 5; 4 x 10 shrinks to 2 x 5.
    expected = np.zeros((10, 20, 3), dtype=np.uint8)
    expected[0:5, 0:10] = RED
    expected[0:5, 14:16] = GREEN
    np.testing.assert_array_equal(canvas, expected)


def test_tiles_fill_rows_of_a_square_grid_in_order() -> None:
    colors = [RED, GREEN, BLUE, (9, 9, 9), (200, 100, 50)]
    images = [solid(4, 4, color, f"i{k}") for k, color in enumerate(colors)]

    canvas = pixels_of(tile(images, width=12, height=12))

    # Five images: a 3 x 3 grid of 4 x 4 cells; image 3 opens the second row.
    for index, color in enumerate(colors):
        row, column = divmod(index, 3)
        assert (
            canvas[row * 4 : row * 4 + 4, column * 4 : column * 4 + 4] == color
        ).all()
    assert not canvas[4:12, 8:12].any()


def test_list_values_are_flattened_in_place_and_none_items_leave_black_cells() -> None:
    red, green, blue = (
        solid(2, 2, RED, "r"),
        solid(2, 2, GREEN, "g"),
        solid(2, 2, BLUE, "b"),
    )

    canvas = pixels_of(tile([red, [None, green], blue], width=4, height=4))

    # Tiles red, None, green, blue in a 2 x 2 grid of 2 x 2 cells.
    assert (canvas[0:2, 0:2] == RED).all()
    assert not canvas[0:2, 2:4].any()
    assert (canvas[2:4, 0:2] == GREEN).all()
    assert (canvas[2:4, 2:4] == BLUE).all()


def test_no_images_give_a_black_canvas_without_sources() -> None:
    canvas = tile([[]], width=6, height=4)

    assert canvas.size_hw == (4, 6)
    assert not pixels_of(canvas).any()
    assert canvas.composite_sources == ()


def test_the_canvas_is_a_new_root_that_maps_each_tile_back_to_its_image() -> None:
    wide, tall = solid(4, 8, RED, "wide"), solid(10, 4, GREEN, "tall")

    canvas = tile([wide, tall], width=20, height=10)

    assert canvas.is_composite and canvas.root.frame_id == canvas.image_id
    placed = {source.image_id: source for source in canvas.composite_sources}
    assert placed["wide"].canvas_xyxy == (0, 0, 10, 5)
    assert placed["tall"].canvas_xyxy == (14, 0, 16, 5)
    assert placed["tall"].index == (1,)
    # The tall tile's centre (15, 2.5) is the centre (2, 5) of its image.
    assert placed["tall"].parent.map_xy(15.0, 2.5) == pytest.approx((2.0, 5.0))


def test_a_grey_tile_is_drawn_in_grey() -> None:
    grey = ImageData.from_tensor(
        solid(3, 3, (60, 60, 60), "g").tensor_image[:1].clone()
    )

    canvas = pixels_of(tile([grey], width=3, height=3))

    assert (canvas == 60).all()


def test_a_tile_too_thin_to_fit_keeps_one_pixel() -> None:
    line = solid(1, 100, RED, "line")

    canvas = tile([line], width=10, height=10)

    # 100 x 1 scaled by 0.1 is 10 x 0.1: one row, centred at y = 4.
    assert canvas.composite_sources[0].canvas_xyxy == (0, 4, 10, 5)
    assert (pixels_of(canvas)[4] == RED).all()


@pytest.mark.parametrize(
    "sizes, width, height",
    [
        ([(48, 64), (30, 50), (64, 20)], 64, 48),  # mostly shrinking
        ([(48, 64), (7, 9)], 101, 37),  # growing, cells that do not divide evenly
        ([(48, 64)] * 5, 90, 70),  # 3 x 3 grid with empty cells
    ],
)
def test_grids_equal_v1(sizes: List[Tuple[int, int]], width: int, height: int) -> None:
    pixels = [scene(*size, seed=index) for index, size in enumerate(sizes)]

    canvas = tile(
        [ImageData.from_numpy_rgb(p) for p in pixels], width=width, height=height
    )
    v1 = GridVisualizationBlockV1().run(
        images=[v1_image(p) for p in pixels], width=width, height=height
    )

    np.testing.assert_array_equal(
        pixels_of(canvas), v1["image"].numpy_image[:, :, ::-1]
    )


def test_the_inputs_are_unchanged() -> None:
    pixels = scene()
    image = ImageData.from_numpy_rgb(pixels)

    tile([image, image], width=20, height=20)

    np.testing.assert_array_equal(pixels_of(image), pixels)


def test_a_single_selector_is_a_list_of_one() -> None:
    validated = spec_of(GridVisualizationBlock).validate_params(
        {"images": "$steps.b.images"}
    )

    assert validated.images == ["$steps.b.images"]


def test_a_canvas_smaller_than_its_grid_is_an_error() -> None:
    images = [solid(2, 2, RED, f"i{k}") for k in range(5)]

    with pytest.raises(
        ValueError, match="need a 3 x 3 grid, which does not fit in 2 x 9"
    ):
        tile(images, width=2, height=9)


def test_a_value_that_is_not_an_image_is_an_error() -> None:
    with pytest.raises(
        ValueError, match="images must be images or lists of images, got int"
    ):
        tile([[solid(2, 2, RED, "r"), 7]], width=4, height=4)


def test_an_invalid_size_fails_alike_as_literal_and_as_selected_value() -> None:
    message = "width must be an integer >= 1, got 0"

    with pytest.raises(ParamsValidationError) as compile_error:
        spec_of(GridVisualizationBlock).validate_params(
            {"images": ["$inputs.a"], "width": 0}
        )
    validated = spec_of(GridVisualizationBlock).validate_params(
        {"images": ["$inputs.a"], "width": "$inputs.width"}
    )
    with pytest.raises(ValueError) as run_error:
        GridVisualizationBlock().run(
            **{**dict(validated), "images": [solid(2, 2, RED, "r")], "width": 0}
        )

    assert message in str(compile_error.value) + str(compile_error.value.__cause__)
    assert message in str(run_error.value)


def v1_image(pixels: np.ndarray, name: Optional[str] = None) -> WorkflowImageData:
    """The V1 form of RGB pixels: a BGR ``WorkflowImageData``."""
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=name or "image"),
        numpy_image=np.ascontiguousarray(pixels[:, :, ::-1]),
    )

    return image
