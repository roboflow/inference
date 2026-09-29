"""Image tiling and letterboxing used by the visualisation sinks.

Copy of `inference.core.utils.drawing.create_tiles` and its helpers, and of the
`np.ndarray` path of `letterbox_image` / `resize_image_keeping_aspect_ratio`
from `inference.core.utils.preprocess`. The sinks materialise tensor frames to
CPU pixels before drawing, so only arrays reach these helpers; any other image
type is rejected with the same error the legacy helpers raise when
`USE_PYTORCH_FOR_PREPROCESSING` is off. Their tensor path stays in
`inference.core.utils.preprocess`.
"""

import itertools
import math
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple

import cv2
import numpy as np

MAX_COLUMNS_FOR_SINGLE_ROW_GRID = 3


def create_tiles(
    images: List[np.ndarray],
    grid_size: Optional[Tuple[Optional[int], Optional[int]]] = None,
    single_tile_size: Optional[Tuple[int, int]] = None,
    tile_scaling: Literal["min", "max", "avg"] = "avg",
    tile_padding_color: Tuple[int, int, int] = (0, 0, 0),
    tile_margin: int = 15,
    tile_margin_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("Could not create image tiles from empty list of images.")
    if single_tile_size is None:
        single_tile_size = _aggregate_images_shape(images=images, mode=tile_scaling)
    resized_images = [
        letterbox_image(
            image=i, desired_size=single_tile_size, color=tile_padding_color
        )
        for i in images
    ]
    grid_size = _establish_grid_size(images=images, grid_size=grid_size)
    if len(images) > grid_size[0] * grid_size[1]:
        raise ValueError(f"Grid of size: {grid_size} cannot fit {len(images)} images.")
    return _generate_tiles(
        images=resized_images,
        grid_size=grid_size,
        single_tile_size=single_tile_size,
        tile_padding_color=tile_padding_color,
        tile_margin=tile_margin,
        tile_margin_color=tile_margin_color,
    )


def _calculate_aggregated_images_shape(
    images: List[np.ndarray], aggregator: Callable[[List[int]], float]
) -> Tuple[int, int]:
    height = round(aggregator([i.shape[0] for i in images]))
    width = round(aggregator([i.shape[1] for i in images]))
    return width, height


SHAPE_AGGREGATION_FUN = {
    "min": partial(_calculate_aggregated_images_shape, aggregator=np.min),
    "max": partial(_calculate_aggregated_images_shape, aggregator=np.max),
    "avg": partial(_calculate_aggregated_images_shape, aggregator=np.average),
}


def _aggregate_images_shape(
    images: List[np.ndarray], mode: Literal["min", "max", "avg"]
) -> Tuple[int, int]:
    if mode not in SHAPE_AGGREGATION_FUN:
        raise ValueError(
            f"Could not aggregate images shape - provided unknown mode: {mode}. "
            f"Supported modes: {list(SHAPE_AGGREGATION_FUN.keys())}."
        )
    return SHAPE_AGGREGATION_FUN[mode](images)


def _establish_grid_size(
    images: List[np.ndarray], grid_size: Optional[Tuple[Optional[int], Optional[int]]]
) -> Tuple[int, int]:
    if grid_size is None or all(e is None for e in grid_size):
        return _negotiate_grid_size(images=images)
    if grid_size[0] is None:
        return math.ceil(len(images) / grid_size[1]), grid_size[1]
    if grid_size[1] is None:
        return grid_size[0], math.ceil(len(images) / grid_size[0])
    return grid_size


def _negotiate_grid_size(images: List[np.ndarray]) -> Tuple[int, int]:
    if len(images) <= MAX_COLUMNS_FOR_SINGLE_ROW_GRID:
        return 1, len(images)
    nearest_sqrt = math.ceil(np.sqrt(len(images)))
    proposed_columns = nearest_sqrt
    proposed_rows = nearest_sqrt
    while proposed_columns * (proposed_rows - 1) >= len(images):
        proposed_rows -= 1
    return proposed_rows, proposed_columns


def _generate_tiles(
    images: List[np.ndarray],
    grid_size: Tuple[int, int],
    single_tile_size: Tuple[int, int],
    tile_padding_color: Tuple[int, int, int],
    tile_margin: int,
    tile_margin_color: Tuple[int, int, int],
) -> np.ndarray:
    rows, columns = grid_size
    # Same rows as `create_batches(sequence=images, batch_size=columns)`.
    row_length = max(columns, 1)
    tiles_elements = [
        images[start : start + row_length]
        for start in range(0, len(images), row_length)
    ]
    while len(tiles_elements[-1]) < columns:
        tiles_elements[-1].append(
            _generate_color_image(shape=single_tile_size, color=tile_padding_color)
        )
    while len(tiles_elements) < rows:
        tiles_elements.append(
            [_generate_color_image(shape=single_tile_size, color=tile_padding_color)]
            * columns
        )
    return _merge_tiles_elements(
        tiles_elements=tiles_elements,
        grid_size=grid_size,
        single_tile_size=single_tile_size,
        tile_margin=tile_margin,
        tile_margin_color=tile_margin_color,
    )


def _merge_tiles_elements(
    tiles_elements: List[List[np.ndarray]],
    grid_size: Tuple[int, int],
    single_tile_size: Tuple[int, int],
    tile_margin: int,
    tile_margin_color: Tuple[int, int, int],
) -> np.ndarray:
    vertical_padding = (
        np.ones((single_tile_size[1], tile_margin, 3)) * tile_margin_color
    )
    merged_rows = [
        np.concatenate(
            list(
                itertools.chain.from_iterable(
                    zip(row, [vertical_padding] * grid_size[1])
                )
            )[:-1],
            axis=1,
        )
        for row in tiles_elements
    ]
    row_width = merged_rows[0].shape[1]
    horizontal_padding = (
        np.ones((tile_margin, row_width, 3), dtype=np.uint8) * tile_margin_color
    )
    rows_with_paddings = []
    for row in merged_rows:
        rows_with_paddings.append(row)
        rows_with_paddings.append(horizontal_padding)
    return np.concatenate(
        rows_with_paddings[:-1],
        axis=0,
    ).astype(np.uint8)


def _generate_color_image(
    shape: Tuple[int, int], color: Tuple[int, int, int]
) -> np.ndarray:
    return np.ones(shape[::-1] + (3,), dtype=np.uint8) * color


def letterbox_image(
    image: np.ndarray,
    desired_size: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Resize and pad image to fit the desired size, preserving its aspect ratio.

    Parameters:
    - image: numpy array representing the image.
    - desired_size: tuple (width, height) representing the target dimensions.
    - color: tuple (B, G, R) representing the color to pad with.

    Returns:
    - letterboxed image.
    """
    resized_img = resize_image_keeping_aspect_ratio(
        image=image,
        desired_size=desired_size,
    )
    new_height, new_width = resized_img.shape[:2]
    top_padding = (desired_size[1] - new_height) // 2
    bottom_padding = desired_size[1] - new_height - top_padding
    left_padding = (desired_size[0] - new_width) // 2
    right_padding = desired_size[0] - new_width - left_padding
    return cv2.copyMakeBorder(
        resized_img,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        cv2.BORDER_CONSTANT,
        value=color,
    )


def resize_image_keeping_aspect_ratio(
    image: np.ndarray,
    desired_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize reserving its aspect ratio.

    Parameters:
    - image: numpy array representing the image.
    - desired_size: tuple (width, height) representing the target dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"Received an image of unknown type, {type(image)}; "
            "This is most likely a bug. Contact Roboflow team through github issues "
            "(https://github.com/roboflow/inference/issues) providing full context of the problem"
        )
    img_ratio = image.shape[1] / image.shape[0]
    desired_ratio = desired_size[0] / desired_size[1]

    # Determine the new dimensions
    if img_ratio >= desired_ratio:
        # Resize by width
        new_width = desired_size[0]
        new_height = int(desired_size[0] / img_ratio)
    else:
        # Resize by height
        new_height = desired_size[1]
        new_width = int(desired_size[1] * img_ratio)

    # Resize the image to new dimensions
    return cv2.resize(image, (new_width, new_height))
