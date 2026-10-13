"""Images side by side: ``v2/grid_visualization@v1``.

Port of V1 ``roboflow_core/grid_visualization@v1``::

    tiles   the images, with each list value flattened in place
    grid    g = ceil(sqrt(len(tiles))) columns and rows
    cell    (width // g) x (height // g); tile k in row k // g, column k % g
    tile    resized to fit its cell keeping its aspect ratio (sizes
            truncated), with OpenCV INTER_AREA; centred in the cell
    canvas  black width x height; no tiles give a black canvas

Pixels equal V1's: ``ImageData.resize(interpolation="area")`` repeats
``cv2.INTER_AREA``. A ``None`` item of a list value leaves its cell black, as
in V1. All work runs on the images' device.

Differences from V1:

- The output is a new root image (``ImageData.composite``) whose
  ``composite_sources`` record each tile's rectangle and its mappings back
  to the source image. V1 gave the grid the first image's metadata, which
  claimed that image's geometry for the whole canvas.
- No caching between calls. V1 kept the last inputs and resized tiles keyed
  by ``id()``; the block is stateless and calls may run concurrently.
- A tile whose fitted side truncates to 0 keeps 1 pixel (V1 failed in
  ``cv2.resize``). A canvas smaller than its grid is a clear error.
"""

import math
from typing import Any, Dict, List, Optional

import torch
from pydantic import Field, StrictInt, field_validator, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    check_literals,
    checked_int,
)
from roboflow_workflows.v2.data.image import CompositeSource, ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, LIST_OF_VALUES_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class GridVisualizationBlock(Block):
    """Arrange images in a square grid on one canvas.

    Example: ``images: ["$inputs.image", "$steps.mask_visualization.image"]``
    with ``width=1280, height=720`` puts the two images side by side in the
    top row of a 2 x 2 grid of 640 x 360 cells.
    """

    type = "v2/grid_visualization@v1"
    prunable = True
    metadata = {
        "name": "Grid Visualization",
        "port_of": "roboflow_core/grid_visualization@v1",
        "section": "visualization",
    }
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="images",
            description=(
                "New width x height RGB canvas, a root image whose "
                "composite_sources place each tile."
            ),
        ),
    }

    class Params(BlockParams):
        images: List[Ref(IMAGE_KIND, LIST_OF_VALUES_KIND)] = Field(
            min_length=1,
            description=(
                "Images to tile, in order: single images and lists of images "
                "(flattened in place), all of the same row. A single selector "
                "is read as a list of one."
            ),
            examples=[["$inputs.image", "$steps.mask_visualization.image"]],
        )
        width: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2560,
            description="Canvas width in pixels; >= 1.",
            examples=[2560, "$inputs.width"],
        )
        height: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1440,
            description="Canvas height in pixels; >= 1.",
            examples=[1440, "$inputs.height"],
        )

        @field_validator("images", mode="before")
        @classmethod
        def _one_selector_is_a_list(cls, value: Any) -> Any:
            # V1 accepted one selector of a list of images.
            images = [value] if isinstance(value, str) else value

            return images

        @model_validator(mode="after")
        def _check_literals(self) -> "GridVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(self, *, images: List[Any], width: int, height: int) -> Dict[str, Any]:
        """Tile ``images`` onto a new canvas.

        Args:
            images: One value per selector: an image or a list of images
                (items may be ``None``).
            width: Canvas width.
            height: Canvas height.

        Returns:
            ``image``: the composite canvas.

        Raises:
            ValueError: On invalid sizes, a canvas too small for its grid,
                a value that is not an image or images on different devices.
        """
        values = dict(width=width, height=height)
        for name, check in _CHECKS.items():
            check(values[name])
        tiles = self._flattened(images)
        present = [tile for tile in tiles if tile is not None]
        devices = {tile.device for tile in present}
        if len(devices) > 1:
            raise ValueError(
                f"{self.type}: images are on different devices "
                f"{sorted(map(str, devices))}; move them to one device first"
            )

        device = devices.pop() if devices else torch.device("cpu")
        canvas = torch.zeros((3, height, width), dtype=torch.uint8, device=device)
        if not tiles:
            result = {"image": ImageData.composite(canvas, sources=())}
            return result

        columns = math.ceil(math.sqrt(len(tiles)))
        cell_width, cell_height = width // columns, height // columns
        if cell_width == 0 or cell_height == 0:
            raise ValueError(
                f"{self.type}: {len(tiles)} images need a {columns} x {columns} "
                f"grid, which does not fit in {width} x {height} pixels; use "
                f"width and height >= {columns}"
            )

        sources = []
        for position, tile in enumerate(tiles):
            if tile is None:
                continue
            row, column = divmod(position, columns)
            scale = min(cell_width / tile.width, cell_height / tile.height)
            fitted_width = max(int(tile.width * scale), 1)
            fitted_height = max(int(tile.height * scale), 1)
            x0 = column * cell_width + (cell_width - fitted_width) // 2
            y0 = row * cell_height + (cell_height - fitted_height) // 2
            resized = tile.resize((fitted_height, fitted_width), interpolation="area")
            # A one-channel tile broadcasts over the RGB canvas.
            canvas[:, y0 : y0 + fitted_height, x0 : x0 + fitted_width] = (
                resized.tensor_image
            )
            sources.append(
                CompositeSource.place(
                    tile,
                    index=(position,),
                    canvas_xyxy=(x0, y0, x0 + fitted_width, y0 + fitted_height),
                )
            )
        result = {"image": ImageData.composite(canvas, sources=sources)}

        return result

    def _flattened(self, images: List[Any]) -> List[Optional[ImageData]]:
        """Return the tiles in order: images, and the items of list values."""
        tiles = []
        for value in images:
            items = value if isinstance(value, (list, tuple)) else [value]
            for item in items:
                if item is not None and not isinstance(item, ImageData):
                    raise ValueError(
                        f"{self.type}: images must be images or lists of "
                        f"images, got {type(item).__name__}"
                    )
                tiles.append(item)

        return tiles


_CHECKS = {
    "width": lambda value: checked_int(value, name="width", minimum=1),
    "height": lambda value: checked_int(value, name="height", minimum=1),
}
