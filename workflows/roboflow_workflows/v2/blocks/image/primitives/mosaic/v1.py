"""Tile one group of images into one canvas per parent.

Canonical type: ``roboflow_core/mosaic@v1``.
"""

import math
from typing import Any, Dict, Optional
import torch
from pydantic import Field, StrictInt
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import CompositeSource, ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)


EMPTY_CANVAS_CHANNELS = 3


def _common_device(images: Batch[ImageData]) -> torch.device:
    devices = {image.device for image in images}
    if len(devices) > 1:
        raise ValueError(
            "roboflow_core/mosaic@v1 needs all images on one device, got "
            f"{sorted(str(device) for device in devices)}"
        )

    (device,) = devices

    return device


class MosaicBlock(Block):
    """Tile one group of images into one canvas per parent.

    Every image is resized with nearest-neighbour sampling to a square tile of
    ``tile_size`` pixels (aspect ratio is not kept). Tiles are placed row by row
    in the group's order on a grid with ``columns`` columns, or on the most
    square grid when ``columns`` is not set. Sparse indices do not leave gaps:
    the canvas holds only the images present in the group.

    The canvas is a new root image. It has no single parent: its
    ``composite_sources`` record, per tile, the source's logical index, id,
    tile rectangle and mappings back to the source and its root. Grayscale
    tiles are repeated to RGB when the group mixes both. All images must be on
    one device, which the canvas uses.

    An empty group produces one blank ``tile_size`` RGB square on the CPU,
    filled with ``background``, with no sources and a ``count`` of zero.

    Both outputs are timed at the last image of the group (``last`` policy), so
    a mosaic of a time window ``[N, T] -> [N]`` carries the window's closing
    timestamp; their source context is kept only when all images share it.
    """

    type = "roboflow_core/mosaic@v1"
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="images",
            context_policy="last",
            description="Canvas of all tiles; blank for an empty group.",
        ),
        "count": Output(
            INTEGER_KIND,
            source="images",
            context_policy="last",
            description="Number of tiled images.",
        ),
    }

    class Params(BlockParams):
        images: Group(IMAGE_KIND) = Field(
            description="Images to tile: the children of one parent, possibly none."
        )
        tile_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=64, ge=1, description="Side length in pixels of each square tile."
        )
        columns: Optional[StrictInt] | Ref(INTEGER_KIND) = Field(
            default=None,
            ge=1,
            description="Number of grid columns; null chooses ceil(sqrt(count)).",
        )
        background: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0, ge=0, le=255, description="Fill value for unused canvas area."
        )

    def run(
        self,
        *,
        images: Batch[ImageData],
        tile_size: int,
        columns: Optional[int],
        background: int,
    ) -> Dict[str, Any]:
        """Tile ``images`` into one canvas.

        Args:
            images: ``Batch`` of the images of one parent.
            tile_size: Side length in pixels of each square tile.
            columns: Grid columns, or ``None`` for ``ceil(sqrt(count))``.
            background: Fill value ``0..255`` of unused canvas area.

        Returns:
            ``image``: the composite canvas. ``count``: number of tiled images.

        Raises:
            ValueError: When the images are on different devices.
        """
        count = len(images)
        if count == 0:
            canvas = torch.full(
                (EMPTY_CANVAS_CHANNELS, tile_size, tile_size),
                background,
                dtype=torch.uint8,
            )
            result = {"image": ImageData.composite(canvas, sources=()), "count": 0}
            return result

        grid_columns = columns or math.ceil(math.sqrt(count))
        grid_rows = math.ceil(count / grid_columns)
        canvas = torch.full(
            (
                max(image.channels for image in images),
                grid_rows * tile_size,
                grid_columns * tile_size,
            ),
            background,
            dtype=torch.uint8,
            device=_common_device(images),
        )
        sources = []
        for position, (index, image) in enumerate(images.iter_with_indices()):
            row, column = divmod(position, grid_columns)
            x0, y0 = column * tile_size, row * tile_size
            tile = image.resize((tile_size, tile_size), interpolation="nearest")
            # A one-channel tile broadcasts over an RGB canvas.
            canvas[:, y0 : y0 + tile_size, x0 : x0 + tile_size] = tile.tensor_image
            sources.append(
                CompositeSource.place(
                    image,
                    index=index,
                    canvas_xyxy=(x0, y0, x0 + tile_size, y0 + tile_size),
                )
            )

        result = {
            "image": ImageData.composite(canvas, sources=sources),
            "count": count,
        }

        return result
