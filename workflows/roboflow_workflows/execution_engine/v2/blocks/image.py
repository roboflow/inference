"""Native V2 CPU image blocks.

Each block is one class that declares its identity, parameters and outputs.
A step may set every parameter to a literal, leave its default, or bind it to
a workflow input or an upstream output. ``run`` receives plain values, or a
``Batch`` for a ``Group`` field, and returns a mapping of ordinary results.

* ``v2/crop``: image and rectangles -> ``crops`` along a new ``regions`` axis,
  and one ``summary`` per image.
* ``v2/invert``: image -> pixel-inverted ``image``.
* ``v2/mosaic``: group of images -> one ``image`` canvas and its ``count`` per
  parent of the group.
* ``v2/has_brightness``: image -> ``keep``, whether the mean pixel value
  reaches a minimum.

Image payloads are RGB ``uint8`` NumPy arrays; see ``IMAGE_KIND``.
"""

import math
from typing import Annotated, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import AfterValidator, Field, StrictFloat, StrictInt
from roboflow_workflows.execution_engine.v2.blocks.kinds import (
    CROP_SUMMARY_KIND,
    IMAGE_CHANNELS,
    IMAGE_KIND,
)
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
)

Rectangle = Tuple[int, int, int, int]


def _require_rectangles(regions: Sequence[Any]) -> Sequence[Any]:
    # Shared by literal and selected regions; checks without converting.
    for rectangle in regions:
        corners = list(rectangle) if isinstance(rectangle, (list, tuple)) else None
        if (
            corners is None
            or len(corners) != 4
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in corners
            )
        ):
            raise ValueError(f"rectangle {rectangle!r} must be four integers")
        x0, y0, x1, y1 = corners
        if x1 < x0 or y1 < y0:
            raise ValueError(f"rectangle {corners} must have x0 <= x1 and y0 <= y1")

    return regions


class CropBlock(Block):
    """Crop rectangles out of one image.

    Rectangles are ``[x0, y0, x1, y1]`` pixel coordinates with exclusive upper
    bounds. Each rectangle is clipped to the image. A rectangle that is empty
    after clipping is omitted from ``crops``; surviving crops keep the position
    of their rectangle as local index, so ``crops`` may be sparse, for example
    ``(0,)`` and ``(2,)``. Crops are contiguous copies, not views of the image.

    ``crops`` is dynamic nesting: which rectangles survive depends on the size
    of each image.
    """

    type = "v2/crop"
    outputs = {
        "crops": Output(
            IMAGE_KIND,
            expand="regions",
            source="image",
            description="Crops indexed by the position of their rectangle.",
        ),
        "summary": Output(
            CROP_SUMMARY_KIND,
            source="image",
            description="Image size, crop count and the kept rectangle positions.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to crop.")
        regions: Annotated[
            List[Tuple[StrictInt, StrictInt, StrictInt, StrictInt]]
            | Ref(LIST_OF_VALUES_KIND),
            AfterValidator(_require_rectangles),
        ] = Field(
            description=(
                "Rectangles [x0, y0, x1, y1] in pixels with x0 <= x1 and "
                "y0 <= y1. Upper bounds are exclusive."
            ),
            examples=[[[0, 0, 64, 64], [32, 32, 96, 96]]],
        )

    def run(self, *, image: np.ndarray, regions: Sequence[Rectangle]) -> Dict[str, Any]:
        """Crop ``regions`` from ``image``.

        Args:
            image: RGB ``uint8`` array of shape ``(height, width, 3)``.
            regions: Rectangles ``(x0, y0, x1, y1)`` in order.

        Returns:
            ``crops``: a ``Batch`` of crop arrays indexed by the positions of
            the rectangles that survived clipping. ``summary``: a mapping with
            ``crop_count``, ``image_height``, ``image_width``, ``kept_regions``
            and ``crop_dimensions``.
        """
        height, width = image.shape[:2]

        crops: List[np.ndarray] = []
        kept_regions: List[int] = []
        for position, (x0, y0, x1, y1) in enumerate(regions):
            clipped_x0, clipped_y0 = max(0, x0), max(0, y0)
            clipped_x1, clipped_y1 = min(width, x1), min(height, y1)
            if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
                continue

            crop = image[clipped_y0:clipped_y1, clipped_x0:clipped_x1].copy(order="C")
            crops.append(crop)
            kept_regions.append(position)

        summary = {
            "crop_count": len(crops),
            "image_height": int(height),
            "image_width": int(width),
            "kept_regions": kept_regions,
            "crop_dimensions": [[int(c.shape[0]), int(c.shape[1])] for c in crops],
        }
        indices = [(position,) for position in kept_regions]
        result = {"crops": Batch.of(crops, indices=indices), "summary": summary}

        return result


class InvertBlock(Block):
    """Invert every pixel of one image (``255 - value``).

    Returns a new array; the input image is not modified.
    """

    type = "v2/invert"
    outputs = {"image": Output(IMAGE_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to invert.")

    def run(self, *, image: np.ndarray) -> Dict[str, Any]:
        """Invert ``image``.

        Args:
            image: RGB ``uint8`` array of shape ``(height, width, 3)``.

        Returns:
            ``image``: a new array equal to ``255 - image``.
        """
        inverted = np.subtract(np.uint8(255), image, dtype=np.uint8)
        result = {"image": inverted}

        return result


class MosaicBlock(Block):
    """Tile one group of images into one canvas per parent.

    Every image is resized with nearest-neighbour sampling to a square tile of
    ``tile_size`` pixels (aspect ratio is not kept). Tiles are placed row by row
    in the group's order on a grid with ``columns`` columns, or on the most
    square grid when ``columns`` is not set. Sparse indices do not leave gaps:
    the canvas holds only the images present in the group.

    An empty group produces one blank ``tile_size`` square filled with
    ``background`` and a ``count`` of zero.
    """

    type = "v2/mosaic"
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="images",
            context_policy="common_or_none",
            description="Canvas of all tiles; blank for an empty group.",
        ),
        "count": Output(
            INTEGER_KIND,
            source="images",
            context_policy="common_or_none",
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
        images: Iterable[np.ndarray],
        tile_size: int,
        columns: Optional[int],
        background: int,
    ) -> Dict[str, Any]:
        """Tile ``images`` into one canvas.

        Args:
            images: ``Batch`` of RGB ``uint8`` images of one parent.
            tile_size: Side length in pixels of each square tile.
            columns: Grid columns, or ``None`` for ``ceil(sqrt(count))``.
            background: Fill value ``0..255`` of unused canvas area.

        Returns:
            ``image``: the canvas. ``count``: number of tiled images.
        """
        tiles = [
            _resize_nearest(image, height=tile_size, width=tile_size)
            for image in images
        ]
        count = len(tiles)
        if count == 0:
            canvas = _blank_canvas(
                rows=1, columns=1, tile_size=tile_size, background=background
            )
            result = {"image": canvas, "count": 0}
            return result

        grid_columns = columns or math.ceil(math.sqrt(count))
        grid_rows = math.ceil(count / grid_columns)
        canvas = _blank_canvas(
            rows=grid_rows,
            columns=grid_columns,
            tile_size=tile_size,
            background=background,
        )
        for position, tile in enumerate(tiles):
            row, column = divmod(position, grid_columns)
            y0, x0 = row * tile_size, column * tile_size
            canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile

        result = {"image": canvas, "count": count}

        return result


class HasBrightnessBlock(Block):
    """Decide whether the mean pixel value of an image reaches a minimum.

    ``keep`` is a plain ``bool``; bind it to the ``condition`` of
    ``v2/continue_if`` to gate later steps.
    """

    type = "v2/has_brightness"
    outputs = {"keep": Output(BOOLEAN_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to measure.")
        minimum: StrictFloat | Ref(FLOAT_KIND) = Field(
            ge=0,
            le=255,
            description="Inclusive threshold of the mean pixel value, 0..255.",
        )

    def run(self, *, image: np.ndarray, minimum: float) -> Dict[str, Any]:
        """Compare the mean pixel value of ``image`` with ``minimum``.

        Args:
            image: RGB ``uint8`` array of shape ``(height, width, 3)``.
            minimum: Inclusive threshold in ``0..255``.

        Returns:
            ``keep``: ``True`` when ``image.mean() >= minimum``.
        """
        keep = bool(float(image.mean()) >= minimum)
        result = {"keep": keep}

        return result


def _blank_canvas(
    *, rows: int, columns: int, tile_size: int, background: int
) -> np.ndarray:
    shape = (rows * tile_size, columns * tile_size, IMAGE_CHANNELS)
    canvas = np.full(shape, background, dtype=np.uint8)

    return canvas


def _resize_nearest(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    rows = (np.arange(height) * source_height) // height
    columns = (np.arange(width) * source_width) // width
    resized = image[rows[:, None], columns[None, :]]

    return resized
