"""Native V2 image blocks.

Each block is one class that declares its identity, parameters and outputs.
A step may set every parameter to a literal, leave its default, or bind it to
a workflow input or an upstream output. ``run`` receives plain values, or a
``Batch`` for a ``Group`` field, and returns a mapping of ordinary results.

* ``v2/crop``: image and rectangles -> ``crops`` along a new ``regions`` axis,
  and one ``summary`` per image.
* ``v2/static_crop``: image and configured rectangles -> ``crops`` along a
  new stationary ``regions`` axis: position ``k`` is always rectangle ``k``.
* ``v2/resize``: image -> ``image`` of a given size.
* ``v2/invert``: image -> pixel-inverted ``image``.
* ``v2/mosaic``: group of images -> one ``image`` canvas and its ``count`` per
  parent of the group, timed at the last image of the group.
* ``v2/has_brightness``: image -> ``keep``, whether the mean pixel value
  reaches a minimum.

Image payloads are ``ImageData``: a ``(channels, height, width)`` ``uint8``
tensor with its provenance; see ``image_data``. Blocks keep tensors on their
device and never modify an input tensor.
"""

import math
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
from pydantic import AfterValidator, Field, StrictFloat, StrictInt
from roboflow_workflows.execution_engine.v2.blocks.image_data import (
    CompositeSource,
    ImageData,
)
from roboflow_workflows.execution_engine.v2.blocks.kinds import (
    CROP_SUMMARY_KIND,
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

EMPTY_CANVAS_CHANNELS = 3


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
    ``(0,)`` and ``(2,)``. Crops are contiguous copies on the image's device,
    each with a new ``image_id``; their provenance places them in the image
    and in its root.

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

    def run(self, *, image: ImageData, regions: Sequence[Rectangle]) -> Dict[str, Any]:
        """Crop ``regions`` from ``image``.

        Args:
            image: Image to crop.
            regions: Rectangles ``(x0, y0, x1, y1)`` in order.

        Returns:
            ``crops``: a ``Batch`` of crops indexed by the positions of the
            rectangles that survived clipping. ``summary``: a mapping with
            ``crop_count``, ``image_height``, ``image_width``, ``kept_regions``
            and ``crop_dimensions``.
        """
        crops: List[ImageData] = []
        kept_regions: List[int] = []
        for position, region in enumerate(regions):
            crop = image.crop(region)
            if crop is None:
                continue
            crops.append(crop)
            kept_regions.append(position)

        summary = {
            "crop_count": len(crops),
            "image_height": image.height,
            "image_width": image.width,
            "kept_regions": kept_regions,
            "crop_dimensions": [list(crop.size_hw) for crop in crops],
        }
        indices = [(position,) for position in kept_regions]
        result = {"crops": Batch.of(crops, indices=indices), "summary": summary}

        return result


class StaticCropBlock(Block):
    """Crop the same configured rectangles out of every image.

    Rectangles are ``[x0, y0, x1, y1]`` pixel coordinates with exclusive upper
    bounds, clipped to the image. Unlike ``v2/crop``, every rectangle keeps its
    position: crop ``k`` always comes from rectangle ``k``. That is why
    ``crops`` is declared stationary, so a window operator may collect each
    region over time (``[N, regions] -> [N, regions, T]``).

    A rectangle entirely outside an image never shifts the others. With
    ``outside="error"`` (default) the step fails and names the rectangle; with
    ``outside="none"`` that position holds ``None``, which downstream blocks
    skip like any missing value.
    """

    type = "v2/static_crop"
    outputs = {
        "crops": Output(
            IMAGE_KIND,
            expand="regions",
            stationary=True,
            source="image",
            description="One crop per configured rectangle, in configured order.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to crop.")
        regions: Annotated[
            List[Tuple[StrictInt, StrictInt, StrictInt, StrictInt]],
            AfterValidator(_require_rectangles),
        ] = Field(
            description=(
                "Fixed rectangles [x0, y0, x1, y1] in pixels with x0 <= x1 and "
                "y0 <= y1. Upper bounds are exclusive. Written in the workflow, "
                "never selected, so positions mean the same region every time."
            ),
            examples=[[[0, 0, 64, 64], [32, 32, 96, 96]]],
        )
        outside: Literal["error", "none"] = Field(
            default="error",
            description=(
                "What a rectangle entirely outside the image produces: a step "
                "error, or None at its position."
            ),
        )

    def run(
        self, *, image: ImageData, regions: Sequence[Rectangle], outside: str
    ) -> Dict[str, Any]:
        """Crop every configured rectangle from ``image``.

        Args:
            image: Image to crop.
            regions: Rectangles ``(x0, y0, x1, y1)`` in configured order.
            outside: ``"error"`` or ``"none"``, for rectangles outside the image.

        Returns:
            ``crops``: a ``Batch`` with one entry per rectangle at its position;
            ``None`` for a rectangle outside the image when ``outside="none"``.

        Raises:
            ValueError: When a rectangle is outside the image and
                ``outside="error"``.
        """
        crops: List[Optional[ImageData]] = []
        for position, region in enumerate(regions):
            crop = image.crop(region)
            if crop is None and outside == "error":
                raise ValueError(
                    f"rectangle {position} {list(region)} lies outside the "
                    f"{image.width}x{image.height} image; set outside='none' to "
                    "keep its position empty"
                )
            crops.append(crop)

        result = {"crops": Batch.of(crops)}

        return result


class ResizeBlock(Block):
    """Resize one image to a fixed width and height.

    The aspect ratio is not kept. The result has a new ``image_id``; its
    provenance records the actual size ratio, so coordinates found in the
    resized image map back to the input and to the root.
    """

    type = "v2/resize"
    outputs = {"image": Output(IMAGE_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to resize.")
        width: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=1, description="Width of the result in pixels."
        )
        height: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=1, description="Height of the result in pixels."
        )
        interpolation: Literal["bilinear", "nearest"] = Field(
            default="bilinear",
            description="Pixel sampling: antialiased bilinear or nearest neighbour.",
        )

    def run(
        self, *, image: ImageData, width: int, height: int, interpolation: str
    ) -> Dict[str, Any]:
        """Resize ``image`` to ``width`` x ``height``.

        Args:
            image: Image to resize.
            width: Width of the result in pixels.
            height: Height of the result in pixels.
            interpolation: ``"bilinear"`` or ``"nearest"``.

        Returns:
            ``image``: the resized image on the input's device.
        """
        resized = image.resize((height, width), interpolation=interpolation)
        result = {"image": resized}

        return result


class InvertBlock(Block):
    """Invert every pixel of one image (``255 - value``).

    Returns a new tensor with the input's identity and provenance; the input
    image is not modified.
    """

    type = "v2/invert"
    outputs = {"image": Output(IMAGE_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to invert.")

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        """Invert ``image``.

        Args:
            image: Image to invert.

        Returns:
            ``image``: the same image with pixels ``255 - value``.
        """
        inverted = image.with_pixels(255 - image.tensor_image)
        result = {"image": inverted}

        return result


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

    type = "v2/mosaic"
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


class HasBrightnessBlock(Block):
    """Decide whether the mean pixel value of an image reaches a minimum.

    ``keep`` is a plain ``bool``; bind it to the ``condition`` of
    ``v2/continue_if`` to gate later steps. The mean is computed on the image's
    device; only that one number is read back.
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

    def run(self, *, image: ImageData, minimum: float) -> Dict[str, Any]:
        """Compare the mean pixel value of ``image`` with ``minimum``.

        Args:
            image: Image to measure.
            minimum: Inclusive threshold in ``0..255``.

        Returns:
            ``keep``: ``True`` when the mean over all pixels and channels is at
            least ``minimum``.
        """
        mean = image.tensor_image.mean(dtype=torch.float32).item()
        keep = bool(mean >= minimum)
        result = {"keep": keep}

        return result


def _common_device(images: Batch[ImageData]) -> torch.device:
    devices = {image.device for image in images}
    if len(devices) > 1:
        raise ValueError(
            "v2/mosaic needs all images on one device, got "
            f"{sorted(str(device) for device in devices)}"
        )

    (device,) = devices

    return device
