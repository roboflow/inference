"""Native V2 CPU image blocks.

Each block is an ordinary Python object: its constructor receives validated
static configuration, and ``run(**named_inputs)`` receives payloads or
``Batch`` views and returns a mapping of output names to ordinary results.
A batch view exposes its group's layout and metadata, which these blocks do
not need. Blocks never see buffers or pulse identifiers: the engine assembles
buffers and derives output layouts and metadata from the declared
:class:`BlockContract`.

Blocks in this module:

* ``v2/crop``: item image -> appended ``crops`` group and a preserved
  ``summary`` mapping. Configured rectangles are clipped to the image; empty
  results are omitted while keeping their configured local index.
* ``v2/invert``: item image -> preserved pixel-inverted ``image``.
* ``v2/mosaic``: batch of images -> one collapsed ``image`` canvas and a
  collapsed ``count``. An empty group yields a blank canvas and count zero.
* ``v2/has_brightness``: item image -> preserved boolean ``keep``.

Image payloads are RGB ``uint8`` NumPy arrays as validated by
:func:`roboflow_workflows.execution_engine.v2.blocks.kinds.is_image_payload`.
"""

import math
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from roboflow_workflows.execution_engine.v2.blocks.kinds import (
    BOOLEAN_KIND,
    CROP_SUMMARY_KIND,
    IMAGE_CHANNELS,
    IMAGE_KIND,
    INTEGER_KIND,
    is_image_payload,
)
from roboflow_workflows.execution_engine.v2.contracts import (
    BlockContract,
    InputSpec,
    OutputSpec,
)
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.errors import ContractError

CROP_BLOCK_NAME = "v2/crop"
INVERT_BLOCK_NAME = "v2/invert"
MOSAIC_BLOCK_NAME = "v2/mosaic"
HAS_BRIGHTNESS_BLOCK_NAME = "v2/has_brightness"

CROP_REGIONS_AXIS = "regions"

Region = Tuple[int, int, int, int]

CROP_CONTRACT = BlockContract(
    reference="image",
    inputs={"image": InputSpec(kind=IMAGE_KIND, view="item")},
    outputs={
        "crops": OutputSpec(
            kind=IMAGE_KIND,
            transform="append",
            axis=CROP_REGIONS_AXIS,
            stationary=False,
        ),
        "summary": OutputSpec(kind=CROP_SUMMARY_KIND, transform="preserve"),
    },
)

INVERT_CONTRACT = BlockContract(
    reference="image",
    inputs={"image": InputSpec(kind=IMAGE_KIND, view="item")},
    outputs={"image": OutputSpec(kind=IMAGE_KIND, transform="preserve")},
)

MOSAIC_CONTRACT = BlockContract(
    reference="images",
    inputs={"images": InputSpec(kind=IMAGE_KIND, view="batch")},
    outputs={
        "image": OutputSpec(kind=IMAGE_KIND, transform="collapse"),
        "count": OutputSpec(kind=INTEGER_KIND, transform="collapse"),
    },
)

HAS_BRIGHTNESS_CONTRACT = BlockContract(
    reference="image",
    inputs={"image": InputSpec(kind=IMAGE_KIND, view="item")},
    outputs={"keep": OutputSpec(kind=BOOLEAN_KIND, transform="preserve")},
)


class CropBlock:
    """Crop configured rectangles out of one image.

    Rectangles are ``[x0, y0, x1, y1]`` pixel coordinates with exclusive upper
    bounds. Each rectangle is clipped to the image. A rectangle that becomes
    empty after clipping is omitted from ``crops``; the surviving crops keep
    the local index of their configured rectangle, so the returned ``Batch``
    may have sparse indices such as ``(0,)`` and ``(2,)``.

    The crops are copies, not views into the parent image.

    The ``crops`` output is declared as dynamic nesting because the set of
    surviving rectangles depends on the image size. A stationary static-crop
    variant is intentionally not provided in this catalogue.
    """

    def __init__(self, *, regions: Sequence[Region]):
        """Create a crop block.

        Args:
            regions: Rectangles ``(x0, y0, x1, y1)`` in configured order.
        """
        self._regions: Tuple[Region, ...] = tuple(regions)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CropBlock":
        """Build a crop block from a step ``config`` mapping.

        Args:
            config: Mapping with the required key ``regions``: a list of
                ``[x0, y0, x1, y1]`` integer rectangles with ``x0 <= x1`` and
                ``y0 <= y1``. Zero-area rectangles are accepted and are
                omitted at run time like clipped-away ones.

        Returns:
            Configured crop block.

        Raises:
            ContractError: If ``regions`` is missing or malformed, or unknown
                configuration keys are present.
        """
        _reject_unknown_config_keys(
            config, allowed={"regions"}, block_name=CROP_BLOCK_NAME
        )
        if "regions" not in config:
            raise ContractError(
                f"{CROP_BLOCK_NAME} requires config key 'regions': a list of "
                "[x0, y0, x1, y1] rectangles"
            )

        regions = _parse_regions(config["regions"])
        block = cls(regions=regions)

        return block

    @property
    def regions(self) -> Tuple[Region, ...]:
        """Configured rectangles in configured order."""
        return self._regions

    def run(self, *, image: np.ndarray) -> Mapping[str, Any]:
        """Crop the configured rectangles from ``image``.

        Args:
            image: RGB ``uint8`` array of shape ``(height, width, 3)``.

        Returns:
            Mapping with ``crops`` (a ``Batch`` of crop arrays whose indices
            are the configured rectangle positions that survived clipping) and
            ``summary`` (a mapping with ``crop_count``, ``image_height``,
            ``image_width``, ``kept_regions`` and ``crop_dimensions``).

        Raises:
            ContractError: If ``image`` is not a valid image payload.
        """
        _require_image(image, argument="image", block_name=CROP_BLOCK_NAME)
        height, width = image.shape[:2]

        crops: List[np.ndarray] = []
        indices: List[Tuple[int, ...]] = []
        for position, (x0, y0, x1, y1) in enumerate(self._regions):
            clipped_x0 = max(0, x0)
            clipped_y0 = max(0, y0)
            clipped_x1 = min(width, x1)
            clipped_y1 = min(height, y1)
            if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
                continue

            crop = image[clipped_y0:clipped_y1, clipped_x0:clipped_x1].copy(order="C")
            crops.append(crop)
            indices.append((position,))

        summary = {
            "crop_count": len(crops),
            "image_height": int(height),
            "image_width": int(width),
            "kept_regions": [index[0] for index in indices],
            "crop_dimensions": [[int(c.shape[0]), int(c.shape[1])] for c in crops],
        }
        result = {"crops": Batch.of(crops, indices=indices), "summary": summary}

        return result


class InvertBlock:
    """Invert every pixel of one image (``255 - value``).

    The block has no configuration and returns a new array; the input image is
    not modified.
    """

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "InvertBlock":
        """Build an invert block from a step ``config`` mapping.

        Args:
            config: Must be empty; the block has no configuration.

        Returns:
            Invert block.

        Raises:
            ContractError: If any configuration key is present.
        """
        _reject_unknown_config_keys(config, allowed=set(), block_name=INVERT_BLOCK_NAME)
        block = cls()

        return block

    def run(self, *, image: np.ndarray) -> Mapping[str, Any]:
        """Invert ``image``.

        Args:
            image: RGB ``uint8`` array of shape ``(height, width, 3)``.

        Returns:
            Mapping with ``image``: a new array equal to ``255 - image``.

        Raises:
            ContractError: If ``image`` is not a valid image payload.
        """
        _require_image(image, argument="image", block_name=INVERT_BLOCK_NAME)
        inverted = np.subtract(np.uint8(255), image, dtype=np.uint8)
        result = {"image": inverted}

        return result


class MosaicBlock:
    """Tile a group of images into one canvas.

    Every image in the group is resized with nearest-neighbour sampling to a
    square tile of ``tile_size`` pixels (aspect ratio is not preserved) and
    placed row by row on a grid with ``columns`` columns. When ``columns`` is
    not configured, the grid is as square as possible.

    Empty-group policy: a group with no images produces one blank canvas of
    ``tile_size`` by ``tile_size`` pixels filled with ``background`` and a
    ``count`` of zero. This is a documented plugin choice; the engine treats
    the empty group as an ordinary valid input.
    """

    def __init__(
        self,
        *,
        tile_size: int = 64,
        columns: Optional[int] = None,
        background: int = 0,
    ):
        """Create a mosaic block.

        Args:
            tile_size: Side length in pixels of each square tile.
            columns: Number of grid columns; ``None`` chooses ``ceil(sqrt(n))``.
            background: Fill value ``0..255`` for unused canvas area and for
                the empty-group canvas.
        """
        self._tile_size = tile_size
        self._columns = columns
        self._background = background

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MosaicBlock":
        """Build a mosaic block from a step ``config`` mapping.

        Args:
            config: Optional keys ``tile_size`` (positive int, default 64),
                ``columns`` (positive int or absent) and ``background``
                (int in ``0..255``, default 0).

        Returns:
            Configured mosaic block.

        Raises:
            ContractError: If a value is out of range or an unknown key is
                present.
        """
        _reject_unknown_config_keys(
            config,
            allowed={"tile_size", "columns", "background"},
            block_name=MOSAIC_BLOCK_NAME,
        )
        tile_size = _positive_int(
            config.get("tile_size", 64), key="tile_size", block_name=MOSAIC_BLOCK_NAME
        )
        columns = config.get("columns")
        if columns is not None:
            columns = _positive_int(
                columns, key="columns", block_name=MOSAIC_BLOCK_NAME
            )
        background = _channel_value(
            config.get("background", 0), key="background", block_name=MOSAIC_BLOCK_NAME
        )

        block = cls(tile_size=tile_size, columns=columns, background=background)

        return block

    @property
    def tile_size(self) -> int:
        """Side length in pixels of each square tile."""
        return self._tile_size

    def run(self, *, images: Iterable[np.ndarray]) -> Mapping[str, Any]:
        """Tile ``images`` into one canvas.

        Args:
            images: ``Batch`` (or any iterable) of RGB ``uint8`` images. The
                engine supplies one trailing-axis group per invocation.

        Returns:
            Mapping with ``image`` (the canvas) and ``count`` (number of tiled
            images). An empty group yields the blank canvas and count 0.

        Raises:
            ContractError: If any element is not a valid image payload.
        """
        tiles = []
        for position, image in enumerate(images):
            _require_image(
                image, argument=f"images[{position}]", block_name=MOSAIC_BLOCK_NAME
            )
            tile = _resize_nearest(image, height=self._tile_size, width=self._tile_size)
            tiles.append(tile)

        count = len(tiles)
        if count == 0:
            canvas = self._blank_canvas(rows=1, columns=1)
            result = {"image": canvas, "count": 0}
            return result

        columns = self._columns or math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)
        canvas = self._blank_canvas(rows=rows, columns=columns)
        for position, tile in enumerate(tiles):
            row, column = divmod(position, columns)
            y0 = row * self._tile_size
            x0 = column * self._tile_size
            canvas[y0 : y0 + self._tile_size, x0 : x0 + self._tile_size] = tile

        result = {"image": canvas, "count": count}

        return result

    def _blank_canvas(self, *, rows: int, columns: int) -> np.ndarray:
        shape = (rows * self._tile_size, columns * self._tile_size, IMAGE_CHANNELS)
        canvas = np.full(shape, self._background, dtype=np.uint8)

        return canvas


class HasBrightnessBlock:
    """Decide whether an image's mean pixel value reaches a minimum.

    Intended as a gate producer: its ``keep`` output is a plain ``bool`` that a
    step's ``when`` selector can reference.
    """

    def __init__(self, *, minimum: float):
        """Create a brightness predicate.

        Args:
            minimum: Inclusive mean-pixel threshold in ``0..255``.
        """
        self._minimum = minimum

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HasBrightnessBlock":
        """Build a brightness predicate from a step ``config`` mapping.

        Args:
            config: Mapping with required key ``minimum``: a number in
                ``0..255``.

        Returns:
            Configured predicate block.

        Raises:
            ContractError: If ``minimum`` is missing or out of range, or an
                unknown key is present.
        """
        _reject_unknown_config_keys(
            config, allowed={"minimum"}, block_name=HAS_BRIGHTNESS_BLOCK_NAME
        )
        if "minimum" not in config:
            raise ContractError(
                f"{HAS_BRIGHTNESS_BLOCK_NAME} requires config key 'minimum' "
                "(mean pixel threshold in 0..255)"
            )

        minimum = config["minimum"]
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
            raise ContractError(
                f"{HAS_BRIGHTNESS_BLOCK_NAME} config 'minimum' must be a number, "
                f"got {type(minimum).__name__}"
            )
        if not 0 <= minimum <= 255:
            raise ContractError(
                f"{HAS_BRIGHTNESS_BLOCK_NAME} config 'minimum' must be in 0..255, "
                f"got {minimum}"
            )

        block = cls(minimum=float(minimum))

        return block

    @property
    def minimum(self) -> float:
        """Inclusive mean-pixel threshold."""
        return self._minimum

    def run(self, *, image: np.ndarray) -> Mapping[str, Any]:
        """Compare the mean pixel value of ``image`` with the threshold.

        Args:
            image: RGB ``uint8`` array of shape ``(height, width, 3)``.

        Returns:
            Mapping with ``keep``: ``True`` when ``image.mean() >= minimum``.

        Raises:
            ContractError: If ``image`` is not a valid image payload.
        """
        _require_image(image, argument="image", block_name=HAS_BRIGHTNESS_BLOCK_NAME)
        keep = bool(float(image.mean()) >= self._minimum)
        result = {"keep": keep}

        return result


def _require_image(payload: Any, *, argument: str, block_name: str) -> None:
    if is_image_payload(payload):
        return

    description = _describe_payload(payload)
    raise ContractError(
        f"{block_name} input '{argument}' must be an RGB uint8 numpy array of "
        f"shape (height, width, 3) with positive size, got {description}"
    )


def _describe_payload(payload: Any) -> str:
    if isinstance(payload, np.ndarray):
        description = f"ndarray(shape={payload.shape}, dtype={payload.dtype})"
    else:
        description = type(payload).__name__

    return description


def _reject_unknown_config_keys(
    config: Mapping[str, Any], *, allowed: set, block_name: str
) -> None:
    if not isinstance(config, Mapping):
        raise ContractError(
            f"{block_name} config must be a mapping, got {type(config).__name__}"
        )

    unknown = sorted(set(config) - allowed)
    if unknown:
        allowed_text = ", ".join(sorted(allowed)) or "(none)"
        raise ContractError(
            f"{block_name} received unknown config keys {unknown}; "
            f"allowed keys: {allowed_text}"
        )


def _parse_regions(value: Any) -> List[Region]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(
            f"{CROP_BLOCK_NAME} config 'regions' must be a list of "
            f"[x0, y0, x1, y1] rectangles, got {type(value).__name__}"
        )

    regions: List[Region] = []
    for position, rectangle in enumerate(value):
        if not isinstance(rectangle, (list, tuple)) or len(rectangle) != 4:
            raise ContractError(
                f"{CROP_BLOCK_NAME} config 'regions[{position}]' must be "
                f"[x0, y0, x1, y1], got {rectangle!r}"
            )

        coordinates = []
        for coordinate in rectangle:
            if isinstance(coordinate, bool) or not isinstance(coordinate, int):
                raise ContractError(
                    f"{CROP_BLOCK_NAME} config 'regions[{position}]' must contain "
                    f"integers, got {rectangle!r}"
                )
            coordinates.append(coordinate)

        x0, y0, x1, y1 = coordinates
        if x1 < x0 or y1 < y0:
            raise ContractError(
                f"{CROP_BLOCK_NAME} config 'regions[{position}]' must satisfy "
                f"x0 <= x1 and y0 <= y1, got {rectangle!r}"
            )
        regions.append((x0, y0, x1, y1))

    return regions


def _positive_int(value: Any, *, key: str, block_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError(
            f"{block_name} config '{key}' must be a positive integer, got {value!r}"
        )

    return value


def _channel_value(value: Any, *, key: str, block_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise ContractError(
            f"{block_name} config '{key}' must be an integer in 0..255, got {value!r}"
        )

    return value


def _resize_nearest(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    rows = (np.arange(height) * source_height) // height
    columns = (np.arange(width) * source_width) // width
    resized = image[rows[:, None], columns[None, :]]

    return resized
