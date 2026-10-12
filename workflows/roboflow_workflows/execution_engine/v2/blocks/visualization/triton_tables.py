"""The ``triton`` backend: one drawing launch per painter call and batch.

The drawing is a list of **paint items** per image, in paint order: the
boxes in detection order, then the labels in detection order. Each item
covers a set of pixels and gives them a color::

    box    border = rect minus hole, one color      (gpu_draw_boxes, square)
    label  opaque pixels of its sprite canvas        (V1 label paste)

The rule that reproduces painting in order, with one writer per pixel::

    item i writes pixel p  <=>  i covers p  and  no later item of the image covers p

The kernel checks the later items i+1 .. later_end-1; ``later_end`` stops
after the last later item whose rect meets item i's rect (host-computed).
So later boxes win over earlier ones, labels over every box and later labels
over earlier ones, as the tensor backend paints. Threads never write the same
pixel, so no ordering, atomics or owner buffer is needed. That also needs the
scenes of one launch to share no byte (``_check_disjoint``).

All geometry is computed on the host from ``DetectionDrawing`` and uploaded
once per launch through a pinned ring (``PinnedTableRing``)::

    scenes   [B, 4]   SCENE_COLUMNS: data_ptr and strides of each scene
    canvases [K]      data_ptr of (h, w, 4) RGBA label canvases
    items    [N, 12]  ITEM_COLUMNS
    pieces   [R, 6]   PIECE_COLUMNS: disjoint rects holding an item's pixels
    tiles    [T]      piece of each tile; the launch grid is (T,)

``_triton_paint`` holds the kernel and is imported on the first launch only.
On a non-CUDA device or without Triton, ``paint`` raises
``DrawingBackendUnavailable`` before painting anything; there is no fallback.

Ported from development example 13 (``compact_kernels``, ``staged_kernels``),
without its counters and profiling ranges.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from roboflow_workflows.core_steps.visualizations.label.v1_tensor import (
    _LabelSprite,
    _staged_upload,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.drawing import (
    BOX_THICKNESS,
    PALETTE,
    DetectionDrawing,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.kernels import (
    DrawingBackendUnavailable,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.labels import (
    LabelPlacement,
)

BOX, LABEL = 0, 1
SCENE_COLUMNS = ("data_ptr", "stride_c", "stride_h", "stride_w")
ITEM_COLUMNS = (
    "image",  # index in the batch
    "kind",  # BOX or LABEL
    "r1",  # rect, inclusive, inside the frame
    "c1",
    "r2",
    "c2",
    "hole_r1",  # box: inner rect, inclusive, inside the frame; empty: r1 > r2
    "hole_c1",
    "hole_r2",
    "hole_c2",
    "paint",  # box: 0xRRGGBB; label: canvas index
    "later_end",  # one past the last later item of the image meeting this rect
)
PIECE_COLUMNS = ("item", "r1", "c1", "width", "pixels", "first_tile")
SCENE_COLUMN = {name: index for index, name in enumerate(SCENE_COLUMNS)}
ITEM_COLUMN = {name: index for index, name in enumerate(ITEM_COLUMNS)}
PIECE_COLUMN = {name: index for index, name in enumerate(PIECE_COLUMNS)}
TILE_PIXELS = 256
TABLE_RING_DEPTH = 4
CANVAS_CAPACITY = 512
_RECT = [ITEM_COLUMN[name] for name in ("r1", "c1", "r2", "c2")]
_HOLE = [ITEM_COLUMN[name] for name in ("hole_r1", "hole_c1", "hole_r2", "hole_c2")]
_EMPTY_HOLE = (0, 0, -1, -1)
_PALETTE_RGB = np.asarray([color.as_rgb() for color in PALETTE.colors], np.int64)


@dataclass
class LaunchTables:
    """Everything one launch reads, before upload.

    Args:
        scenes: Target CHW uint8 tensors, in batch order.
        canvases: RGBA label canvases, indexed by a label item's ``paint``.
        items: ``(N, len(ITEM_COLUMNS))`` int64 paint items, image-major.
        pieces: ``(R, len(PIECE_COLUMNS))`` int64 pixel pieces.
        tile_piece: ``(T,)`` int64 piece of each tile.
    """

    scenes: List[torch.Tensor]
    canvases: List[torch.Tensor]
    items: np.ndarray
    pieces: np.ndarray
    tile_piece: np.ndarray

    @property
    def tile_count(self) -> int:
        """Programs of the launch grid."""
        return int(self.tile_piece.shape[0])

    def packed(self) -> Tuple[np.ndarray, Dict[str, slice]]:
        """One int64 vector of every table, each section 16-byte aligned.

        Returns:
            The vector and the slice of each section (``scenes``,
            ``canvases``, ``items``, ``pieces``, ``tiles``).
        """
        scene_rows = []
        for scene in self.scenes:
            stride_c, stride_h, stride_w = scene.stride()
            row = {
                "data_ptr": scene.data_ptr(),
                "stride_c": stride_c,
                "stride_h": stride_h,
                "stride_w": stride_w,
            }
            scene_rows.append([row[name] for name in SCENE_COLUMNS])
        sections = {
            "scenes": np.asarray(scene_rows, dtype=np.int64).reshape(-1),
            "canvases": np.asarray(
                [canvas.data_ptr() for canvas in self.canvases], dtype=np.int64
            ),
            "items": self.items.reshape(-1),
            "pieces": self.pieces.reshape(-1),
            "tiles": self.tile_piece,
        }
        parts, slices, offset = [], {}, 0
        for name, values in sections.items():
            padded = len(values) + len(values) % 2
            part = np.zeros(padded, dtype=np.int64)
            part[: len(values)] = values
            parts.append(part)
            slices[name] = slice(offset, offset + len(values))
            offset += padded
        vector = np.concatenate(parts)

        return vector, slices


def box_items(
    drawings: Sequence[DetectionDrawing],
    frame_sizes: Sequence[Tuple[int, int]],
    *,
    thickness: int,
) -> np.ndarray:
    """Paint items of every box of the batch, image-major, ``later_end`` unset.

    ``gpu_draw_boxes`` with square corners paints four bands per box: top and
    bottom over the full width, left and right between them. Their union is
    exactly ``rect`` minus ``hole``::

        rect = rows [y1, y2],          cols [x1, x2]
        hole = rows [y1 + t, y2 - t],  cols [x1 + t, x2 - t]
        x1 = min(xa, xb) - t // 2,  x2 = max(xa, xb) + t // 2   (y the same)

    Both are clipped to the frame; boxes with nothing inside are dropped.

    Args:
        drawings: Prepared detections per image.
        frame_sizes: ``(height, width)`` per image.
        thickness: Border width ``t``.

    Returns:
        ``(N, len(ITEM_COLUMNS))`` int64 items.

    Raises:
        ValueError: On a negative class id (as ``ColorPalette.by_idx``).
    """
    counts = [len(drawing) for drawing in drawings]
    if sum(counts) == 0:
        return np.zeros((0, len(ITEM_COLUMNS)), dtype=np.int64)

    # Same integer conversion as the tensor backend: xyxy.astype(int).
    xy = np.concatenate(
        [drawing.xyxy.astype(int).reshape(-1, 4) for drawing in drawings]
    ).astype(np.int64)
    class_id = np.concatenate([drawing.class_id for drawing in drawings])
    class_id = class_id.astype(np.int64)
    if (class_id < 0).any():
        raise ValueError("idx argument should not be negative")

    image = np.repeat(np.arange(len(drawings)), counts)
    sizes = np.asarray(frame_sizes, dtype=np.int64).reshape(-1, 2)
    last_row = sizes[image, 0] - 1
    last_col = sizes[image, 1] - 1
    outer = thickness // 2
    y1 = np.minimum(xy[:, 1], xy[:, 3]) - outer
    y2 = np.maximum(xy[:, 1], xy[:, 3]) + outer
    x1 = np.minimum(xy[:, 0], xy[:, 2]) - outer
    x2 = np.maximum(xy[:, 0], xy[:, 2]) + outer

    rect = _clip(y1, x1, y2, x2, last_row=last_row, last_col=last_col)
    hole = _clip(
        y1 + thickness,
        x1 + thickness,
        y2 - thickness,
        x2 - thickness,
        last_row=last_row,
        last_col=last_col,
    )
    hole_empty = (hole[:, 0] > hole[:, 2]) | (hole[:, 1] > hole[:, 3])
    hole[hole_empty] = _EMPTY_HOLE
    rgb = _PALETTE_RGB[class_id % len(_PALETTE_RGB)]
    paint = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]

    items = np.zeros((len(image), len(ITEM_COLUMNS)), dtype=np.int64)
    items[:, ITEM_COLUMN["image"]] = image
    items[:, ITEM_COLUMN["kind"]] = BOX
    items[:, _RECT] = rect
    items[:, _HOLE] = hole
    items[:, ITEM_COLUMN["paint"]] = paint
    inside = (rect[:, 0] <= rect[:, 2]) & (rect[:, 1] <= rect[:, 3])
    items = items[inside]

    return items


def _clip(
    r1: np.ndarray,
    c1: np.ndarray,
    r2: np.ndarray,
    c2: np.ndarray,
    *,
    last_row: np.ndarray,
    last_col: np.ndarray,
) -> np.ndarray:
    clipped = np.stack(
        [
            np.maximum(r1, 0),
            np.maximum(c1, 0),
            np.minimum(r2, last_row),
            np.minimum(c2, last_col),
        ],
        axis=1,
    )

    return clipped


def later_ends(items: np.ndarray) -> np.ndarray:
    """``later_end`` of every item: the later items its pixels must check.

    Args:
        items: Image-major paint items in paint order.

    Returns:
        Per item ``i``, one past the last later item of the same image whose
        rect meets ``i``'s rect; ``i + 1`` when there is none.
    """
    ends = np.arange(1, len(items) + 1, dtype=np.int64)
    if len(items) == 0:
        return ends

    images = items[:, ITEM_COLUMN["image"]]
    starts = np.flatnonzero(np.r_[True, images[1:] != images[:-1]])
    for start, stop in zip(starts, np.r_[starts[1:], len(items)]):
        r1, c1, r2, c2 = items[start:stop][:, _RECT].T
        meet = (
            (r1[:, None] <= r2[None, :])
            & (r2[:, None] >= r1[None, :])
            & (c1[:, None] <= c2[None, :])
            & (c2[:, None] >= c1[None, :])
        )
        meet = np.triu(meet, k=1)
        last = meet.shape[1] - 1 - np.argmax(meet[:, ::-1], axis=1)
        ends[start:stop] = np.where(
            meet.any(axis=1), start + last + 1, ends[start:stop]
        )

    return ends


def item_pieces(items: np.ndarray) -> np.ndarray:
    """Disjoint rects holding exactly the pixels of every item.

    A label is one piece (its rect; transparent texels are skipped by the
    kernel). A box is ``rect`` minus ``hole``: top and bottom strips over the
    full width, left and right strips beside the hole. A box without a hole
    is one piece.

    Args:
        items: ``(N, len(ITEM_COLUMNS))`` paint items.

    Returns:
        ``(R, len(PIECE_COLUMNS))`` int64 pieces, ``first_tile`` filled.
    """
    r1, c1, r2, c2 = items[:, _RECT].T
    h1, hc1, h2, hc2 = items[:, _HOLE].T
    # Without a hole (labels, solid boxes): top strip = whole rect, others empty.
    no_hole = (items[:, ITEM_COLUMN["kind"]] == LABEL) | (h1 > h2)
    h1 = np.where(no_hole, r2 + 1, h1)
    h2 = np.where(no_hole, r2, h2)
    hc1 = np.where(no_hole, c1, hc1)
    hc2 = np.where(no_hole, c2, hc2)

    index = np.arange(len(items))
    strips = [
        (index, r1, c1, h1 - 1, c2),  # top
        (index, h2 + 1, c1, r2, c2),  # bottom
        (index, h1, c1, h2, hc1 - 1),  # left
        (index, h1, hc2 + 1, h2, c2),  # right
    ]
    item, top, left, bottom, right = (np.concatenate(column) for column in zip(*strips))
    height = bottom - top + 1
    width = right - left + 1
    keep = (height > 0) & (width > 0)
    columns = {
        "item": item,
        "r1": top,
        "c1": left,
        "width": width,
        "pixels": height * width,
    }

    pieces = np.zeros((int(keep.sum()), len(PIECE_COLUMNS)), dtype=np.int64)
    for name, values in columns.items():
        pieces[:, PIECE_COLUMN[name]] = values[keep]
    tiles = _tiles_of(pieces)
    pieces[:, PIECE_COLUMN["first_tile"]] = np.cumsum(tiles) - tiles

    return pieces


def tile_table(pieces: np.ndarray) -> np.ndarray:
    """Piece index of every tile of ``TILE_PIXELS`` pixels.

    Args:
        pieces: Output of ``item_pieces``.

    Returns:
        ``(T,)`` int64 piece indices.
    """
    tiles = _tiles_of(pieces)
    tile_piece = np.repeat(np.arange(len(pieces), dtype=np.int64), tiles)

    return tile_piece


def _tiles_of(pieces: np.ndarray) -> np.ndarray:
    tiles = -(-pieces[:, PIECE_COLUMN["pixels"]] // TILE_PIXELS)

    return tiles


class SpriteCanvases:
    """RGBA canvases of label sprites: built once per sprite, LRU-bounded.

    A canvas spans the sprite's opaque bounding box: ``(h, w, 4)`` uint8,
    RGB from the sprite, alpha 255 on its pixels and 0 elsewhere. Entries
    keep their sprite alive, so a sprite id is never reused for another.

    Canvases used by a launch stay referenced until the next ``begin_batch``
    (``batch_canvases``). The painter waits for its stream before it
    returns, so an evicted canvas is never freed under a queued kernel.

    Args:
        capacity: Most cached canvases.
    """

    def __init__(self, *, capacity: int) -> None:
        self._capacity = capacity
        self._entries: "OrderedDict[int, Tuple[_LabelSprite, torch.Tensor]]" = (
            OrderedDict()
        )
        self._pending_uploads: List[Tuple[torch.Tensor, Any]] = []
        self.batch_canvases: List[torch.Tensor] = []
        # id -> (sprite, index); holding the sprite keeps its id unique.
        self._batch_index: Dict[int, Tuple[_LabelSprite, int]] = {}

    @property
    def cached_canvases(self) -> int:
        """Number of canvases in the LRU cache (at most the capacity)."""
        return len(self._entries)

    def begin_batch(self) -> None:
        """Forget the previous launch's canvases (its painter has waited)."""
        self.batch_canvases = []
        self._batch_index = {}
        while self._pending_uploads and self._pending_uploads[0][1].query():
            self._pending_uploads.pop(0)

    def index_of(self, sprite: _LabelSprite, *, device: torch.device) -> int:
        """The batch canvas index of ``sprite``, building its canvas if new.

        Args:
            sprite: A sprite with at least one pixel.
            device: Device of the scenes.

        Returns:
            Index into ``batch_canvases``.
        """
        known = self._batch_index.get(id(sprite))
        if known is not None and known[0] is sprite:
            return known[1]

        entry = self._entries.get(id(sprite))
        if entry is not None and entry[0] is sprite:
            self._entries.move_to_end(id(sprite))
            canvas = entry[1]
        else:
            canvas = self._build(sprite, device=device)
            self._entries[id(sprite)] = (sprite, canvas)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
        self.batch_canvases.append(canvas)
        index = len(self.batch_canvases) - 1
        self._batch_index[id(sprite)] = (sprite, index)

        return index

    def _build(self, sprite: _LabelSprite, *, device: torch.device) -> torch.Tensor:
        height = sprite.row_max - sprite.row_min + 1
        width = sprite.col_max - sprite.col_min + 1
        texels = (sprite.rows.astype(np.int64) - sprite.row_min) * width + (
            sprite.cols.astype(np.int64) - sprite.col_min
        )
        # Pinned non-blocking upload; the staging is kept until its copy ran.
        pending = self._pending_uploads if device.type == "cuda" else None
        texels_dev = _staged_upload(texels, device, pending)
        opaque = torch.full(
            (sprite.pixel_count, 1), 255, dtype=torch.uint8, device=device
        )
        canvas = torch.zeros((height * width, 4), dtype=torch.uint8, device=device)
        canvas[texels_dev] = torch.cat([sprite.colors_dev, opaque], dim=1)
        shaped = canvas.view(height, width, 4)

        return shaped


class _RingSlot:
    __slots__ = ("host", "device", "event")

    def __init__(self) -> None:
        self.host: Optional[torch.Tensor] = None
        self.device: Optional[torch.Tensor] = None
        self.event: Optional[torch.cuda.Event] = None


class PinnedTableRing:
    """Upload small host tables without a host wait.

    Each slot pairs a pinned host slab with a device slab. ``upload`` writes
    the values into the next slot's host slab and queues a non-blocking copy
    on the current stream; an event recorded behind the copy guards the host
    slab. Before a slot is rewritten, its previous copy must have run: the
    ring checks the event and waits only if it has not fired. A painter
    makes one upload per call and waits for its stream before returning, so
    that wait does not happen in steady state.

    The returned device tensor is valid for work queued later on the same
    stream, until the slot comes round again ``depth`` uploads later.

    Args:
        depth: Number of slots.
    """

    def __init__(self, *, depth: int) -> None:
        self._slots = [_RingSlot() for _ in range(depth)]
        self._cursor = 0

    def upload(self, values: np.ndarray, *, device: torch.device) -> torch.Tensor:
        """Queue a copy of 1-D int64 ``values`` to ``device`` on the current stream.

        Args:
            values: 1-D int64 array.
            device: Target device. On CPU the copy is a plain copy.

        Returns:
            A device tensor of ``len(values)`` elements, a view of the slot.
        """
        slot = self._slots[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._slots)
        if slot.event is not None and not slot.event.query():
            slot.event.synchronize()

        count = int(values.shape[0])
        self._fit(slot, count=count, device=device)
        slot.host[:count].numpy()[:] = values
        device_view = slot.device[:count]
        device_view.copy_(slot.host[:count], non_blocking=True)
        if device.type == "cuda":
            slot.event = slot.event or torch.cuda.Event()
            slot.event.record(torch.cuda.current_stream(device))

        return device_view

    def _fit(self, slot: _RingSlot, *, count: int, device: torch.device) -> None:
        # Grow geometrically. The host slab's last copy has run (checked above).
        if (
            slot.device is not None
            and slot.device.device == device
            and slot.device.numel() >= count
        ):
            return

        capacity = 64 if slot.device is None else slot.device.numel()
        while capacity < count:
            capacity *= 2
        if slot.device is not None and slot.device.is_cuda:
            # Kernels queued on this stream may still read the old device slab.
            slot.device.record_stream(torch.cuda.current_stream(slot.device.device))
        pin = device.type == "cuda"
        slot.host = torch.empty(capacity, dtype=torch.int64, pin_memory=pin)
        slot.device = torch.empty(capacity, dtype=torch.int64, device=device)


class TritonKernels:
    """Boxes and labels of a whole batch in one Triton launch.

    One instance holds one sprite placement cache, one canvas cache and one
    pinned table ring. Not thread-safe: one painter call at a time, and the
    caller waits for its stream before the next call.

    Args:
        canvas_capacity: Most cached label canvases.
    """

    name = "triton"

    def __init__(self, *, canvas_capacity: int = CANVAS_CAPACITY) -> None:
        self.placement = LabelPlacement()
        self.canvases = SpriteCanvases(capacity=canvas_capacity)
        self._ring = PinnedTableRing(depth=TABLE_RING_DEPTH)
        self._launch: Optional[Callable[..., None]] = None

    def paint(
        self,
        scenes: Sequence[torch.Tensor],
        drawings: Sequence[DetectionDrawing],
        *,
        boxes: bool,
        labels: bool,
    ) -> None:
        """Queue one launch drawing boxes and/or labels on every scene.

        Args:
            scenes: CHW RGB uint8 CUDA tensors, any strides without
                self-overlap, sharing no byte with each other; changed in
                place.
            drawings: Their prepared detections, in the same order.
            boxes: Draw box borders.
            labels: Paste labels, over the boxes.

        Raises:
            DrawingBackendUnavailable: Not on CUDA, or Triton cannot be
                imported. Nothing is painted.
            ValueError: On a bad scene, scenes that may share bytes, a
                negative class id, or a label outside the frame. Nothing is
                painted.
        """
        if not scenes:
            return

        device = scenes[0].device
        if device.type != "cuda":
            raise DrawingBackendUnavailable(
                f"the triton drawing backend runs on CUDA only, got {device}"
            )
        launch = self._launcher()

        tables = self.build_tables(scenes, drawings, boxes=boxes, labels=labels)
        if tables.tile_count == 0:
            return

        vector, slices = tables.packed()
        slab = self._ring.upload(vector, device=device)
        launch(
            {name: slab[part] for name, part in slices.items()},
            tiles=tables.tile_count,
        )

    def build_tables(
        self,
        scenes: Sequence[torch.Tensor],
        drawings: Sequence[DetectionDrawing],
        *,
        boxes: bool,
        labels: bool,
    ) -> LaunchTables:
        """Host tables of one launch; builds canvases of new label sprites.

        Args:
            scenes: CHW RGB uint8 tensors on one device.
            drawings: Their prepared detections, in the same order.
            boxes: Include box items.
            labels: Include label items.

        Returns:
            The tables; ``scenes`` and canvases are referenced, not copied.

        Raises:
            ValueError: As ``paint``.
        """
        if len(scenes) != len(drawings):
            raise ValueError(f"{len(scenes)} scenes but {len(drawings)} drawings")
        for scene in scenes:
            _check_scene(scene, device=scenes[0].device)
        _check_disjoint(scenes)

        self.canvases.begin_batch()
        sizes = [(int(scene.shape[1]), int(scene.shape[2])) for scene in scenes]
        parts = [np.zeros((0, len(ITEM_COLUMNS)), dtype=np.int64)]
        if boxes:
            parts.append(box_items(drawings, sizes, thickness=BOX_THICKNESS))
        if labels:
            parts.append(self._label_items(scenes, drawings))
        items = np.concatenate(parts)
        # Image-major; boxes before labels; detection order kept.
        order = np.lexsort(
            (
                np.arange(len(items)),
                items[:, ITEM_COLUMN["kind"]],
                items[:, ITEM_COLUMN["image"]],
            )
        )
        items = items[order]
        items[:, ITEM_COLUMN["later_end"]] = later_ends(items)
        pieces = item_pieces(items)

        tables = LaunchTables(
            scenes=list(scenes),
            canvases=list(self.canvases.batch_canvases),
            items=items,
            pieces=pieces,
            tile_piece=tile_table(pieces),
        )

        return tables

    def _label_items(
        self, scenes: Sequence[torch.Tensor], drawings: Sequence[DetectionDrawing]
    ) -> np.ndarray:
        images, rects, canvases = [], [], []
        for image, (scene, drawing) in enumerate(zip(scenes, drawings)):
            height, width = int(scene.shape[1]), int(scene.shape[2])
            sprites, origins = self.placement.sprites(
                drawing, frame_hw=(height, width), device=scene.device
            )
            for sprite, (origin_x, origin_y) in zip(sprites, origins):
                if sprite.pixel_count == 0:
                    continue
                top, left = origin_y + sprite.row_min, origin_x + sprite.col_min
                bottom, right = origin_y + sprite.row_max, origin_x + sprite.col_max
                if top < 0 or left < 0 or bottom >= height or right >= width:
                    raise ValueError("label sprite pixels fall outside the frame")
                images.append(image)
                rects.append((top, left, bottom, right))
                canvases.append(self.canvases.index_of(sprite, device=scene.device))

        items = np.zeros((len(images), len(ITEM_COLUMNS)), dtype=np.int64)
        items[:, ITEM_COLUMN["image"]] = images
        items[:, ITEM_COLUMN["kind"]] = LABEL
        items[:, _RECT] = np.asarray(rects, dtype=np.int64).reshape(-1, 4)
        items[:, _HOLE] = _EMPTY_HOLE
        items[:, ITEM_COLUMN["paint"]] = canvases

        return items

    def _launcher(self) -> Callable[..., None]:
        if self._launch is None:
            try:
                from roboflow_workflows.execution_engine.v2.blocks.visualization import (  # noqa: E501
                    _triton_paint,
                )
            except ImportError as error:
                raise DrawingBackendUnavailable(
                    f"the triton drawing backend needs Triton: {error}"
                ) from error
            self._launch = _triton_paint.launch

        return self._launch


def _check_scene(scene: torch.Tensor, *, device: torch.device) -> None:
    if scene.dtype != torch.uint8 or scene.dim() != 3 or scene.shape[0] != 3:
        raise ValueError(f"scene must be a CHW RGB uint8 tensor, got {scene.shape}")
    if scene.device != device:
        raise ValueError(f"scenes on several devices: {device} and {scene.device}")
    # Distinct pixels must be distinct bytes, or one thread's write leaks.
    extent = 1
    for size, stride in sorted(zip(scene.shape, scene.stride()), key=lambda x: x[1]):
        if size > 1 and stride < extent:
            raise ValueError(f"scene strides {scene.stride()} overlap in memory")
        extent = max(extent, stride * size)


def _check_disjoint(scenes: Sequence[torch.Tensor]) -> None:
    """Refuse scenes of one launch that may share a byte.

    Sharing a storage is fine; sharing a byte is not. Each channel plane of
    a scene lies within one byte span (``s*``: strides)::

        plane c = [ptr + c*s_c,  ptr + c*s_c + (h-1)*s_h + (w-1)*s_w]

    If no span of one scene meets a span of another, they share no byte.
    This accepts separate images, the images of one batch tensor and row
    bands of one image. It is conservative: interleaved views (e.g. left
    and right halves of an image) are refused although their pixels differ.
    Host metadata only: no copy, no sync.
    """
    spans = []
    for image, scene in enumerate(scenes):
        _, height, width = scene.shape
        if height == 0 or width == 0:
            continue
        stride_c, stride_h, stride_w = scene.stride()
        plane = (height - 1) * stride_h + (width - 1) * stride_w
        for channel in range(3):
            start = scene.data_ptr() + channel * stride_c
            spans.append((start, start + plane, image))

    # Sweep by start. While no two scenes have met, every span still open at
    # ``start`` belongs to one scene: the owner of the furthest end so far.
    spans.sort()
    reach, owner = -1, -1
    for start, end, image in spans:
        if start <= reach and owner != image:
            raise ValueError(
                f"scenes {owner} and {image} may share bytes; the triton "
                "backend paints a batch at once and needs disjoint scenes"
            )
        if end > reach:
            reach, owner = end, image
