"""Triton kernel of the ``triton`` drawing backend (tables: ``triton_tables``).

Imported only by ``TritonKernels`` on its first launch, so tensor-backend users
never import this module or compile its kernel. (The triton package itself may
still be imported by torch when it is installed.) Ported unchanged from
development example 13 ``triton_paint``.

One program paints one tile of ``TILE_PIXELS`` pixels of one piece::

    pixel k of the piece   -> row r, column c of its image
    own    = box: always;  label: canvas alpha at (r, c) != 0
    hidden = some later item of the same image covers (r, c)
    write the item's color where own and not hidden

Only items ``item + 1 .. later_end - 1`` are checked (``later_end``: host
bound, see ``triton_tables``), and of those, items whose rect misses the
tile's rows and columns are skipped with one scalar test.

Table offsets are module ``tl.constexpr`` globals taken from the column
tables by name, so the kernel and the host cannot disagree on a column.
"""

import torch
import triton
import triton.language as tl
from roboflow_workflows.execution_engine.v2.blocks.visualization.triton_tables import (  # noqa: E501
    BOX,
    ITEM_COLUMN,
    ITEM_COLUMNS,
    LABEL,
    PIECE_COLUMN,
    PIECE_COLUMNS,
    SCENE_COLUMN,
    SCENE_COLUMNS,
    TILE_PIXELS,
)

_TILE = tl.constexpr(TILE_PIXELS)
_BOX = tl.constexpr(BOX)
_LABEL = tl.constexpr(LABEL)

_SCENE_WIDTH = tl.constexpr(len(SCENE_COLUMNS))
_DATA_PTR = tl.constexpr(SCENE_COLUMN["data_ptr"])
_STRIDE_C = tl.constexpr(SCENE_COLUMN["stride_c"])
_STRIDE_H = tl.constexpr(SCENE_COLUMN["stride_h"])
_STRIDE_W = tl.constexpr(SCENE_COLUMN["stride_w"])

_ITEM_WIDTH = tl.constexpr(len(ITEM_COLUMNS))
_IMAGE = tl.constexpr(ITEM_COLUMN["image"])
_KIND = tl.constexpr(ITEM_COLUMN["kind"])
_R1 = tl.constexpr(ITEM_COLUMN["r1"])
_C1 = tl.constexpr(ITEM_COLUMN["c1"])
_R2 = tl.constexpr(ITEM_COLUMN["r2"])
_C2 = tl.constexpr(ITEM_COLUMN["c2"])
_HOLE_R1 = tl.constexpr(ITEM_COLUMN["hole_r1"])
_HOLE_C1 = tl.constexpr(ITEM_COLUMN["hole_c1"])
_HOLE_R2 = tl.constexpr(ITEM_COLUMN["hole_r2"])
_HOLE_C2 = tl.constexpr(ITEM_COLUMN["hole_c2"])
_PAINT = tl.constexpr(ITEM_COLUMN["paint"])  # box: 0xRRGGBB; label: canvas index
_LATER_END = tl.constexpr(ITEM_COLUMN["later_end"])

_PIECE_WIDTH = tl.constexpr(len(PIECE_COLUMNS))
_P_ITEM = tl.constexpr(PIECE_COLUMN["item"])
_P_TOP = tl.constexpr(PIECE_COLUMN["r1"])
_P_LEFT = tl.constexpr(PIECE_COLUMN["c1"])
_P_WIDTH = tl.constexpr(PIECE_COLUMN["width"])
_P_PIXELS = tl.constexpr(PIECE_COLUMN["pixels"])
_P_FIRST_TILE = tl.constexpr(PIECE_COLUMN["first_tile"])


@triton.jit
def _covered_by_later(
    items,
    canvases,
    first,
    end,
    rows,
    cols,
    load_mask,
    tile_r1,
    tile_r2,
    tile_c1,
    tile_c2,
):
    # True where an item in [first, end) covers (rows, cols). Label alpha is
    # loaded only on ``load_mask`` lanes.
    covered = rows < 0  # all False
    for later in range(first, end):
        entry = items + later * _ITEM_WIDTH
        r1 = tl.load(entry + _R1)
        c1 = tl.load(entry + _C1)
        r2 = tl.load(entry + _R2)
        c2 = tl.load(entry + _C2)
        if (r1 <= tile_r2) & (r2 >= tile_r1) & (c1 <= tile_c2) & (c2 >= tile_c1):
            inside = (rows >= r1) & (rows <= r2) & (cols >= c1) & (cols <= c2)
            if tl.load(entry + _KIND) == _LABEL:
                canvas = tl.load(canvases + tl.load(entry + _PAINT))
                canvas = canvas.to(tl.pointer_type(tl.uint8))
                texel = ((rows - r1) * (c2 - c1 + 1) + (cols - c1)) * 4
                alpha = tl.load(canvas + texel + 3, mask=inside & load_mask, other=0)
                inside = inside & (alpha != 0)
            else:
                hole_r1 = tl.load(entry + _HOLE_R1)
                hole_c1 = tl.load(entry + _HOLE_C1)
                hole_r2 = tl.load(entry + _HOLE_R2)
                hole_c2 = tl.load(entry + _HOLE_C2)
                in_hole = (
                    (rows >= hole_r1)
                    & (rows <= hole_r2)
                    & (cols >= hole_c1)
                    & (cols <= hole_c2)
                )
                inside = inside & ~in_hole
            covered = covered | inside

    return covered


@triton.jit
def _paint_tiles(scenes, canvases, items, pieces, tiles):
    program = tl.program_id(0)
    piece_entry = pieces + tl.load(tiles + program) * _PIECE_WIDTH
    item = tl.load(piece_entry + _P_ITEM)
    top = tl.load(piece_entry + _P_TOP)
    left = tl.load(piece_entry + _P_LEFT)
    width = tl.load(piece_entry + _P_WIDTH)
    pixels = tl.load(piece_entry + _P_PIXELS)
    first_pixel = (program - tl.load(piece_entry + _P_FIRST_TILE)) * _TILE

    k = first_pixel + tl.arange(0, _TILE)
    live = k < pixels
    rows = top + k // width
    cols = left + k % width
    last_pixel = tl.minimum(first_pixel + _TILE, pixels) - 1
    tile_r1 = top + first_pixel // width
    tile_r2 = top + last_pixel // width
    tile_c1 = left
    tile_c2 = left + width - 1

    entry = items + item * _ITEM_WIDTH
    kind = tl.load(entry + _KIND)
    paint = tl.load(entry + _PAINT)
    later_end = tl.load(entry + _LATER_END)
    scene = scenes + tl.load(entry + _IMAGE) * _SCENE_WIDTH
    base = tl.load(scene + _DATA_PTR).to(tl.pointer_type(tl.uint8))
    stride_c = tl.load(scene + _STRIDE_C)
    offsets = rows * tl.load(scene + _STRIDE_H) + cols * tl.load(scene + _STRIDE_W)

    if kind == _BOX:
        hidden = _covered_by_later(
            items,
            canvases,
            item + 1,
            later_end,
            rows,
            cols,
            live,
            tile_r1,
            tile_r2,
            tile_c1,
            tile_c2,
        )
        write = live & ~hidden
        zero = tl.zeros([_TILE], dtype=tl.int64)
        for channel in tl.static_range(3):
            value = (zero + ((paint >> (16 - 8 * channel)) & 255)).to(tl.uint8)
            tl.store(base + channel * stride_c + offsets, value, mask=write)
    else:
        canvas = tl.load(canvases + paint).to(tl.pointer_type(tl.uint8))
        r1 = tl.load(entry + _R1)
        c1 = tl.load(entry + _C1)
        c2 = tl.load(entry + _C2)
        texel = ((rows - r1) * (c2 - c1 + 1) + (cols - c1)) * 4
        own = live & (tl.load(canvas + texel + 3, mask=live, other=0) != 0)
        hidden = _covered_by_later(
            items,
            canvases,
            item + 1,
            later_end,
            rows,
            cols,
            own,
            tile_r1,
            tile_r2,
            tile_c1,
            tile_c2,
        )
        write = own & ~hidden
        for channel in tl.static_range(3):
            value = tl.load(canvas + texel + channel, mask=write, other=0)
            tl.store(base + channel * stride_c + offsets, value, mask=write)


def launch(tables: dict, *, tiles: int) -> None:
    """Queue the drawing kernel on the current stream.

    Args:
        tables: Device int64 views ``scenes``, ``canvases``, ``items``,
            ``pieces`` and ``tiles`` (``LaunchTables.packed`` sections).
        tiles: Number of tiles, the grid size.
    """
    with torch.cuda.device(tables["items"].device):
        _paint_tiles[(tiles,)](
            tables["scenes"],
            tables["canvases"],
            tables["items"],
            tables["pieces"],
            tables["tiles"],
        )
