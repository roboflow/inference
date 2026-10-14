"""Validated host decoding of native compressed COCO RLE masks.

pycocotools decodes compressed counts without checking that the runs cover the
image. Short counts can decode without an error into an invalid mask whose
uncovered pixels are undefined. Every V2 decode of ``InstancesRLEMasks``
therefore checks the counts of each decoded row first.

``check_rle_masks`` validates rows for a caller that decodes with its own native
helper. ``decode_rle_masks`` validates and decodes on the host into one
``(rows, H, W)`` bool NumPy array; nothing is copied to an accelerator.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from roboflow_workflows.v2.data.errors import ContractError

from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    coco_rle_masks_to_torch_mask,
)


def check_rle_masks(
    masks: InstancesRLEMasks, *, rows: Optional[Sequence[int]] = None
) -> None:
    """Check the compressed counts of the given rows against the encoded ``mask_size``.

    Args:
        masks: Native compressed masks.
        rows: Row positions to check; all rows when ``None``.

    Raises:
        ContractError: When a row's counts are not compressed COCO counts
            covering exactly ``H * W`` pixels. The message names the row.
    """
    size = masks.mask_size or masks.image_size
    size_hw = (int(size[0]), int(size[1]))
    checked_rows = range(len(masks.masks)) if rows is None else rows
    for row in checked_rows:
        check_rle_counts(masks.masks[row], size_hw=size_hw, row=row)


def decode_rle_masks(
    masks: InstancesRLEMasks, *, rows: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Validate, then decode the given rows on the host.

    Args:
        masks: Native compressed masks.
        rows: Row positions to decode, in this order; all rows when ``None``.

    Returns:
        A ``(len(rows), H, W)`` bool array.

    Raises:
        ContractError: On malformed counts; see ``check_rle_masks``.
    """
    check_rle_masks(masks, rows=rows)
    if rows is not None:
        masks = InstancesRLEMasks(
            image_size=masks.image_size,
            mask_size=masks.mask_size,
            masks=[masks.masks[row] for row in rows],
        )
    if not masks.masks:
        return np.empty((0, *(masks.mask_size or masks.image_size)), dtype=bool)
    decoded = coco_rle_masks_to_numpy_mask(masks)

    return decoded


def decode_rle_tensor(
    masks: InstancesRLEMasks, *, device: torch.device
) -> torch.Tensor:
    """Validate encoded raster counts before decoding on the requested device."""
    check_rle_masks(masks)
    if not masks.masks:
        return torch.empty(
            (0, *(masks.mask_size or masks.image_size)), dtype=torch.bool, device=device
        )
    return coco_rle_masks_to_torch_mask(masks, device=device)


def check_rle_counts(
    counts: Union[bytes, str], *, size_hw: Tuple[int, int], row: int
) -> None:
    """Reject counts that do not describe exactly one ``H x W`` mask.

    Walks the compressed COCO format (LEB128-like 6-bit characters, runs
    after the second delta-coded against the run two before) and checks that
    the runs are non-negative and cover exactly ``H * W`` pixels.

    Args:
        counts: One row's compressed counts.
        size_hw: Image size ``(H, W)`` the counts must cover.
        row: Row position named in error messages.

    Raises:
        ContractError: On a non-compressed or wrongly sized count sequence.
    """
    if isinstance(counts, str):
        if not counts.isascii():
            raise ContractError(f"RLE row {row}: counts are not ASCII")
        counts = counts.encode("ascii")
    if not isinstance(counts, bytes):
        raise ContractError(
            f"RLE row {row}: counts must be bytes or str, got {type(counts).__name__}"
        )

    runs = []
    position = 0
    while position < len(counts):
        value, shift, more = 0, 0, True
        while more:
            if position >= len(counts):
                raise ContractError(f"RLE row {row}: counts end inside a run length")
            code = counts[position] - 48
            if not 0 <= code < 64:
                raise ContractError(
                    f"RLE row {row}: byte {counts[position]!r} at {position} is not "
                    "a compressed COCO count"
                )
            value |= (code & 0x1F) << shift
            more = bool(code & 0x20)
            position += 1
            shift += 5
            if not more and code & 0x10:
                value |= -1 << shift
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise ContractError(f"RLE row {row}: negative run length {value}")
        runs.append(value)

    height, width = size_hw
    if sum(runs) != height * width:
        raise ContractError(
            f"RLE row {row}: counts cover {sum(runs)} pixels; image_size "
            f"{(height, width)} needs {height * width}"
        )


__all__ = ["check_rle_counts", "check_rle_masks", "decode_rle_masks"]
