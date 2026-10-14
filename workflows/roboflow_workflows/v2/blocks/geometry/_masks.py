"""Host mask windows and filled-polygon masks shared by the geometry blocks.

``roboflow_core/bounding_rect@v1`` and ``roboflow_core/dynamic_zone@v1`` both read each instance
mask as a small host window, fit a polygon with OpenCV, and replace the mask
by that filled polygon::

    windows = mask_windows(mask)            # (N, H) + (N, W) any() to host,
                                            # then one window per instance
    polygon = <block's own OpenCV fit of each window>
    mask    = polygon_masks(mask, polygons) # None keeps the row's mask

Dense masks keep their dtype and device; only the filled windows come back.
RLE masks are host data; they are decoded and re-encoded on the host.
"""

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.v2.data.rle import decode_rle_masks

from inference_models.models.base.types import InstancesRLEMasks

# (x0, y0) of a host mask window and the window itself.
Window = Tuple[Tuple[int, int], np.ndarray]


def mask_windows(mask: Any) -> List[Optional[Window]]:
    """Return the smallest host window holding each instance.

    Args:
        mask: ``(N, H, W)`` dense mask tensor or ``InstancesRLEMasks``.

    Returns:
        Per instance, ``((x0, y0), uint8 pixels)``; ``None`` for an empty mask.
    """
    if isinstance(mask, InstancesRLEMasks):
        host_masks = decode_rle_masks(mask)
        any_in_row = host_masks.any(axis=2)
        any_in_column = host_masks.any(axis=1)
    else:
        host_masks = None
        any_in_row = mask.any(dim=2).cpu().numpy()
        any_in_column = mask.any(dim=1).cpu().numpy()

    windows: List[Optional[Window]] = []
    for instance, (rows, columns) in enumerate(zip(any_in_row, any_in_column)):
        ys, xs = np.flatnonzero(rows), np.flatnonzero(columns)
        if ys.size == 0:
            windows.append(None)
            continue

        y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
        if host_masks is None:
            pixels = mask[instance, y0:y1, x0:x1].cpu().numpy()
        else:
            pixels = host_masks[instance, y0:y1, x0:x1]
        windows.append(((x0, y0), pixels.astype(np.uint8)))

    return windows


def polygon_masks(mask: Any, *, polygons: List[Optional[np.ndarray]]) -> Any:
    """Replace each row's mask by its filled polygon, clipped to the frame.

    Args:
        mask: ``(N, H, W)`` dense mask tensor or ``InstancesRLEMasks``; not
            modified.
        polygons: Per row, ``(K, 2)`` integer ``[x, y]`` vertices in frame
            pixels, or ``None`` to keep that row's mask.

    Returns:
        A new mask of the same type; dense masks keep dtype and device.
    """
    if isinstance(mask, InstancesRLEMasks):
        return _polygon_rle_masks(mask, polygons=polygons)

    height, width = int(mask.shape[1]), int(mask.shape[2])
    filled = mask.clone()
    for row, polygon in enumerate(polygons):
        if polygon is None:
            continue
        (x0, y0), window = _filled_window(polygon, size_hw=(height, width))
        filled[row].zero_()
        filled[row, y0 : y0 + window.shape[0], x0 : x0 + window.shape[1]] = (
            torch.from_numpy(window).to(device=mask.device, dtype=mask.dtype)
        )

    return filled


def _polygon_rle_masks(
    mask: InstancesRLEMasks, *, polygons: List[Optional[np.ndarray]]
) -> InstancesRLEMasks:
    height, width = mask.image_size
    encoded = []
    for row, polygon in enumerate(polygons):
        if polygon is None:
            encoded.append(mask.masks[row])
            continue
        canvas = np.zeros((height, width), dtype=np.uint8)
        (x0, y0), window = _filled_window(polygon, size_hw=(height, width))
        canvas[y0 : y0 + window.shape[0], x0 : x0 + window.shape[1]] = window
        encoded.append(mask_utils.encode(np.asfortranarray(canvas))["counts"])

    converted = InstancesRLEMasks(image_size=mask.image_size, masks=encoded)

    return converted


def _filled_window(polygon: np.ndarray, *, size_hw: Tuple[int, int]) -> Window:
    # Fill the polygon inside its own clipped bounding window; filling a
    # shifted polygon on a window equals filling it on the full canvas.
    height, width = size_hw
    x0, y0 = (int(v) for v in np.clip(polygon.min(axis=0), 0, [width, height]))
    x1, y1 = (int(v) for v in np.clip(polygon.max(axis=0) + 1, 0, [width, height]))
    window = np.zeros((max(y1 - y0, 0), max(x1 - x0, 0)), dtype=np.uint8)
    if window.size:
        shifted = (polygon - np.array([x0, y0])).astype(np.int32)
        cv2.fillPoly(window, [shifted], color=(1,))

    return (x0, y0), window
