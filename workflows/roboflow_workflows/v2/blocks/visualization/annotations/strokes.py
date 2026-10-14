"""Solid strokes (lines, rectangles, arcs, discs) drawn exactly as V1 draws them.

V1 draws strokes with OpenCV into the frame. The thick-line rules of OpenCV
(rounded joins, fixed-point ends) are its own, so the reference reproduces
them by asking OpenCV which pixels a stroke covers, on a host mask, and then
writing only those pixels on the image's device::

    host   scratch uint8 mask of the frame size (zero pages until touched)
           per stroke: cv2 draws value 1 in frame coordinates; the stroke's
           scan rectangles are read back with np.nonzero, then cleared
    host   strokes in order; a pixel keeps the color of its last stroke
    device one upload of (pixel index, color) pairs, one indexed write

Frame coordinates are used unchanged, so OpenCV clips at the same frame edges
as V1. Each stroke's scan rectangles must contain every pixel it can touch;
callers pass the geometry's bounds and the stroke's reach beyond them.

Cost per image: host work proportional to the scanned area, one host-to-
device copy of ``2 x pixels`` int64 values, no device-to-host copy and no
wait. The scratch mask is allocated per call (no shared state).
"""

from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch

Rect = Tuple[int, int, int, int]
DrawFunction = Callable[[np.ndarray], None]


class Strokes:
    """Collect strokes of one frame on the host; write them to a device image.

    Args:
        size_hw: Frame size ``(H, W)``.
    """

    def __init__(self, size_hw: Tuple[int, int]) -> None:
        self._height, self._width = (int(size_hw[0]), int(size_hw[1]))
        self._scratch = np.zeros((self._height, self._width), dtype=np.uint8)
        self._pixels: List[np.ndarray] = []
        self._colors: List[int] = []

    def add(
        self,
        draw: DrawFunction,
        *,
        color: Sequence[int],
        scan: Sequence[Rect],
    ) -> np.ndarray:
        """Rasterize one stroke now and remember its pixels and color.

        Args:
            draw: Draws the stroke with value ``1`` into the ``(H, W)`` uint8
                mask it receives, in frame coordinates (e.g. a ``cv2`` call).
            color: RGB color of the stroke.
            scan: Half-open ``(x0, y0, x1, y1)`` rectangles holding every
                pixel the stroke can touch; clipped to the frame here.

        Returns:
            The stroke's flat pixel indices (``y * W + x``), sorted.
        """
        draw(self._scratch)
        found = []
        for x0, y0, x1, y1 in scan:
            x0, y0 = max(int(x0), 0), max(int(y0), 0)
            x1, y1 = min(int(x1), self._width), min(int(y1), self._height)
            if x1 <= x0 or y1 <= y0:
                continue
            window = self._scratch[y0:y1, x0:x1]
            rows, columns = np.nonzero(window)
            found.append((rows + y0).astype(np.int64) * self._width + columns + x0)
            window[:] = 0

        pixels = np.unique(np.concatenate(found)) if found else np.zeros(0, np.int64)
        self._pixels.append(pixels)
        red, green, blue = (int(channel) for channel in color)
        self._colors.append((red << 16) | (green << 8) | blue)

        return pixels

    def __len__(self) -> int:
        return len(self._pixels)

    def upload(self, device: torch.device) -> "DeviceStrokes":
        """Copy every stroke's pixels and color to ``device`` in one transfer.

        Returns:
            The strokes on the device, in the order they were added.
        """
        counts = [len(pixels) for pixels in self._pixels]
        flat = np.concatenate(self._pixels) if self._pixels else np.zeros(0, np.int64)
        colors = np.repeat(np.asarray(self._colors, dtype=np.int64), counts)
        packed = torch.from_numpy(np.stack([flat, colors])).to(device)
        device_strokes = DeviceStrokes(packed=packed, counts=counts)

        return device_strokes

    def paint(self, scene: torch.Tensor) -> None:
        """Write every stroke into ``scene`` in order; the last stroke wins.

        Args:
            scene: Contiguous ``(3, H, W)`` uint8 image, changed in place.
        """
        if not any(len(pixels) for pixels in self._pixels):
            return

        flat = np.concatenate(self._pixels)
        colors = np.repeat(
            np.asarray(self._colors, dtype=np.int64),
            [len(pixels) for pixels in self._pixels],
        )
        # Keep each pixel's last occurrence, so later strokes cover earlier ones.
        _, first_reversed = np.unique(flat[::-1], return_index=True)
        last = len(flat) - 1 - first_reversed
        packed = torch.from_numpy(np.stack([flat[last], colors[last]]))
        _write(scene, packed.to(scene.device))


class DeviceStrokes:
    """Strokes already on the device; write them one at a time.

    Args:
        packed: ``(2, total)`` int64: flat pixel indices, then packed RGB.
        counts: Pixels per stroke, in order.
    """

    def __init__(self, packed: torch.Tensor, counts: Sequence[int]) -> None:
        self._packed = packed
        self._offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    def paint_one(self, scene: torch.Tensor, index: int) -> None:
        """Write stroke ``index`` into ``scene`` (contiguous ``(3, H, W)``)."""
        start, stop = int(self._offsets[index]), int(self._offsets[index + 1])
        if stop > start:
            _write(scene, self._packed[:, start:stop])


def rectangle_scan(rect: Rect, *, reach: int) -> List[Rect]:
    """Scan rectangles of a rectangle outline: its four edge strips.

    Args:
        rect: Integer ``(x1, y1, x2, y2)`` corners as passed to ``cv2``
            (inclusive).
        reach: How far the stroke can extend from an edge, in pixels.

    Returns:
        Up to four half-open rectangles covering the outline.
    """
    x1, y1, x2, y2 = rect
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    outer = (left - reach, top - reach, right + reach + 1, bottom + reach + 1)
    if right - left <= 2 * reach + 1 or bottom - top <= 2 * reach + 1:
        return [outer]

    strips = [
        (outer[0], outer[1], outer[2], top + reach + 1),
        (outer[0], bottom - reach, outer[2], outer[3]),
        (outer[0], top + reach + 1, left + reach + 1, bottom - reach),
        (right - reach, top + reach + 1, outer[2], bottom - reach),
    ]

    return strips


def bounds_scan(points: np.ndarray, *, reach: int) -> List[Rect]:
    """Scan rectangle of any stroke: its points' bounds grown by ``reach``.

    Args:
        points: ``(K, 2)`` integer x, y points of the geometry.
        reach: How far the stroke can extend from them.

    Returns:
        One half-open rectangle, or none for no points.
    """
    if len(points) == 0:
        return []

    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    rect = (
        int(x_min) - reach,
        int(y_min) - reach,
        int(x_max) + reach + 1,
        int(y_max) + reach + 1,
    )

    return [rect]


def stroke_reach(thickness: int) -> int:
    """Pixels a ``cv2`` stroke of ``thickness`` can reach beyond its geometry."""
    reach = max(int(thickness), 1) + 2

    return reach


def _write(scene: torch.Tensor, packed: torch.Tensor) -> None:
    indices, colors = packed[0], packed[1]
    rgb = torch.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255])
    scene.view(3, -1)[:, indices] = rgb.to(torch.uint8)
