"""Text labels drawn exactly as V1's supervision label annotators draw them.

Anti-aliased OpenCV text blends into the pixels below it, so labels cannot be
precomputed as solid strokes. The reference exchanges only label-sized
patches with the device::

    host   label geometry (cv2.getTextSize) -> one region per label
           patches: the label regions, or their enclosing rectangle when
           that costs less: overlapping regions hold more pixels than it, or
           on CUDA many small patches cost more kernels than the extra pixels
    device -> host   one transfer of all patches of the image
    host   a zeroed frame-coordinate canvas receives the patches; every
           label is drawn in order with cv2 (rounded background, LINE_AA text)
    host -> device   one transfer back; patches are written in place

Overlapping patches need no merging. Every patch is read before any label is
drawn, so overlapping patches put the same scene pixels on the canvas, and a
later label blends over an earlier one there. They are written back from the
finished canvas, so overlapping patches also write the same pixels.

Labels are drawn in frame coordinates, so OpenCV clips them at the same frame
edges as V1. The device-to-host copy waits for the queued work of the
caller's stream (no device-wide synchronization).

Two V1 layouts are provided: ``detection_labels`` (``sv.LabelAnnotator``,
used by the label and classification-label blocks) and ``vertex_labels``
(``sv.VertexLabelAnnotator``, used by keypoint ``vertex_label``).
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch

FONT = cv2.FONT_HERSHEY_SIMPLEX
POSITIONS = (
    "CENTER",
    "CENTER_LEFT",
    "CENTER_RIGHT",
    "TOP_CENTER",
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_CENTER",
    "BOTTOM_RIGHT",
)
RGB = Tuple[int, int, int]
Rect = Tuple[int, int, int, int]

# Heuristic dispatch cost of an extra CUDA patch, expressed as pixels moved.
# This balances extra gather/write-back launches against copying a larger region;
# it is not a device calibration. CPU and MPS retain the area-only policy.
_CUDA_PATCH_PIXELS = 6144


@dataclass(frozen=True)
class TextStyle:
    """How label text is set.

    Args:
        scale: OpenCV font scale.
        thickness: Text stroke thickness.
        padding: Pixels between text and background edge, and between lines.
        border_radius: Background corner radius.
    """

    scale: float
    thickness: int
    padding: int
    border_radius: int


@dataclass(frozen=True)
class Label:
    """One label: a background and text lines, in frame pixels.

    Args:
        background: Inclusive ``(x1, y1, x2, y2)`` of the background.
        background_rgb: Background color.
        text_rgb: Text color.
        lines: ``(text, x, y)`` per line; ``(x, y)`` is the text origin.
        style: Text style.
    """

    background: Rect
    background_rgb: RGB
    text_rgb: RGB
    lines: Tuple[Tuple[str, int, int], ...]
    style: TextStyle

    def region(self) -> Rect:
        """Half-open ``(x0, y0, x1, y1)`` holding every pixel this label touches."""
        x1, y1, x2, y2 = self.background
        reach = self.style.thickness + 3
        left, top = min(x1, x2), min(y1, y2)
        right, bottom = max(x1, x2) + 1, max(y1, y2) + 1
        for text, x, y in self.lines:
            (width, height), baseline = cv2.getTextSize(
                text, FONT, self.style.scale, self.style.thickness
            )
            left, top = min(left, x - reach), min(top, y - height - reach)
            right = max(right, x + width + reach)
            bottom = max(bottom, y + baseline + reach)

        return left, top, right, bottom


def detection_labels(
    texts: Sequence[str],
    anchors: np.ndarray,
    *,
    position: str,
    style: TextStyle,
    background_rgb: Sequence[RGB],
    text_rgb: RGB,
) -> List[Label]:
    """Lay out labels as ``sv.LabelAnnotator`` (no smart position) does.

    Args:
        texts: One text per label; ``\\n`` starts a new line.
        anchors: ``(N, 2)`` anchor points, truncated to integers as V1 does.
        position: One of ``POSITIONS`` or ``CENTER_OF_MASS``.
        style: Text style.
        background_rgb: Background color per label.
        text_rgb: Text color of every label.

    Returns:
        The labels, in drawing order.
    """
    labels = []
    for text, anchor, background in zip(
        texts, np.asarray(anchors).astype(int), background_rgb
    ):
        lines = str(text).splitlines() or [""]
        sizes = [_text_size(line, style=style) for line in lines]
        width = max(size[0] for size in sizes) + 2 * style.padding
        height = (
            sum(size[1] for size in sizes)
            + (len(sizes) - 1) * style.padding
            + 2 * style.padding
        )
        x1, y1, x2, y2 = background_rect(
            int(anchor[0]),
            int(anchor[1]),
            width=width,
            height=height,
            position=position,
        )

        placed = []
        current_y = y1 + style.padding
        for line in lines:
            if not line:
                current_y += _text_size("Tg", style=style)[1] + style.padding
                continue
            line_height = _text_size(line, style=style)[1]
            placed.append((line, x1 + style.padding, current_y + line_height))
            current_y += line_height + style.padding

        labels.append(
            Label(
                background=(x1, y1, x2, y2),
                background_rgb=tuple(background),
                text_rgb=text_rgb,
                lines=tuple(placed),
                style=style,
            )
        )

    return labels


def vertex_labels(
    texts: Sequence[str],
    anchors: Sequence[Tuple[int, int]],
    *,
    style: TextStyle,
    background_rgb: RGB,
    text_rgb: RGB,
) -> List[Label]:
    """Lay out labels as ``sv.VertexLabelAnnotator`` (no smart position) does.

    Args:
        texts: One single-line text per vertex.
        anchors: Integer ``(x, y)`` centres.
        style: Text style; ``border_radius`` rounds the background.
        background_rgb: Background color of every label.
        text_rgb: Text color of every label.

    Returns:
        The labels, in drawing order.
    """
    labels = []
    for text, (x, y) in zip(texts, anchors):
        width, height = _text_size(text, style=style)
        box = (x - width // 2, y - height // 2, x + width // 2, y + height // 2)
        padded = (
            box[0] - style.padding,
            box[1] - style.padding,
            box[2] + style.padding,
            box[3] + style.padding,
        )
        labels.append(
            Label(
                background=padded,
                background_rgb=background_rgb,
                text_rgb=text_rgb,
                lines=((text, box[0], box[3]),),
                style=style,
            )
        )

    return labels


def paint_labels(scene: torch.Tensor, labels: Sequence[Label]) -> None:
    """Draw labels into ``scene`` in order, exchanging only their patches.

    Args:
        scene: Contiguous ``(3, H, W)`` uint8 RGB image, changed in place.
        labels: Labels in frame pixels.
    """
    height, width = int(scene.shape[1]), int(scene.shape[2])
    regions = [_clipped(label.region(), height=height, width=width) for label in labels]
    patches = _patch_plan(
        [region for region in regions if region is not None],
        patch_pixels=_CUDA_PATCH_PIXELS if scene.is_cuda else 0,
    )
    if not patches:
        return

    on_device = [scene[:, y0:y1, x0:x1].reshape(-1) for x0, y0, x1, y1 in patches]
    host = torch.cat(on_device).cpu().numpy()

    canvas_height = max(patch[3] for patch in patches)
    canvas_width = max(patch[2] for patch in patches)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    offset = 0
    for x0, y0, x1, y1 in patches:
        size = 3 * (y1 - y0) * (x1 - x0)
        canvas[y0:y1, x0:x1] = (
            host[offset : offset + size].reshape(3, y1 - y0, x1 - x0).transpose(1, 2, 0)
        )
        offset += size

    for label in labels:
        _draw(canvas, label)

    drawn = np.concatenate(
        [
            np.ascontiguousarray(canvas[y0:y1, x0:x1].transpose(2, 0, 1)).reshape(-1)
            for x0, y0, x1, y1 in patches
        ]
    )
    uploaded = torch.from_numpy(drawn).to(scene.device)
    offset = 0
    for x0, y0, x1, y1 in patches:
        size = 3 * (y1 - y0) * (x1 - x0)
        scene[:, y0:y1, x0:x1] = uploaded[offset : offset + size].view(
            3, y1 - y0, x1 - x0
        )
        offset += size


def _text_size(text: str, *, style: TextStyle) -> Tuple[int, int]:
    (width, height), _ = cv2.getTextSize(text, FONT, style.scale, style.thickness)

    return width, height


def background_rect(x: int, y: int, *, width: int, height: int, position: str) -> Rect:
    """Place a label background at its anchor, as ``sv`` does.

    Ports ``sv.annotators.utils.resolve_text_background_xyxy``. Plain and rich
    labels share it, so both place backgrounds identically.

    Args:
        x: Anchor x, in frame pixels.
        y: Anchor y, in frame pixels.
        width: Background width.
        height: Background height.
        position: One of ``POSITIONS`` or ``CENTER_OF_MASS``.

    Returns:
        The background ``(x1, y1, x2, y2)``.

    Raises:
        ValueError: If ``position`` is unknown.
    """
    if position == "TOP_LEFT":
        return x, y - height, x + width, y
    if position == "TOP_RIGHT":
        return x - width, y - height, x, y
    if position == "TOP_CENTER":
        return x - width // 2, y - height, x + width // 2, y
    if position in ("CENTER", "CENTER_OF_MASS"):
        return x - width // 2, y - height // 2, x + width // 2, y + height // 2
    if position == "BOTTOM_LEFT":
        return x, y, x + width, y + height
    if position == "BOTTOM_RIGHT":
        return x - width, y, x, y + height
    if position == "BOTTOM_CENTER":
        return x - width // 2, y, x + width // 2, y + height
    if position == "CENTER_LEFT":
        return x - width, y - height // 2, x, y + height // 2
    if position == "CENTER_RIGHT":
        return x, y - height // 2, x + width, y + height // 2

    raise ValueError(f"unknown text position {position!r}")


def _draw(canvas: np.ndarray, label: Label) -> None:
    # sv LabelAnnotator.draw_rounded_rectangle, then cv2.putText per line.
    x1, y1, x2, y2 = label.background
    radius = min(label.style.border_radius, min(x2 - x1, y2 - y1) // 2)
    color = tuple(int(channel) for channel in label.background_rgb)
    for corner_a, corner_b in (
        ((x1 + radius, y1), (x2 - radius, y2)),
        ((x1, y1 + radius), (x2, y2 - radius)),
    ):
        cv2.rectangle(canvas, corner_a, corner_b, color, -1)
    for centre in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(canvas, centre, radius, color, -1)

    text_color = tuple(int(channel) for channel in label.text_rgb)
    for text, x, y in label.lines:
        cv2.putText(
            canvas,
            text,
            (x, y),
            FONT,
            label.style.scale,
            text_color,
            label.style.thickness,
            cv2.LINE_AA,
        )


def _clipped(region: Rect, *, height: int, width: int) -> "Rect | None":
    x0, y0, x1, y1 = region
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if x1 <= x0 or y1 <= y0:
        return None

    return x0, y0, x1, y1


def _patch_plan(regions: List[Rect], *, patch_pixels: int = 0) -> List[Rect]:
    # One walk over the regions. Separate patches cost their summed area
    # (overlaps count twice) plus patch_pixels for each patch after the first;
    # one enclosing patch costs its area. Send whichever costs less.
    if not regions:
        return []

    enclosing = (
        min(region[0] for region in regions),
        min(region[1] for region in regions),
        max(region[2] for region in regions),
        max(region[3] for region in regions),
    )
    separate = (
        sum(_area(region) for region in regions) + (len(regions) - 1) * patch_pixels
    )
    if separate > _area(enclosing):
        return [enclosing]

    return regions


def _area(region: Rect) -> int:
    x0, y0, x1, y1 = region

    return (x1 - x0) * (y1 - y0)
