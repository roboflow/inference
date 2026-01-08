from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final, List, Optional, Tuple

import cv2
import numpy as np

ANCHORS: Final[Dict[str, Tuple[float, float]]] = {
    "top_left": (0.0, 0.0),
    "top_center": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "center_left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "center_right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0),
    "bottom_center": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}


def calculate_relative_position(
    anchor: str,
    offset_x: int,
    offset_y: int,
    box_width: int,
    box_height: int,
    img_width: int,
    img_height: int,
) -> Tuple[int, int]:
    """Calculate the top-left corner position for a box positioned relative to an image anchor.

    Args:
        anchor: Anchor point name (e.g., "top_left", "center", "bottom_right")
        offset_x: Horizontal offset from anchor point (positive = right)
        offset_y: Vertical offset from anchor point (positive = down)
        box_width: Width of the box to position
        box_height: Height of the box to position
        img_width: Width of the image
        img_height: Height of the image

    Returns:
        Tuple of (x, y) coordinates for the top-left corner of the box

    Raises:
        ValueError: If anchor is not recognized
    """
    key = anchor.lower()
    try:
        ax, ay = ANCHORS[key]
    except KeyError as e:
        raise ValueError(
            f"Unknown anchor: {anchor!r}. Must be one of {sorted(ANCHORS.keys())}"
        ) from e

    anchor_x = int(round(ax * img_width))
    anchor_y = int(round(ay * img_height))

    box_x = anchor_x - int(round(ax * box_width)) + offset_x
    box_y = anchor_y - int(round(ay * box_height)) + offset_y

    return box_x, box_y


@dataclass(frozen=True)
class TextLayout:
    lines: List[str]
    line_widths: List[int]
    max_width: int
    ref_height: int
    line_advance: int
    line_spacing: int
    box_x: int
    box_y: int
    box_w: int
    box_h: int


def clamp_box(
    box_x: int, box_y: int, box_w: int, box_h: int, img_w: int, img_h: int
) -> Tuple[int, int]:
    """Clamp box position to image bounds."""
    box_x = 0 if box_w > img_w else max(0, min(box_x, img_w - box_w))
    box_y = 0 if box_h > img_h else max(0, min(box_y, img_h - box_h))
    return box_x, box_y


def align_offset(text_align: str, max_width: int, line_width: int) -> int:
    """Calculate horizontal offset for text alignment."""
    if text_align == "center":
        return (max_width - line_width) // 2
    elif text_align == "right":
        return max_width - line_width
    else:  # left
        return 0


def compute_layout(
    *,
    formatted_text: str,
    font,
    font_scale: float,
    font_thickness: int,
    padding: int,
    position_mode: str,
    position_x: int,
    position_y: int,
    anchor: str,
    offset_x: int,
    offset_y: int,
    img_w: int,
    img_h: int,
) -> TextLayout:
    """Compute text layout including dimensions and position."""
    lines = formatted_text.split("\n") if formatted_text else [""]
    (_, ref_h), ref_base = cv2.getTextSize("Ag", font, font_scale, font_thickness)
    line_advance = ref_h + ref_base
    line_spacing = max(1, int(round(0.25 * line_advance)))

    line_widths = [
        (
            cv2.getTextSize(line, font, font_scale, font_thickness)[0][0]
            if line.strip()
            else 0
        )
        for line in lines
    ]
    max_width = max(line_widths, default=0)

    num_lines = len(lines)
    total_h = num_lines * line_advance + max(0, num_lines - 1) * line_spacing

    box_w = max_width + 2 * padding
    box_h = total_h + 2 * padding

    if position_mode == "absolute":
        box_x, box_y = position_x, position_y
    else:
        box_x, box_y = calculate_relative_position(
            anchor=anchor,
            offset_x=offset_x,
            offset_y=offset_y,
            box_width=box_w,
            box_height=box_h,
            img_width=img_w,
            img_height=img_h,
        )

    box_x, box_y = clamp_box(box_x, box_y, box_w, box_h, img_w, img_h)

    return TextLayout(
        lines=lines,
        line_widths=line_widths,
        max_width=max_width,
        ref_height=ref_h,
        line_advance=line_advance,
        line_spacing=line_spacing,
        box_x=box_x,
        box_y=box_y,
        box_w=box_w,
        box_h=box_h,
    )


def draw_text_lines(
    img: np.ndarray,
    *,
    layout: TextLayout,
    padding: int,
    text_align: str,
    font,
    font_scale: float,
    font_thickness: int,
    color_bgr: Tuple[int, int, int],
) -> None:
    """Draw text lines on the image."""
    img_h, img_w = img.shape[:2]
    current_y = layout.box_y + padding
    base_x = layout.box_x + padding

    for i, line in enumerate(layout.lines):
        if line.strip():
            w = layout.line_widths[i]
            text_x = base_x + align_offset(text_align, layout.max_width, w)
            text_y = current_y + layout.ref_height

            if text_y > 0 and current_y < img_h and text_x < img_w:
                cv2.putText(
                    img,
                    line,
                    (text_x, text_y),
                    font,
                    font_scale,
                    color_bgr,
                    font_thickness,
                    cv2.LINE_AA,
                )

        current_y += layout.line_advance
        if i < len(layout.lines) - 1:
            current_y += layout.line_spacing


def draw_rounded_rectangle(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    radius: int,
) -> None:
    """Draw a filled rounded rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Early return for invalid coordinates
    if x2 <= x1 or y2 <= y1:
        return

    max_radius = min((x2 - x1) // 2, (y2 - y1) // 2)
    radius = min(radius, max_radius)

    if radius <= 0:
        cv2.rectangle(img, pt1, pt2, color, -1)
        return

    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    cv2.ellipse(
        img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1
    )
    cv2.ellipse(
        img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1
    )
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1)


def draw_background_with_alpha(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float,
    border_radius: int,
) -> None:
    """Draw a filled rectangle with alpha blending using overlay compositing.

    Uses proper overlay-based alpha blending for smooth antialiased edges,
    especially important for rounded rectangles.

    Process:
    1. Extract the affected region
    2. Create overlay and draw shape on it
    3. Alpha-blend overlay with original region
    4. Write blended result back
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # Clamp to image bounds
    img_h, img_w = img.shape[:2]
    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(img_w, x2)
    y2_clamped = min(img_h, y2)

    if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
        return

    # Extract the region of interest
    roi = img[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

    # Create overlay for just this region
    overlay = roi.copy()

    roi_w = x2_clamped - x1_clamped
    roi_h = y2_clamped - y1_clamped

    # Draw the shape onto the overlay (coordinates relative
    # to ROI and OpenCV uses inclusive coordinates,
    # so max index is size - 1
    if border_radius > 0:
        draw_rounded_rectangle(
            overlay,
            (0, 0),
            (roi_w - 1, roi_h - 1),
            color,
            border_radius,
        )
    else:
        cv2.rectangle(
            overlay,
            (0, 0),
            (roi_w - 1, roi_h - 1),
            color,
            -1,
        )

    # Alpha blend: result = overlay * alpha + original * (1 - alpha)
    blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)

    # Write blended result back to image
    img[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = blended


def draw_background(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    bg_color_bgr: Optional[Tuple[int, int, int]],
    background_opacity: float,
    border_radius: int,
) -> None:
    """Draw background rectangle with optional transparency and rounded corners."""
    if bg_color_bgr is None or x2 <= x1 or y2 <= y1:
        return

    if background_opacity > 0.0:
        if background_opacity < 1.0:
            # Alpha blending required
            draw_background_with_alpha(
                img,
                (x1, y1),
                (x2, y2),
                bg_color_bgr,
                background_opacity,
                border_radius,
            )
        else:
            # Fully opaque - use direct drawing
            # OpenCV uses inclusive coordinates, so subtract 1 from exclusive end coords
            if border_radius > 0:
                draw_rounded_rectangle(
                    img,
                    (x1, y1),
                    (x2 - 1, y2 - 1),
                    bg_color_bgr,
                    border_radius,
                )
            else:
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2 - 1, y2 - 1),
                    bg_color_bgr,
                    -1,
                )
