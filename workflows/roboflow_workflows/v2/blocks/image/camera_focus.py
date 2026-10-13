"""Score image sharpness and draw camera aids: ``v2/camera_focus@v2``.

Port of V1 ``roboflow_core/camera_focus@v2`` (its device path)::

    gray      = OpenCV's uint8 BGR2GRAY, on the image's device
    tenengrad = Sobel_x(gray)^2 + Sobel_y(gray)^2      (3x3, reflected border)
    focus_measure        = mean(tenengrad)
    bbox_focus_measures  = mean(tenengrad[box]) per detection box (truncated
                           to whole pixels and clipped), None for a box with
                           no pixels inside the image

Higher values mean sharper edges. With any overlay enabled the output image
is a copy with, in this order: zebra stripes on under/overexposed pixels,
green focus peaking (top 70 % of the Tenengrad range), a centre crosshair, a
composition grid and a HUD panel (focus value, histograms). With every
overlay off the input image is returned as is. Geometry never changes.

Host work: the crosshair and HUD are drawn by OpenCV on small patches copied
to the host; the histograms, focus value and box means reach the host in one
transfer, the boxes in another. Pixels are identical to V1's OpenCV output
except where the HUD prints the focus value: the mean is accumulated in
float64, so a value on a rounding boundary of ``%.1f`` can print differently.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import Field
from roboflow_workflows.v2.blocks.image._resample import require_same_image
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, LIST_OF_VALUES_KIND
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

GRID_DIVISIONS = {"None": 0, "2x2": 2, "3x3": 3, "4x4": 4, "5x5": 5}

# OpenCV 4.x uint8 BGR2GRAY fixed point (color_rgb2gray RY15 / GY15 / BY15).
GRAY_WEIGHTS_RGB = (9798, 19235, 3735)
GRAY_SHIFT = 15

ZEBRA_SPACING = 8
ZEBRA_OPACITY = 0.5
UNDEREXPOSED_RGB = (0, 0, 255)
OVEREXPOSED_RGB = (255, 0, 0)
PEAKING_RGB = (0, 255, 0)
PEAKING_OPACITY = 0.6
PEAKING_THRESHOLD_PERCENT = 30.0
GRID_LEVEL = 128
# Extra pixels around a cv2-drawn patch, so antialiasing is not cut off.
PATCH_MARGIN = 4

HUD_FOCUS_LABEL = "TFM Focus"
HUD_EXPOSURE_LABEL = "Exposure"
# Histogram line colours in BGR, for the B, G and R channels, then gray.
HUD_HISTOGRAM_BGR = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX


class CameraFocusBlock(Block):
    """Measure focus with the Tenengrad score; see the module docstring.

    Example: a checkerboard scores high; the same board after a strong
    Gaussian blur scores far lower, and a uniform gray image scores 0.
    """

    type = "v2/camera_focus@v2"
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description="The image with the enabled overlays; the input when none.",
        ),
        "focus_measure": Output(
            FLOAT_KIND, description="Mean Tenengrad value of the whole image."
        ),
        "bbox_focus_measures": Output(
            LIST_OF_VALUES_KIND,
            description=(
                "Mean Tenengrad value inside each detection box, in row order; "
                "None for a box outside the image. Empty without detections."
            ),
        ),
    }
    prunable = True

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to measure.")
        underexposed_threshold_percent: float = Field(
            default=3.0,
            ge=0.0,
            le=100.0,
            description="Brightness percent below which zebra stripes are blue.",
        )
        overexposed_threshold_percent: float = Field(
            default=97.0,
            ge=0.0,
            le=100.0,
            description="Brightness percent above which zebra stripes are red.",
        )
        show_zebra_warnings: bool = Field(
            default=True, description="Stripe under- and overexposed pixels."
        )
        grid_overlay: Literal["None", "2x2", "3x3", "4x4", "5x5"] = Field(
            default="3x3", description="Composition grid, or 'None'."
        )
        show_hud: bool = Field(
            default=True, description="Panel with the focus value and histograms."
        )
        show_focus_peaking: bool = Field(
            default=True, description="Tint the sharpest pixels green."
        )
        show_center_marker: bool = Field(
            default=True, description="Crosshair at the image centre."
        )
        detections: Optional[
            Ref(OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND)
        ] = Field(
            default=None,
            description="Optional detections of this image to score box by box.",
        )

    def run(
        self,
        *,
        image: ImageData,
        underexposed_threshold_percent: float,
        overexposed_threshold_percent: float,
        show_zebra_warnings: bool,
        grid_overlay: str,
        show_hud: bool,
        show_focus_peaking: bool,
        show_center_marker: bool,
        detections: Any,
    ) -> Dict[str, Any]:
        """Measure ``image`` and draw the enabled overlays.

        Args:
            image: Image to measure.
            underexposed_threshold_percent: Zebra lower brightness percent.
            overexposed_threshold_percent: Zebra upper brightness percent.
            show_zebra_warnings: Draw exposure stripes.
            grid_overlay: Grid name.
            show_hud: Draw the HUD panel.
            show_focus_peaking: Draw focus peaking.
            show_center_marker: Draw the crosshair.
            detections: ``Detections``, ``InstanceDetections`` or ``None``.

        Returns:
            ``image``, ``focus_measure`` and ``bbox_focus_measures``.

        Raises:
            ContractError: When the detections describe another image.
        """
        boxes = None
        if detections is not None and len(detections.xyxy) > 0:
            require_same_image(detections, image=image, consumer=self.type)
            boxes = detections.xyxy.detach().cpu().numpy().reshape(-1, 4)

        pixels = image.tensor_image
        gray = gray_plane(pixels)
        focus = tenengrad(gray)
        focus_value, box_values = _mean_focus(focus, boxes=boxes)

        grid_divisions = GRID_DIVISIONS[grid_overlay]
        if not (
            show_zebra_warnings
            or show_hud
            or show_focus_peaking
            or show_center_marker
            or grid_divisions > 0
        ):
            return {
                "image": image,
                "focus_measure": focus_value,
                "bbox_focus_measures": box_values,
            }

        _, height, width = pixels.shape
        # A gray image is drawn on in color, as V1's GRAY2BGR.
        drawn = pixels.expand(3, height, width).clone()
        if show_zebra_warnings:
            _draw_zebra(
                drawn,
                gray=gray,
                under=int(underexposed_threshold_percent * 255 / 100),
                over=int(overexposed_threshold_percent * 255 / 100),
            )
        if show_focus_peaking:
            _draw_peaking(drawn, focus=focus)
        if show_center_marker:
            _draw_center_marker(drawn)
        for index in range(1, grid_divisions):
            drawn[:, :, width * index // grid_divisions] = GRID_LEVEL
            drawn[:, height * index // grid_divisions, :] = GRID_LEVEL
        if show_hud:
            _draw_hud(drawn, source=pixels, gray=gray, focus_value=focus_value)

        result = {
            "image": image.with_pixels(drawn),
            "focus_measure": focus_value,
            "bbox_focus_measures": box_values,
        }

        return result


def gray_plane(pixels: torch.Tensor) -> torch.Tensor:
    """Convert to grayscale exactly as OpenCV's uint8 ``BGR2GRAY``.

    Args:
        pixels: ``(C, H, W)`` uint8 image, one channel or RGB.

    Returns:
        ``(H, W)`` uint8 tensor on the same device; a one-channel image's own
        plane (a view).
    """
    if pixels.shape[0] == 1:
        return pixels[0]

    values = pixels.to(torch.int32)
    weighted = sum(
        values[channel] * weight for channel, weight in enumerate(GRAY_WEIGHTS_RGB)
    )
    gray = ((weighted + (1 << (GRAY_SHIFT - 1))) >> GRAY_SHIFT).to(torch.uint8)

    return gray


def tenengrad(gray: torch.Tensor) -> torch.Tensor:
    """Squared 3x3 Sobel gradient magnitude, exact for uint8 input.

    The border is reflected without repeating the edge pixel (OpenCV's
    ``BORDER_REFLECT_101``); an axis one pixel long repeats it.

    Args:
        gray: ``(H, W)`` uint8 image.

    Returns:
        ``(H, W)`` float32 ``gx^2 + gy^2``.
    """
    height, width = gray.shape
    plane = gray.to(torch.float32)[None, None]
    plane = F.pad(plane, (1, 1, 0, 0), mode="reflect" if width > 1 else "replicate")
    plane = F.pad(plane, (0, 0, 1, 1), mode="reflect" if height > 1 else "replicate")
    sobel = torch.tensor(
        [
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        ],
        device=gray.device,
    )
    gradients = F.conv2d(plane, sobel)[0]
    measure = gradients[0] * gradients[0] + gradients[1] * gradients[1]

    return measure


@dataclass(frozen=True)
class _HudLayout:
    """Panel geometry for one frame size and focus text (V1's layout)."""

    x: int
    y: int
    width: int
    height: int
    padding: int
    histogram_width: int
    histogram_height: int
    font_scale: float
    small_font_scale: float
    thickness: int
    line_spacing: int
    section_spacing: int
    focus_label_hw: Tuple[int, int]
    focus_value_hw: Tuple[int, int]
    exposure_label_hw: Tuple[int, int]
    focus_text: str


def _mean(values: torch.Tensor) -> torch.Tensor:
    # MPS has no float64; its tree reduction is close to numpy's pairwise mean.
    if values.device.type == "mps":
        return values.mean()
    return values.sum(dtype=torch.float64) / values.numel()


def _mean_focus(
    focus: torch.Tensor, *, boxes: Optional[np.ndarray]
) -> Tuple[float, List[Optional[float]]]:
    """Whole-image and per-box means, brought to the host together."""
    height, width = focus.shape
    means = [_mean(focus)]
    box_values: List[Optional[float]] = []
    measured = []
    for x1, y1, x2, y2 in boxes if boxes is not None else []:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))
        if x2 > x1 and y2 > y1:
            measured.append(len(box_values))
            means.append(_mean(focus[y1:y2, x1:x2]))
        box_values.append(None)
    host = torch.stack(means).cpu().tolist()
    for slot, value in zip(measured, host[1:]):
        box_values[slot] = value

    return host[0], box_values


def _blend(
    drawn: torch.Tensor,
    *,
    mask: torch.Tensor,
    color_rgb: Tuple[int, int, int],
    opacity: float,
) -> None:
    """``drawn[mask] = drawn * (1 - opacity) + color * opacity``, in place.

    Through a 256-entry table built with V1's numpy float64 expression, so
    blended pixels match V1 exactly.
    """
    levels = np.arange(256, dtype=np.uint8)
    table = torch.from_numpy(
        np.stack(
            [
                (levels * (1 - opacity) + np.uint8(channel) * opacity).astype(np.uint8)
                for channel in color_rgb
            ]
        )
    ).to(drawn.device)
    for channel in range(3):
        plane = drawn[channel]
        blended = table[channel][plane.long()]
        plane.copy_(torch.where(mask, blended, plane))


def _draw_zebra(
    drawn: torch.Tensor, *, gray: torch.Tensor, under: int, over: int
) -> None:
    height, width = gray.shape
    rows = torch.arange(height, device=gray.device).view(height, 1)
    columns = torch.arange(width, device=gray.device).view(1, width)
    stripes = ((rows + columns) // ZEBRA_SPACING) % 2 == 0
    _blend(
        drawn,
        mask=(gray < under) & stripes,
        color_rgb=UNDEREXPOSED_RGB,
        opacity=ZEBRA_OPACITY,
    )
    _blend(
        drawn,
        mask=(gray > over) & stripes,
        color_rgb=OVEREXPOSED_RGB,
        opacity=ZEBRA_OPACITY,
    )


def _draw_peaking(drawn: torch.Tensor, *, focus: torch.Tensor) -> None:
    # Focus values are sums of squared integers: a non-zero maximum is >= 1,
    # so clamp(min=1) only matters for an all-zero frame, which V1 skipped.
    normalized = (focus / focus.max().clamp(min=1) * 255).to(torch.uint8)
    threshold = int(PEAKING_THRESHOLD_PERCENT * 255 / 100)
    _blend(
        drawn,
        mask=normalized > threshold,
        color_rgb=PEAKING_RGB,
        opacity=PEAKING_OPACITY,
    )


def _overlay_scale(height: int, width: int) -> float:
    scale = min(height, width) / 720

    return max(0.4, min(scale, 2.5))


def _draw_center_marker(drawn: torch.Tensor) -> None:
    _, height, width = drawn.shape
    scale = _overlay_scale(height, width)
    center_x, center_y = width // 2, height // 2
    size, thickness = int(20 * scale), max(1, int(1.6 * scale))
    reach = size + thickness + PATCH_MARGIN

    def draw(patch: np.ndarray, origin: Tuple[int, int]) -> None:
        x, y = center_x - origin[0], center_y - origin[1]
        white = (255, 255, 255)
        cv2.line(patch, (x - size, y), (x + size, y), white, thickness)
        cv2.line(patch, (x, y - size), (x, y + size), white, thickness)

    _edit_patch(
        drawn,
        rect=(center_x - reach, center_y - reach, center_x + reach, center_y + reach),
        draw=draw,
    )


def _hud_layout(height: int, width: int, focus_value: float) -> _HudLayout:
    scale = _overlay_scale(height, width)
    font_scale, small_font_scale = 0.5 * scale, 0.4 * scale
    thickness = max(1, int(1.6 * scale))
    focus_text = f"{focus_value:.1f}"

    def text_hw(text: str, size: float) -> Tuple[int, int]:
        (text_width, text_height), _ = cv2.getTextSize(text, HUD_FONT, size, thickness)
        return text_height, text_width

    focus_label_hw = text_hw(HUD_FOCUS_LABEL, font_scale)
    focus_value_hw = text_hw(focus_text, font_scale)
    exposure_label_hw = text_hw(HUD_EXPOSURE_LABEL, small_font_scale)
    padding = int(14 * scale)
    histogram_width, histogram_height = int(180 * scale), int(50 * scale)
    line_spacing, section_spacing = int(6 * scale), int(12 * scale)
    content_width = max(
        focus_label_hw[1] + focus_value_hw[1] + int(20 * scale),
        exposure_label_hw[1],
        histogram_width,
    )
    content_height = (
        focus_label_hw[0]
        + section_spacing
        + exposure_label_hw[0]
        + line_spacing
        + histogram_height
    )
    margin = int(12 * scale)
    layout = _HudLayout(
        x=margin,
        y=margin,
        width=content_width + padding * 2,
        height=content_height + padding * 2,
        padding=padding,
        histogram_width=histogram_width,
        histogram_height=histogram_height,
        font_scale=font_scale,
        small_font_scale=small_font_scale,
        thickness=thickness,
        line_spacing=line_spacing,
        section_spacing=section_spacing,
        focus_label_hw=focus_label_hw,
        focus_value_hw=focus_value_hw,
        exposure_label_hw=exposure_label_hw,
        focus_text=focus_text,
    )

    return layout


def _draw_hud(
    drawn: torch.Tensor,
    *,
    source: torch.Tensor,
    gray: torch.Tensor,
    focus_value: float,
) -> None:
    _, height, width = drawn.shape
    layout = _hud_layout(height, width, focus_value)
    # Histograms of B, G, R (color images only) and gray, in one transfer.
    planes = [source[channel] for channel in (2, 1, 0)] if source.shape[0] == 3 else []
    counts = torch.stack(
        [torch.bincount(plane.reshape(-1), minlength=256) for plane in [*planes, gray]]
    )
    histograms = counts.to(torch.float32).cpu().numpy()
    colors = HUD_HISTOGRAM_BGR[:3] if planes else ()

    def draw(patch: np.ndarray, origin: Tuple[int, int]) -> None:
        _draw_hud_panel(
            patch,
            layout=layout,
            origin=origin,
            histograms=list(zip(histograms, (*colors, HUD_HISTOGRAM_BGR[3]))),
        )

    _edit_patch(
        drawn,
        rect=(
            layout.x - PATCH_MARGIN,
            layout.y - PATCH_MARGIN,
            layout.x + layout.width + PATCH_MARGIN + 1,
            layout.y + layout.height + PATCH_MARGIN + 1,
        ),
        draw=draw,
    )


def _draw_hud_panel(
    patch: np.ndarray,
    *,
    layout: _HudLayout,
    origin: Tuple[int, int],
    histograms: List[Tuple[np.ndarray, Tuple[int, int, int]]],
) -> None:
    """Draw V1's HUD onto a BGR patch whose top-left is ``origin`` in the frame."""
    x, y = layout.x - origin[0], layout.y - origin[1]
    right, bottom = x + layout.width, y + layout.height
    shaded = patch.copy()
    cv2.rectangle(shaded, (x, y), (right, bottom), (0, 0, 0), -1)
    cv2.addWeighted(shaded, 0.7, patch, 0.3, 0, patch)
    cv2.rectangle(patch, (x, y), (right, bottom), (80, 80, 80), 1)

    text_x = x + layout.padding
    cursor_y = y + layout.padding + layout.focus_label_hw[0]
    white = (255, 255, 255)
    _outlined_text(
        patch, HUD_FOCUS_LABEL, (text_x, cursor_y), layout.font_scale, white, layout
    )
    value_x = right - layout.padding - layout.focus_value_hw[1]
    _outlined_text(
        patch, layout.focus_text, (value_x, cursor_y), layout.font_scale, white, layout
    )
    divider_y = cursor_y + int(layout.section_spacing * 0.5)
    cv2.line(
        patch,
        (x + layout.padding, divider_y),
        (right - layout.padding, divider_y),
        (60, 60, 60),
        1,
    )

    cursor_y += layout.section_spacing + layout.exposure_label_hw[0]
    _outlined_text(
        patch,
        HUD_EXPOSURE_LABEL,
        (text_x, cursor_y),
        layout.small_font_scale,
        (180, 180, 180),
        layout,
    )

    top = cursor_y + layout.line_spacing
    base = top + layout.histogram_height - 1
    cv2.rectangle(
        patch, (text_x, top), (text_x + layout.histogram_width, base), (0, 0, 0), -1
    )
    xs = np.linspace(text_x, text_x + layout.histogram_width - 1, 256).astype(np.int32)
    for counts, color in histograms:
        peak = counts.max()
        if peak > 0:
            heights = (counts / peak * layout.histogram_height).astype(np.int32)
            points = np.column_stack([xs, base - heights]).astype(np.int32)
            cv2.polylines(patch, [points], False, color, max(1, layout.thickness))


def _outlined_text(
    patch: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int],
    layout: _HudLayout,
) -> None:
    cv2.putText(
        patch, text, position, HUD_FONT, font_scale, (0, 0, 0), layout.thickness + 2
    )
    cv2.putText(patch, text, position, HUD_FONT, font_scale, color, layout.thickness)


def _edit_patch(
    drawn: torch.Tensor,
    *,
    rect: Tuple[int, int, int, int],
    draw: Callable[[np.ndarray, Tuple[int, int]], None],
) -> None:
    """Let OpenCV draw in place on one frame region, copied to the host.

    The region is clipped to the frame first, so a stroke past the frame
    edge is clipped where a full-frame call would clip it. ``draw`` receives
    a BGR HWC patch and its top-left position in the frame.
    """
    _, height, width = drawn.shape
    x1, y1 = max(0, rect[0]), max(0, rect[1])
    x2, y2 = min(width, rect[2]), min(height, rect[3])
    if x2 <= x1 or y2 <= y1:
        return

    region = drawn[:, y1:y2, x1:x2]
    patch = np.ascontiguousarray(region.flip(0).permute(1, 2, 0).cpu().numpy())
    draw(patch, (x1, y1))
    region.copy_(torch.from_numpy(patch).to(drawn.device).permute(2, 0, 1).flip(0))
