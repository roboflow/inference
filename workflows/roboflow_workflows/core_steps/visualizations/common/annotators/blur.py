from typing import Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from supervision.detection.compact_mask import CompactMask


class MaskAwareBlurAnnotator(sv.BlurAnnotator):
    """Blur annotator that follows segmentation masks when detections carry them.

    `sv.BlurAnnotator` blurs each detection's bounding box and never reads
    `Detections.mask`, so instance segmentation predictions were blurred as
    rectangles. With masks present, this annotator blurs the region covering both
    the box and the mask's own extent (masks may reach past their box), then
    copies the blurred pixels back only where the mask is on. Without masks it
    blurs boxes exactly like `sv.BlurAnnotator`.

    Args:
        kernel_size (Optional[int]): Size of the average pooling kernel. When
            `None`, a dynamic size of one-third of each box's shorter side is used.
        padding (int): Extra pixels blurred around each detection. Boxes grow by
            this many pixels on every side; masks grow outward by this many
            pixels. `0` blurs exactly the box or mask.
    """

    def __init__(self, kernel_size: Optional[int] = None, padding: int = 0):
        super().__init__(kernel_size=kernel_size)
        if padding < 0:
            raise ValueError(f"padding must be >= 0, got {padding}.")
        self.padding = padding

    def annotate(self, scene: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Blur each detection's mask, or its box when it has no mask, in place.

        Args:
            scene (np.ndarray): Image to blur, modified in place.
            detections (sv.Detections): Detections to blur. Dense masks and
                `CompactMask` masks are both supported; `CompactMask` masks are
                read as box-sized crops.

        Returns:
            np.ndarray: `scene`, with each detection blurred.
        """
        if detections.mask is None:
            if self.padding == 0:
                return super().annotate(scene=scene, detections=detections)
            padded = sv.Detections(
                xyxy=detections.xyxy
                + np.array([-1, -1, 1, 1], dtype=np.float64) * self.padding
            )
            return super().annotate(scene=scene, detections=padded)
        image_height, image_width = scene.shape[:2]
        clipped_xyxy = sv.clip_boxes(
            xyxy=detections.xyxy, resolution_wh=(image_width, image_height)
        ).astype(int)
        for index, (x1, y1, x2, y2) in enumerate(clipped_xyxy):
            mask_crop = _mask_crop(detections.mask, index)
            if mask_crop is None:
                continue
            crop, crop_x, crop_y = mask_crop
            crop_height, crop_width = crop.shape
            region_x1 = max(min(x1, crop_x) - self.padding, 0)
            region_y1 = max(min(y1, crop_y) - self.padding, 0)
            region_x2 = min(max(x2, crop_x + crop_width) + self.padding, image_width)
            region_y2 = min(max(y2, crop_y + crop_height) + self.padding, image_height)
            if region_x2 <= region_x1 or region_y2 <= region_y1:
                continue
            inside = np.zeros((region_y2 - region_y1, region_x2 - region_x1), bool)
            _paste(inside, crop, crop_x - region_x1, crop_y - region_y1)
            if self.padding > 0:
                inside = _grow(inside, self.padding)
            if not inside.any():
                continue
            kernel_size = (
                self.kernel_size
                if self.kernel_size is not None
                else _dynamic_kernel_size(region_x1, region_y1, region_x2, region_y2)
            )
            roi = scene[region_y1:region_y2, region_x1:region_x2]
            blurred = cv2.blur(roi, (kernel_size, kernel_size))
            roi[inside] = blurred[inside]
        return scene


def _mask_crop(masks, index: int) -> Optional[Tuple[np.ndarray, int, int]]:
    """Return one detection's mask trimmed to its pixels, with its (x, y) origin.

    `CompactMask` is read with `crop()`, so no full-frame array is built. Returns
    `None` when the mask has no pixels.
    """
    if isinstance(masks, CompactMask):
        crop = masks.crop(index)
        origin_x, origin_y = (int(value) for value in masks.offsets[index])
    else:
        crop = np.asarray(masks[index], dtype=bool)
        origin_x, origin_y = 0, 0
    rows = np.flatnonzero(crop.any(axis=1))
    if rows.size == 0:
        return None
    columns = np.flatnonzero(crop.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    return crop[top:bottom, left:right], origin_x + left, origin_y + top


def _paste(target: np.ndarray, crop: np.ndarray, x: int, y: int) -> None:
    """Copy `crop` into `target` at `(x, y)`, dropping parts that fall outside."""
    target_height, target_width = target.shape
    crop_height, crop_width = crop.shape
    left, top = max(x, 0), max(y, 0)
    right = min(x + crop_width, target_width)
    bottom = min(y + crop_height, target_height)
    if right <= left or bottom <= top:
        return
    target[top:bottom, left:right] |= crop[top - y : bottom - y, left - x : right - x]


def _grow(mask: np.ndarray, padding: int) -> np.ndarray:
    """Grow a boolean mask outward by `padding` pixels with a round kernel."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * padding + 1, 2 * padding + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _dynamic_kernel_size(x1: int, y1: int, x2: int, y2: int) -> int:
    """Blur kernel size for a region: one-third of its shorter side, at least 1.

    The rule `sv.BlurAnnotator` applies when `kernel_size` is `None`, inlined so
    this module does not import a helper from supervision's private
    `annotators.utils` namespace.
    """
    return max(1, min(y2 - y1, x2 - x1) // 3)
