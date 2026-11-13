"""Mask geometry shared by both Perspective Correction implementations."""

from typing import Optional, Tuple

import cv2 as cv
import numpy as np
import supervision as sv

# OpenCV's perspective warp uses signed 16-bit coordinates in remap.
_OPENCV_WARP_DIMENSION_LIMIT = np.iinfo(np.int16).max


def _warp_instance_mask(
    mask: np.ndarray,
    *,
    transform: np.ndarray,
    resolution_wh: Tuple[int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Warp the mask and return its bounds, or None for an empty source mask."""
    width, height = resolution_wh
    source_mask = mask.astype(np.uint8)
    contours, hierarchy = cv.findContours(
        image=source_mask, mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        empty_mask = np.zeros((height, width), dtype=bool)
        return empty_mask, None

    vertices = cv.perspectiveTransform(
        src=np.concatenate(contours, axis=0).astype(np.float32), m=transform
    ).reshape(-1, 2)
    box = np.rint(sv.polygon_to_xyxy(polygon=vertices)).astype(np.int32)

    has_complex_mask = len(contours) > 1 or len(contours[0]) < 3
    can_warp_raster = (
        max(*source_mask.shape, width, height) < _OPENCV_WARP_DIMENSION_LIMIT
    )
    if has_complex_mask and can_warp_raster:
        warped = cv.warpPerspective(
            src=source_mask, M=transform, dsize=resolution_wh, flags=cv.INTER_NEAREST
        )
        x, y, raster_width, raster_height = cv.boundingRect(warped)
        if raster_width:
            # Include visible pixels introduced by resampling at larger scales.
            box[:2] = np.minimum(box[:2], (x, y))
            box[2:] = np.maximum(box[2:], (x + raster_width - 1, y + raster_height - 1))
    else:
        # Keep the original single-polygon rasterization and support canvases
        # outside the raster warp's limits. CCOMP keeps holes with their exterior.
        offsets = np.cumsum([len(contour) for contour in contours[:-1]], dtype=np.intp)
        polygons = np.split(np.rint(vertices).astype(np.int32), offsets)
        groups: dict[int, list[np.ndarray]] = {}
        for index, (polygon, links) in enumerate(zip(polygons, hierarchy[0])):
            parent = int(links[3])
            group = index if parent == -1 else parent
            groups.setdefault(group, []).append(polygon)

        warped = np.zeros((height, width), dtype=np.uint8)
        for group_polygons in groups.values():
            cv.fillPoly(img=warped, pts=group_polygons, color=1)

    warped_mask = warped.astype(bool)
    return warped_mask, box
