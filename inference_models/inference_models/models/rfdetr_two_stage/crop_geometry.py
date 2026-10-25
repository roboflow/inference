"""Top-down crop geometry for stage 2 of the RF-DETR two-stage keypoint model.

The exported stage-2 graph is keypoints only: it consumes fixed-size, normalized
crops and returns keypoints in normalized crop coordinates. Cutting the crop
out of the image and mapping the keypoints back is the caller's job, and it
must match the geometry the model was trained with. This module mirrors the
trainer's ``crop_and_resize`` and ``invert_crop_keypoints`` transforms: the
same aspect extension, context padding, UDP
denominators and affine warp, expressed as one ``normalized crop -> image``
affine so the inverse mapping and the ``s_norm`` window area fall out of it.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class TopDownCropConfig:
    """The ``crop`` block of the stage-2 export manifest."""

    input_size: Tuple[int, int]  # (height, width)
    context_padding: float
    preserve_aspect: bool
    udp: bool


@dataclass(frozen=True)
class CropGeometry:
    """Maps normalized crop coordinates in [0, 1] to source image pixels."""

    normalized_to_image: np.ndarray  # (2, 3) float32 affine

    @property
    def window_area(self) -> float:
        """Area in image pixels of the window the crop was cut from."""
        area = abs(
            float(np.linalg.det(self.normalized_to_image[:, :2].astype(np.float64)))
        )
        return max(area, 1.0)

    def to_image(self, xy_normalized: np.ndarray) -> np.ndarray:
        """(K, 2) normalized crop coordinates -> (K, 2) image pixels."""
        points = np.asarray(xy_normalized, dtype=np.float32).reshape(-1, 2)
        homogeneous = np.concatenate(
            [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
        )
        return (homogeneous @ self.normalized_to_image.T).astype(np.float32)


def crop_object(
    image_rgb: np.ndarray,
    box_xyxy: Sequence[float],
    config: TopDownCropConfig,
) -> Tuple[np.ndarray, CropGeometry]:
    """Cut one object's crop the way the stage-2 model was trained to see it.

    Returns the ``(H, W, 3)`` uint8 crop at ``config.input_size`` and the
    geometry that maps the model's normalized output back into ``image_rgb``.
    """
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    box_width, box_height = x2 - x1, y2 - y1
    out_height, out_width = config.input_size
    if config.preserve_aspect:
        return _crop_with_aspect_window(
            image_rgb,
            center=(x1 + box_width * 0.5, y1 + box_height * 0.5),
            half_size=(
                box_width * 0.5 * config.context_padding,
                box_height * 0.5 * config.context_padding,
            ),
            out_size=(out_height, out_width),
            udp=config.udp,
        )
    return _crop_axis_aligned(
        image_rgb,
        box=(x1, y1, x2, y2),
        image_size=(height, width),
        out_size=(out_height, out_width),
    )


def s_norm_for_box(box_xyxy: Sequence[float], geometry: CropGeometry) -> float:
    """``sqrt(object_box_area / crop_window_area)``, the model's per-crop scale input."""
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    object_area = max((x2 - x1) * (y2 - y1), 0.0)
    return float(np.sqrt(object_area / geometry.window_area))


def _crop_with_aspect_window(
    image_rgb: np.ndarray,
    center: Tuple[float, float],
    half_size: Tuple[float, float],
    out_size: Tuple[int, int],
    udp: bool,
) -> Tuple[np.ndarray, CropGeometry]:
    out_height, out_width = out_size
    center_x, center_y = center
    half_width, half_height = half_size
    aspect = out_width / out_height
    if half_width < aspect * half_height:
        half_width = aspect * half_height
    else:
        half_height = half_width / aspect
    source = np.asarray(
        [
            [center_x, center_y],
            [center_x + half_width, center_y],
            [center_x, center_y + half_height],
        ],
        dtype=np.float32,
    )
    denominator_width = float(out_width - 1) if udp else float(out_width)
    denominator_height = float(out_height - 1) if udp else float(out_height)
    destination = np.asarray(
        [
            [denominator_width * 0.5, denominator_height * 0.5],
            [denominator_width, denominator_height * 0.5],
            [denominator_width * 0.5, denominator_height],
        ],
        dtype=np.float32,
    )
    affine = cv2.getAffineTransform(source, destination)
    crop = cv2.warpAffine(
        image_rgb,
        affine,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    inverse = cv2.invertAffineTransform(affine).astype(np.float32)
    # Normalized output -> crop pixels (scaled by the UDP-aware denominators)
    # -> image pixels through the inverse warp, folded into one affine.
    normalized_to_image = np.concatenate(
        (
            inverse[:, :2]
            @ np.diag([denominator_width, denominator_height]).astype(np.float32),
            inverse[:, 2:3],
        ),
        axis=1,
    )
    return crop, CropGeometry(normalized_to_image=normalized_to_image)


def _crop_axis_aligned(
    image_rgb: np.ndarray,
    box: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    out_size: Tuple[int, int],
) -> Tuple[np.ndarray, CropGeometry]:
    height, width = image_size
    out_height, out_width = out_size
    x1, y1, x2, y2 = box
    x0 = max(0, int(np.floor(x1)))
    y0 = max(0, int(np.floor(y1)))
    x1_clipped = min(width, int(np.ceil(x2)))
    y1_clipped = min(height, int(np.ceil(y2)))
    if x1_clipped <= x0 or y1_clipped <= y0:
        crop = np.zeros(
            (out_height, out_width, image_rgb.shape[2]), dtype=image_rgb.dtype
        )
        geometry = CropGeometry(
            normalized_to_image=np.asarray(
                [[1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)]], dtype=np.float32
            )
        )
        return crop, geometry
    source = image_rgb[y0:y1_clipped, x0:x1_clipped]
    source_height, source_width = source.shape[:2]
    crop = cv2.resize(source, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
    geometry = CropGeometry(
        normalized_to_image=np.asarray(
            [
                [float(source_width), 0.0, float(x0)],
                [0.0, float(source_height), float(y0)],
            ],
            dtype=np.float32,
        )
    )
    return crop, geometry


def full_image_box(image: np.ndarray) -> List[float]:
    """The box covering a whole image, for treating the image itself as one crop."""
    height, width = image.shape[:2]
    return [0.0, 0.0, float(width), float(height)]
