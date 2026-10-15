"""Colourise an anomaly map and blend it over an image.

Anomaly map values share the units of the model's anomaly score and are
unbounded above, so every value is placed on a fixed saturating scale anchored
to the model's threshold: ``t = v / (v + threshold)``. Zero is transparent, the
threshold sits at the ramp's midpoint, and the ramp saturates as values grow.
The scale depends only on the model, never on the image, so the same colour
means the same evidence on every image and normal images show their map too.
This is local anomaly evidence, not a defect mask.

The ``ROBOFLOW`` ramp reproduces the stops the Roboflow app draws in its
inference preview; the other ramps are OpenCV colour maps over the same scale.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np

ROBOFLOW_COLOR_MAP = "ROBOFLOW"

# Stops on the [0, 1) scale, where 0.5 is the saved threshold: (position, (r, g, b), alpha).
ROBOFLOW_COLOR_STOPS: List[Tuple[float, Tuple[int, int, int], float]] = [
    (0.0, (255, 214, 0), 0.0),
    (0.25, (255, 214, 0), 0.55),
    (0.5, (255, 122, 0), 0.85),
    (1.0, (220, 30, 30), 1.0),
]

OPENCV_COLOR_MAPS: Dict[str, int] = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "HOT": cv2.COLORMAP_HOT,
}

SUPPORTED_COLOR_MAPS = (ROBOFLOW_COLOR_MAP, *OPENCV_COLOR_MAPS)


def normalize_anomaly_map(anomaly_map: np.ndarray, threshold: float) -> np.ndarray:
    """Map raw anomaly values onto ``[0, 1)``: 0 at or below zero, 0.5 at the threshold.

    Args:
        anomaly_map (np.ndarray): Raw map in the units of the anomaly score.
        threshold (float): The model's decision threshold; must be positive and finite.

    Returns:
        A float32 array of the same shape with values in ``[0, 1)``.

    Raises:
        ValueError: If ``threshold`` is not a positive finite number.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            f"Anomaly threshold must be a positive finite number to anchor the "
            f"heatmap ramp, got {threshold!r}"
        )
    positive = np.maximum(anomaly_map.astype(np.float32, copy=False), 0.0)
    normalized = positive / (positive + np.float32(threshold))
    return normalized.astype(np.float32, copy=False)


def roboflow_ramp(normalized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Colour a normalised map with the Roboflow app's yellow-to-red ramp.

    Args:
        normalized (np.ndarray): Values in ``[0, 1]`` (clipped otherwise).

    Returns:
        ``(bgr, alpha)``: a uint8 ``(h, w, 3)`` BGR image and a float32 ``(h, w)``
        alpha in ``[0, 1]`` interpolated from the ramp stops.
    """
    t = np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)
    positions = [stop[0] for stop in ROBOFLOW_COLOR_STOPS]
    channels = []
    for channel_index in range(3):
        channel_values = [stop[1][channel_index] for stop in ROBOFLOW_COLOR_STOPS]
        channels.append(np.interp(t, positions, channel_values))
    alphas = [stop[2] for stop in ROBOFLOW_COLOR_STOPS]
    alpha = np.interp(t, positions, alphas).astype(np.float32)
    rgb = np.stack(channels, axis=-1)
    bgr = np.rint(rgb[..., ::-1]).astype(np.uint8)
    return bgr, alpha


def colorize_anomaly_map(
    normalized: np.ndarray, color_map: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Colour a normalised map with the named ramp.

    Args:
        normalized (np.ndarray): Values in ``[0, 1]``.
        color_map (str): ``"ROBOFLOW"`` or one of the OpenCV names in
            ``OPENCV_COLOR_MAPS``.

    Returns:
        ``(bgr, alpha)`` as :func:`roboflow_ramp`. OpenCV ramps are opaque
        (alpha 1 everywhere); the blend opacity alone controls their strength.

    Raises:
        ValueError: If ``color_map`` is not supported.
    """
    name = str(color_map).upper()
    if name == ROBOFLOW_COLOR_MAP:
        return roboflow_ramp(normalized)
    if name not in OPENCV_COLOR_MAPS:
        raise ValueError(
            f"Unsupported anomaly heatmap color_map {color_map!r}; "
            f"expected one of {list(SUPPORTED_COLOR_MAPS)}"
        )
    scaled = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(scaled, OPENCV_COLOR_MAPS[name])
    alpha = np.ones(normalized.shape, dtype=np.float32)
    return bgr, alpha


class AnomalyHeatmapAnnotator:
    """Blend a colourised anomaly map over a BGR scene."""

    def __init__(self, color_map: str, opacity: float):
        if str(color_map).upper() not in SUPPORTED_COLOR_MAPS:
            raise ValueError(
                f"Unsupported anomaly heatmap color_map {color_map!r}; "
                f"expected one of {list(SUPPORTED_COLOR_MAPS)}"
            )
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"opacity must be within [0, 1], got {opacity!r}")
        self.color_map = str(color_map).upper()
        self.opacity = float(opacity)

    def annotate(
        self, scene: np.ndarray, anomaly_map: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Draw the map onto ``scene`` in place and return it.

        Args:
            scene (np.ndarray): uint8 BGR ``(h, w, 3)`` or grayscale ``(h, w)``
                image; a grayscale scene is promoted to BGR.
            anomaly_map (np.ndarray): 2D raw map at any resolution; it is
                stretched to the scene with bilinear interpolation.
            threshold (float): The threshold anchoring the ramp.

        Returns:
            The annotated uint8 BGR image.

        Raises:
            ValueError: If the map is not 2D or the threshold is invalid.
        """
        anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
        if anomaly_map.ndim != 2:
            raise ValueError(
                f"Anomaly map must be 2D, got shape {tuple(anomaly_map.shape)}"
            )
        if scene.ndim == 2:
            scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
        height, width = scene.shape[:2]

        normalized = normalize_anomaly_map(anomaly_map, threshold=threshold)
        if normalized.shape != (height, width):
            normalized = cv2.resize(
                normalized, (width, height), interpolation=cv2.INTER_LINEAR
            )
        bgr, alpha = colorize_anomaly_map(normalized, color_map=self.color_map)

        weight = (alpha * self.opacity)[..., None]
        blended = (
            scene.astype(np.float32) * (1.0 - weight) + bgr.astype(np.float32) * weight
        )
        np.copyto(scene, np.rint(blended).astype(np.uint8))
        return scene
