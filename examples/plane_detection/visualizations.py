"""Visualization utilities for plane detection results."""

from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_label_colors(n_labels: int) -> np.ndarray:
    """Generate n_labels distinct RGB colors (0-255)."""
    colors = []
    for i in range(max(n_labels, 1)):
        hue = (i * 0.618033988749895) % 1.0  # golden ratio for spreading
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(colors, dtype=np.uint8)


def get_plane_visualization_fig(
    input_image_path: Path,
    label_img: np.ndarray,
    opacity: float = 0.5,
) -> go.Figure:
    """Visualize input image with label overlay using plotly."""
    img = cv2.imread(str(input_image_path))
    if img is None:
        raise ValueError(f"Could not load image: {input_image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H_label, W_label = label_img.shape
    H_img, W_img = img_rgb.shape[:2]
    if (H_img, W_img) != (H_label, W_label):
        img_rgb = cv2.resize(img_rgb, (W_label, H_label), interpolation=cv2.INTER_LINEAR)

    img_float = img_rgb.astype(np.float64) / 255.0

    unique_labels = np.unique(label_img)
    unique_labels = unique_labels[unique_labels >= 0]
    n_labels = len(unique_labels)
    colors = get_label_colors(n_labels)

    overlay = img_float.copy()
    for i, lbl in enumerate(unique_labels):
        mask = label_img == lbl
        if not np.any(mask):
            continue
        color = colors[i].astype(np.float64) / 255.0
        alpha = opacity * mask.astype(np.float64)[:, :, np.newaxis]
        overlay = overlay * (1 - alpha) + color * alpha

    overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Input image", "Plane overlay"))
    fig.add_trace(go.Image(z=img_rgb), row=1, col=1)
    fig.add_trace(go.Image(z=overlay_uint8), row=1, col=2)
    fig.update_layout(
        width=1200,
        height=600,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return fig
