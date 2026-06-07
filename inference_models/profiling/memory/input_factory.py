from __future__ import annotations

from typing import List

import numpy as np


def build_random_rgb_images(
    batch_size: int,
    *,
    height: int,
    width: int,
) -> List[np.ndarray]:
    """Build deterministic random RGB images for profiling runs.

    Uses a fixed NumPy seed so repeated runs see the same pixel values.

    Args:
        batch_size: Number of images in the batch.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        List of ``uint8`` arrays shaped ``(height, width, 3)``.
    """
    rng = np.random.default_rng(0)
    images = []

    for _ in range(batch_size):
        image = rng.integers(
            0,
            255,
            size=(height, width, 3),
            dtype=np.uint8,
        )
        images.append(
            image,
        )

    return images
