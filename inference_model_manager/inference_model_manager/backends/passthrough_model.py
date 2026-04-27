"""Passthrough model for benchmarking infrastructure overhead."""

from __future__ import annotations

import numpy as np


class _DummyDetections:
    """Minimal Detections-like object."""

    __slots__ = ("xyxy", "confidence", "class_id")

    def __init__(self):
        self.xyxy = np.zeros((1, 4), dtype=np.float32)
        self.confidence = np.array([0.99], dtype=np.float32)
        self.class_id = np.array([0], dtype=np.int64)


class PassthroughModel:
    """Returns dummy detections instantly. No weights, no GPU, no base class."""

    @property
    def class_names(self):
        return ["dummy"]

    @property
    def max_batch_size(self):
        return 64

    def infer(self, images, **kwargs):
        if isinstance(images, list):
            return [_DummyDetections() for _ in images]
        return _DummyDetections()
