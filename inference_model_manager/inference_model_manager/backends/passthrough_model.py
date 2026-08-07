"""Passthrough model for benchmarking infrastructure overhead."""

from __future__ import annotations

import torch


class _DummyDetections:
    """Minimal Detections-like object.

    Uses torch tensors (not numpy) so the subprocess worker's result
    normalization (``result.xyxy.cpu()`` etc.) is a no-op rather than an
    AttributeError. Keeps the hot path branch-free.
    """

    __slots__ = ("xyxy", "confidence", "class_id")

    def __init__(self):
        self.xyxy = torch.zeros((1, 4), dtype=torch.float32)
        self.confidence = torch.tensor([0.99], dtype=torch.float32)
        self.class_id = torch.tensor([0], dtype=torch.int64)


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
