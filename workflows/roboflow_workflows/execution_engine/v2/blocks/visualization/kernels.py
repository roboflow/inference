"""Drawing backends: how boxes and labels get into image tensors, in place.

Both backends paint the same pixels from the same ``DetectionDrawing`` values
(example 13 checked it on Jetson Thor; here a CPU stand-in checks the triton
tables). Each paints a batch of scenes in place, boxes before labels, on the
stream chosen by the painter block. The block owns the final completion wait;
the tensor backend's per-image box uploads also block the host::

    backend  where                 per call of a painter
    tensor   CPU and CUDA          per image: gpu_draw_boxes (one pageable box
                                   table upload, which drains the stream) and
                                   gpu_paste_label_sprites (pinned, non-blocking)
    triton   CUDA with Triton      one pinned non-blocking table upload and
                                   one kernel launch for the whole batch

``tensor`` reuses the V1 tensor primitives unchanged. ``triton`` is example
13's single-launch kernel; its module and Triton itself are imported on the
first triton call only. There is no fallback between backends.

A backend instance holds its caches (sprites, canvases, table rings) and is
not thread-safe: one painter call at a time.
"""

from typing import Protocol, Sequence, Tuple

import numpy as np
import torch
from roboflow_workflows.core_steps.visualizations.bounding_box.v1_tensor import (
    gpu_draw_boxes,
)
from roboflow_workflows.core_steps.visualizations.label.v1_tensor import (
    _PinnedSlabRing,
    gpu_paste_label_sprites,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.drawing import (
    BOX_THICKNESS,
    PALETTE,
    DetectionDrawing,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.labels import (
    LabelPlacement,
)

BACKENDS: Tuple[str, ...] = ("tensor", "triton")


class DrawingBackendUnavailable(RuntimeError):
    """The selected drawing backend cannot run here; nothing was painted."""


class DrawingKernels(Protocol):
    """What a painter block needs of a backend."""

    name: str

    def paint(
        self,
        scenes: Sequence[torch.Tensor],
        drawings: Sequence[DetectionDrawing],
        *,
        boxes: bool,
        labels: bool,
    ) -> None:
        """Paint ``scenes`` in place from ``drawings``: boxes, then labels."""


class TensorKernels:
    """Paint with the V1 tensor primitives, image after image."""

    name = "tensor"

    def __init__(self) -> None:
        self.placement = LabelPlacement()
        self._table_ring = _PinnedSlabRing()

    def paint(
        self,
        scenes: Sequence[torch.Tensor],
        drawings: Sequence[DetectionDrawing],
        *,
        boxes: bool,
        labels: bool,
    ) -> None:
        """Paint every scene in place: boxes, then labels; image after image.

        Args:
            scenes: Contiguous CHW RGB uint8 tensors on one device.
            drawings: Their prepared detections, in the same order.
            boxes: Draw box borders.
            labels: Paste labels, after the boxes.
        """
        for scene, drawing in zip(scenes, drawings):
            if boxes:
                self._boxes(scene, drawing)
            if labels:
                self._labels(scene, drawing)

    def _boxes(self, scene: torch.Tensor, drawing: DetectionDrawing) -> None:
        if len(drawing) == 0:
            return

        colors_rgb = np.asarray(
            [PALETTE.by_idx(int(one)).as_rgb() for one in drawing.class_id],
            dtype=np.uint8,
        )
        gpu_draw_boxes(scene, drawing.xyxy.astype(int), colors_rgb, BOX_THICKNESS)

    def _labels(self, scene: torch.Tensor, drawing: DetectionDrawing) -> None:
        sprites, origins = self.placement.sprites(
            drawing,
            frame_hw=(int(scene.shape[1]), int(scene.shape[2])),
            device=scene.device,
        )
        gpu_paste_label_sprites(scene, sprites, origins, self._table_ring)


def create_kernels(backend: str) -> DrawingKernels:
    """A new backend instance; only that backend's module is imported.

    Args:
        backend: One of ``BACKENDS``.

    Returns:
        ``TensorKernels`` or ``triton_tables.TritonKernels``.

    Raises:
        ValueError: On an unknown backend.
    """
    if backend == "tensor":
        kernels = TensorKernels()
    elif backend == "triton":
        from roboflow_workflows.execution_engine.v2.blocks.visualization.triton_tables import (  # noqa: E501
            TritonKernels,
        )

        kernels = TritonKernels()
    else:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

    return kernels
