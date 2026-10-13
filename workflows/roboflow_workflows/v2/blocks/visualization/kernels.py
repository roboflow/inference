"""How the painters get boxes and labels into image tensors, in place.

``TensorKernels`` reuses the V1 tensor primitives unchanged, on CPU and CUDA.
It paints a batch of scenes in place, image after image, boxes before labels,
on the stream chosen by the painter block. The block owns the final
completion wait; per image, these uploads happen::

    gpu_draw_boxes            one pageable box table upload (drains the stream)
    gpu_paste_label_sprites   pinned, non-blocking

An instance holds its caches (sprites, table ring) and is not thread-safe:
one painter call at a time.
"""

from typing import Sequence

import numpy as np
import torch
from roboflow_workflows.core_steps.visualizations.bounding_box.v1_tensor import (
    gpu_draw_boxes,
)
from roboflow_workflows.core_steps.visualizations.label.v1_tensor import (
    _PinnedSlabRing,
    gpu_paste_label_sprites,
)
from roboflow_workflows.v2.blocks.visualization.drawing import (
    BOX_THICKNESS,
    PALETTE,
    DetectionDrawing,
)
from roboflow_workflows.v2.blocks.visualization.labels import (
    LabelPlacement,
)


class TensorKernels:
    """Paint with the V1 tensor primitives, image after image."""

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
