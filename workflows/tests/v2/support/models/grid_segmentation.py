"""A grid-backed instance-segmentation producer double, for contract tests.

This double overrides the compact implementation's producer hook to supply
known low-resolution grid views, so tests can count demand and materialization.
Real compact backend integration is exercised separately.

It is NOT a compact model backend and says nothing about the speed of one:
the fake's grid is half the image size and box-shaped::

    image (H, W)  ->  candidates i < round(4 * brightness)
                      grid (Q, H/2, W/2) probabilities, 1.0 inside box i / 2
                      rows kept: confidence 0.9 - 0.1 * i >= threshold

Image sides must be even.
"""

from typing import Any, Dict, List, Tuple

import torch
from roboflow_workflows.v2.blocks.models.common import ModelInput
from roboflow_workflows.v2.blocks.models.roboflow_platform.instance_segmentation_model.v4 import (
    CompactMasks,
    InstanceSegmentationBlock,
)
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry, SegmentationView

from inference_models.models.base.object_detection import Detections
from tests.v2.support.models.fake_task_models import (
    CLASS_NAMES,
    InstanceModel,
    box_masks,
    row_count,
)

GRID_STRIDE = 2
GRID_THRESHOLD = 0.5


class GridInstanceModel(InstanceModel):
    """``InstanceModel`` that can also decode to a mask grid per image."""

    def post_process_grid(
        self, raw: torch.Tensor, sizes: list, **options: Any
    ) -> List[Tuple[Detections, torch.Tensor, torch.Tensor, MaskGridGeometry]]:
        """Per image: kept detections, the grid of every candidate, kept rows."""
        threshold = self._record(**options)
        decoded = []
        for brightness, (height, width) in zip(raw.tolist(), sizes):
            index = torch.arange(row_count(brightness))
            xyxy = torch.stack([index, index, width - index, height - index], dim=1)
            confidence = 0.9 - 0.1 * index.float()
            rows = torch.nonzero(confidence >= threshold).flatten()
            grid_hw = (height // GRID_STRIDE, width // GRID_STRIDE)
            all_scores = box_masks(xyxy // GRID_STRIDE, size_hw=grid_hw).float()
            detections = Detections(
                xyxy=xyxy[rows].float(),
                class_id=(index % len(CLASS_NAMES))[rows],
                confidence=confidence[rows],
                image_metadata=self._note(),
            )
            decoded.append((detections, all_scores, rows, _geometry(height, width)))

        return decoded


class GridInstanceSegmentationBlock(CompactMasks, InstanceSegmentationBlock):
    """The real block with a grid producer; records what it was asked.

    ``asked`` holds every ``wants`` answer of the last call, also copied to
    the model as ``demand`` for tests that cannot reach an engine-built block;
    ``produced`` counts the grid views the producer built.
    """

    type = "test/grid_instance_segmentation@v1"
    implementations = ()  # This fixture supplies its own producer.

    def __init__(self, *, models: Any):
        super().__init__(models=models)
        self.asked: Dict[str, bool] = {}
        self.produced = 0

    def wants(self, output: str) -> bool:
        answer = super().wants(output)
        self.asked[output] = answer

        return answer

    def _segmentations(
        self, network_input: ModelInput, raw_output: Any
    ) -> List[SegmentationView]:
        network_input.model.demand = dict(self.asked)
        decoded = network_input.model.post_process_grid(
            raw_output, network_input.metadata, **_grid_options(network_input.options)
        )
        views = [
            SegmentationView.from_selected_rows(
                all_scores,
                rows=rows,
                detections=detections,
                score_type="probabilities",
                threshold=GRID_THRESHOLD,
                geometry=geometry,
            )
            for detections, all_scores, rows, geometry in decoded
        ]
        self.produced += len(views)

        return views


def _grid_options(options: Dict[str, Any]) -> Dict[str, Any]:
    # This fixture uses its separate grid producer instead of post_process.
    grid_options = {
        key: value for key, value in options.items() if key != "mask_format"
    }

    return grid_options


def _geometry(height: int, width: int) -> MaskGridGeometry:
    geometry = MaskGridGeometry(
        grid_size_hw=(height // GRID_STRIDE, width // GRID_STRIDE),
        inference_size_hw=(height, width),
        padding_ltrb=(0, 0, 0, 0),
        scale_xy=(1.0, 1.0),
        crop_size_hw=(height, width),
        original_size_hw=(height, width),
        static_crop_xywh=(0, 0, width, height),
    )

    return geometry
