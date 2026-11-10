"""CUDA parity of crop visualization for parent boxes crossing a crop (optional).

Skipped without CUDA. The CPU cases of ``test_crop_window`` on a CUDA parent,
child and predictions must keep the output on the device, leave the inputs
unchanged and equal the CPU output: exact at scale 1, within 1 level when
resampled (float32 weights may round differently on the device).
"""

from typing import List, Tuple

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.visualization.annotations import (
    CropVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData

from .support import detections, pixels_of, run_v2, scene

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CHILD_XYXY = (16, 12, 48, 40)
STYLE = {
    "color_palette": "CUSTOM",
    "custom_colors": ["#FF00FF"],
    "border_thickness": 1,
}
# Crossing left, top and both; then entirely left, above and both.
ROWS: List[Tuple[int, int, int, int]] = [
    (6, 18, 26, 30),
    (24, 2, 36, 20),
    (8, 4, 24, 20),
    (0, 16, 12, 36),
    (20, 0, 40, 10),
    (0, 0, 10, 8),
]


def child_and_rows(device: str):
    pixels = torch.from_numpy(scene()).permute(2, 0, 1).contiguous().to(device)
    parent = ImageData.from_tensor(pixels, image_id="parent")
    child = parent.crop(CHILD_XYXY, image_id="child")
    rows = detections(
        parent, [(*map(float, row), 0, 0.9) for row in ROWS], device=device
    )

    return child, rows


@pytest.mark.parametrize("position", ["BOTTOM_RIGHT", "CENTER"])
@pytest.mark.parametrize("scale_factor, levels", [(1.0, 0), (2.0, 1)])
def test_cuda_crossing_and_outside_rows_match_cpu(
    position: str, scale_factor: float, levels: int
) -> None:
    child, rows = child_and_rows("cuda")
    child_before = child.tensor_image.clone()
    boxes_before = rows.xyxy.clone()
    cpu_child, cpu_rows = child_and_rows("cpu")
    parameters = dict(position=position, scale_factor=scale_factor, **STYLE)

    output = run_v2(CropVisualizationBlock, image=child, predictions=rows, **parameters)

    expected = run_v2(
        CropVisualizationBlock, image=cpu_child, predictions=cpu_rows, **parameters
    )
    assert output.tensor_image.is_cuda
    difference = np.abs(
        pixels_of(output).astype(np.int16) - pixels_of(expected).astype(np.int16)
    )
    assert int(difference.max()) <= levels
    assert (pixels_of(expected) != pixels_of(cpu_child)).any()
    assert torch.equal(child.tensor_image, child_before)
    assert torch.equal(rows.xyxy, boxes_before)
