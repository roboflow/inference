"""``non_max_merge`` validates and decodes RLE masks once per call.

The grouping and the mask union share one host decode. Four rows, two
overlapping pairs of class 0, merge into two rows::

    rows 0, 1   overlapping squares near the top-left       -> group 1
    rows 2, 3   overlapping squares near the bottom-right   -> group 2
"""

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.v2.blocks.geometry.overlap import non_max_merge
from roboflow_workflows.v2.data import rle
from roboflow_workflows.v2.data.rle import decode_rle_masks

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks

SIZE_HW = (64, 64)
BOXES = [(0, 0, 30, 30), (2, 2, 32, 32), (34, 34, 60, 60), (35, 35, 61, 61)]


def squares() -> np.ndarray:
    dense = np.zeros((len(BOXES), *SIZE_HW), dtype=bool)
    for row, (x0, y0, x1, y1) in enumerate(BOXES):
        dense[row, y0:y1, x0:x1] = True

    return dense


def rle_predictions() -> InstanceDetections:
    counts = [
        mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"]
        for mask in squares()
    ]
    predictions = InstanceDetections(
        xyxy=torch.tensor(BOXES, dtype=torch.float32),
        class_id=torch.zeros(len(BOXES), dtype=torch.int64),
        confidence=torch.tensor([0.9, 0.8, 0.7, 0.6]),
        mask=InstancesRLEMasks(image_size=SIZE_HW, masks=counts),
    )

    return predictions


@pytest.fixture
def rle_work(monkeypatch: pytest.MonkeyPatch) -> dict:
    work = {"rows_walked": 0, "decodes": 0}
    walk = rle.check_rle_counts
    decode = rle.coco_rle_masks_to_numpy_mask

    def counting_walk(*args, **kwargs):
        work["rows_walked"] += 1
        return walk(*args, **kwargs)

    def counting_decode(*args, **kwargs):
        work["decodes"] += 1
        return decode(*args, **kwargs)

    monkeypatch.setattr(rle, "check_rle_counts", counting_walk)
    monkeypatch.setattr(rle, "coco_rle_masks_to_numpy_mask", counting_decode)

    return work


@pytest.mark.parametrize("use_masks", [True, False])
def test_rle_rows_are_walked_and_decoded_once(rle_work, use_masks):
    merged = non_max_merge(rle_predictions(), threshold=0.5, use_masks=use_masks)

    assert len(merged.xyxy) == 2
    assert rle_work == {"rows_walked": 4, "decodes": 1}


def test_shared_decode_keeps_the_group_unions():
    dense = squares()

    merged = non_max_merge(rle_predictions(), threshold=0.5, use_masks=True)

    assert isinstance(merged.mask, InstancesRLEMasks)
    assert merged.mask.image_size == SIZE_HW
    assert np.array_equal(
        decode_rle_masks(merged.mask),
        np.stack([dense[[0, 1]].any(0), dense[[2, 3]].any(0)]),
    )
