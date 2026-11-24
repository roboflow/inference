"""Merge and combine keep rows on an accelerator (CUDA or MPS, when present)."""

import dataclasses

import pytest
import torch
from roboflow_workflows.v2.blocks.combine import combine_predictions, merge_rows
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from .scene import PEOPLE, VEHICLES, boxes, dense

ACCELERATORS = [
    name
    for name, available in (
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    )
    if available
]


def moved(prediction, device, *, namespace):
    fields = {
        "xyxy": prediction.xyxy.to(device),
        "class_id": prediction.class_id.to(device),
        "confidence": prediction.confidence.to(device),
    }
    if isinstance(getattr(prediction, "mask", None), torch.Tensor):
        fields["mask"] = prediction.mask.to(device)
    return with_row_identity(
        dataclasses.replace(prediction, **fields), namespace=namespace
    )


@pytest.mark.skipif(not ACCELERATORS, reason="no CUDA or MPS device")
@pytest.mark.parametrize("device", ACCELERATORS)
def test_outputs_and_row_ids_stay_on_the_device(device: str) -> None:
    # given
    people = moved(boxes(PEOPLE, namespace=None), device, namespace="people")
    vehicles = moved(boxes(VEHICLES, namespace=None), device, namespace="vehicles")
    picked = select_predictions(vehicles, indices=torch.tensor([2, 0], device=device))

    # when
    merged = merge_rows(vehicles, class_name="traffic")
    combined = combine_predictions(people, vehicles)
    same_result = combine_predictions(picked, vehicles)
    masks = combine_predictions(
        moved(dense(PEOPLE, namespace=None), device, namespace="a"),
        moved(dense(VEHICLES, namespace=None), device, namespace="b"),
    )

    # then
    assert merged.xyxy.device.type == merged.class_id.device.type == device
    assert merged.xyxy.cpu().tolist() == [[12.0, 2.0, 58.0, 38.0]]
    assert combined.xyxy.device.type == device
    assert row_record(same_result).ids.device.type == device
    assert row_record(same_result).host_ids() == [2, 0, 0, 1, 2]
    assert masks.mask.device.type == device
