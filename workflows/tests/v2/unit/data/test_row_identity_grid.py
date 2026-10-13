"""Row identity on compact SegmentationViews.

A view keeps the native detections' row record. Selection follows the shared
``take_rows`` path, so ids move with the rows without an upscale. Tensor ids
are view inputs: they join stream readiness and ``record_stream`` lifetime.
"""

import pytest
import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.masks import grid
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.row_identity import (
    ROW_RECORD_KEY,
    require_row_record,
    row_record,
    with_row_identity,
)

from inference_models.models.base.object_detection import Detections


def _detections(rows=3, *, device="cpu"):
    return Detections(
        xyxy=torch.tensor([[0.0, 0.0, 2.0, 2.0]], device=device).repeat(rows, 1),
        class_id=torch.arange(rows, device=device),
        confidence=torch.full((rows,), 0.5, device=device),
        image_metadata={"parent_id": "frame"},
    )


def _view(detections, *, device="cpu"):
    rows = int(detections.xyxy.shape[0])
    return SegmentationView(
        detections=detections,
        scores=torch.randn(rows, 4, 4, device=device),
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.identity((4, 4)),
    )


def _ids(prediction):
    return row_record(prediction).host_ids()


def test_select_chain_keeps_rows_without_materializing():
    view = _view(with_row_identity(_detections(), namespace="r1"))

    sorted_rows = view.select(torch.tensor([2, 0, 1]))
    repeated = sorted_rows.select(torch.tensor([1, 1]))
    empty = repeated.select(torch.tensor([], dtype=torch.int64))

    for picked in (sorted_rows, repeated, empty):
        assert picked.materialization_count == 0
        assert row_record(picked.detections).namespace == "r1"
    assert _ids(sorted_rows.detections) == [2, 0, 1]
    assert _ids(repeated.detections) == [0, 0]
    assert empty.detections.image_metadata[ROW_RECORD_KEY]["ids"] is None
    assert _ids(empty.detections) == []


def test_materialized_rows_keep_their_ids_and_masks():
    view = _view(with_row_identity(_detections()))
    dense = view.full_res()

    picked = view.select(torch.tensor([2, 2, 0]))
    picked_dense = picked.full_res()

    assert _ids(picked_dense) == [2, 2, 0]
    assert torch.equal(picked_dense.mask, dense.mask[[2, 2, 0]])


def test_caller_indices_mutated_after_select_do_not_move_ids():
    view = _view(with_row_identity(_detections()))
    indices = torch.tensor([2, 0])

    picked = view.select(indices)
    indices[0] = 1

    assert _ids(picked.detections) == [2, 0]


def test_view_without_record_stays_unknown():
    view = _view(_detections())

    picked = view.select(torch.tensor([1]))

    assert row_record(picked.detections) is None
    assert row_record(picked.full_res()) is None
    with pytest.raises(ContractError, match="needs predictions with a row identity"):
        require_row_record(picked.detections, consumer="v2/test")


def test_tensor_ids_are_ordered_with_the_view_inputs(monkeypatch):
    ordered = []
    monkeypatch.setattr(
        grid,
        "order_current_stream",
        lambda readiness, tensors: ordered.append(list(tensors)),
    )
    implicit = _view(with_row_identity(_detections()))
    picked = implicit.select(torch.tensor([1, 0]))
    ids = picked.detections.image_metadata[ROW_RECORD_KEY]["ids"]
    ordered.clear()

    implicit.detections
    picked.full_res()

    implicit_inputs, *picked_inputs = ordered
    assert len(implicit_inputs) == 4
    assert picked_inputs
    for tensors in picked_inputs:
        assert len(tensors) == 5 and any(tensor is ids for tensor in tensors)


def test_stale_record_is_rejected_at_view_construction():
    detections = with_row_identity(_detections())
    detections.image_metadata[ROW_RECORD_KEY]["rows"] = 5

    with pytest.raises(ContractError, match="describes 5 rows"):
        _view(detections)


def test_ids_on_another_device_than_the_view_are_rejected():
    # Meta tensors have no data, so construction reads shapes and devices only.
    meta = torch.device("meta")
    detections = with_row_identity(_detections(device=meta))
    detections.image_metadata[ROW_RECORD_KEY]["ids"] = torch.arange(3)

    with pytest.raises(ContractError, match="ids"):
        _view(detections, device=meta)
