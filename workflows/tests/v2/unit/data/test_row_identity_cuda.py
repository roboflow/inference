"""Row identity on CUDA: no host syncs, ids stay on the rows' device.

Skipped without CUDA. Out-of-range CUDA indices trigger a device-side assert
that poisons the CUDA context, so that case runs in a child interpreter.
"""

import subprocess
import sys
import textwrap

import pytest
import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import (
    ROW_RECORD_KEY,
    row_record,
    with_row_identity,
)

from inference_models.models.base.object_detection import Detections

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


def _detections(rows=4):
    device = torch.device("cuda")
    return Detections(
        xyxy=torch.arange(rows * 4, dtype=torch.float32, device=device).reshape(
            rows, 4
        ),
        class_id=torch.arange(rows, device=device),
        confidence=torch.linspace(0.9, 0.1, rows, device=device),
        image_metadata={"parent_id": "frame"},
    )


def test_mint_validate_and_index_selection_never_sync():
    source = _detections()
    first_indices = torch.tensor([3, 1, 1], device="cuda")
    second_indices = torch.tensor([2, 0], device="cuda")
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        identified = with_row_identity(source)
        first = select_predictions(identified, indices=first_indices)
        second = select_predictions(first, indices=second_indices)
        OBJECT_DETECTION_PREDICTION_KIND.check(second)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    ids = second.image_metadata[ROW_RECORD_KEY]["ids"]
    assert ids.device.type == "cuda" and ids.dtype == torch.int64
    assert row_record(second).host_ids() == [1, 3]


def test_view_selection_on_cuda_keeps_ids_without_sync_or_materialization():
    view = SegmentationView(
        detections=with_row_identity(_detections()),
        scores=torch.randn(4, 4, 4, device="cuda"),
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.identity((4, 4)),
    )
    first_rows = torch.tensor([3, 1, 1], device="cuda")
    second_rows = torch.tensor([2, 0], device="cuda")
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("error")
    try:
        picked = view.select(first_rows).select(second_rows)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        dense = picked.full_res()
    side.synchronize()

    assert picked.materialization_count == 1
    assert row_record(picked.detections).host_ids() == [1, 3]
    assert row_record(dense).host_ids() == [1, 3]


def test_ids_follow_cuda_selection():
    identified = with_row_identity(_detections())

    picked = select_predictions(
        identified, mask=torch.tensor([True, False, True, True], device="cuda")
    )
    again = select_predictions(picked, indices=[2, 2, 0])

    assert row_record(again).host_ids() == [3, 3, 0]


def test_caller_cuda_indices_are_not_aliased():
    identified = with_row_identity(_detections())
    indices = torch.tensor([2, 0], device="cuda")

    picked = select_predictions(identified, indices=indices)
    indices.fill_(3)

    assert row_record(picked).host_ids() == [2, 0]


def test_ids_and_rectangles_cross_once(monkeypatch):
    picked = select_predictions(with_row_identity(_detections()), indices=[3, 0])
    record = row_record(picked)
    copies = []
    original_cpu = torch.Tensor.cpu

    def counting_cpu(self, *args, **kwargs):
        copies.append(self.device.type)
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counting_cpu)
    ids, rectangles = record.host_ids_with(picked.xyxy.round().to(torch.int32))

    assert copies == ["cuda"]
    assert ids == [3, 0]
    assert rectangles[0] == [12, 13, 14, 15]


def test_ids_on_another_device_than_the_rows_are_rejected():
    identified = with_row_identity(_detections())
    identified.image_metadata[ROW_RECORD_KEY]["ids"] = torch.arange(4)

    with pytest.raises(
        ContractError, match="is on cpu, but the prediction rows are on cuda"
    ):
        OBJECT_DETECTION_PREDICTION_KIND.check(identified)


def test_out_of_range_cuda_indices_fail_in_a_child_process():
    script = textwrap.dedent("""
        import torch
        from roboflow_workflows.v2.data.prediction_ops import select_predictions
        from roboflow_workflows.v2.data.row_identity import with_row_identity
        from inference_models.models.base.object_detection import Detections

        device = torch.device("cuda")
        detections = with_row_identity(Detections(
            xyxy=torch.zeros(2, 4, device=device),
            class_id=torch.zeros(2, dtype=torch.int64, device=device),
            confidence=torch.zeros(2, device=device),
        ))
        picked = select_predictions(
            detections, indices=torch.tensor([5], device=device)
        )
        torch.cuda.synchronize()
        """)

    child = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )

    assert child.returncode != 0
