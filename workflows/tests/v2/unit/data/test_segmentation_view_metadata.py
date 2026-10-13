"""Row metadata replacements preserve borrowed masks and independent caches."""

import pytest
import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.prediction_ops import RowSelection
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections


def _view(*, dense=False, dtype=torch.float32):
    detections = with_row_identity(
        Detections(
            xyxy=torch.tensor([[2, 2, 12, 12], [20, 2, 30, 12]], dtype=torch.float64),
            class_id=torch.tensor([0, 1]),
            confidence=torch.tensor([0.9, 0.8]),
            image_metadata={"parent_id": "test-frame"},
            bboxes_metadata=[{"original": 0}, {"original": 1}],
        ),
        namespace="mask-rows",
    )
    if dense:
        predictions = InstanceDetections(
            xyxy=detections.xyxy,
            class_id=detections.class_id,
            confidence=detections.confidence,
            mask=torch.ones((2, 40, 60), dtype=dtype),
            image_metadata=detections.image_metadata,
            bboxes_metadata=detections.bboxes_metadata,
        )
        return SegmentationView.from_dense(predictions)

    geometry = MaskGridGeometry(
        grid_size_hw=(20, 30),
        inference_size_hw=(40, 60),
        padding_ltrb=(0, 0, 0, 0),
        scale_xy=(1.0, 1.0),
        crop_size_hw=(40, 60),
        original_size_hw=(40, 60),
        static_crop_xywh=(0, 0, 60, 40),
    )
    view = SegmentationView(
        detections=detections,
        scores=torch.ones((2, 20, 30), dtype=dtype),
        score_type="probabilities",
        threshold=0.4,
        geometry=geometry,
    )

    return view


@pytest.mark.parametrize(
    "dense,dtype",
    [
        (False, torch.float16),
        (False, torch.float64),
        (True, torch.bool),
        (True, torch.uint8),
    ],
)
@pytest.mark.parametrize("indices", [[0, 1], [1, 0, 1], []])
def test_metadata_replacement_preserves_storage_rows_and_view_semantics(
    dense, dtype, indices, monkeypatch
):
    view = _view(dense=dense, dtype=dtype).select(RowSelection.of(2, indices=indices))
    supplied = [{"tracker_id": 100 + i} for i in range(len(view))]
    original_rows = [dict(row) for row in view.detections.bboxes_metadata or []]

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "metadata replacement must not materialize, copy or transfer tensors"
        )

    with monkeypatch.context() as guard:
        guard.setattr(SegmentationView, "full_res", forbidden)
        for operation in ("clone", "cpu", "to"):
            guard.setattr(torch.Tensor, operation, forbidden)

        changed = view.with_row_metadata(supplied)
    assert changed is not view
    assert changed.scores is view.scores
    assert changed.scores.dtype == dtype and changed.scores.device == view.scores.device
    assert changed.geometry == view.geometry
    assert changed.threshold == view.threshold and changed.score_type == view.score_type
    assert changed.is_dense_fallback == dense
    assert changed.materialization_count == view.materialization_count == 0
    for field in ("xyxy", "class_id", "confidence", "image_metadata"):
        assert getattr(changed.detections, field) is getattr(view.detections, field)
    assert row_record(changed.detections).host_ids() == indices
    assert changed.detections.bboxes_metadata == supplied
    assert changed.detections.bboxes_metadata is not supplied
    assert view.detections.bboxes_metadata == original_rows
    if supplied:
        assert changed.detections.bboxes_metadata[0] is not supplied[0]
        changed.detections.bboxes_metadata[0]["tracker_id"] = -1
        assert supplied[0]["tracker_id"] == 100
        supplied[0]["caller_change"] = True
        assert "caller_change" not in changed.detections.bboxes_metadata[0]
        assert view.detections.bboxes_metadata == original_rows


@pytest.mark.parametrize("dense", [False, True])
def test_metadata_can_be_removed_without_changing_source_or_dense_boundary(dense):
    view = _view(dense=dense, dtype=torch.bool if dense else torch.float32)
    changed = view.with_row_metadata(None)
    assert changed.detections.bboxes_metadata is None
    assert view.detections.bboxes_metadata == [{"original": 0}, {"original": 1}]
    dense_predictions = changed.full_res()
    assert dense_predictions.bboxes_metadata is None
    assert dense_predictions.mask.shape == (2, 40, 60)
    if dense:
        assert dense_predictions.mask is view.scores
        assert changed.materialization_count == 0


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize(
    "metadata,message",
    [
        ([], "0 rows for 2 instances"),
        ([{}, {}, {}], "3 rows for 2 instances"),
        ([{}, None], "only dictionaries"),
        (({}, {}), "list of dictionaries or None"),
    ],
)
def test_invalid_row_metadata_is_rejected_without_changing_view(
    dense, metadata, message
):
    view = _view(dense=dense, dtype=torch.bool if dense else torch.float32)
    with pytest.raises(ContractError, match=message):
        view.with_row_metadata(metadata)
    assert view.detections.bboxes_metadata == [{"original": 0}, {"original": 1}]
    assert view.materialization_count == 0


def test_new_metadata_has_fresh_materialization_cache_and_dense_metadata():
    view = _view()
    first = view.full_res()
    assert view.materialization_count == 1 and view.is_full_res_materialized()
    changed = view.with_row_metadata([{"tracker_id": 7}, {"tracker_id": 8}])
    assert not changed.is_full_res_materialized()
    assert changed.materialization_count == 0
    assert changed.scores is view.scores
    second = changed.full_res()
    assert changed.materialization_count == 1 and view.materialization_count == 1
    assert second is changed.full_res() and first is view.full_res()
    assert second is not first
    assert torch.equal(first.mask, second.mask)
    assert second.bboxes_metadata == [{"tracker_id": 7}, {"tracker_id": 8}]
    assert first.bboxes_metadata == [{"original": 0}, {"original": 1}]
