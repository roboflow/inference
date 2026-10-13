"""Tracker state and prediction recovery, independent of tracker heuristics."""

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
import supervision as sv
import torch
from pycocotools import mask as coco_mask
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.tracking.base import (
    ROW_INDEX_KEY,
    TRACKER_OUTPUTS,
    InstanceCache,
    SourceTrackers,
    TrackerCapacityError,
    host_boxes,
    kept_rows,
    tracking_result,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks


def _image(source="a", *, fps=30):
    metadata = VideoMetadata(
        video_identifier=source,
        fps=30,
        frame_number=0,
        frame_timestamp=datetime(2026, 10, 13),
    )
    metadata.fps = fps
    image = ImageData.from_tensor(
        torch.zeros((3, 40, 60), dtype=torch.uint8), video_metadata=metadata
    )

    return image


def _prediction(family="boxes"):
    boxes = Detections(
        xyxy=torch.tensor(
            [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], dtype=torch.float32
        ),
        confidence=torch.tensor([0.9, 0.8, 0.7]),
        class_id=torch.tensor([0, 1, 2]),
        image_metadata={"parent_id": "frame", "class_names": {0: "a", 1: "b", 2: "c"}},
        bboxes_metadata=[{"row": i, "custom": f"value-{i}"} for i in range(3)],
    )
    boxes = with_row_identity(boxes, namespace="test-rows")
    if family == "boxes":
        return boxes

    masks = torch.zeros((3, 40, 60), dtype=torch.bool)
    masks[0, :10, :10] = True
    masks[1, :10, 20:30] = True
    masks[2, :10, 40:50] = True
    if family == "keypoints":
        points = KeyPoints(
            xy=boxes.xyxy[:, :2].unsqueeze(1),
            confidence=boxes.confidence.unsqueeze(1),
            class_id=boxes.class_id,
            key_points_metadata=[{"point_row": i} for i in range(3)],
            covariance=torch.arange(12, dtype=torch.float32).reshape(3, 1, 2, 2),
            detection_confidence=boxes.confidence,
        )
        return points, boxes

    if family == "grid":
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
            detections=boxes,
            scores=masks[:, ::2, ::2].float(),
            score_type="logits",
            threshold=0.0,
            geometry=geometry,
        )
        return view

    mask = masks
    if family == "rle":
        mask = InstancesRLEMasks(
            image_size=(40, 60),
            masks=[
                coco_mask.encode(np.asfortranarray(m.numpy().astype(np.uint8)))[
                    "counts"
                ]
                for m in masks
            ],
        )
    predictions = InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        image_metadata=boxes.image_metadata,
        bboxes_metadata=boxes.bboxes_metadata,
        mask=mask,
    )
    if family == "dense_view":
        return SegmentationView.from_dense(predictions)

    return predictions


def _boxes(prediction):
    if isinstance(prediction, SegmentationView):
        return prediction.detections

    return prediction[1] if isinstance(prediction, tuple) else prediction


def test_fifo_cache_does_not_refresh_seen_ids_and_reports_positions():
    cache = InstanceCache(2)
    assert cache.split([1, 2]) == ([0, 1], [])
    assert cache.split([1, 3, 2, 1]) == ([1, 3], [0, 2])
    # Seeing 2 did not refresh it, so inserting 1 evicted 2.
    assert cache.split([2]) == ([0], [])
    assert len(cache) == 2
    with pytest.raises(ValueError, match="at least 1"):
        InstanceCache(0)


@pytest.mark.parametrize("fps", [None, 0, -1, float("nan"), float("inf"), True])
def test_invalid_source_rate_creates_no_state(fps):
    states = SourceTrackers(block_type="test")
    with pytest.raises(ValueError, match="fps"):
        states.state_for(
            _image(fps=fps),
            max_sources=2,
            frame_rate=None,
            instances_cache_size=2,
            create_tracker=lambda rate: object(),
        )
    assert states == {}


def test_source_capacity_mapping_first_frame_and_rate_override():
    states = SourceTrackers(block_type="test")
    rates = []

    def create(rate):
        rates.append(rate)
        return object()

    first = states.state_for(
        _image(fps=0),
        max_sources=1,
        frame_rate=15.0,
        instances_cache_size=2,
        create_tracker=create,
    )
    again = states.state_for(
        _image(),
        max_sources=1,
        frame_rate=60.0,
        instances_cache_size=10,
        create_tracker=create,
    )
    assert again is first and rates == [15.0]
    assert first.instances.size == 2 and list(states) == ["a"]
    with pytest.raises(TypeError):
        states["b"] = first
    with pytest.raises(TrackerCapacityError, match="max_sources=1"):
        states.state_for(
            _image("b"),
            max_sources=1,
            frame_rate=None,
            instances_cache_size=2,
            create_tracker=create,
        )
    assert dict(states) == {"a": first} and rates == [15.0]


def test_missing_identifier_creates_no_state():
    states = SourceTrackers(block_type="test")
    with pytest.raises(ValueError, match="video_identifier"):
        states.state_for(
            _image(None),
            max_sources=1,
            frame_rate=30.0,
            instances_cache_size=2,
            create_tracker=lambda rate: object(),
        )
    assert states == {}


def test_host_boxes_handles_bfloat16_and_kept_positions():
    prediction = replace(_prediction(), xyxy=_prediction().xyxy.to(torch.bfloat16))
    host = host_boxes(prediction)
    assert host.xyxy.dtype == np.float32
    assert host.data[ROW_INDEX_KEY].tolist() == [0, 1, 2]
    host.tracker_id = np.array([7, -1, 9])
    rows, ids = kept_rows(host)
    assert rows.tolist() == [0, 2] and ids.tolist() == [7, 9]
    host.tracker_id = None
    assert not len(kept_rows(host)[0])
    assert not len(kept_rows(sv.Detections.empty())[0])
    with pytest.raises(ValueError, match="without Detections"):
        host_boxes((_prediction("keypoints")[0], None))


@pytest.mark.parametrize(
    "family", ["boxes", "dense", "rle", "grid", "dense_view", "keypoints"]
)
def test_recovery_preserves_fields_lineage_and_partition_without_materialization(
    family, monkeypatch
):
    prediction = _prediction(family)
    original = _boxes(prediction)
    before_metadata = [dict(row) for row in original.bboxes_metadata]
    if isinstance(prediction, SegmentationView):

        def forbidden(*args, **kwargs):
            raise AssertionError("tracking must not call full_res")

        monkeypatch.setattr(SegmentationView, "full_res", forbidden)

    cache = InstanceCache(10)
    cache.split([12])
    new_positions, seen_positions = cache.split([12, 10])
    outputs = tracking_result(
        prediction,
        rows=np.array([2, 0]),
        tracker_ids=np.array([12, 10]),
        new_positions=new_positions,
        seen_positions=seen_positions,
        wanted_outputs=set(TRACKER_OUTPUTS),
    )
    for name, expected_rows, expected_ids in [
        ("tracked_detections", [2, 0], [12, 10]),
        ("new_instances", [0], [10]),
        ("already_seen_instances", [2], [12]),
    ]:
        recovered = outputs[name]
        boxes = _boxes(recovered)
        assert torch.equal(boxes.xyxy, original.xyxy[expected_rows])
        assert torch.equal(boxes.class_id, original.class_id[expected_rows])
        assert torch.equal(boxes.confidence, original.confidence[expected_rows])
        assert [r["custom"] for r in boxes.bboxes_metadata] == [
            f"value-{i}" for i in expected_rows
        ]
        assert all(
            r is not original.bboxes_metadata[i]
            for r, i in zip(boxes.bboxes_metadata, expected_rows)
        )
        assert (
            boxes.xyxy.dtype == original.xyxy.dtype
            and boxes.xyxy.device == original.xyxy.device
        )
        assert [r["row"] for r in boxes.bboxes_metadata] == expected_rows
        assert [r["tracker_id"] for r in boxes.bboxes_metadata] == expected_ids
        assert boxes.image_metadata["parent_id"] == "frame"
        assert row_record(boxes).namespace == "test-rows"
        assert row_record(boxes).host_ids() == expected_rows
        if isinstance(recovered, SegmentationView):
            assert (
                recovered.materialization_count == prediction.materialization_count == 0
            )
            assert recovered.is_dense_fallback == prediction.is_dense_fallback
            assert torch.equal(recovered.scores, prediction.scores[expected_rows])
            assert recovered.geometry == prediction.geometry
        elif isinstance(recovered, InstanceDetections):
            if family == "rle":
                assert recovered.mask.masks == [
                    prediction.mask.masks[i] for i in expected_rows
                ]
            else:
                assert torch.equal(recovered.mask, prediction.mask[expected_rows])
        elif isinstance(recovered, tuple):
            assert torch.equal(recovered[0].xy, prediction[0].xy[expected_rows])
            assert torch.equal(
                recovered[0].covariance, prediction[0].covariance[expected_rows]
            )
            assert recovered[0].key_points_metadata == [
                prediction[0].key_points_metadata[i] for i in expected_rows
            ]
    assert original.bboxes_metadata == before_metadata
    assert row_record(original).host_ids() == [0, 1, 2]


@pytest.mark.parametrize(
    "family", ["boxes", "dense", "rle", "grid", "dense_view", "keypoints"]
)
def test_empty_results_keep_family_and_metadata(family):
    prediction = _prediction(family)
    outputs = tracking_result(
        prediction,
        rows=np.array([], dtype=np.int64),
        tracker_ids=np.array([], dtype=np.int64),
        new_positions=[],
        seen_positions=[],
        wanted_outputs=set(TRACKER_OUTPUTS),
    )
    for result in outputs.values():
        assert type(result) is type(prediction)
        assert len(_boxes(result)) == 0 and _boxes(result).bboxes_metadata is None
        assert row_record(_boxes(result)).host_ids() == []


def test_pure_output_builder_does_not_select_undemanded_payloads(monkeypatch):
    from roboflow_workflows.v2.blocks.tracking import base

    def forbidden(*args, **kwargs):
        raise AssertionError("undemanded payload selected")

    monkeypatch.setattr(base, "_tracked_rows", forbidden)
    assert (
        tracking_result(
            _prediction("dense"),
            rows=np.array([1]),
            tracker_ids=np.array([7]),
            new_positions=[0],
            seen_positions=[],
            wanted_outputs=set(),
        )
        == {}
    )


def test_output_construction_does_not_change_partition_inputs():
    new_positions, seen_positions = [1], [0]
    rows, ids = np.array([2, 0]), np.array([12, 10])
    arguments = dict(
        rows=rows,
        tracker_ids=ids,
        new_positions=new_positions,
        seen_positions=seen_positions,
        wanted_outputs=set(TRACKER_OUTPUTS),
    )
    first = tracking_result(_prediction(), **arguments)
    second = tracking_result(_prediction(), **arguments)
    assert new_positions == [1] and seen_positions == [0]
    assert rows.tolist() == [2, 0] and ids.tolist() == [12, 10]
    for name in TRACKER_OUTPUTS:
        assert first[name].bboxes_metadata == second[name].bboxes_metadata
