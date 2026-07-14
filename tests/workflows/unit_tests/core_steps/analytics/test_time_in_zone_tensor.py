import datetime

import numpy as np
import pytest
import torch

from inference.core.workflows.core_steps.analytics._zone_geometry import LeanPolygonZone
from inference.core.workflows.core_steps.analytics.time_in_zone.v1_tensor import (
    TimeInZoneBlockV1,
)
from inference.core.workflows.core_steps.analytics.time_in_zone.v2_tensor import (
    TimeInZoneBlockV2,
)
from inference.core.workflows.core_steps.analytics.time_in_zone.v3_tensor import (
    TimeInZoneBlockV3,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections


def _video_metadata() -> VideoMetadata:
    return VideoMetadata(
        video_identifier="tensor-zone-test",
        frame_number=1,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )


def _workflow_image(metadata: VideoMetadata) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=np.zeros((32, 32, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


def _detections() -> Detections:
    return Detections(
        xyxy=torch.tensor(
            [[1.0, 1.0, 5.0, 5.0], [20.0, 20.0, 25.0, 25.0]],
            dtype=torch.float32,
        ),
        class_id=torch.tensor([0, 0]),
        confidence=torch.tensor([0.9, 0.8]),
        image_metadata={"class_names": {0: "object"}},
        bboxes_metadata=[{"tracker_id": 11}, {"tracker_id": 22}],
    )


@pytest.mark.parametrize(
    "block_type",
    [TimeInZoneBlockV1, TimeInZoneBlockV2, TimeInZoneBlockV3],
)
def test_tensor_time_in_zone_uses_float32_host_geometry(
    monkeypatch, block_type
) -> None:
    detections = _detections()
    metadata = _video_metadata()
    calls = []

    def trigger(_zone, xyxy_host):
        assert isinstance(xyxy_host, np.ndarray)
        assert xyxy_host.dtype == np.float32
        np.testing.assert_array_equal(xyxy_host, detections.xyxy.numpy())
        calls.append(xyxy_host)
        return np.array([True, False], dtype=bool)

    monkeypatch.setattr(LeanPolygonZone, "trigger", trigger)
    kwargs = {
        "image": _workflow_image(metadata),
        "detections": detections,
        "zone": [[0, 0], [0, 10], [10, 10], [10, 0]],
        "triggering_anchor": "CENTER",
        "remove_out_of_zone_detections": True,
        "reset_out_of_zone_detections": True,
    }
    if block_type is TimeInZoneBlockV1:
        kwargs["metadata"] = metadata

    result = block_type().run(**kwargs)["timed_detections"]

    assert len(calls) == 1
    assert isinstance(result.xyxy, torch.Tensor)
    assert result.xyxy.device == detections.xyxy.device
    assert result.xyxy.tolist() == [[1.0, 1.0, 5.0, 5.0]]
    assert result.bboxes_metadata[0]["time_in_zone"] == 0.0


def test_tensor_time_in_zone_v3_reduces_multiple_zone_triggers(
    monkeypatch,
) -> None:
    detections = _detections()
    metadata = _video_metadata()
    trigger_results = iter(
        [
            np.array([True, False], dtype=bool),
            np.array([False, True], dtype=bool),
        ]
    )

    def trigger(_zone, xyxy_host):
        assert isinstance(xyxy_host, np.ndarray)
        return next(trigger_results)

    monkeypatch.setattr(LeanPolygonZone, "trigger", trigger)
    result = TimeInZoneBlockV3().run(
        image=_workflow_image(metadata),
        detections=detections,
        zone=[
            [[0, 0], [0, 10], [10, 10], [10, 0]],
            [[15, 15], [15, 30], [30, 30], [30, 15]],
        ],
        triggering_anchor="CENTER",
        remove_out_of_zone_detections=True,
        reset_out_of_zone_detections=True,
    )["timed_detections"]

    assert isinstance(result.xyxy, torch.Tensor)
    assert result.xyxy.tolist() == detections.xyxy.tolist()
