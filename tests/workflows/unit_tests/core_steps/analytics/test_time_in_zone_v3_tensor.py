import datetime

import numpy as np
import pytest
import supervision as sv

torch = pytest.importorskip("torch")

from inference.core.workflows.core_steps.analytics.time_in_zone import (  # noqa: E402
    v3_tensor,
)
from inference.core.workflows.execution_engine.constants import (  # noqa: E402
    TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (  # noqa: E402
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections  # noqa: E402

TimeInZoneBlockV3 = v3_tensor.TimeInZoneBlockV3


def _native_detections(xyxy: np.ndarray, tracker_ids: np.ndarray) -> Detections:
    n = len(xyxy)
    return Detections(
        xyxy=torch.as_tensor(xyxy, dtype=torch.float32),
        class_id=torch.zeros(n, dtype=torch.int64),
        confidence=torch.ones(n, dtype=torch.float32),
        bboxes_metadata=[{"tracker_id": int(value)} for value in tracker_ids],
    )


def test_time_in_zone_v3_matches_supervision_and_old_bookkeeping() -> None:
    zones = [
        [[10, 10], [10, 30], [30, 30], [30, 10]],
        [[50, 50], [50, 70], [70, 70], [70, 50]],
    ]
    frames = [
        (
            np.array([[15, 15, 17, 17], [35, 35, 37, 37], [55, 55, 57, 57]]),
            np.array([1, 2, 3]),
        ),
        (
            np.array([[35, 35, 37, 37], [15, 15, 17, 17], [55, 55, 57, 57]]),
            np.array([1, 2, 3]),
        ),
        (
            np.array([[15, 15, 17, 17], [16, 16, 18, 18], [80, 80, 82, 82]]),
            np.array([1, 2, 3]),
        ),
    ]
    oracle_zones = [
        sv.PolygonZone(np.asarray(polygon), triggering_anchors=(sv.Position.CENTER,))
        for polygon in zones
    ]
    tracked_ids_in_zone = {}
    block = TimeInZoneBlockV3()

    for frame_number, (xyxy, tracker_ids) in enumerate(frames, start=10):
        metadata = VideoMetadata(
            video_identifier="tensor-zone",
            frame_number=frame_number,
            fps=1,
            frame_timestamp=datetime.datetime.now(datetime.timezone.utc),
            comes_from_video_file=True,
        )
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="root"),
            numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
            video_metadata=metadata,
        )
        sv_input = sv.Detections(xyxy=xyxy.astype(float), tracker_id=tracker_ids)
        is_in_any_zone = np.any(
            [polygon_zone.trigger(sv_input) for polygon_zone in oracle_zones], axis=0
        )
        surviving_mask = np.zeros(len(xyxy), dtype=bool)
        surviving_times = {}
        ts_end = frame_number / metadata.fps
        for index, is_in_zone, tracker_id in zip(
            range(len(xyxy)), is_in_any_zone, tracker_ids
        ):
            time_in_zone = 0.0
            if is_in_zone:
                ts_start = tracked_ids_in_zone.setdefault(int(tracker_id), ts_end)
                time_in_zone = ts_end - ts_start
            elif int(tracker_id) in tracked_ids_in_zone:
                # Keep the old reset=False deletion quirk as the oracle.
                del tracked_ids_in_zone[int(tracker_id)]
            surviving_mask[index] = True
            surviving_times[index] = time_in_zone

        result = block.run(
            image=image,
            detections=_native_detections(xyxy, tracker_ids),
            zone=zones,
            triggering_anchor="CENTER",
            remove_out_of_zone_detections=False,
            reset_out_of_zone_detections=False,
        )["timed_detections"]

        np.testing.assert_array_equal(result.xyxy.numpy(), xyxy[surviving_mask])
        assert [item["tracker_id"] for item in result.bboxes_metadata] == (
            tracker_ids[surviving_mask].tolist()
        )
        assert [
            item[TIME_IN_ZONE_KEY_IN_SV_DETECTIONS] for item in result.bboxes_metadata
        ] == [surviving_times[index] for index in np.flatnonzero(surviving_mask)]
