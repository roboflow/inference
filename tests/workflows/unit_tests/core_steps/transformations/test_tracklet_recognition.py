import datetime

import numpy as np
import pytest
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.core_steps.transformations.tracklet_recognition import (
    v1 as tracklet_recognition,
)
from inference.core.workflows.core_steps.transformations.tracklet_recognition.v1 import (
    _STORES,
    DETECTION_ID_KEY,
    RECOGNITION_AGE_SECONDS_KEY,
    RECOGNITION_CLASS_NAME_KEY,
    RECOGNITION_LOCKED_KEY,
    RECOGNITION_UPDATED_KEY,
    TrackletRecognitionCacheBlockV1,
    TrackletRecognitionCacheManifest,
    TrackletRecognitionGateBlockV1,
    TrackletRecognitionStore,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def setup_function() -> None:
    _STORES.clear()


def test_tracklet_recognition_gate_throttles_and_skips_locked_tracklets() -> None:
    # given
    gate = TrackletRecognitionGateBlockV1()
    cache = TrackletRecognitionCacheBlockV1()
    frame0 = _image(frame_number=0, fps=10)
    detections = _detections(
        detection_ids=["det-4", "det-5"],
        tracker_ids=[4, 5],
    )

    # when
    first_gate_result = gate.run(
        image=frame0,
        detections=detections,
        recognition_interval_seconds=0.2,
        state_id="plates",
    )

    # then
    assert first_gate_result["detections_to_recognize"].tracker_id.tolist() == [4, 5]

    # when - update tracklet 4 once at t=0.0s
    cache.run(
        image=frame0,
        detections=first_gate_result["all_detections"],
        recognized_detections=first_gate_result["detections_to_recognize"][:1],
        recognition_predictions=[
            _classification_prediction(parent_id="det-4", class_name="ABC123")
        ],
        lock_after_consistent_results=2,
        state_id="plates",
        update_detection_class=True,
    )
    throttled_result = gate.run(
        image=_image(frame_number=1, fps=10),
        detections=detections,
        recognition_interval_seconds=0.2,
        state_id="plates",
    )

    # then - tracklet 4 is throttled, tracklet 5 has never been recognized
    assert throttled_result["detections_to_recognize"].tracker_id.tolist() == [5]

    # when - same result for tracklet 4 arrives after the interval and locks it
    due_result = gate.run(
        image=_image(frame_number=2, fps=10),
        detections=detections,
        recognition_interval_seconds=0.2,
        state_id="plates",
    )
    assert due_result["detections_to_recognize"].tracker_id.tolist() == [4, 5]
    cache.run(
        image=_image(frame_number=2, fps=10),
        detections=due_result["all_detections"],
        recognized_detections=due_result["detections_to_recognize"][:1],
        recognition_predictions=[
            _classification_prediction(parent_id="det-4", class_name="ABC123")
        ],
        lock_after_consistent_results=2,
        state_id="plates",
        update_detection_class=True,
    )
    after_lock_result = gate.run(
        image=_image(frame_number=10, fps=10),
        detections=detections,
        recognition_interval_seconds=0.2,
        state_id="plates",
    )

    # then
    assert after_lock_result["detections_to_recognize"].tracker_id.tolist() == [5]


def test_tracklet_recognition_cache_emits_cached_result_without_new_recognition() -> (
    None
):
    # given
    gate = TrackletRecognitionGateBlockV1()
    cache = TrackletRecognitionCacheBlockV1()
    detections = _detections(
        detection_ids=["det-4"],
        tracker_ids=[4],
        class_names=["vehicle"],
    )
    gate_result = gate.run(
        image=_image(frame_number=0, fps=10),
        detections=detections,
        recognition_interval_seconds=0.2,
        state_id="plates",
    )

    # when
    first_cache_result = cache.run(
        image=_image(frame_number=0, fps=10),
        detections=gate_result["all_detections"],
        recognized_detections=gate_result["detections_to_recognize"],
        recognition_predictions=[
            _classification_prediction(parent_id="det-4", class_name="ABC123")
        ],
        lock_after_consistent_results=5,
        state_id="plates",
        update_detection_class=True,
    )
    second_cache_result = cache.run(
        image=_image(frame_number=1, fps=10),
        detections=gate_result["all_detections"],
        recognized_detections=sv.Detections.empty(),
        recognition_predictions=[],
        lock_after_consistent_results=5,
        state_id="plates",
        update_detection_class=True,
    )

    # then
    first_detections = first_cache_result["recognized_detections"]
    assert first_detections[RECOGNITION_CLASS_NAME_KEY].tolist() == ["ABC123"]
    assert first_detections[RECOGNITION_UPDATED_KEY].tolist() == [True]
    assert first_detections.data[CLASS_NAME_DATA_FIELD].tolist() == ["ABC123"]
    assert first_cache_result["recognition_results"] == [
        {
            "tracker_id": 4,
            "text": "ABC123",
            "confidence": 0.95,
            "class_id": 7,
            "locked": False,
            "updated_this_frame": True,
            "age_seconds": 0.0,
        }
    ]

    second_detections = second_cache_result["recognized_detections"]
    assert second_detections[RECOGNITION_CLASS_NAME_KEY].tolist() == ["ABC123"]
    assert second_detections[RECOGNITION_UPDATED_KEY].tolist() == [False]
    assert second_detections[RECOGNITION_LOCKED_KEY].tolist() == [False]
    assert second_detections[RECOGNITION_AGE_SECONDS_KEY].tolist() == [0.1]
    assert second_detections.data[CLASS_NAME_DATA_FIELD].tolist() == ["ABC123"]


def test_tracklet_recognition_cache_accepts_crop_level_predictions() -> None:
    assert TrackletRecognitionCacheManifest.get_input_dimensionality_offsets() == {
        "detections": 0,
        "recognized_detections": 0,
        "recognition_predictions": 1,
    }


def test_tracklet_recognition_cache_treats_scalar_ocr_string_as_single_prediction() -> (
    None
):
    # given
    gate = TrackletRecognitionGateBlockV1()
    cache = TrackletRecognitionCacheBlockV1()
    detections = _detections(
        detection_ids=["det-4"],
        tracker_ids=[4],
        class_names=["stall"],
    )
    gate_result = gate.run(
        image=_image(frame_number=0, fps=10),
        detections=detections,
        recognition_interval_seconds=0.2,
        state_id="plates",
    )

    # when
    cache_result = cache.run(
        image=_image(frame_number=0, fps=10),
        detections=gate_result["all_detections"],
        recognized_detections=gate_result["detections_to_recognize"],
        recognition_predictions="647",
        lock_after_consistent_results=5,
        state_id="plates",
        update_detection_class=True,
    )

    # then
    result_detections = cache_result["recognized_detections"]
    assert result_detections[RECOGNITION_CLASS_NAME_KEY].tolist() == ["647"]
    assert result_detections.data[CLASS_NAME_DATA_FIELD].tolist() == ["647"]
    assert cache_result["recognition_results"][0]["text"] == "647"


def test_tracklet_recognition_cache_can_use_status_as_class_id() -> None:
    # given
    gate = TrackletRecognitionGateBlockV1()
    cache = TrackletRecognitionCacheBlockV1()
    detections = _detections(
        detection_ids=["det-4"],
        tracker_ids=[4],
        class_names=["stall"],
    )
    gate_result = gate.run(
        image=_image(frame_number=0, fps=10),
        detections=detections,
        recognition_interval_seconds=0.2,
        state_id="plates",
    )

    # when - a recognition result updates this tracklet
    updated_result = cache.run(
        image=_image(frame_number=0, fps=10),
        detections=gate_result["all_detections"],
        recognized_detections=gate_result["detections_to_recognize"],
        recognition_predictions=[
            _classification_prediction(parent_id="det-4", class_name="ABC123")
        ],
        lock_after_consistent_results=2,
        state_id="plates",
        update_detection_class=True,
        update_class_id_from_cache_status=True,
    )

    # then - class text remains the OCR result, while class_id is the update status
    updated_detections = updated_result["recognized_detections"]
    assert updated_detections.data[CLASS_NAME_DATA_FIELD].tolist() == ["ABC123"]
    assert updated_detections[RECOGNITION_CLASS_NAME_KEY].tolist() == ["ABC123"]
    assert updated_detections.class_id.tolist() == [1]

    # when - no new recognition result is produced and the tracklet is still unlocked
    cached_result = cache.run(
        image=_image(frame_number=1, fps=10),
        detections=gate_result["all_detections"],
        recognized_detections=sv.Detections.empty(),
        recognition_predictions=[],
        lock_after_consistent_results=2,
        state_id="plates",
        update_detection_class=True,
        update_class_id_from_cache_status=True,
    )

    # then
    cached_detections = cached_result["recognized_detections"]
    assert cached_detections.data[CLASS_NAME_DATA_FIELD].tolist() == ["ABC123"]
    assert cached_detections.class_id.tolist() == [0]

    # when - the same recognition result arrives again and locks the tracklet
    locked_result = cache.run(
        image=_image(frame_number=2, fps=10),
        detections=gate_result["all_detections"],
        recognized_detections=gate_result["detections_to_recognize"],
        recognition_predictions=[
            _classification_prediction(parent_id="det-4", class_name="ABC123")
        ],
        lock_after_consistent_results=2,
        state_id="plates",
        update_detection_class=True,
        update_class_id_from_cache_status=True,
    )

    # then - locked status takes precedence over updated-this-frame
    locked_detections = locked_result["recognized_detections"]
    assert locked_detections.data[CLASS_NAME_DATA_FIELD].tolist() == ["ABC123"]
    assert locked_detections[RECOGNITION_UPDATED_KEY].tolist() == [True]
    assert locked_detections[RECOGNITION_LOCKED_KEY].tolist() == [True]
    assert locked_detections.class_id.tolist() == [2]


def test_tracklet_recognition_store_is_bounded() -> None:
    # given
    store = TrackletRecognitionStore(
        max_videos=1,
        max_tracklets_per_video=2,
        tracklet_state_ttl_seconds=1.0,
    )

    # when - a third tracklet is inserted after tracklet 1 was touched
    store.set(
        video_id="video-a",
        tracker_id=1,
        class_name="A",
        class_id=0,
        confidence=1.0,
        timestamp_seconds=0.0,
        consistency_window=1,
    )
    store.set(
        video_id="video-a",
        tracker_id=2,
        class_name="B",
        class_id=1,
        confidence=1.0,
        timestamp_seconds=0.0,
        consistency_window=1,
    )
    assert (
        store.get(video_id="video-a", tracker_id=1, timestamp_seconds=0.5) is not None
    )
    store.set(
        video_id="video-a",
        tracker_id=3,
        class_name="C",
        class_id=2,
        confidence=1.0,
        timestamp_seconds=0.5,
        consistency_window=1,
    )

    # then - the least recently used tracklet is evicted
    assert (
        store.get(video_id="video-a", tracker_id=1, timestamp_seconds=0.5) is not None
    )
    assert store.get(video_id="video-a", tracker_id=2, timestamp_seconds=0.5) is None
    assert (
        store.get(video_id="video-a", tracker_id=3, timestamp_seconds=0.5) is not None
    )

    # when - a second video is inserted
    store.set(
        video_id="video-b",
        tracker_id=4,
        class_name="D",
        class_id=3,
        confidence=1.0,
        timestamp_seconds=0.5,
        consistency_window=1,
    )

    # then - the least recently used video is evicted
    assert store.get(video_id="video-a", tracker_id=1, timestamp_seconds=0.5) is None
    assert (
        store.get(video_id="video-b", tracker_id=4, timestamp_seconds=0.5) is not None
    )


def test_tracklet_recognition_store_prunes_stale_tracklets() -> None:
    # given
    store = TrackletRecognitionStore(tracklet_state_ttl_seconds=1.0)
    store.set(
        video_id="video",
        tracker_id=1,
        class_name="A",
        class_id=0,
        confidence=1.0,
        timestamp_seconds=0.0,
        consistency_window=1,
    )

    # then
    assert store.get(video_id="video", tracker_id=1, timestamp_seconds=1.0) is not None
    assert store.get(video_id="video", tracker_id=1, timestamp_seconds=2.1) is None


def test_tracklet_recognition_store_registry_is_bounded(monkeypatch) -> None:
    # given
    monkeypatch.setattr(tracklet_recognition, "MAX_STATE_IDS", 2)
    _STORES.clear()
    first_key = ("workspace", "workflow", "run", "first")
    second_key = ("workspace", "workflow", "run", "second")
    third_key = ("workspace", "workflow", "run", "third")

    # when
    first_store = tracklet_recognition._get_store(state_key=first_key)
    second_store = tracklet_recognition._get_store(state_key=second_key)
    assert tracklet_recognition._get_store(state_key=first_key) is first_store
    third_store = tracklet_recognition._get_store(state_key=third_key)

    # then
    assert list(_STORES.keys()) == [first_key, third_key]
    assert _STORES[first_key] is first_store
    assert _STORES[third_key] is third_store
    assert second_store not in _STORES.values()


def test_tracklet_recognition_state_is_scoped_by_platform_context() -> None:
    # given
    gate_first_run = TrackletRecognitionGateBlockV1(
        workspace_id="workspace",
        workflow_id="workflow",
        execution_session_id="run-a",
    )
    cache_first_run = TrackletRecognitionCacheBlockV1(
        workspace_id="workspace",
        workflow_id="workflow",
        execution_session_id="run-a",
    )
    gate_second_run = TrackletRecognitionGateBlockV1(
        workspace_id="workspace",
        workflow_id="workflow",
        execution_session_id="run-b",
    )
    detections = _detections(detection_ids=["det-4"], tracker_ids=[4])
    first_gate_result = gate_first_run.run(
        image=_image(frame_number=0, fps=10),
        detections=detections,
        recognition_interval_seconds=10.0,
        state_id="plates",
    )
    cache_first_run.run(
        image=_image(frame_number=0, fps=10),
        detections=first_gate_result["all_detections"],
        recognized_detections=first_gate_result["detections_to_recognize"],
        recognition_predictions=[
            _classification_prediction(parent_id="det-4", class_name="ABC123")
        ],
        lock_after_consistent_results=5,
        state_id="plates",
        update_detection_class=True,
    )

    # when
    first_run_result = gate_first_run.run(
        image=_image(frame_number=1, fps=10),
        detections=detections,
        recognition_interval_seconds=10.0,
        state_id="plates",
    )
    second_run_result = gate_second_run.run(
        image=_image(frame_number=1, fps=10),
        detections=detections,
        recognition_interval_seconds=10.0,
        state_id="plates",
    )

    # then
    assert first_run_result["detections_to_recognize"].tracker_id.tolist() == []
    assert second_run_result["detections_to_recognize"].tracker_id.tolist() == [4]


def test_tracklet_recognition_global_tracklet_limit_is_bounded(monkeypatch) -> None:
    # given
    monkeypatch.setattr(tracklet_recognition, "MAX_TOTAL_TRACKLETS", 2)
    first_store = tracklet_recognition._get_store(
        state_key=("workspace", "workflow", "run", "first")
    )
    second_store = tracklet_recognition._get_store(
        state_key=("workspace", "workflow", "run", "second")
    )

    # when
    first_store.set(
        video_id="video",
        tracker_id=1,
        class_name="A",
        class_id=0,
        confidence=1.0,
        timestamp_seconds=0.0,
        consistency_window=1,
    )
    first_store.set(
        video_id="video",
        tracker_id=2,
        class_name="B",
        class_id=1,
        confidence=1.0,
        timestamp_seconds=0.0,
        consistency_window=1,
    )
    second_store.set(
        video_id="video",
        tracker_id=3,
        class_name="C",
        class_id=2,
        confidence=1.0,
        timestamp_seconds=0.0,
        consistency_window=1,
    )
    tracklet_recognition.enforce_global_tracklet_limit()

    # then
    assert sum(store.count_tracklets() for store in _STORES.values()) == 2
    assert (
        first_store.get(video_id="video", tracker_id=1, timestamp_seconds=0.0) is None
    )
    assert (
        first_store.get(video_id="video", tracker_id=2, timestamp_seconds=0.0)
        is not None
    )
    assert (
        second_store.get(video_id="video", tracker_id=3, timestamp_seconds=0.0)
        is not None
    )


def test_tracklet_recognition_rejects_large_consistency_window() -> None:
    # given
    store = TrackletRecognitionStore()

    # when / then
    with pytest.raises(ValueError, match="lock_after_consistent_results"):
        tracklet_recognition.update_store_from_predictions(
            store=store,
            video_id="video",
            timestamp_seconds=0.0,
            recognized_detections=_detections(
                detection_ids=["det-4"],
                tracker_ids=[4],
            ),
            recognition_predictions=["ABC123"],
            consistency_window=tracklet_recognition.MAX_CONSISTENCY_WINDOW + 1,
        )


def _detections(
    detection_ids,
    tracker_ids,
    class_names=None,
) -> sv.Detections:
    if class_names is None:
        class_names = ["object"] * len(detection_ids)
    return sv.Detections(
        xyxy=np.asarray(
            [[10 + i, 10 + i, 20 + i, 20 + i] for i in range(len(detection_ids))],
            dtype=float,
        ),
        confidence=np.asarray([0.9] * len(detection_ids), dtype=float),
        class_id=np.asarray([0] * len(detection_ids), dtype=int),
        tracker_id=np.asarray(tracker_ids, dtype=int),
        data={
            CLASS_NAME_DATA_FIELD: np.asarray(class_names, dtype=object),
            DETECTION_ID_KEY: np.asarray(detection_ids, dtype=object),
        },
    )


def _classification_prediction(parent_id: str, class_name: str) -> dict:
    return {
        "top": class_name,
        "predictions": [
            {
                "class": class_name,
                "class_id": 7,
                "confidence": 0.95,
            }
        ],
        "parent_id": parent_id,
    }


def _image(frame_number: int, fps: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=np.zeros((100, 100, 3), dtype=np.uint8),
        video_metadata=VideoMetadata(
            video_identifier="video",
            frame_number=frame_number,
            frame_timestamp=datetime.datetime.fromtimestamp(
                0, tz=datetime.timezone.utc
            ),
            fps=fps,
            comes_from_video_file=True,
        ),
    )
