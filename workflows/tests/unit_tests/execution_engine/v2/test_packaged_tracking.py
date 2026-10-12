"""Packaged ``v2/byte_tracker@v1``: per-source ByteTrack over native predictions.

Reference results come from stock ``sv.ByteTrack`` and from the unchanged V1
tensor helpers, run on the same rows. The block must match them per source,
whatever the interleaving of sources.
"""

import math
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.common.tensor_native import (
    take_prediction_by_indices,
)
from roboflow_workflows.core_steps.transformations.byte_tracker.v3_tensor import (
    _track_and_recover_indices,
    _write_tracker_ids,
)
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.execution_engine.v2.blocks.detection import (
    ObjectDetectionBlock,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.blocks.tracking import (
    DEFAULT_MAX_SOURCES,
    ByteTrackerBlock,
    TrackerCapacityError,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.pipelining import PipelineOptions
from roboflow_workflows.execution_engine.v2.plan import CompileOptions
from roboflow_workflows.execution_engine.v2.updates import RESET, compare_plans

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

FRAMES = 30
SOURCES = ("cam_a", "cam_b", "cam_c")
START = datetime(2026, 10, 10, 12, 0, 0)
EMPTY_FRAMES = {0: (12, 13), 1: (), 2: (5,)}
CLASS_NAMES = {0: "car", 1: "person"}


def _rows(source: int, frame: int) -> np.ndarray:
    """Rows ``x0 y0 x1 y1 confidence class``; objects move, one disappears."""
    if frame in EMPTY_FRAMES[source]:
        return np.zeros((0, 6), dtype=np.float32)

    rows = []
    for obj in range(3):
        if obj == 1 and 6 <= frame < 9:
            continue
        x = 10.0 + 40 * obj + (1.5 + source) * frame
        y = 30.0 + 20 * obj + 5 * source
        confidence = 0.2 if obj == 2 and frame % 4 == 0 else 0.9 - 0.1 * obj
        rows.append([x, y, x + 20, y + 25, confidence, obj % 2])
    # Reverse so ByteTrack's output order differs from input order.
    rows = np.asarray(rows[::-1], dtype=np.float32)

    return rows


def _detections(source: int, frame: int) -> Detections:
    rows = torch.from_numpy(_rows(source, frame))
    detections = Detections(
        xyxy=rows[:, :4],
        confidence=rows[:, 4],
        class_id=rows[:, 5].to(torch.int64),
        image_metadata={"class_names": CLASS_NAMES, "parent_id": f"{source}-{frame}"},
        bboxes_metadata=[{"row": row} for row in range(len(rows))] or None,
    )

    return detections


def _image(source: int, frame: int, *, fps: Optional[float] = 30) -> ImageData:
    pixels = torch.zeros((3, 8, 8), dtype=torch.uint8)
    pixels[0, 0, 0] = source
    pixels[1, 0, 0] = frame
    metadata = VideoMetadata(
        video_identifier=SOURCES[source],
        frame_number=frame,
        frame_timestamp=START + timedelta(seconds=frame / 30),
        fps=fps,
    )
    image = ImageData.from_tensor(
        pixels, image_id=f"{SOURCES[source]}-{frame}", video_metadata=metadata
    )

    return image


def _track(
    block: ByteTrackerBlock, image: ImageData, predictions: Any, **params: Any
) -> Any:
    arguments = {
        "track_activation_threshold": 0.25,
        "lost_track_buffer": 30,
        "minimum_matching_threshold": 0.8,
        "minimum_consecutive_frames": 1,
        "max_sources": DEFAULT_MAX_SOURCES,
        **params,
    }
    tracked = block.run(image=image, predictions=predictions, **arguments)

    return tracked["tracked_detections"]


def _v1_reference(
    source: int, *, frame_rate: float = 30, lost_track_buffer: int = 30
) -> List[Tuple[List[float], List[int]]]:
    """Per frame: (kept xyxy flattened, tracker ids) with the V1 helpers."""
    tracker = sv.ByteTrack(frame_rate=frame_rate, lost_track_buffer=lost_track_buffer)
    reference = []
    for frame in range(FRAMES):
        detections = _detections(source, frame)
        kept, tracker_ids = _track_and_recover_indices(
            tracker=tracker, detections=detections
        )
        tracked = take_prediction_by_indices(detections, kept)
        _write_tracker_ids(tracked, tracker_ids)
        reference.append(_summary(tracked))

    return reference


def _summary(tracked: Detections) -> Tuple[List[float], List[int]]:
    ids = [row["tracker_id"] for row in tracked.bboxes_metadata or []]
    summary = (tracked.xyxy.flatten().tolist(), ids)

    return summary


def test_single_source_matches_the_v1_tensor_helpers_frame_by_frame() -> None:
    block = ByteTrackerBlock()

    results = [
        _summary(_track(block, _image(0, frame), _detections(0, frame)))
        for frame in range(FRAMES)
    ]

    assert results == _v1_reference(0)
    assert any(ids for _, ids in results)


def test_interleaved_sources_are_tracked_independently() -> None:
    block = ByteTrackerBlock()
    results: Dict[int, list] = {source: [] for source in range(3)}

    # Sources interleave in a changing order; each still sees its own frames
    # in order.
    for frame in range(FRAMES):
        order = (0, 1, 2) if frame % 2 else (2, 0, 1)
        for source in order:
            tracked = _track(block, _image(source, frame), _detections(source, frame))
            results[source].append(_summary(tracked))

    for source in range(3):
        assert results[source] == _v1_reference(source)


def test_output_rows_follow_bytetrack_order_and_keep_row_metadata() -> None:
    block = ByteTrackerBlock()
    detections = _detections(0, 1)

    tracked = _track(block, _image(0, 1), detections)

    stock = sv.ByteTrack(frame_rate=30).update_with_detections(
        sv.Detections(
            xyxy=detections.xyxy.numpy(),
            confidence=detections.confidence.numpy(),
            class_id=detections.class_id.numpy(),
            data={"row": np.arange(len(detections))},
        )
    )
    expected_rows = stock.data["row"].tolist()
    assert [row["row"] for row in tracked.bboxes_metadata] == expected_rows
    assert [row["tracker_id"] for row in tracked.bboxes_metadata] == (
        stock.tracker_id.tolist()
    )
    assert torch.equal(tracked.xyxy, detections.xyxy[expected_rows])


def test_empty_frame_returns_an_empty_prediction_and_ages_tracks() -> None:
    block = ByteTrackerBlock()
    for frame in range(12):
        before = _track(block, _image(0, frame), _detections(0, frame))

    empty = _track(block, _image(0, 12), _detections(0, 12))

    assert isinstance(empty, Detections)
    assert len(empty) == 0 and empty.xyxy.shape == (0, 4)
    assert empty.image_metadata["class_names"] == CLASS_NAMES
    after = _track(block, _image(0, 14), _detections(0, 14))
    # Tracks survive the two empty frames inside lost_track_buffer.
    assert set(_summary(after)[1]) <= set(_summary(before)[1])
    assert _summary(after)[1]


def test_lost_track_buffer_expires_tracks_across_empty_frames() -> None:
    block = ByteTrackerBlock()
    first = _track(block, _image(0, 0), _detections(0, 0), lost_track_buffer=1)

    for frame in range(1, 4):
        _track(block, _image(0, frame), _detections(0, 12), lost_track_buffer=1)
    later = _track(block, _image(0, 4), _detections(0, 0), lost_track_buffer=1)

    assert not set(_summary(first)[1]) & set(_summary(later)[1])


def test_input_prediction_is_not_mutated() -> None:
    block = ByteTrackerBlock()
    detections = _detections(1, 3)
    xyxy = detections.xyxy.clone()
    row_dicts = detections.bboxes_metadata
    row_copies = [dict(row) for row in row_dicts]
    image_metadata = dict(detections.image_metadata)

    tracked = _track(block, _image(1, 3), detections)

    assert tracked is not detections
    assert torch.equal(detections.xyxy, xyxy)
    assert detections.bboxes_metadata is row_dicts
    assert detections.bboxes_metadata == row_copies
    assert all("tracker_id" not in row for row in detections.bboxes_metadata)
    assert detections.image_metadata == image_metadata
    assert tracked.image_metadata == image_metadata


def test_input_without_row_metadata_stays_without_it() -> None:
    block = ByteTrackerBlock()
    detections = _detections(0, 1)
    detections.bboxes_metadata = None

    tracked = _track(block, _image(0, 1), detections)

    assert detections.bboxes_metadata is None
    assert all("tracker_id" in row for row in tracked.bboxes_metadata)


def test_instance_segmentation_keeps_its_family_and_masks() -> None:
    block = ByteTrackerBlock()
    detections = _detections(0, 1)
    rows = len(detections)
    masks = torch.zeros((rows, 64, 64), dtype=torch.bool)
    for row in range(rows):
        masks[row, row, row] = True
    segmentation = InstanceDetections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        mask=masks,
        image_metadata=detections.image_metadata,
    )

    tracked = _track(block, _image(0, 1), segmentation)

    assert isinstance(tracked, InstanceDetections)
    assert len(tracked) == rows
    INSTANCE_SEGMENTATION_PREDICTION_KIND.check(tracked)
    for position in range(len(tracked)):
        source_row = int(
            torch.nonzero((detections.xyxy == tracked.xyxy[position]).all(dim=1)).item()
        )
        assert torch.equal(tracked.mask[position], masks[source_row])


def test_capacity_error_comes_before_any_state_change() -> None:
    block = ByteTrackerBlock()
    _track(block, _image(0, 0), _detections(0, 0), max_sources=2)
    _track(block, _image(1, 0), _detections(1, 0), max_sources=2)
    trackers = dict(block._trackers)

    with pytest.raises(TrackerCapacityError, match="max_sources=2"):
        _track(block, _image(2, 0), _detections(2, 0), max_sources=2)

    assert block._trackers == trackers
    # Existing sources keep tracking as if the rejected frame never happened.
    results = [
        _summary(_track(block, _image(0, frame), _detections(0, frame)))
        for frame in range(1, FRAMES)
    ]
    assert results == _v1_reference(0)[1:]


def test_raising_the_bound_admits_more_sources() -> None:
    block = ByteTrackerBlock()
    _track(block, _image(0, 0), _detections(0, 0), max_sources=1)

    _track(block, _image(1, 0), _detections(1, 0), max_sources=2)

    assert sorted(block._trackers) == ["cam_a", "cam_b"]


def test_default_bound_is_256_sources() -> None:
    block = ByteTrackerBlock()
    for source in range(DEFAULT_MAX_SOURCES):
        image = _image(0, 0)
        image.video_metadata.video_identifier = f"source-{source}"
        _track(block, image, _detections(0, 0))

    overflow = _image(0, 0)
    overflow.video_metadata.video_identifier = "one-too-many"
    with pytest.raises(TrackerCapacityError):
        _track(block, overflow, _detections(0, 0))

    assert len(block._trackers) == DEFAULT_MAX_SOURCES


UNKNOWN_FPS = [None, 0, -30.0, math.nan, math.inf]


@pytest.mark.parametrize("fps", UNKNOWN_FPS)
def test_missing_fps_is_rejected_and_points_to_frame_rate(fps) -> None:
    block = ByteTrackerBlock()

    with pytest.raises(ValueError, match="fps") as caught:
        _track(block, _image(0, 0, fps=fps), _detections(0, 0))

    assert "frame_rate" in str(caught.value) and "cam_a" in str(caught.value)
    assert block._trackers == {}


@pytest.mark.parametrize("fps", UNKNOWN_FPS)
def test_frame_rate_tracks_a_source_without_fps_and_leaves_its_metadata(
    fps,
) -> None:
    block = ByteTrackerBlock()
    images = [_image(0, frame, fps=fps) for frame in range(FRAMES)]
    # repr, because NaN != NaN.
    metadata_before = [repr(image.video_metadata) for image in images]

    results = [
        _summary(_track(block, image, _detections(0, frame), frame_rate=30.0))
        for frame, image in enumerate(images)
    ]

    assert results == _v1_reference(0)
    assert [repr(image.video_metadata) for image in images] == metadata_before


# lost_track_buffer=2 keeps a lost track for int(rate / 30 * 2) processed
# frames: 4 at 60 fps, 1 at 15 fps. Object 1 is missing for frames 6-8, so
# the two rates give it different tracker ids afterwards.
SHORT_BUFFER = 2


def test_rates_that_differ_give_different_references() -> None:
    assert _v1_reference(0, frame_rate=60, lost_track_buffer=SHORT_BUFFER) != (
        _v1_reference(0, frame_rate=15, lost_track_buffer=SHORT_BUFFER)
    )


@pytest.mark.parametrize(
    "fps, frame_rate, expected_rate",
    [(60.0, None, 60), (60.0, 15.0, 15), (0, 15.0, 15)],
)
def test_frame_rate_overrides_known_fps_and_null_uses_it(
    fps, frame_rate, expected_rate
) -> None:
    block = ByteTrackerBlock()

    results = [
        _summary(
            _track(
                block,
                _image(0, frame, fps=fps),
                _detections(0, frame),
                lost_track_buffer=SHORT_BUFFER,
                frame_rate=frame_rate,
            )
        )
        for frame in range(FRAMES)
    ]

    assert results == _v1_reference(
        0, frame_rate=expected_rate, lost_track_buffer=SHORT_BUFFER
    )


@pytest.mark.parametrize("frame_rate", [0, -1.0, math.nan, math.inf, True])
def test_invalid_frame_rate_in_a_direct_call_creates_no_tracker(frame_rate) -> None:
    block = ByteTrackerBlock()

    with pytest.raises(ValueError, match="frame_rate"):
        _track(block, _image(0, 0), _detections(0, 0), frame_rate=frame_rate)

    assert block._trackers == {}


def test_missing_video_identifier_is_rejected() -> None:
    block = ByteTrackerBlock()
    image = ImageData.from_tensor(torch.zeros((3, 8, 8), dtype=torch.uint8))

    with pytest.raises(ValueError, match="video_identifier"):
        _track(block, image, _detections(0, 0))

    assert block._trackers == {}


def test_tracker_declares_the_native_detection_families() -> None:
    block_kinds = {
        kind.name for kind in ByteTrackerBlock.outputs["tracked_detections"].kinds
    }

    assert block_kinds == {
        OBJECT_DETECTION_PREDICTION_KIND.name,
        INSTANCE_SEGMENTATION_PREDICTION_KIND.name,
        RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND.name,
    }


class SourceKeyedModel:
    """Batched fake detector: pixel (0, 0) encodes source and frame."""

    class_names = ["car", "person"]

    def __init__(self, *, jitter_seconds: float = 0.0):
        self.batch_sizes: List[int] = []
        self._jitter = jitter_seconds
        self._random = random.Random(7)

    def pre_process(self, images, input_color_format=None):
        self.batch_sizes.append(len(images))
        keys = [(int(image[0, 0, 0]), int(image[1, 0, 0])) for image in images]

        return keys, None

    def forward(self, keys):
        if self._jitter:
            time.sleep(self._random.uniform(0, self._jitter))

        return keys

    def post_process(self, keys, metadata, confidence=None):
        predictions = []
        for source, frame in keys:
            detections = _detections(source, frame)
            detections.bboxes_metadata = None
            detections.image_metadata = {}
            predictions.append(detections)

        return predictions


def _tracking_plan(
    mode: str, *, max_sources: int = DEFAULT_MAX_SOURCES, **tracker_params: Any
):
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
        "steps": [
            {
                "type": "v2/object_detection@v1",
                "name": "detector",
                "image": "$inputs.image",
                "model_id": "fake",
                "confidence": 0.0,
            },
            {
                "type": "v2/byte_tracker@v1",
                "name": "tracker",
                "image": "$inputs.image",
                "predictions": "$steps.detector.predictions",
                "max_sources": max_sources,
                **tracker_params,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "tracked_detections",
                "selector": "$steps.tracker.tracked_detections",
            }
        ],
    }
    catalogue = Catalogue(
        [ObjectDetectionBlock, ByteTrackerBlock],
        kinds=[
            IMAGE_KIND,
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            FLOAT_KIND,
        ],
    )
    plan = compile_workflow(
        definition,
        catalogue=catalogue,
        options=CompileOptions(block_execution=mode),
    )

    return plan


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_batched_detector_feeds_per_source_trackers(mode: str) -> None:
    model = SourceKeyedModel()
    session = _tracking_plan(mode).create_session({"models": {"fake": model}})
    results: Dict[int, list] = {0: [], 1: []}

    for frame in range(FRAMES):
        sources = (0, 1) if frame % 3 else (1, 0)
        images = [_image(source, frame) for source in sources]
        rows = session.run({"image": images}).rows()
        for source, row in zip(sources, rows):
            tracked = row["tracked_detections"]
            assert tracked.image_metadata["parent_id"] == f"{SOURCES[source]}-{frame}"
            assert tracked.image_metadata["class_names"] == CLASS_NAMES
            results[source].append(_summary(tracked))

    assert model.batch_sizes == [2] * FRAMES
    for source in (0, 1):
        assert results[source] == _v1_reference(source)


def test_pipelined_runs_keep_each_source_in_frame_order() -> None:
    # Jitter in forward lets detector phases of different runs overlap; the
    # tracker must still see every source's frames in submission order.
    model = SourceKeyedModel(jitter_seconds=0.004)
    session = _tracking_plan("phases").create_session({"models": {"fake": model}})
    submitted = []
    with session.pipeline(options=PipelineOptions(max_in_flight=3)) as pipeline:
        for frame in range(FRAMES):
            sources = (0, 1) if frame % 3 else (1, 0)
            images = [_image(source, frame) for source in sources]
            submitted.append((sources, pipeline.submit({"image": images})))
        results: Dict[int, list] = {0: [], 1: []}
        for sources, future in submitted:
            rows = future.result().rows()
            for source, row in zip(sources, rows):
                results[source].append(_summary(row["tracked_detections"]))

    for source in (0, 1):
        assert results[source] == _v1_reference(source)


def test_workflow_capacity_overflow_names_the_bound() -> None:
    session = _tracking_plan("run", max_sources=1).create_session(
        {"models": {"fake": SourceKeyedModel()}}
    )
    session.run({"image": [_image(0, 0)]})

    with pytest.raises(StepExecutionError) as caught:
        session.run({"image": [_image(1, 0)]})

    assert isinstance(caught.value.__cause__, TrackerCapacityError)
    assert "max_sources=1" in str(caught.value)


def test_workflow_frame_rate_tracks_sources_without_fps_independently() -> None:
    session = _tracking_plan("run", frame_rate=30.0).create_session(
        {"models": {"fake": SourceKeyedModel()}}
    )
    results: Dict[int, list] = {0: [], 1: []}

    for frame in range(FRAMES):
        images = [_image(source, frame, fps=0) for source in (1, 0)]
        rows = session.run({"image": images}).rows()
        for source, row in zip((1, 0), rows):
            tracked = row["tracked_detections"]
            assert tracked.image_metadata["parent_id"] == f"{SOURCES[source]}-{frame}"
            results[source].append(_summary(tracked))

    for source in (0, 1):
        assert results[source] == _v1_reference(source)


def test_workflow_without_frame_rate_fails_on_a_source_without_fps() -> None:
    session = _tracking_plan("run").create_session(
        {"models": {"fake": SourceKeyedModel()}}
    )

    with pytest.raises(StepExecutionError) as caught:
        session.run({"image": [_image(0, 0, fps=0)]})

    assert isinstance(caught.value.__cause__, ValueError)
    assert "frame_rate" in str(caught.value)


@pytest.mark.parametrize("frame_rate", [0, -30.0, math.nan, math.inf, "30", True])
def test_invalid_workflow_frame_rate_is_rejected_at_compile(frame_rate) -> None:
    with pytest.raises(ParamsValidationError) as caught:
        _tracking_plan("run", frame_rate=frame_rate)

    assert "frame_rate" in str(caught.value)


def test_integer_frame_rate_is_accepted() -> None:
    _tracking_plan("run", frame_rate=30)


def test_editing_frame_rate_requires_a_processing_reset() -> None:
    base = _tracking_plan("run")

    for edited in (_tracking_plan("run", frame_rate=25.0), base):
        diff = compare_plans(_tracking_plan("run", frame_rate=30.0), edited)

        assert diff.kind == RESET
        assert [(c.name, c.reason) for c in diff.changes] == [
            ("$steps.tracker", "params_changed")
        ]
