"""The four ``trackers`` package blocks, run against the real library.

Fixture: three objects moving on a 120x240 frame at 30 fps.

    row  start box           step per frame    confidence
    A    [10, 10, 40, 40]    x +4              0.9
    B    [100, 20, 130, 50]  y +3              0.85
    C    [200, 60, 230, 90]  x -5              0.4

Tracker ids come from a counter shared by all trackers of one algorithm in the
process, so tests compare ids by first appearance, never absolutely.
"""

import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pytest
import supervision as sv
import torch
import trackers
from pycocotools import mask as coco_mask
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.tracking.base import TRACKER_OUTPUTS
from roboflow_workflows.v2.blocks.tracking.trackers_botsort.v1 import BoTSORTTrackerBlock
from roboflow_workflows.v2.blocks.tracking.trackers_bytetrack.v1 import ByteTrackTrackerBlock
from roboflow_workflows.v2.blocks.tracking.trackers_ocsort.v1 import OCSORTTrackerBlock
from roboflow_workflows.v2.blocks.tracking.trackers_sort.v1 import SORTTrackerBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity
from roboflow_workflows.v2.engine import NoExecutionContextError
from roboflow_workflows.v2.engine.errors import EventEmissionError

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

BLOCKS = [
    SORTTrackerBlock,
    ByteTrackTrackerBlock,
    OCSORTTrackerBlock,
    BoTSORTTrackerBlock,
]
LIBRARY_CLASSES = {
    SORTTrackerBlock: trackers.SORTTracker,
    ByteTrackTrackerBlock: trackers.ByteTrackTracker,
    OCSORTTrackerBlock: trackers.OCSORTTracker,
    BoTSORTTrackerBlock: trackers.BoTSORTTracker,
}
START = torch.tensor(
    [[10, 10, 40, 40], [100, 20, 130, 50], [200, 60, 230, 90]], dtype=torch.float32
)
STEP = torch.tensor([[4, 0, 4, 0], [0, 3, 0, 3], [-5, 0, -5, 0]], dtype=torch.float32)
CONFIDENCE = torch.tensor([0.9, 0.85, 0.4])
NAMES = "ABC"
HEIGHT, WIDTH = 120, 240

# Reported objects per frame 0..4 with V1 defaults, verified on trackers 2.4.0.
V1_DEFAULT_PATTERNS = {
    SORTTrackerBlock: ["", "", "ABC", "ABC", "ABC"],
    ByteTrackTrackerBlock: ["", "AB", "AB", "AB", "AB"],
    # Warm-up reports early; C (0.4) is below high_conf_det_threshold 0.6.
    OCSORTTrackerBlock: ["", "AB", "AB", "AB", "AB"],
    BoTSORTTrackerBlock: ["", "AB", "AB", "AB", "AB"],
}

V1_DEFAULTS = {
    SORTTrackerBlock: dict(
        minimum_iou_threshold=0.3,
        minimum_consecutive_frames=3,
        lost_track_buffer=30,
        track_activation_threshold=0.25,
    ),
    ByteTrackTrackerBlock: dict(
        minimum_iou_threshold=0.1,
        minimum_consecutive_frames=2,
        lost_track_buffer=30,
        track_activation_threshold=0.7,
        high_conf_det_threshold=0.6,
    ),
    OCSORTTrackerBlock: dict(
        minimum_iou_threshold=0.3,
        minimum_consecutive_frames=3,
        lost_track_buffer=30,
        high_conf_det_threshold=0.6,
        direction_consistency_weight=0.2,
        delta_t=3,
    ),
    BoTSORTTrackerBlock: dict(
        minimum_iou_threshold_first_assoc=0.2,
        minimum_iou_threshold_second_assoc=0.5,
        minimum_iou_threshold_unconfirmed_assoc=0.3,
        minimum_consecutive_frames=2,
        lost_track_buffer=30,
        track_activation_threshold=0.7,
        high_conf_det_threshold=0.6,
        enable_cmc=False,
        cmc_method="sparseOptFlow",
        cmc_downscale=2,
        instant_first_frame_activation=False,
    ),
}


def _image(source="a", *, fps=30.0):
    metadata = VideoMetadata(
        video_identifier=source,
        fps=30,
        frame_number=0,
        frame_timestamp=datetime(2026, 10, 13),
    )
    metadata.fps = fps
    image = ImageData.from_tensor(
        torch.zeros((3, HEIGHT, WIDTH), dtype=torch.uint8), video_metadata=metadata
    )

    return image


def _boxes_at(frame):
    xyxy = START + frame * STEP
    boxes = Detections(
        xyxy=xyxy,
        confidence=CONFIDENCE.clone(),
        class_id=torch.tensor([0, 1, 2]),
        image_metadata={"parent_id": f"frame-{frame}"},
        bboxes_metadata=[{"name": name, "custom": frame} for name in NAMES],
    )
    boxes = with_row_identity(boxes, namespace="moving")

    return boxes


def _box_masks(xyxy):
    masks = torch.zeros((len(xyxy), HEIGHT, WIDTH), dtype=torch.bool)
    for mask, (x1, y1, x2, y2) in zip(masks, xyxy.long().tolist()):
        mask[y1:y2, x1:x2] = True

    return masks


def _prediction(frame, family="boxes"):
    boxes = _boxes_at(frame)
    if family == "boxes":
        return boxes

    if family == "keypoints":
        points = KeyPoints(
            xy=boxes.xyxy[:, :2].unsqueeze(1),
            confidence=boxes.confidence.unsqueeze(1),
            class_id=boxes.class_id,
            key_points_metadata=[{"point": name} for name in NAMES],
        )
        return points, boxes

    masks = _box_masks(boxes.xyxy)
    if family == "grid":
        geometry = MaskGridGeometry(
            grid_size_hw=(HEIGHT // 2, WIDTH // 2),
            inference_size_hw=(HEIGHT, WIDTH),
            padding_ltrb=(0, 0, 0, 0),
            scale_xy=(1.0, 1.0),
            crop_size_hw=(HEIGHT, WIDTH),
            original_size_hw=(HEIGHT, WIDTH),
            static_crop_xywh=(0, 0, WIDTH, HEIGHT),
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
            image_size=(HEIGHT, WIDTH),
            masks=[
                coco_mask.encode(np.asfortranarray(m.numpy().astype(np.uint8)))[
                    "counts"
                ]
                for m in masks
            ],
        )
    instances = InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        image_metadata=boxes.image_metadata,
        bboxes_metadata=boxes.bboxes_metadata,
        mask=mask,
    )
    if family == "dense_view":
        return SegmentationView.from_dense(instances)

    return instances


def _detections(prediction):
    if isinstance(prediction, SegmentationView):
        return prediction.detections

    return prediction[1] if isinstance(prediction, tuple) else prediction


def _run(block, frame=0, *, source="a", prediction=None, fps=30.0, **overrides):
    params = dict(V1_DEFAULTS[type(block)], max_sources=4, frame_rate=None)
    params.update(overrides)
    result = block.run(
        image=_image(source, fps=fps),
        predictions=_prediction(frame) if prediction is None else prediction,
        **params,
    )

    return result


def _names(prediction):
    names = "".join(
        sorted(row["name"] for row in _detections(prediction).bboxes_metadata or [])
    )

    return names


def _ids(prediction):
    ids = [row["tracker_id"] for row in _detections(prediction).bboxes_metadata or []]

    return ids


def _ids_by_name(prediction):
    rows = _detections(prediction).bboxes_metadata or []
    ids = {row["name"]: row["tracker_id"] for row in rows}

    return ids


def _first_appearance(ids):
    order = {}
    for value in ids:
        order.setdefault(value, len(order))
    relative = [order[value] for value in ids]

    return relative


def _library_run(block_type, frames):
    """Track the fixture with the bare library tracker, as V1 did."""
    library = LIBRARY_CLASSES[block_type]
    tracker = library(frame_rate=30.0, **V1_DEFAULTS[block_type])
    reported = []
    for frame in frames:
        boxes = _boxes_at(frame)
        tracked = tracker.update(
            sv.Detections(
                xyxy=boxes.xyxy.numpy(),
                confidence=boxes.confidence.numpy(),
                class_id=boxes.class_id.numpy(),
            )
        )
        kept = tracked.tracker_id != -1
        reported.append(
            (tracked.xyxy[kept].tolist(), tracked.tracker_id[kept].tolist())
        )

    return reported


def test_params_defaults_are_the_v1_defaults():
    for block_type, defaults in V1_DEFAULTS.items():
        fields = block_type.Params.model_fields
        assert {name: fields[name].default for name in defaults} == defaults
        assert fields["instances_cache_size"].default == 16384
        assert fields["max_sources"].default == 256
        assert fields["frame_rate"].default is None


@pytest.mark.parametrize("block_type", BLOCKS)
def test_first_frame_constructs_library_tracker_with_knobs_and_rate(block_type):
    block = block_type()
    created = []
    library = LIBRARY_CLASSES[block_type]
    assert block._tracker_class is library

    def record(**kwargs):
        created.append(kwargs)
        return library(**kwargs)

    block._tracker_class = record
    _run(block, frame_rate=60.0)
    _run(block, minimum_consecutive_frames=7)
    assert created == [dict(V1_DEFAULTS[block_type], frame_rate=60.0)]
    tracker = block._trackers["a"].tracker
    assert isinstance(tracker, library)
    # lost_track_buffer 30 frames at 30 fps scaled to the 60 fps source.
    assert tracker.maximum_frames_without_update == 60


@pytest.mark.parametrize("block_type", BLOCKS)
def test_v1_defaults_confirm_moving_objects_like_the_library(block_type):
    block = block_type()
    reported = [_run(block, frame)["tracked_detections"] for frame in range(5)]
    assert [_names(p) for p in reported] == V1_DEFAULT_PATTERNS[block_type]
    library = _library_run(block_type, range(5))
    block_ids = [_ids(p) for p in reported]
    library_ids = [ids for _, ids in library]
    assert _first_appearance(sum(block_ids, [])) == _first_appearance(
        sum(library_ids, [])
    )
    for prediction, (library_xyxy, _) in zip(reported, library):
        assert _detections(prediction).xyxy.tolist() == library_xyxy


def test_botsort_instant_first_frame_activation_reports_frame_zero():
    block = BoTSORTTrackerBlock()
    first = _run(block, instant_first_frame_activation=True)
    assert _names(first["tracked_detections"]) == "AB"
    assert _names(first["new_instances"]) == "AB"


@pytest.mark.parametrize(
    "family", ["boxes", "dense", "rle", "grid", "dense_view", "keypoints"]
)
def test_ocsort_keeps_row_metadata_masks_keypoints_and_view_laziness(
    family, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("tracking must not call full_res")

    monkeypatch.setattr(SegmentationView, "full_res", forbidden)
    block = OCSORTTrackerBlock()
    for frame in range(3):
        prediction = _prediction(frame, family)
        outputs = _run(block, frame, prediction=prediction)
    original = _detections(prediction)
    before = [dict(row) for row in original.bboxes_metadata]
    tracked = outputs["tracked_detections"]
    boxes = _detections(tracked)
    rows = [NAMES.index(row["name"]) for row in boxes.bboxes_metadata]
    assert sorted(rows) == [0, 1]  # C is dropped below high_conf_det_threshold.
    assert torch.equal(boxes.xyxy, original.xyxy[rows])
    assert torch.equal(boxes.confidence, original.confidence[rows])
    assert [row["custom"] for row in boxes.bboxes_metadata] == [2, 2]
    assert boxes.image_metadata["parent_id"] == "frame-2"
    assert row_record(boxes).host_ids() == rows
    assert original.bboxes_metadata == before
    if isinstance(tracked, SegmentationView):
        assert tracked.materialization_count == prediction.materialization_count == 0
        assert torch.equal(tracked.scores, prediction.scores[rows])
        assert tracked.geometry == prediction.geometry
    elif isinstance(tracked, tuple):
        assert torch.equal(tracked[0].xy, prediction[0].xy[rows])
        assert [row["point"] for row in tracked[0].key_points_metadata] == [
            NAMES[i] for i in rows
        ]
    elif family == "rle":
        assert tracked.mask.masks == [prediction.mask.masks[i] for i in rows]
    elif family == "dense":
        assert torch.equal(tracked.mask, prediction.mask[rows])


@pytest.mark.parametrize("block_type", BLOCKS)
def test_zero_demand_still_advances_tracker_and_fifo(block_type, monkeypatch):
    reference = block_type()
    limited = block_type()
    monkeypatch.setattr(limited, "_wanted_outputs", lambda: set())
    for frame in range(3):
        expected = _run(reference, frame)
        assert _run(limited, frame) == {}
    assert len(limited._trackers["a"].instances) == len(
        reference._trackers["a"].instances
    )
    monkeypatch.setattr(limited, "_wanted_outputs", lambda: set(limited.outputs))
    expected = _run(reference, 3)
    actual = _run(limited, 3)
    for name in TRACKER_OUTPUTS:
        assert _names(actual[name]) == _names(expected[name])
    assert _names(actual["already_seen_instances"])
    assert _names(actual["new_instances"]) == ""


@pytest.mark.parametrize("block_type", BLOCKS)
def test_interleaved_sources_have_own_trackers_and_first_frame_knobs(block_type):
    block = block_type()
    for frame in range(3):
        for source in ("b", "a"):
            outputs = _run(
                block,
                frame,
                source=source,
                instances_cache_size=8,
                # Later values must not retune a source's existing tracker.
                minimum_consecutive_frames=1 if frame == 0 else 9,
            )
            if frame == 2:
                assert _names(outputs["tracked_detections"])
    assert block._trackers["a"].tracker is not block._trackers["b"].tracker
    assert block._trackers["a"].instances.size == 8
    a_ids = set(_ids_by_name(_run(block, 3, source="a")["tracked_detections"]).values())
    b_ids = set(_ids_by_name(_run(block, 3, source="b")["tracked_detections"]).values())
    # One shared id counter per algorithm: sources never reuse each other's ids.
    assert a_ids and b_ids and not a_ids & b_ids


@pytest.mark.parametrize("fps", [None, 0.0, float("nan")])
def test_source_without_usable_fps_fails_and_creates_no_state(fps):
    block = SORTTrackerBlock()
    with pytest.raises(ValueError, match="fps"):
        _run(block, fps=fps)
    assert len(block._trackers) == 0
    _run(block, fps=fps, frame_rate=60.0)
    assert block._trackers["a"].tracker.maximum_frames_without_update == 60


def test_keypoints_without_boxes_fail_and_create_no_state():
    block = SORTTrackerBlock()
    points, _ = _prediction(0, "keypoints")
    with pytest.raises(ValueError):
        _run(block, prediction=(points, None))
    assert len(block._trackers) == 0


@pytest.mark.parametrize("block_type", BLOCKS)
def test_empty_frames_age_tracks_and_a_lost_object_returns_with_a_new_id(
    block_type,
):
    block = block_type()
    for frame in range(4):
        before = _ids_by_name(
            _run(block, frame, lost_track_buffer=1)["tracked_detections"]
        )
    assert before
    empty = select_predictions(_prediction(4), indices=[])
    for frame in range(4, 8):
        outputs = _run(block, frame, prediction=empty, lost_track_buffer=1)
        assert all(len(_detections(value)) == 0 for value in outputs.values())
    after = {}
    for frame in range(8, 12):
        after = _ids_by_name(_run(block, frame)["tracked_detections"])
    assert after.keys() == before.keys()
    assert not set(after.values()) & set(before.values())


def test_cache_eviction_makes_a_still_tracked_id_new_again():
    block = ByteTrackTrackerBlock()
    for frame in range(3):
        outputs = _run(block, frame, instances_cache_size=1)
    # Two ids alternate through a one-id cache, so both are new every frame.
    assert _names(outputs["new_instances"]) == "AB"
    assert _names(outputs["already_seen_instances"]) == ""


@pytest.mark.parametrize("block_type", BLOCKS)
def test_plain_call_falls_back_only_for_missing_context(block_type, monkeypatch):
    block = block_type()
    module = sys.modules[block_type.__module__]

    def absent():
        raise NoExecutionContextError("plain call")

    monkeypatch.setattr(module, "get_execution_context", absent)
    assert set(_run(block)) == set(TRACKER_OUTPUTS)

    def broken():
        raise EventEmissionError("bad engine context")

    monkeypatch.setattr(module, "get_execution_context", broken)
    with pytest.raises(EventEmissionError, match="bad engine context"):
        _run(block)


@pytest.mark.parametrize("block_type", BLOCKS)
def test_missing_trackers_package_fails_on_construction_with_install_hint(
    block_type, monkeypatch
):
    monkeypatch.setitem(sys.modules, "trackers", None)
    with pytest.raises(ImportError, match=r"pip install 'trackers~=2\.4\.0'"):
        block_type()


def test_importing_the_blocks_does_not_import_trackers():
    code = (
        "import sys\n"
        "import roboflow_workflows.v2.blocks.tracking.trackers_sort.v1\n"
        "import roboflow_workflows.v2.blocks.tracking.trackers_bytetrack.v1\n"
        "import roboflow_workflows.v2.blocks.tracking.trackers_ocsort.v1\n"
        "import roboflow_workflows.v2.blocks.tracking.trackers_botsort.v1\n"
        "assert 'trackers' not in sys.modules, 'trackers imported eagerly'\n"
    )
    # Keep this process's import path so the child loads the same sources.
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
