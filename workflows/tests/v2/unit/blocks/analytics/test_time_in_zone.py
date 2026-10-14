"""``roboflow_core/time_in_zone@v3`` called directly.

The oracle is latest V1 ``roboflow_core/time_in_zone@v3`` (tensor variant);
the task's randomized parity probe compares whole sequences with it. These
tests pin the contract by hand: frame time, zone membership, the
remove/reset flags, documented V2 differences and explicit source and track
capacity.

``ZONE`` is the square ``0..100``; a row centred at ``IN`` is inside it and
one centred at ``OUT`` is not.
"""

import math
from datetime import datetime, timedelta
from typing import Any, List, Optional, Sequence

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.analytics.time_in_zone import v3 as time_in_zone
from roboflow_workflows.v2.blocks.analytics.time_in_zone.v3 import MAX_TRACKED_SOURCES, TimeInZoneBlock, TimeInZoneCapacityError
from roboflow_workflows.v2.data.batch import SampleContext
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks.grid import SegmentationView
from roboflow_workflows.v2.engine.context import ExecutionContext, use_execution_context

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "mps",
        id="mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="needs MPS"
        ),
    ),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
]
T0 = datetime(2026, 10, 13)
ZONE = [[0, 0], [0, 100], [100, 100], [100, 0]]
OTHER_ZONE = [[200, 0], [200, 100], [300, 100], [300, 0]]
IN, OUT, IN_OTHER = (50.0, 50.0), (150.0, 50.0), (250.0, 50.0)
UNSET = object()


class _Scope:
    """Call scope where only the image carries a sample context."""

    def __init__(self, image_sample: Optional[SampleContext]) -> None:
        self._image_sample = image_sample

    def sample_at(self, index, *, field=None):
        return self._image_sample if field == "image" else None


def _file_frame(number: int, *, fps: Any = 10.0, video: str = "cam-a") -> ImageData:
    """Frame of a video file; its timestamp is deliberately far from number/fps."""
    metadata = VideoMetadata(
        video_identifier=video,
        frame_number=number,
        frame_timestamp=T0 + timedelta(seconds=1000 + number),
        fps=fps,
        comes_from_video_file=True,
    )
    image = ImageData.from_tensor(
        torch.zeros((3, 4, 4), dtype=torch.uint8),
        image_id=f"{video}-{number}",
        video_metadata=metadata,
    )

    return image


def _stream_frame(seconds: float, *, number: int = 1, video: str = "cam-a"):
    metadata = VideoMetadata(
        video_identifier=video,
        frame_number=number,
        frame_timestamp=T0 + timedelta(seconds=seconds),
        fps=30.0,
        comes_from_video_file=False,
    )
    image = ImageData.from_tensor(
        torch.zeros((3, 4, 4), dtype=torch.uint8),
        image_id=f"{video}-{number}",
        video_metadata=metadata,
    )

    return image


def _tracked(
    centres: Sequence[tuple],
    tracker_ids: Sequence[Any],
    *,
    device: str = "cpu",
) -> Detections:
    """One 10 x 10 box per row centred at each ``(x, y)``."""
    xyxy = torch.tensor(
        [[x - 5, y - 5, x + 5, y + 5] for x, y in centres], dtype=torch.float32
    ).reshape(-1, 4)
    detections = Detections(
        xyxy=xyxy.to(device),
        class_id=torch.zeros(len(centres), dtype=torch.int64, device=device),
        confidence=torch.full((len(centres),), 0.5, device=device),
        image_metadata={"class_names": {0: "person"}},
        bboxes_metadata=[
            {"detection_id": f"d{row}", "tracker_id": tracker_id}
            for row, tracker_id in enumerate(tracker_ids)
        ],
    )

    return detections


def _call(
    block: TimeInZoneBlock,
    predictions: Any,
    *,
    frame: Optional[ImageData] = None,
    zone: Any = ZONE,
    anchor: str = "CENTER",
    remove: Any = True,
    reset: Any = True,
    source: Any = UNSET,
    max_sources: Any = UNSET,
) -> dict:
    """One call; ``source`` runs it inside an engine context."""
    arguments = {
        "image": frame or _file_frame(1),
        "predictions": predictions,
        "zone": zone,
        "triggering_anchor": anchor,
        "remove_out_of_zone_detections": remove,
        "reset_out_of_zone_detections": reset,
    }
    if max_sources is not UNSET:
        arguments["max_sources"] = max_sources
    if source is UNSET:
        result = block.run(**arguments)
        return result

    sample = SampleContext(source_id=source) if source is not None else None
    context = ExecutionContext(
        step_path=("timer",),
        block_type=TimeInZoneBlock.type,
        session_id="session",
        run_id="run",
        indices=((0,),),
        call_scope=_Scope(sample),
    )
    with use_execution_context(context):
        result = block.run(**arguments)

    return result


def _times(result: dict) -> List[Optional[float]]:
    timed = result["timed_detections"]
    rows = timed.detections if isinstance(timed, SegmentationView) else timed
    times = [row.get("time_in_zone") for row in rows.bboxes_metadata or []]

    return [None if t is None else round(t, 9) for t in times]


def _ids(result: dict) -> List[int]:
    rows = result["timed_detections"]
    tracker_ids = [row["tracker_id"] for row in rows.bboxes_metadata or []]

    return tracker_ids


def _sequence(steps: Sequence[tuple], **kwargs: Any) -> List[List[Optional[float]]]:
    """``(frame number, centres, tracker ids)`` per call on one 10 fps file."""
    block = TimeInZoneBlock()
    results = [
        _times(_call(block, _tracked(centres, ids), frame=_file_frame(n), **kwargs))
        for n, centres, ids in steps
    ]

    return results


def _params(**overrides: Any) -> TimeInZoneBlock.Params:
    fields = {
        "image": "$inputs.image",
        "predictions": "$steps.tracker.tracked_detections",
        "zone": ZONE,
    }
    params = TimeInZoneBlock.Params(**{**fields, **overrides})

    return params


def _retained_tracks(block: TimeInZoneBlock, source: str = "cam-a") -> List[int]:
    # White box: the tracks whose entry time the source still remembers.
    tracks = list(block._entries[source])

    return tracks


# -------------------------------------------------------------- declaration
def test_declaration_keeps_v1_output_and_adds_count_and_capacity():
    assert TimeInZoneBlock.type == "roboflow_core/time_in_zone@v3"
    assert set(TimeInZoneBlock.outputs) == {"timed_detections", "count_in_zone"}
    assert TimeInZoneBlock.__block_spec__.prunable is False
    params = _params()
    assert params.triggering_anchor == "CENTER"
    assert params.remove_out_of_zone_detections is True
    assert params.reset_out_of_zone_detections is True
    assert params.max_sources == MAX_TRACKED_SOURCES == 256


@pytest.mark.parametrize(
    "zone",
    [[[0, 0], [1, 1]], [ZONE, [[0, 0], [1, 1]]], [[0, 0], [1, 1], [2, None]], "zone"],
)
def test_invalid_literal_zone_fails_at_compile_time(zone):
    with pytest.raises(ValidationError):
        _params(zone=zone)


@pytest.mark.parametrize(
    "field, value",
    [
        ("triggering_anchor", "CENTER_OF_MASS"),
        ("remove_out_of_zone_detections", "yes"),
        ("reset_out_of_zone_detections", 1),
    ],
)
def test_invalid_literal_anchor_or_flag_fails_at_compile_time(field, value):
    with pytest.raises(ValidationError):
        _params(**{field: value})


def test_zone_anchor_and_flags_may_be_selectors():
    params = _params(
        zone="$inputs.zone",
        triggering_anchor="$inputs.anchor",
        remove_out_of_zone_detections="$inputs.remove",
        reset_out_of_zone_detections="$inputs.reset",
    )

    assert params.zone == "$inputs.zone"


@pytest.mark.parametrize("limit", [0, -1, True, 2.0, "2", "$inputs.limit"])
def test_source_capacity_is_a_positive_literal_integer(limit):
    with pytest.raises(ValidationError):
        _params(max_sources=limit)

    with pytest.raises(ValueError, match="max_sources must be a positive integer"):
        _call(TimeInZoneBlock(), _tracked([IN], [1]), max_sources=limit)


# ---------------------------------------------------------------- timing
def test_docstring_example_on_a_10_fps_file():
    steps = [(n, [OUT if n == 6 else IN], [1]) for n in (3, 4, 5, 6, 7)]

    assert _sequence(steps) == [[0.0], [0.1], [0.2], [], [0.0]]


def test_streams_use_the_frame_timestamp():
    block = TimeInZoneBlock()
    seconds = [0.0, 0.25, 1.5]

    results = [
        _call(block, _tracked([IN], [1]), frame=_stream_frame(s, number=1))
        for s in seconds
    ]

    assert [_times(r) for r in results] == [[0.0], [0.25], [1.5]]


@pytest.mark.parametrize("fps", [None, 0.0, -5.0, math.nan, math.inf])
def test_video_file_without_a_usable_fps_uses_the_frame_timestamp(fps):
    block = TimeInZoneBlock()
    _call(block, _tracked([IN], [1]), frame=_file_frame(1, fps=fps))

    result = _call(block, _tracked([IN], [1]), frame=_file_frame(4, fps=fps))

    assert _times(result) == [3.0]


def test_image_without_video_metadata_fails():
    image = ImageData.from_tensor(torch.zeros((3, 4, 4), dtype=torch.uint8))

    with pytest.raises(ValueError, match="needs video_metadata"):
        _call(TimeInZoneBlock(), _tracked([IN], [1]), frame=image)


def test_restarted_file_restarts_the_entry_instead_of_negative_time():
    steps = [(10, [IN], [1]), (12, [IN], [1]), (2, [IN], [1]), (5, [IN], [1])]

    assert _sequence(steps) == [[0.0], [0.2], [0.0], [0.3]]


def test_times_are_python_floats_in_seconds():
    block = TimeInZoneBlock()
    _call(block, _tracked([IN], [1]), frame=_file_frame(1))

    result = _call(block, _tracked([IN], [1]), frame=_file_frame(16))

    value = result["timed_detections"].bboxes_metadata[0]["time_in_zone"]
    assert type(value) is float and value == pytest.approx(1.5)


# ---------------------------------------------------------- zone membership
def test_a_row_in_any_polygon_is_in_zone_and_entries_are_per_track():
    steps = [(1, [IN], [1]), (2, [IN_OTHER], [1]), (3, [OUT], [1])]

    times = _sequence(steps, zone=[ZONE, OTHER_ZONE])

    # Moving between polygons keeps the entry: zones are one union.
    assert times == [[0.0], [0.1], []]


def test_an_empty_zone_list_has_no_row_in_zone():
    block = TimeInZoneBlock()

    removed = _call(block, _tracked([IN], [1]), zone=[])
    kept = _call(block, _tracked([IN], [1]), zone=[], remove=False)

    assert _times(removed) == [] and removed["count_in_zone"] == 0
    assert _times(kept) == [0.0] and kept["count_in_zone"] == 0


def test_the_triggering_anchor_must_be_inside():
    block = TimeInZoneBlock()
    # Centre inside, bottom edge at y = 105 outside.
    row = _tracked([(50.0, 100.0)], [1])

    centre = _call(block, row)
    bottom = _call(TimeInZoneBlock(), row, anchor="BOTTOM_CENTER")

    assert centre["count_in_zone"] == 1 and bottom["count_in_zone"] == 0


def test_zone_may_change_per_frame():
    block = TimeInZoneBlock()
    _call(block, _tracked([IN], [1]), frame=_file_frame(1))

    result = _call(
        block, _tracked([IN], [1]), frame=_file_frame(3), zone=[ZONE, OTHER_ZONE]
    )

    assert _times(result) == [0.2]


def test_anchor_may_change_per_frame_keeping_entries():
    block = TimeInZoneBlock()
    # Box y 95..105 over the zone's bottom edge at y = 100.
    row = [(50.0, 100.0)]
    _call(block, _tracked(row, [1]), frame=_file_frame(1))

    top = _call(block, _tracked(row, [1]), frame=_file_frame(3), anchor="TOP_CENTER")
    bottom = _call(
        block, _tracked(row, [1]), frame=_file_frame(4), anchor="BOTTOM_CENTER"
    )

    assert _times(top) == [0.2]
    assert bottom["count_in_zone"] == 0


# ------------------------------------------------------ remove / reset flags
LEAVE_AND_RETURN = [
    (1, [IN, OUT], [1, 2]),
    (2, [OUT, OUT], [1, 2]),
    (3, [IN, OUT], [1, 2]),
]


@pytest.mark.parametrize(
    "remove, reset, expected",
    [
        (True, True, [[0.0], [], [0.0]]),
        (True, False, [[0.0], [], [0.2]]),
        (False, True, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        # V2 difference: V1 reset the entry here anyway and reported 0.0.
        (False, False, [[0.0, 0.0], [0.0, 0.0], [0.2, 0.0]]),
    ],
)
def test_remove_and_reset_flags(remove, reset, expected):
    assert _sequence(LEAVE_AND_RETURN, remove=remove, reset=reset) == expected


def test_count_in_zone_counts_rows_in_zone_whatever_is_output():
    block = TimeInZoneBlock()
    rows = _tracked([IN, OUT, IN], [1, 2, 3])

    removed = _call(block, rows)
    kept = _call(TimeInZoneBlock(), rows, remove=False)

    assert removed["count_in_zone"] == kept["count_in_zone"] == 2
    assert _ids(removed) == [1, 3] and _ids(kept) == [1, 2, 3]


def test_runtime_flags_must_be_booleans():
    with pytest.raises(ValueError, match="remove_out_of_zone_detections must be"):
        _call(TimeInZoneBlock(), _tracked([IN], [1]), remove="false")
    with pytest.raises(ValueError, match="reset_out_of_zone_detections must be"):
        _call(TimeInZoneBlock(), _tracked([IN], [1]), reset=0)


# ---------------------------------------------------------- rows and families
def test_input_rows_are_not_modified_and_metadata_is_copied():
    block = TimeInZoneBlock()
    rows = _tracked([IN, OUT], [1, 2])

    result = _call(block, rows, remove=False)

    assert rows.bboxes_metadata == [
        {"detection_id": "d0", "tracker_id": 1},
        {"detection_id": "d1", "tracker_id": 2},
    ]
    assert result["timed_detections"].bboxes_metadata == [
        {"detection_id": "d0", "tracker_id": 1, "time_in_zone": 0.0},
        {"detection_id": "d1", "tracker_id": 2, "time_in_zone": 0.0},
    ]
    assert torch.equal(result["timed_detections"].xyxy, rows.xyxy)


@pytest.mark.parametrize("device", DEVICES)
def test_rows_stay_on_the_boxes_device(device):
    block = TimeInZoneBlock()
    _call(block, _tracked([IN, OUT], [1, 2], device=device), frame=_file_frame(1))

    result = _call(
        block, _tracked([IN, OUT], [1, 2], device=device), frame=_file_frame(2)
    )

    assert result["timed_detections"].xyxy.device.type == device
    assert _ids(result) == [1] and _times(result) == [0.1]


def _segmented(centres: Sequence[tuple], tracker_ids: Sequence[int]):
    boxes = _tracked(centres, tracker_ids)
    masks = torch.zeros((len(centres), 8, 8), dtype=torch.bool)
    for row in range(len(centres)):
        masks[row, row, row] = True
    dense = InstanceDetections(
        xyxy=boxes.xyxy,
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        mask=masks,
        image_metadata=boxes.image_metadata,
        bboxes_metadata=boxes.bboxes_metadata,
    )

    return dense


def test_instance_masks_stay_aligned_with_kept_rows():
    predictions = _segmented([OUT, IN], [1, 2])

    result = _call(TimeInZoneBlock(), predictions)

    timed = result["timed_detections"]
    assert isinstance(timed, InstanceDetections)
    assert torch.equal(timed.mask, predictions.mask[1:2])
    assert _times(result) == [0.0]


def test_segmentation_view_stays_a_view_without_materializing():
    view = SegmentationView.from_dense(_segmented([OUT, IN], [1, 2]))

    result = _call(TimeInZoneBlock(), view)

    timed = result["timed_detections"]
    assert isinstance(timed, SegmentationView)
    assert timed.materialization_count == 0
    assert _times(result) == [0.0]
    assert view.detections.bboxes_metadata[1] == {"detection_id": "d1", "tracker_id": 2}


# ------------------------------------------------------------------- state
def test_sources_are_timed_separately():
    block = TimeInZoneBlock()
    _call(block, _tracked([IN], [1]), frame=_file_frame(1, video="cam-a"))

    result_b = _call(block, _tracked([IN], [1]), frame=_file_frame(5, video="cam-b"))
    result_a = _call(block, _tracked([IN], [1]), frame=_file_frame(5, video="cam-a"))

    assert _times(result_b) == [0.0] and _times(result_a) == [0.4]


def test_source_falls_back_to_sample_source_then_default():
    block = TimeInZoneBlock()
    unnamed = _file_frame(1, video="")

    _call(block, _tracked([IN], [1]), frame=unnamed, source="rtsp-1")
    _call(block, _tracked([IN], [1]), frame=unnamed)

    assert set(block._entries) == {"rtsp-1", "default"}


def test_empty_prediction_is_returned_as_given_and_keeps_entries():
    block = TimeInZoneBlock()
    _call(block, _tracked([IN], [1]), frame=_file_frame(1))
    empty = _tracked([], [])

    result = _call(block, empty, frame=_file_frame(2))
    after = _call(block, _tracked([IN], [1]), frame=_file_frame(3))

    assert result["timed_detections"] is empty and result["count_in_zone"] == 0
    assert _times(after) == [0.2]


def test_rows_without_tracker_id_fail_without_changing_state():
    block = TimeInZoneBlock()
    _call(block, _tracked([IN], [1]), frame=_file_frame(1))

    with pytest.raises(ValueError, match="needs tracked predictions"):
        _call(block, _tracked([OUT, IN], [1, None]), frame=_file_frame(2))

    assert _times(_call(block, _tracked([IN], [1]), frame=_file_frame(3))) == [0.2]


def test_reset_state_forgets_entries_but_keeps_cached_masks():
    block = TimeInZoneBlock()
    _call(block, _tracked([IN], [1]), frame=_file_frame(1))

    block.reset_state()
    result = _call(block, _tracked([IN], [1]), frame=_file_frame(3))

    assert _times(result) == [0.0]
    assert len(block._masks) == 1


def test_polygon_mask_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(time_in_zone, "ZONE_CACHE_SIZE", 2)
    block = TimeInZoneBlock()
    zones = [[[0, 0], [0, 100], [100 + k, 100], [100 + k, 0]] for k in range(3)]

    for zone in zones:
        _call(block, _tracked([IN], [1]), zone=zone)

    assert len(block._masks) == 2


# ---------------------------------------------------------------- capacity
def test_retained_entries_keep_the_most_recently_seen_tracks(monkeypatch):
    monkeypatch.setattr(time_in_zone, "MAX_TRACKS_PER_SOURCE", 3)
    block = TimeInZoneBlock()
    _call(block, _tracked([IN] * 3, [1, 2, 3]), frame=_file_frame(1))
    _call(block, _tracked([IN, IN], [4, 2]), frame=_file_frame(2))

    assert _retained_tracks(block) == [3, 4, 2]
    # Track 1 was forgotten: it enters again; track 3 keeps its entry.
    result = _call(block, _tracked([IN, IN], [1, 3]), frame=_file_frame(4))

    assert _times(result) == [0.0, 0.3]
    assert _retained_tracks(block) == [2, 1, 3]


def test_kept_entry_of_a_track_seen_outside_counts_as_recent(monkeypatch):
    monkeypatch.setattr(time_in_zone, "MAX_TRACKS_PER_SOURCE", 2)
    block = TimeInZoneBlock()
    _call(block, _tracked([IN, IN], [1, 2]), frame=_file_frame(1), reset=False)

    # Track 1 is seen outside; track 3 enters. Absent track 2 is forgotten.
    _call(block, _tracked([OUT, IN], [1, 3]), frame=_file_frame(2), reset=False)
    result = _call(block, _tracked([IN], [1]), frame=_file_frame(3), reset=False)

    assert _retained_tracks(block) == [3, 1]
    assert _times(result) == [0.2]


def test_a_call_with_more_tracks_than_the_bound_fails_before_any_change(monkeypatch):
    monkeypatch.setattr(time_in_zone, "MAX_TRACKS_PER_SOURCE", 3)
    block = TimeInZoneBlock()
    _call(block, _tracked([IN] * 3, [1, 2, 3]), frame=_file_frame(1))

    with pytest.raises(TimeInZoneCapacityError, match="MAX_TRACKS_PER_SOURCE=3"):
        _call(block, _tracked([OUT] * 4, [1, 2, 3, 4]), frame=_file_frame(2))
    with pytest.raises(TimeInZoneCapacityError, match="4 distinct tracker ids"):
        _call(block, _tracked([IN] * 4, [1, 2, 3, 4]), frame=_file_frame(2, video="b"))

    assert set(block._entries) == {"cam-a"}
    assert _retained_tracks(block) == [1, 2, 3]
    result = _call(block, _tracked([IN] * 4, [1, 2, 3, 3]), frame=_file_frame(3))
    assert _times(result) == [0.2, 0.2, 0.2, 0.2]


def test_source_capacity_failure_changes_nothing_and_keeps_every_source():
    block = TimeInZoneBlock()
    for video in ("cam-a", "cam-b"):
        _call(
            block,
            _tracked([IN], [1]),
            frame=_file_frame(1, video=video),
            max_sources=2,
        )
    rejected = _file_frame(1, video="cam-c")

    with pytest.raises(TimeInZoneCapacityError, match="max_sources=2"):
        _call(block, _tracked([IN], [1]), frame=rejected, max_sources=2)
    with pytest.raises(TimeInZoneCapacityError, match="'cam-c'"):
        _call(block, _tracked([], []), frame=rejected, max_sources=2)

    assert set(block._entries) == {"cam-a", "cam-b"}
    for video in ("cam-a", "cam-b"):
        result = _call(
            block,
            _tracked([IN], [1]),
            frame=_file_frame(3, video=video),
            max_sources=2,
        )
        assert _times(result) == [0.2]


def test_source_capacity_is_freed_by_reset():
    block = TimeInZoneBlock()
    _call(block, _tracked([IN], [1]), max_sources=1)

    with pytest.raises(TimeInZoneCapacityError):
        _call(
            block, _tracked([IN], [1]), frame=_file_frame(1, video="b"), max_sources=1
        )

    block.reset_state()
    _call(block, _tracked([IN], [1]), frame=_file_frame(1, video="b"), max_sources=1)
    assert set(block._entries) == {"b"}
