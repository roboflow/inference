"""``v2/line_counter@v2`` called directly.

The oracle is latest V1 ``roboflow_core/line_counter@v2`` (tensor variant);
the task's randomized parity probe compares whole sequences with it. These
tests pin the contract by hand: crossing rule, per-source state, documented
V2 differences and explicit source and track capacity.

Line ``LINE`` is horizontal at ``y = 50`` drawn left to right, so a row
above it is on side +1 and moving up crosses in.
"""

from datetime import datetime, timedelta
from typing import Any, List, Optional, Sequence

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.analytics import line_counter
from roboflow_workflows.v2.blocks.analytics.line_counter import (
    MAX_TRACKED_SOURCES,
    LineCounterBlock,
    LineCounterCapacityError,
)
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
LINE = [[0, 50], [100, 50]]
ABOVE, BELOW = 30.0, 70.0
UNSET = object()


class _Scope:
    """Call scope where only the image carries a sample context."""

    def __init__(self, image_sample: Optional[SampleContext]) -> None:
        self._image_sample = image_sample

    def sample_at(self, index, *, field=None):
        return self._image_sample if field == "image" else None


def _frame(number: int = 1, *, video: str = "cam-a") -> ImageData:
    metadata = VideoMetadata(
        video_identifier=video,
        frame_number=number,
        frame_timestamp=T0 + timedelta(seconds=number),
    )
    image = ImageData.from_tensor(
        torch.zeros((3, 4, 4), dtype=torch.uint8),
        image_id=f"{video}-{number}",
        video_metadata=metadata,
    )

    return image


def _tracked(
    centres_y: Sequence[float],
    tracker_ids: Sequence[Any],
    *,
    x: float = 50.0,
    device: str = "cpu",
) -> Detections:
    """One 10 x 10 box per row centred at ``(x, y)``."""
    xyxy = torch.tensor(
        [[x - 5, y - 5, x + 5, y + 5] for y in centres_y], dtype=torch.float32
    ).reshape(-1, 4)
    detections = Detections(
        xyxy=xyxy.to(device),
        class_id=torch.zeros(len(centres_y), dtype=torch.int64, device=device),
        confidence=torch.full((len(centres_y),), 0.5, device=device),
        image_metadata={"class_names": {0: "car"}},
        bboxes_metadata=[
            {"detection_id": f"d{row}", "tracker_id": tracker_id}
            for row, tracker_id in enumerate(tracker_ids)
        ],
    )

    return detections


def _call(
    block: LineCounterBlock,
    predictions: Any,
    *,
    frame: Optional[ImageData] = None,
    line: Any = LINE,
    anchor: Optional[str] = "CENTER",
    source: Any = UNSET,
    max_sources: Any = UNSET,
) -> dict:
    """One call; ``source`` runs it inside an engine context."""
    arguments = {
        "image": frame or _frame(),
        "predictions": predictions,
        "line_segment": line,
        "triggering_anchor": anchor,
    }
    if max_sources is not UNSET:
        arguments["max_sources"] = max_sources
    if source is UNSET:
        result = block.run(**arguments)
        return result

    sample = SampleContext(source_id=source) if source is not None else None
    context = ExecutionContext(
        step_path=("counter",),
        block_type=LineCounterBlock.type,
        session_id="session",
        run_id="run",
        indices=((0,),),
        call_scope=_Scope(sample),
    )
    with use_execution_context(context):
        result = block.run(**arguments)

    return result


def _counts(result: dict) -> tuple:
    return result["count_in"], result["count_out"]


def _ids(prediction: Any) -> List[int]:
    rows = (
        prediction.detections
        if isinstance(prediction, SegmentationView)
        else prediction
    )
    tracker_ids = [row["tracker_id"] for row in rows.bboxes_metadata or []]

    return tracker_ids


def _params(**overrides: Any) -> LineCounterBlock.Params:
    fields = {
        "image": "$inputs.image",
        "predictions": "$steps.tracker.tracked_detections",
        "line_segment": LINE,
    }
    params = LineCounterBlock.Params(**{**fields, **overrides})

    return params


def _retained_tracks(block: LineCounterBlock, source: str = "cam-a") -> List[int]:
    # White box: the tracks whose last side the source still remembers.
    tracks = list(block._sources[source].sides)

    return tracks


# -------------------------------------------------------------- declaration
def test_declaration_keeps_v1_outputs_and_adds_capacity():
    assert LineCounterBlock.type == "v2/line_counter@v2"
    assert set(LineCounterBlock.outputs) == {
        "count_in",
        "count_out",
        "detections_in",
        "detections_out",
    }
    assert LineCounterBlock.__block_spec__.prunable is False
    assert _params().max_sources == MAX_TRACKED_SOURCES == 256
    assert _params().triggering_anchor is None


@pytest.mark.parametrize(
    "line", [[[0, 0]], [[0, 0], [0, 0]], [[0, 0], [1, True]], "not a ref"]
)
def test_invalid_literal_line_fails_at_compile_time(line):
    with pytest.raises(ValidationError):
        _params(line_segment=line)


def test_invalid_literal_anchor_fails_at_compile_time():
    with pytest.raises(ValidationError):
        _params(triggering_anchor="CENTER_OF_MASS")


def test_line_and_anchor_may_be_selectors():
    params = _params(line_segment="$inputs.line", triggering_anchor="$inputs.anchor")

    assert params.line_segment == "$inputs.line"


@pytest.mark.parametrize("limit", [0, -1, True, 2.0, "2", "$inputs.limit"])
def test_source_capacity_is_a_positive_literal_integer(limit):
    with pytest.raises(ValidationError):
        _params(max_sources=limit)

    with pytest.raises(ValueError, match="max_sources must be a positive integer"):
        _call(LineCounterBlock(), _tracked([ABOVE], [1]), max_sources=limit)


# ------------------------------------------------------------------ counting
def test_docstring_example_counts_out_then_in():
    block = LineCounterBlock()
    results = [
        _call(block, _tracked([y], [7]), frame=_frame(number))
        for number, y in enumerate([40, 60, 60, 45], start=1)
    ]

    assert [_counts(r) for r in results] == [(0, 0), (0, 1), (0, 1), (1, 1)]
    assert [_ids(r["detections_out"]) for r in results] == [[], [7], [], []]
    assert [_ids(r["detections_in"]) for r in results] == [[], [], [], [7]]


def test_first_recorded_side_never_counts():
    block = LineCounterBlock()

    result = _call(block, _tracked([BELOW], [1]))

    assert _counts(result) == (0, 0)


def test_without_anchor_the_whole_box_must_cross():
    block = LineCounterBlock()
    centres = [40.0, 52.0, 56.0]  # box 10 high: straddles at 52, crossed at 56

    results = [
        _call(block, _tracked([y], [1]), anchor=None, frame=_frame(n))
        for n, y in enumerate(centres)
    ]

    assert [_counts(r) for r in results] == [(0, 0), (0, 0), (0, 1)]


def test_rows_beside_the_segment_are_not_recorded():
    block = LineCounterBlock()

    _call(block, _tracked([ABOVE], [1], x=150.0))
    result = _call(block, _tracked([BELOW], [1], x=150.0))

    assert _counts(result) == (0, 0)


def test_crossing_rows_keep_input_order_and_metadata():
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE, BELOW, ABOVE, BELOW], [1, 2, 3, 4]))
    predictions = _tracked([BELOW, ABOVE, BELOW, ABOVE], [1, 2, 3, 4])

    result = _call(block, predictions)

    assert _counts(result) == (2, 2)
    assert _ids(result["detections_out"]) == [1, 3]
    assert _ids(result["detections_in"]) == [2, 4]
    assert result["detections_in"].bboxes_metadata[0]["detection_id"] == "d1"
    assert predictions.bboxes_metadata[0] == {"detection_id": "d0", "tracker_id": 1}


def test_a_tracker_id_repeated_in_one_call_is_recorded_per_row_like_v1():
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE], [1]))

    result = _call(block, _tracked([BELOW, ABOVE], [1, 1]))

    assert _counts(result) == (1, 1)


def test_tracker_ids_may_be_numeric_strings_or_floats():
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE], ["5"]))

    result = _call(block, _tracked([BELOW], [5.0]))

    assert _counts(result) == (0, 1)


@pytest.mark.parametrize("device", DEVICES)
def test_crossing_rows_stay_on_the_boxes_device(device):
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE], [1], device=device))

    result = _call(block, _tracked([BELOW], [1], device=device))

    assert result["detections_out"].xyxy.device.type == device
    assert result["detections_out"].xyxy.tolist() == [[45.0, 65.0, 55.0, 75.0]]


# -------------------------------------------------------------- families
def _segmented(centres_y: Sequence[float], tracker_ids: Sequence[int]):
    boxes = _tracked(centres_y, tracker_ids)
    masks = torch.zeros((len(centres_y), 8, 8), dtype=torch.bool)
    for row in range(len(centres_y)):
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


def test_instance_masks_stay_aligned_with_crossing_rows():
    block = LineCounterBlock()
    _call(block, _segmented([ABOVE, ABOVE], [1, 2]))
    predictions = _segmented([ABOVE, BELOW], [1, 2])

    result = _call(block, predictions)

    crossed = result["detections_out"]
    assert isinstance(crossed, InstanceDetections)
    assert _ids(crossed) == [2]
    assert torch.equal(crossed.mask, predictions.mask[1:2])


def test_segmentation_view_stays_a_view_without_materializing():
    block = LineCounterBlock()
    _call(block, SegmentationView.from_dense(_segmented([ABOVE, ABOVE], [1, 2])))
    view = SegmentationView.from_dense(_segmented([BELOW, ABOVE], [1, 2]))

    result = _call(block, view)

    crossed = result["detections_out"]
    assert isinstance(crossed, SegmentationView)
    assert _ids(crossed) == [1]
    assert crossed.materialization_count == 0
    assert view.materialization_count == 0


# ------------------------------------------------------------------- state
def test_sources_are_counted_separately_by_video_identifier():
    block = LineCounterBlock()
    for video in ("cam-a", "cam-b"):
        _call(block, _tracked([ABOVE], [1]), frame=_frame(1, video=video))

    result_a = _call(block, _tracked([BELOW], [1]), frame=_frame(2, video="cam-a"))
    result_b = _call(block, _tracked([ABOVE], [1]), frame=_frame(2, video="cam-b"))

    assert _counts(result_a) == (0, 1)
    assert _counts(result_b) == (0, 0)


def test_source_falls_back_to_sample_source_then_default():
    block = LineCounterBlock()
    unnamed = _frame(1, video="")
    _call(block, _tracked([ABOVE], [1]), frame=unnamed, source="rtsp-1")
    _call(block, _tracked([ABOVE], [1]), frame=unnamed)

    in_context = _call(block, _tracked([BELOW], [1]), frame=unnamed, source="rtsp-1")
    no_sample = _call(block, _tracked([BELOW], [1]), frame=unnamed, source=None)

    assert set(block._sources) == {"rtsp-1", "default"}
    assert _counts(in_context) == (0, 1)
    assert _counts(no_sample) == (0, 1)


def test_empty_prediction_passes_through_and_keeps_sides():
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE], [1]))
    empty = _tracked([], [])

    result = _call(block, empty)
    after = _call(block, _tracked([BELOW], [1]))

    assert result["detections_in"] is empty and result["detections_out"] is empty
    assert _counts(result) == (0, 0)
    assert _counts(after) == (0, 1)


def test_rows_without_tracker_id_fail_without_changing_state():
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE], [1]))
    untracked = _tracked([BELOW, BELOW], [1, None])

    with pytest.raises(ValueError, match="needs tracked predictions"):
        _call(block, untracked)

    assert _counts(_call(block, _tracked([BELOW], [1]))) == (0, 1)


def test_rows_without_any_metadata_fail():
    predictions = _tracked([ABOVE], [1])
    untracked = Detections(
        xyxy=predictions.xyxy,
        class_id=predictions.class_id,
        confidence=predictions.confidence,
    )

    with pytest.raises(ValueError, match="needs tracked predictions"):
        _call(LineCounterBlock(), untracked)


def test_runtime_line_and_anchor_are_checked():
    block = LineCounterBlock()

    with pytest.raises(ValueError, match="exactly 2 points"):
        _call(block, _tracked([ABOVE], [1]), line=[[0, 0]])
    with pytest.raises(ValueError, match="triggering_anchor must be one of"):
        _call(block, _tracked([ABOVE], [1]), anchor="MIDDLE")

    assert block._sources == {}


def test_changed_line_restarts_the_source_counts_and_history():
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE], [1]))
    assert _counts(_call(block, _tracked([BELOW], [1]))) == (0, 1)

    moved = [[0, 90], [100, 90]]
    first = _call(block, _tracked([ABOVE], [1]), line=moved)
    second = _call(block, _tracked([95.0], [1]), line=moved)

    assert _counts(first) == (0, 0)
    assert _counts(second) == (0, 1)


def test_same_line_given_as_tuples_or_floats_keeps_the_counts():
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE], [1]))
    _call(block, _tracked([BELOW], [1]))

    result = _call(block, _tracked([BELOW], [1]), line=((0.0, 50.0), (100.0, 50.0)))

    assert _counts(result) == (0, 1)


def test_reset_state_forgets_every_source():
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE], [1]))
    _call(block, _tracked([BELOW], [1]))

    block.reset_state()
    result = _call(block, _tracked([ABOVE], [1]))

    assert _counts(result) == (0, 0)


# ---------------------------------------------------------------- capacity
def test_retained_history_keeps_the_most_recently_recorded_tracks(monkeypatch):
    monkeypatch.setattr(line_counter, "MAX_TRACKS_PER_SOURCE", 3)
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE] * 3, [1, 2, 3]))
    _call(block, _tracked([ABOVE, ABOVE], [4, 2]))

    assert _retained_tracks(block) == [3, 4, 2]
    # Track 1 was forgotten: its next side is a first side again.
    result = _call(block, _tracked([BELOW, BELOW], [1, 3]))

    assert _counts(result) == (0, 1)
    assert _ids(result["detections_out"]) == [3]
    assert len(_retained_tracks(block)) == 3


def test_rows_beside_the_segment_take_no_history_slot(monkeypatch):
    monkeypatch.setattr(line_counter, "MAX_TRACKS_PER_SOURCE", 2)
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE, ABOVE], [1, 2]))

    _call(block, _tracked([ABOVE], [3], x=150.0))

    assert _retained_tracks(block) == [1, 2]


def test_a_call_with_more_tracks_than_the_bound_fails_before_any_change(monkeypatch):
    monkeypatch.setattr(line_counter, "MAX_TRACKS_PER_SOURCE", 3)
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE] * 3, [1, 2, 3]))

    with pytest.raises(LineCounterCapacityError, match="MAX_TRACKS_PER_SOURCE=3"):
        _call(block, _tracked([BELOW] * 4, [1, 2, 3, 4]))
    with pytest.raises(LineCounterCapacityError, match="4 distinct tracker ids"):
        _call(block, _tracked([BELOW] * 4, [1, 2, 3, 4]), frame=_frame(video="new"))

    assert set(block._sources) == {"cam-a"}
    assert _retained_tracks(block) == [1, 2, 3]
    # Exactly the bound, repeated ids included, is accepted.
    result = _call(block, _tracked([BELOW] * 4, [1, 2, 3, 3]))
    assert _counts(result) == (0, 3)


def test_source_capacity_failure_changes_nothing_and_keeps_every_source():
    block = LineCounterBlock()
    for video in ("cam-a", "cam-b"):
        _call(block, _tracked([ABOVE], [1]), frame=_frame(video=video), max_sources=2)
    rejected = _frame(video="cam-c")

    with pytest.raises(LineCounterCapacityError, match="max_sources=2"):
        _call(block, _tracked([ABOVE], [1]), frame=rejected, max_sources=2)
    with pytest.raises(LineCounterCapacityError, match="'cam-c'"):
        _call(block, _tracked([], []), frame=rejected, max_sources=2)

    assert set(block._sources) == {"cam-a", "cam-b"}
    for video in ("cam-a", "cam-b"):
        result = _call(
            block, _tracked([BELOW], [1]), frame=_frame(video=video), max_sources=2
        )
        assert _counts(result) == (0, 1)


def test_source_capacity_is_read_per_call_and_freed_by_reset():
    block = LineCounterBlock()
    _call(block, _tracked([ABOVE], [1]), max_sources=1)
    _call(block, _tracked([ABOVE], [1]), frame=_frame(video="cam-b"), max_sources=2)

    with pytest.raises(LineCounterCapacityError):
        _call(block, _tracked([ABOVE], [1]), frame=_frame(video="cam-c"), max_sources=2)

    block.reset_state()
    _call(block, _tracked([ABOVE], [1]), frame=_frame(video="cam-c"), max_sources=1)
    assert set(block._sources) == {"cam-c"}


def test_default_source_capacity_accepts_256_sources():
    block = LineCounterBlock()
    for camera in range(MAX_TRACKED_SOURCES):
        _call(block, _tracked([], []), frame=_frame(video=f"cam-{camera}"))

    with pytest.raises(LineCounterCapacityError, match="max_sources=256"):
        _call(block, _tracked([], []), frame=_frame(video="one-too-many"))
