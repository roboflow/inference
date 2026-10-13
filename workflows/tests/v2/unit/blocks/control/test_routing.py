"""``v2/switch_case@v1``, ``v2/delta_filter@v1`` and ``v2/rate_limiter@v1``.

Calls run inside a hand-built execution context whose scope answers the
source and observation time of the call's index, as the engine's does.
"""

import threading
from datetime import datetime
from fractions import Fraction
from typing import Any, Optional

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.control.routing import (
    ClockMismatchError,
    DeltaFilterBlock,
    RateLimiterBlock,
    SourceCapacityError,
    SwitchCaseBlock,
)
from roboflow_workflows.v2.data.batch import (
    SampleContext,
    TemporalContext,
    TimeSpan,
    Timestamp,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.context import (
    ExecutionContext,
    use_execution_context,
)
from roboflow_workflows.v2.engine.declaration import Select, Stop

from inference_models.models.base.classification import ClassificationPrediction
from inference_models.models.base.object_detection import Detections

NEXT = ["$steps.next"]
FRAME_TIMESTAMP = datetime(2026, 1, 1)
CASES = {"dog": "$steps.dogs", "Car": "$steps.cars", "1.0": "$steps.one"}


class _Scope:
    """Engine scope stand-in: one source and one time for every index.

    With ``only_field``, just that Params field carries them; the whole call
    merges to ``None``, as when another argument has no context.
    """

    def __init__(
        self,
        sample: Optional[SampleContext],
        temporal: Optional[TemporalContext],
        *,
        only_field: Optional[str] = None,
    ) -> None:
        self._sample = sample
        self._temporal = temporal
        self._only_field = only_field

    def sample_at(self, index, *, field: Optional[str] = None):
        return self._sample if self._answers(field) else None

    def temporal_at(self, index, *, field: Optional[str] = None):
        return self._temporal if self._answers(field) else None

    def _answers(self, field: Optional[str]) -> bool:
        return self._only_field is None or field == self._only_field


def _call(
    block,
    *,
    source: Optional[str] = None,
    seconds: Optional[float] = None,
    clock: str = "engine.monotonic",
    index=(0,),
    run_id: str = "run",
    only_field: Optional[str] = None,
    **arguments: Any,
) -> bool:
    sample = SampleContext(source_id=source) if source is not None else None
    temporal = None
    if seconds is not None:
        stamp = Timestamp(round(seconds * 1000), Fraction(1, 1000), clock)
        temporal = TemporalContext(observed_coverage=stamp)
    context = ExecutionContext(
        step_path=("step",),
        block_type=block.type,
        session_id="session",
        run_id=run_id,
        indices=(index,),
        call_scope=_Scope(sample, temporal, only_field=only_field),
    )
    with use_execution_context(context):
        decision = block.run(**arguments)
    assert isinstance(decision, Select)

    return not isinstance(decision, Stop)


def _frame(
    video: Optional[str] = None, *, frame: int = 0, fps: Optional[float] = 10.0
) -> ImageData:
    metadata = None
    if video is not None:
        metadata = VideoMetadata(
            video_identifier=video,
            frame_number=frame,
            frame_timestamp=FRAME_TIMESTAMP,
            fps=fps,
        )
    pixels = torch.zeros((3, 4, 4), dtype=torch.uint8)
    image = ImageData.from_tensor(
        pixels, image_id=f"{video}-{frame}", video_metadata=metadata
    )

    return image


# --- switch_case -----------------------------------------------------------


def _route(value: Any, *, case_insensitive: bool = False, default=()) -> tuple:
    block = SwitchCaseBlock()
    with use_execution_context(
        ExecutionContext(step_path=("s",), block_type=block.type, session_id="s")
    ):
        decision = block.run(
            value=value,
            cases=CASES,
            case_insensitive=case_insensitive,
            default_next_steps=list(default),
        )

    return decision.targets


@pytest.mark.parametrize(
    "value, targets",
    [
        ("dog", ("$steps.dogs",)),
        ("Car", ("$steps.cars",)),
        ("car", ()),  # case-sensitive by default
        (1.0, ("$steps.one",)),
        (1, ()),  # str(1) is "1", not "1.0"
        (torch.tensor(1.0), ("$steps.one",)),  # 0-d tensor -> its scalar
    ],
)
def test_switch_matches_str_of_the_value(value, targets) -> None:
    assert _route(value) == targets


def test_switch_matches_true_and_none_by_their_string() -> None:
    block = SwitchCaseBlock()
    cases = {"True": "$steps.yes", "None": "$steps.missing"}
    with use_execution_context(
        ExecutionContext(step_path=("s",), block_type=block.type, session_id="s")
    ):
        routed = [
            block.run(
                value=value,
                cases=cases,
                case_insensitive=False,
                default_next_steps=[],
            ).targets
            for value in (True, None, "true")
        ]

    assert routed == [("$steps.yes",), ("$steps.missing",), ()]


def test_case_insensitive_matches_either_side_lowered() -> None:
    assert _route("CAR", case_insensitive=True) == ("$steps.cars",)
    assert _route("DOG", case_insensitive=True) == ("$steps.dogs",)


def test_no_match_selects_every_default_step_or_stops() -> None:
    assert _route("cat", default=["$steps.a", "$steps.b"]) == ("$steps.a", "$steps.b")
    assert _route("cat") == ()


def test_a_tensor_with_dimensions_fails_with_its_shape() -> None:
    with pytest.raises(TypeError, match=r"shape \(2,\)"):
        _route(torch.tensor([1.0, 2.0]))


def test_a_target_used_twice_fails_validation() -> None:
    with pytest.raises(ValidationError, match="Duplicated targets"):
        SwitchCaseBlock.Params(
            value="x",
            cases={"a": "$steps.t"},
            default_next_steps=["$steps.t"],
        )


def test_keys_colliding_without_case_fail_only_when_case_insensitive() -> None:
    cases = {"Red": "$steps.r1", "red": "$steps.r2"}
    SwitchCaseBlock.Params(value="x", cases=cases)

    with pytest.raises(ValidationError, match="collide"):
        SwitchCaseBlock.Params(value="x", cases=cases, case_insensitive=True)


def test_switch_literal_values_keep_their_type() -> None:
    params = SwitchCaseBlock.Params(value=1, cases={})

    assert params.value == 1 and type(params.value) is int


# --- delta_filter ----------------------------------------------------------


def _changed(block, value: Any, image: Optional[ImageData] = None, **context) -> bool:
    changed = _call(
        block,
        image=image if image is not None else _frame("cam-a"),
        value=value,
        next_steps=NEXT,
        max_sources=context.pop("max_sources", 256),
        **context,
    )

    return changed


def test_delta_passes_the_first_value_and_then_only_changes() -> None:
    block = DeltaFilterBlock()

    assert [_changed(block, value) for value in [3, 3, 4, 4, 3]] == [
        True,
        False,
        True,
        False,
        True,
    ]


def test_delta_first_none_does_not_pass_and_keeps_no_record() -> None:
    block = DeltaFilterBlock()

    assert [_changed(block, value) for value in [None, None, 0, None]] == [
        False,
        False,
        True,
        True,
    ]


@pytest.mark.parametrize(
    "first, second, same",
    [
        (0, False, True),
        (1, 1.0, True),
        (1, True, True),
        ("", None, False),
        ("1", 1, False),
        ([1, 2], [1.0, 2.0], True),
        ([1, 2], (1, 2), False),
        ({"a": [1]}, {"a": [1]}, True),
        ({"a": 1}, {"b": 1}, False),
        (torch.tensor([1, 2]), torch.tensor([1.0, 2.0]), True),
        (torch.tensor([1, 2]), torch.tensor([[1, 2]]), False),
        (torch.tensor(3), 3, True),
        (torch.tensor([3]), 3, False),
        (np.array([1, 2]), np.array([1, 2]), True),
        (np.array([1, 2]), [1, 2], False),
        (np.float32(0.5), 0.5, True),
    ],
)
def test_delta_equality_follows_python_and_compares_containers_by_element(
    first, second, same
) -> None:
    block = DeltaFilterBlock()
    _changed(block, first)

    assert _changed(block, second) is not same


def test_delta_compares_predictions_by_their_tensors() -> None:
    def detections(x: float) -> Detections:
        return Detections(
            xyxy=torch.tensor([[x, 0.0, 4.0, 4.0]]),
            class_id=torch.tensor([1]),
            confidence=torch.tensor([0.9]),
        )

    block = DeltaFilterBlock()
    passed = [_changed(block, detections(x)) for x in (0.0, 0.0, 1.0)]
    classification = ClassificationPrediction(
        torch.tensor([0]), torch.tensor([[0.8, 0.2]])
    )

    assert passed == [True, False, True]
    assert _changed(block, classification) is True


def test_delta_values_without_a_bool_equality_fail_the_call() -> None:
    class Weird:
        def __eq__(self, other):
            return "maybe"

    block = DeltaFilterBlock()
    _changed(block, Weird())

    with pytest.raises(TypeError, match="Weird and Weird"):
        _changed(block, Weird())


def test_delta_keeps_one_value_per_video_and_interleaves_freely() -> None:
    block = DeltaFilterBlock()
    a, b = _frame("cam-a"), _frame("cam-b")

    steps = [(a, 1), (b, 1), (a, 1), (b, 2), (a, 2), (b, 2)]

    assert [_changed(block, value, image) for image, value in steps] == [
        True,
        True,
        False,
        True,
        True,
        False,
    ]


def test_delta_without_video_metadata_uses_the_source_then_the_position() -> None:
    block = DeltaFilterBlock()
    plain = _frame()

    by_source = [_changed(block, 1, plain, source=s) for s in ("x", "y", "x")]
    by_position = [_changed(block, 1, plain, index=(n,)) for n in (0, 1, 0)]

    assert by_source == [True, True, False]
    assert by_position == [True, True, False]


def test_delta_reads_the_source_of_the_image_not_of_the_whole_call() -> None:
    block = DeltaFilterBlock()
    plain = _frame()
    # (position, source, value): per source nothing changes after the first two.
    calls = [((0,), "x", 1), ((1,), "y", 2), ((0,), "y", 2), ((1,), "x", 1)]

    passed = [
        _changed(block, value, plain, index=index, source=source, only_field="image")
        for index, source, value in calls
    ]

    assert passed == [True, True, False, False]


def test_delta_crops_of_one_frame_compare_in_crop_order() -> None:
    block = DeltaFilterBlock()
    frame = _frame("cam-a")

    passed = [_changed(block, v, frame, index=(0, k)) for k, v in enumerate("AAB")]

    assert passed == [True, False, True]


def test_delta_capacity_fails_without_changes_and_known_lanes_keep_working() -> None:
    block = DeltaFilterBlock()
    _changed(block, 1, _frame("a"), max_sources=1)

    with pytest.raises(SourceCapacityError, match="max_sources=1"):
        _changed(block, 1, _frame("b"), max_sources=1)
    assert _changed(block, 2, _frame("a"), max_sources=1)
    assert list(block._values) == [("video", "a")]


def test_delta_a_new_block_after_reset_remembers_nothing() -> None:
    old = DeltaFilterBlock()
    _changed(old, 5)

    assert _changed(DeltaFilterBlock(), 5)


# --- rate_limiter ----------------------------------------------------------


def _allowed(
    block,
    *,
    cooldown: float = 1.0,
    image: Optional[ImageData] = None,
    scope: str = "source",
    max_sources: int = 256,
    **context,
) -> bool:
    allowed = _call(
        block,
        cooldown_seconds=cooldown,
        depends_on=None,
        next_steps=NEXT,
        video_reference_image=image,
        scope=scope,
        max_sources=max_sources,
        **context,
    )

    return allowed


def test_limiter_passes_at_the_cooldown_boundary_not_before() -> None:
    block = RateLimiterBlock()
    times = [0.0, 0.999, 1.0, 1.5, 2.0]

    passed = [_allowed(block, source="cam", seconds=t) for t in times]

    assert passed == [True, False, True, False, True]


def test_limiter_with_zero_cooldown_passes_equal_times() -> None:
    block = RateLimiterBlock()

    assert [_allowed(block, cooldown=0, source="c", seconds=5) for _ in "ab"] == [
        True,
        True,
    ]


def test_limiter_throttles_a_backward_time_and_keeps_the_last_pass() -> None:
    block = RateLimiterBlock()

    passed = [_allowed(block, source="c", seconds=t) for t in (10.0, 3.0, 10.5, 11.0)]

    assert passed == [True, False, False, True]


def test_limiter_counts_video_time_exactly_from_frame_number_and_fps() -> None:
    block = RateLimiterBlock()
    frames = [0, 9, 10, 19, 20]

    passed = [
        _allowed(block, image=_frame("v", frame=n, fps=10.0), source="ignored")
        for n in frames
    ]
    ntsc = RateLimiterBlock()
    ntsc_passed = [
        _allowed(ntsc, cooldown=100.0, image=_frame("n", frame=n, fps=29.97))
        for n in (0, 2996, 2997)
    ]

    assert passed == [True, False, True, False, True]
    assert ntsc_passed == [True, False, True]


def test_video_time_ignores_observation_time() -> None:
    block = RateLimiterBlock()

    # Processed far apart, but only 0.5 s apart in the video.
    first = _allowed(block, image=_frame("v", frame=0), seconds=0.0)
    second = _allowed(block, image=_frame("v", frame=5), seconds=100.0)

    assert (first, second) == (True, False)


@pytest.mark.parametrize("fps", [None, 0.0, float("nan")])
def test_a_reference_without_a_usable_fps_falls_back_to_observation_time(
    fps,
) -> None:
    block = RateLimiterBlock()

    passed = [
        _allowed(block, image=_frame("v", frame=n, fps=fps), seconds=t)
        for n, t in ((0, 0.0), (50, 0.5), (1, 1.0))
    ]

    assert passed == [True, False, True]


def test_a_reference_lane_that_changes_clock_fails_without_changes() -> None:
    block = RateLimiterBlock()
    _allowed(block, image=_frame("v", frame=0), seconds=0.0)

    with pytest.raises(ClockMismatchError, match="video-time:v"):
        _allowed(block, image=_frame("v", frame=50, fps=0.0), seconds=50.0)
    assert block._last_passes[("video", "v")] == ("video-time:v", 0)


def test_each_source_has_its_own_cooldown() -> None:
    block = RateLimiterBlock()
    calls = [("a", 0.0), ("b", 0.2), ("a", 0.5), ("b", 1.2), ("a", 1.0)]

    passed = [_allowed(block, source=s, seconds=t) for s, t in calls]

    assert passed == [True, True, False, True, True]


@pytest.mark.parametrize(
    "with_image, lane_field",
    [(True, "video_reference_image"), (False, "depends_on")],
)
def test_limiter_takes_source_and_time_from_its_lane_field(
    with_image: bool, lane_field: str
) -> None:
    block = RateLimiterBlock()
    image = _frame() if with_image else None
    # (position, source, seconds): positions swap sources between calls.
    calls = [
        ((0,), "a", 0.0),
        ((1,), "b", 0.6),
        ((0,), "b", 1.2),
        ((1,), "a", 0.7),
        ((0,), "a", 1.0),
    ]

    passed = [
        _allowed(
            block,
            image=image,
            index=index,
            source=source,
            seconds=seconds,
            only_field=lane_field,
        )
        for index, source, seconds in calls
    ]

    assert passed == [True, True, False, False, True]


def test_global_scope_keeps_one_cooldown_for_every_source() -> None:
    block = RateLimiterBlock()
    calls = [("a", 0.0), ("b", 0.2), ("b", 1.0), ("a", 1.5)]

    passed = [_allowed(block, scope="global", source=s, seconds=t) for s, t in calls]

    assert passed == [True, False, True, False]


def test_global_scope_rejects_video_time_of_two_videos() -> None:
    block = RateLimiterBlock()
    _allowed(block, scope="global", image=_frame("a", frame=0))

    with pytest.raises(ClockMismatchError, match="scope='global'"):
        _allowed(block, scope="global", image=_frame("b", frame=99))


def test_without_temporal_context_one_run_is_one_moment() -> None:
    block = RateLimiterBlock()

    same_run = [_allowed(block, index=(n,), run_id="r1") for n in (0, 0)]
    other_position = _allowed(block, index=(1,), run_id="r1")

    assert same_run == [True, False]
    assert other_position is True


def test_a_temporal_span_is_timed_by_its_end() -> None:
    block = RateLimiterBlock()
    start = Timestamp(0, Fraction(1), "engine.monotonic")
    end = Timestamp(2, Fraction(1), "engine.monotonic")
    _allowed(block, source="c", seconds=0.5)
    context = ExecutionContext(
        step_path=("step",),
        block_type=block.type,
        session_id="session",
        run_id="run",
        indices=((0,),),
        call_scope=_Scope(
            SampleContext(source_id="c"),
            TemporalContext(observed_coverage=TimeSpan(start, end)),
        ),
    )

    with use_execution_context(context):
        decision = block.run(
            cooldown_seconds=1.0,
            depends_on=None,
            next_steps=NEXT,
            video_reference_image=None,
            scope="source",
            max_sources=4,
        )

    assert decision.targets == tuple(NEXT)


def test_an_infinite_cooldown_passes_once_per_lane() -> None:
    block = RateLimiterBlock()
    inf = float("inf")

    passed = [_allowed(block, cooldown=inf, source="c", seconds=t) for t in (0, 1e6)]

    assert passed == [True, False]


def test_limiter_capacity_fails_and_nothing_is_reclaimed_by_time() -> None:
    block = RateLimiterBlock()
    _allowed(block, source="a", seconds=0.0, max_sources=1)

    with pytest.raises(SourceCapacityError, match="lane 2 of max_sources=1"):
        _allowed(block, source="b", seconds=1000.0, max_sources=1)
    assert list(block._last_passes) == [("source", "a")]


@pytest.mark.parametrize("cooldown", [-1, float("nan"), True])
def test_cooldown_must_be_a_number_of_at_least_zero(cooldown) -> None:
    with pytest.raises(ValidationError):
        RateLimiterBlock.Params(
            cooldown_seconds=cooldown, depends_on="$inputs.x", next_steps=NEXT
        )


def test_limiter_defaults_are_v1_cooldown_and_per_source_scope() -> None:
    params = RateLimiterBlock.Params(depends_on="$inputs.x", next_steps=NEXT)

    assert (params.cooldown_seconds, params.scope, params.max_sources) == (
        1.0,
        "source",
        256,
    )


def test_concurrent_sources_each_get_exactly_one_pass_per_moment() -> None:
    block = RateLimiterBlock()
    results = []
    barrier = threading.Barrier(8)

    def worker(source: str) -> None:
        barrier.wait()
        results.append(_allowed(block, source=source, seconds=1.0))

    threads = [threading.Thread(target=worker, args=(f"s{n % 4}",)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [False] * 4 + [True] * 4
    assert len(block._last_passes) == 4
