"""Version 2 coverage, background scheduling, and typed result tests."""

import gc
import threading
import weakref
from unittest.mock import MagicMock

import pytest
from roboflow_workflows.core_steps.common.deserializers import (
    deserialize_action_recognition_result_kind,
)
from roboflow_workflows.core_steps.common.serializers import (
    serialize_action_recognition_result_kind,
    serialize_wildcard_kind,
)
from roboflow_workflows.core_steps.common.serializers_tensor import (
    serialize_wildcard_kind as serialize_tensor_wildcard_kind,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition import (
    v2 as video_classification_module,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    MAX_TIMELINE_ACTIONS,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v2 import (
    ActionRecognitionModelBlockV2,
    BlockManifest,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v2_tensor import (
    ActionRecognitionModelBlockV2 as TensorActionRecognitionModelBlockV2,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v2_tensor import (
    BlockManifest as TensorBlockManifest,
)
from roboflow_workflows.errors import RuntimeInputError
from roboflow_workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
    ActionRecognitionResult,
)
from roboflow_workflows.execution_engine.entities.types import (
    ACTION_RECOGNITION_RESULT_KIND,
)

from inference_models import (
    ActionRecognitionPrediction as ModelActionRecognitionPrediction,
)

from .helpers import _FakeActionRecognitionModel, _make_frame, _model_segment
from .helpers import _run as _run_output
from .helpers import _timeline_entry, make_block


def _make_block(responses=None, tensor=False, model_class_names=None):
    block_type = (
        TensorActionRecognitionModelBlockV2 if tensor else ActionRecognitionModelBlockV2
    )
    return make_block(
        block_type, responses=responses, model_class_names=model_class_names
    )


def _run(block, frame, **kwargs):
    return _run_output(block, frame, **kwargs)["predictions"]


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_has_one_typed_predictions_output(manifest_type):
    manifest = manifest_type.model_validate(
        {
            "type": "roboflow_core/roboflow_action_recognition_model@v2",
            "name": "actions",
            "images": "$inputs.image",
            "model_id": "cosmos-3-edge",
        }
    )

    assert manifest.background_inference is False
    (output,) = manifest_type.describe_outputs()
    assert output.name == "predictions"
    assert output.kind == [ACTION_RECOGNITION_RESULT_KIND]


def test_result_wire_contract_and_wildcard_round_trip():
    entity = ActionRecognitionPrediction(
        start_frame_idx=3,
        end_frame_idx=9,
        class_name="walk",
        class_id=2,
    )
    result = ActionRecognitionResult(
        timeline=[entity],
        analyzed_from_frame=1,
        analyzed_until_frame=9,
        error_status="gap notice",
    )

    serialized = serialize_action_recognition_result_kind(result)

    assert serialized == {
        "timeline": [
            {
                "start_frame_idx": 3,
                "end_frame_idx": 9,
                "class": "walk",
                "class_id": 2,
            }
        ],
        "analyzed_from_frame": 1,
        "analyzed_until_frame": 9,
        "error_status": "gap notice",
    }
    assert serialize_wildcard_kind(value=result) == serialized
    assert serialize_tensor_wildcard_kind(value=result) == serialized
    assert (
        deserialize_action_recognition_result_kind(
            parameter="predictions", value=serialized
        )
        == result
    )
    assert (
        deserialize_action_recognition_result_kind(
            parameter="predictions", value=result
        )
        is result
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"timeline": "bad"},
        {"timeline": [{"class": "walk"}]},
        {"timeline": [], "analyzed_until_frame": "not-an-int"},
    ],
)
def test_result_deserializer_rejects_malformed_values(value):
    with pytest.raises(RuntimeInputError):
        deserialize_action_recognition_result_kind(parameter="predictions", value=value)


@pytest.mark.parametrize(
    "reset_kwargs",
    [
        {"class_filter": ("run",)},
        {"window_seconds": 2.0},
        {"stride_seconds": 1.0},
    ],
)
def test_window_defining_input_change_clears_all_state(reset_kwargs):
    block, model = _make_block(responses=[[_model_segment("walk")]])
    _run(block, _make_frame(0))
    _run(block, _make_frame(1))
    result = _run(block, _make_frame(2))
    assert result.timeline
    assert result.analyzed_from_frame is not None
    assert result.analyzed_until_frame is not None

    class_filter = reset_kwargs.get("class_filter", ("walk", "run"))
    window_seconds = reset_kwargs.get("window_seconds", 1.0)
    stride_seconds = reset_kwargs.get("stride_seconds", 0.5)
    reset_result = _run(
        block,
        _make_frame(3),
        class_filter=class_filter,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )

    assert reset_result.timeline == []
    assert reset_result.error_status == ""
    assert reset_result.analyzed_from_frame is None
    assert reset_result.analyzed_until_frame is None
    assert len(model.calls) == 1

    reset_fire_frame = 3 + max(1, round(stride_seconds * 4.0))
    for frame_number in range(4, reset_fire_frame + 1):
        result = _run(
            block,
            _make_frame(frame_number),
            class_filter=class_filter,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        )

    assert result.timeline == []
    assert len(model.calls) == 2


def test_predictions_timeline_list_is_isolated_per_frame():
    block, _ = _make_block(responses=[[_model_segment("walk")]])
    _run(block, _make_frame(0))
    _run(block, _make_frame(2))
    idle = _run(block, _make_frame(3))
    bookkeeping = next(iter(block._video_bookkeeping.values()))

    assert idle.timeline is not bookkeeping.timeline_snapshot
    idle.timeline.append(_timeline_entry(0, 0, "appended"))
    idle.timeline.pop(0)
    later = _run(block, _make_frame(4))

    assert [entry.class_name for entry in bookkeeping.timeline_snapshot] == ["walk"]
    assert [entry.class_name for entry in later.timeline] == ["walk"]


@pytest.mark.parametrize("tensor", [False, True])
def test_constructed_predictions_serialize_like_a_validated_result(tensor):
    block, _ = _make_block(responses=[[_model_segment("walk")]], tensor=tensor)
    frame_kwargs = {"tensor_rgb_color": [1, 2, 3]} if tensor else {}
    _run(block, _make_frame(0, **frame_kwargs))
    result = _run_output(block, _make_frame(2, **frame_kwargs))
    assert set(result) == {"predictions"}
    predictions = result["predictions"]
    validated = ActionRecognitionResult(
        timeline=[entry.model_dump() for entry in predictions.timeline],
        analyzed_from_frame=predictions.analyzed_from_frame,
        analyzed_until_frame=predictions.analyzed_until_frame,
        error_status=predictions.error_status,
    )
    expected = validated.model_dump(by_alias=True)

    assert predictions.timeline
    assert predictions.model_fields_set == validated.model_fields_set
    assert serialize_action_recognition_result_kind(predictions) == expected
    assert serialize_wildcard_kind(value=predictions) == expected
    assert serialize_tensor_wildcard_kind(value=predictions) == expected
    assert predictions.model_dump_json(by_alias=True) == validated.model_dump_json(
        by_alias=True
    )


def test_build_output_does_not_revalidate_timeline_entries(monkeypatch):
    bookkeeping = video_classification_module._ActionRecognitionBookkeeping()
    bookkeeping.timeline_snapshot = [
        _timeline_entry(index, index, f"action_{index}")
        for index in range(MAX_TIMELINE_ACTIONS)
    ]
    bookkeeping.analyzed_from_frame = 0
    bookkeeping.analyzed_until_frame = 10
    block, _ = _make_block()

    # BaseModel.__init__ runs the full model validator; model_construct does not.
    def _fail_on_validation(*args, **kwargs):
        raise AssertionError("per-frame output re-validated the timeline")

    monkeypatch.setattr(ActionRecognitionResult, "__init__", _fail_on_validation)
    monkeypatch.setattr(ActionRecognitionResult, "model_validate", _fail_on_validation)
    output = block._build_output(bookkeeping=bookkeeping, error_status="")

    assert output["predictions"].timeline == bookkeeping.timeline_snapshot
    assert all(
        built is original
        for built, original in zip(
            output["predictions"].timeline, bookkeeping.timeline_snapshot
        )
    )


def _run_file(block, *, frame_numbers, total_frames, **run_kwargs):
    results = [
        _run(
            block,
            _make_frame(
                frame_number,
                comes_from_video_file=True,
                total_frames=total_frames,
            ),
            **run_kwargs,
        )
        for frame_number in frame_numbers
    ]
    return results


def test_analyzed_bounds_follow_successful_classifications():
    # The cadence anchors at frame 1 and fires at 3, 5 and 7. The call at 5
    # fails, so the marks must stay on 1-3 until the call at 7 returns.
    block, _ = _make_block(responses=[[], RuntimeError("model failure"), []])

    results = [_run(block, _make_frame(number)) for number in range(1, 8)]

    assert [result.analyzed_from_frame for result in results] == [
        None,
        None,
        1,
        1,
        1,
        1,
        5,
    ]
    assert [result.analyzed_until_frame for result in results] == [
        None,
        None,
        3,
        3,
        3,
        3,
        7,
    ]
    assert results[4].error_status == "model failure"


def test_frame_rollback_clears_analyzed_bounds():
    block, _ = _make_block(responses=[[]])
    for number in (1, 2):
        _run(block, _make_frame(number))
    analyzed = _run(block, _make_frame(3))

    restarted = _run(block, _make_frame(1))

    assert analyzed.analyzed_from_frame == 1
    assert analyzed.analyzed_until_frame == 3
    assert restarted.analyzed_from_frame is None
    assert restarted.analyzed_until_frame is None


def test_last_file_frame_classifies_the_tail_between_scheduled_calls():
    # Frame 6 triggers the tail call with samples 3 and 5.
    block, model = _make_block(
        responses=[[], [], [_model_segment("walk", start_frame_idx=0, end_frame_idx=1)]]
    )

    results = _run_file(block, frame_numbers=range(1, 7), total_frames=6)

    assert len(model.calls) == 3
    assert results[-2].analyzed_until_frame == 5
    assert results[-1].analyzed_until_frame == 5
    (action,) = results[-1].timeline
    assert (action.class_name, action.start_frame_idx, action.end_frame_idx) == (
        "walk",
        3,
        5,
    )


@pytest.mark.parametrize(
    "supports_observed_duration, expected_mark", [(False, 3), (True, 4)]
)
def test_scheduled_call_reports_coverage_without_detections(
    supports_observed_duration, expected_mark
):
    block, model = _make_block()
    model.supports_observed_duration = supports_observed_duration

    for number in range(1, 5):
        result = _run(block, _make_frame(number), stride_seconds=0.75)

    assert len(model.calls) == 1
    assert result.error_status == ""
    assert result.timeline == []
    assert result.analyzed_until_frame == expected_mark


def test_file_shorter_than_one_stride_still_gets_a_result():
    block, model = _make_block(responses=[[_model_segment("run")]])

    results = _run_file(
        block,
        frame_numbers=range(1, 7),
        total_frames=6,
        stride_seconds=10.0,
    )

    assert len(model.calls) == 1
    assert results[-1].analyzed_until_frame == 5
    assert len(results[-1].timeline) == 1


@pytest.mark.parametrize(
    "comes_from_video_file, total_frames",
    [(False, 6), (None, 6), (True, None)],
)
def test_sources_without_a_known_last_frame_get_no_tail_call(
    comes_from_video_file, total_frames
):
    block, model = _make_block()

    for number in range(1, 7):
        _run(
            block,
            _make_frame(
                number,
                comes_from_video_file=comes_from_video_file,
                total_frames=total_frames,
            ),
        )

    assert len(model.calls) == 2


def test_tail_call_runs_once_when_the_container_count_is_too_low():
    # The container reports 4 frames but 8 arrive. Calls: scheduled at 3, the
    # tail at 4, then the cadence from 4 gives 6 and 8. Frames 5 and 7 are
    # also at or past the count and must not classify again.
    block, model = _make_block()

    _run_file(block, frame_numbers=range(1, 9), total_frames=4)

    assert len(model.calls) == 4


def test_scheduled_call_on_the_last_frame_is_not_repeated():
    block, model = _make_block()

    results = _run_file(block, frame_numbers=range(1, 6), total_frames=5)

    assert len(model.calls) == 2
    assert results[-1].analyzed_until_frame == 5


def test_single_frame_file_gets_no_tail_call():
    block, model = _make_block()

    results = _run_file(block, frame_numbers=[1], total_frames=1)

    assert model.calls == []
    assert results[-1].analyzed_until_frame is None


def test_tail_call_waits_for_the_minimum_sampled_frames():
    # Four source frames at half the sample rate buffer two samples.
    block, model = _make_block()

    results = _run_file(
        block,
        frame_numbers=range(1, 5),
        total_frames=4,
        stride_seconds=10.0,
        min_frames=3,
    )

    assert model.calls == []
    assert results[-1].analyzed_until_frame is None


class _GatedActionRecognitionModel(_FakeActionRecognitionModel):
    """Holds each call until the test opens the gate, like a slow model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = threading.Event()
        self.entered = threading.Event()

    def infer(self, frames, class_names=None, fps=None, **kwargs):
        self.entered.set()
        assert self.gate.wait(timeout=5), "the test never opened the gate"
        result = super().infer(frames, class_names=class_names, fps=fps, **kwargs)
        return result


def _make_background_block(responses=None):
    block, _ = _make_block()
    model = _GatedActionRecognitionModel(responses=responses)
    block._model = model
    return block, model


def _finish_pending_call(block, model, video_id="stream-0"):
    # The future is the only signal that the step thread will see the result.
    pending = block._video_bookkeeping[video_id].pending
    model.gate.set()
    pending.future.exception(timeout=5)
    model.gate.clear()


def _run_background(block, frame_number, **run_kwargs):
    result = _run(
        block,
        _make_frame(frame_number),
        background_inference=True,
        **run_kwargs,
    )
    return result


@pytest.mark.parametrize(
    "supports_observed_duration, expected_mark", [(False, 3), (True, 4)]
)
def test_background_call_returns_at_once_and_lands_on_a_later_frame(
    supports_observed_duration, expected_mark
):
    block, model = _make_background_block(
        responses=[[_model_segment("walk", start_frame_idx=0, end_frame_idx=1)]]
    )
    model.supports_observed_duration = supports_observed_duration
    try:
        for number in (1, 2, 3):
            _run_background(block, number, stride_seconds=0.75)

        fired = _run_background(block, 4, stride_seconds=0.75)
        assert model.entered.wait(timeout=5)
        # The model is still inside its call: the step did not wait for it.
        assert fired.timeline == []
        assert fired.analyzed_until_frame is None

        _finish_pending_call(block, model)
        landed = _run_background(block, 5, stride_seconds=0.75)
    finally:
        model.gate.set()
        block.close()
        # A second close() must be a no-op.
        block.close()

    # Frame 3 was the last sample, frame 4 triggered the call, and frame 5
    # collected the result.
    assert landed.analyzed_until_frame == expected_mark
    assert [entry.class_name for entry in landed.timeline] == ["walk"]
    assert landed.error_status == ""


def test_background_call_due_while_busy_waits_and_takes_the_newest_window():
    # Scheduled calls: 3, then 5. The first one is still running on 5 and 6,
    # so the second starts on 7 and classifies the window that ends there.
    block, model = _make_background_block()
    try:
        for number in range(1, 7):
            _run_background(block, number)
        assert len(model.calls) == 0

        _finish_pending_call(block, model)
        late = _run_background(block, 7)
        _finish_pending_call(block, model)
        after = _run_background(block, 8)
    finally:
        model.gate.set()
        block.close()

    # A frame's blue channel holds its number: the late call saw frames 5 and 7.
    classified = [[frame[0, 0, 2] for frame in call["frames"]] for call in model.calls]
    assert classified == [[1, 3], [5, 7]]
    assert late.analyzed_until_frame == 3
    assert after.analyzed_until_frame == 7
    # The late window's first sample is frame 5, so frame 4 is in no window:
    # the skipped call on 5 was the one that covered it.
    assert "No window covers source frames 4-4." in late.error_status


@pytest.mark.parametrize(
    "stride_seconds, late_frame, observed_duration, uncovered, analyzed_until",
    [
        # The first call ends on frame 3 and runs until frame 12, so the next
        # window covers 9-12.
        (0.5, 12, False, "4-8", 3),
        # The stride is twice the window: the first call ends on frame 9 and
        # runs until frame 20, so the next window covers 17-20. The stride
        # skips 10-13 anyway; the notice names the whole gap.
        (2.0, 20, False, "10-16", 9),
        # The stride equals the window: the first call ends on frame 5 and
        # runs until frame 11. The next window's first sample is frame 9, so
        # frame 8 is uncovered even though the nominal window starts there.
        (1.0, 11, False, "6-8", 5),
        # Frame 4 fires the previous call. A sample-index model covers through
        # frame 3, while an observed-duration model covers through frame 4.
        (0.75, 11, False, "4-8", 3),
        (0.75, 11, True, "5-8", 4),
    ],
)
def test_late_background_call_names_the_frames_no_window_covers(
    stride_seconds, late_frame, observed_duration, uncovered, analyzed_until
):
    block, model = _make_background_block()
    model.supports_observed_duration = observed_duration
    try:
        for number in range(1, late_frame):
            _run_background(block, number, stride_seconds=stride_seconds)

        _finish_pending_call(block, model)
        late = _run_background(block, late_frame, stride_seconds=stride_seconds)
    finally:
        model.gate.set()
        block.close()

    assert f"No window covers source frames {uncovered}." in late.error_status
    assert late.analyzed_until_frame == analyzed_until


def test_streams_of_one_step_take_turns_on_the_background_worker():
    block, model = _make_background_block()
    try:
        for number in (1, 2, 3):
            for video_id in ("camera-a", "camera-b"):
                _run(
                    block,
                    _make_frame(number, video_id=video_id),
                    background_inference=True,
                )
        assert model.entered.wait(timeout=5)
        first = block._video_bookkeeping["camera-a"].pending.future
        second = block._video_bookkeeping["camera-b"].pending.future

        # One model, one device: the second camera's call waits its turn.
        assert first.running() and not second.running()

        model.gate.set()
        second.result(timeout=5)
        results = [
            _run(
                block,
                _make_frame(4, video_id=video_id),
                background_inference=True,
            )
            for video_id in ("camera-a", "camera-b")
        ]
    finally:
        model.gate.set()
        block.close()

    assert [result.analyzed_until_frame for result in results] == [3, 3]
    assert [result.error_status for result in results] == ["", ""]


def test_background_model_failure_surfaces_on_the_frame_that_collects_it():
    block, model = _make_background_block(responses=[RuntimeError("model failure"), []])
    try:
        for number in (1, 2, 3):
            fired = _run_background(block, number)

        _finish_pending_call(block, model)
        failed = _run_background(block, 4)
        _run_background(block, 5)
        _finish_pending_call(block, model)
        recovered = _run_background(block, 6)
    finally:
        model.gate.set()
        block.close()

    assert fired.error_status == ""
    assert failed.error_status == "model failure"
    assert failed.analyzed_until_frame is None
    assert recovered.error_status == ""
    assert recovered.analyzed_until_frame == 5


def test_video_file_classifies_inside_the_step_even_with_background_on():
    block, _ = _make_block(responses=[[_model_segment("run")]])

    results = _run_file(
        block,
        frame_numbers=range(1, 4),
        total_frames=30,
        background_inference=True,
    )

    # The frame that fired already carries its result.
    assert results[-1].analyzed_until_frame == 3
    assert len(results[-1].timeline) == 1


def test_stream_restart_drops_the_result_of_the_call_it_left_behind():
    block, model = _make_background_block(
        responses=[[_model_segment("walk")], [_model_segment("run")]]
    )
    try:
        for number in (1, 2, 3):
            _run_background(block, number)
        stale_call = block._video_bookkeeping["stream-0"].pending
        assert model.entered.wait(timeout=5)

        restarted = _run_background(block, 1)
        model.gate.set()
        stale_call.future.result(timeout=5)
        after = _run_background(block, 2)
    finally:
        model.gate.set()
        block.close()

    assert restarted.timeline == []
    assert after.timeline == []
    assert after.analyzed_until_frame is None


def _submit_camera_window(block, video_id, *, bgr_color=None):
    for number in (1, 2, 3):
        frame = _make_frame(number, video_id=video_id, bgr_color=bgr_color)
        _run(block, frame, background_inference=True)

    return block._video_bookkeeping[video_id].pending


def test_failed_camera_does_not_retain_another_cameras_completed_window():
    block, model = _make_background_block(
        responses=[RuntimeError("camera A failed"), []]
    )
    try:
        _submit_camera_window(block, "camera-a")
        assert model.entered.wait(timeout=5)
        pending = _submit_camera_window(block, "camera-b")
        frame_refs = [weakref.ref(sample.frame) for sample in pending.sampled]
        del pending

        model.gate.set()
        # The single worker reaches this task after it finishes draining both calls.
        block._worker._pool.submit(lambda: None).result(timeout=5)
        landed = _run(
            block, _make_frame(4, video_id="camera-b"), background_inference=True
        )
        assert landed.analyzed_until_frame == 3

        _run(block, _make_frame(1, video_id="camera-b"), background_inference=True)
        assert all(frame_ref() is None for frame_ref in frame_refs)

        failed = _run(
            block, _make_frame(4, video_id="camera-a"), background_inference=True
        )
        assert failed.error_status == "camera A failed"
    finally:
        model.gate.set()
        block.close()


def test_repeated_restarts_release_queued_frames_while_another_camera_runs():
    block, model = _make_background_block()
    try:
        _submit_camera_window(block, "camera-a")
        assert model.entered.wait(timeout=5)

        futures = []
        frame_refs = []
        for generation in range(4):
            pending = _submit_camera_window(
                block, "camera-b", bgr_color=[generation, 0, 0]
            )
            futures.append(pending.future)
            frame_refs.append(weakref.ref(pending.sampled[0].frame))
            del pending

        assert all(future.cancelled() for future in futures[:-1])
        assert all(frame_ref() is None for frame_ref in frame_refs[:-1])
        assert frame_refs[-1]() is not None

        model.gate.set()
        futures[-1].result(timeout=5)
        landed = _run(
            block, _make_frame(4, video_id="camera-b"), background_inference=True
        )
        assert landed.analyzed_until_frame == 3
        assert len(model.calls) == 2
        assert model.calls[-1]["frames"][0][0, 0, 2] == 3
    finally:
        model.gate.set()
        block.close()


@pytest.mark.parametrize("reason", ["class_filter", "model_id", "eviction"])
def test_discarding_stream_state_releases_a_waiting_call(reason, monkeypatch):
    monkeypatch.setattr(video_classification_module, "MAX_TRACKED_VIDEOS", 2)
    block, model = _make_background_block()
    try:
        _submit_camera_window(block, "camera-a")
        assert model.entered.wait(timeout=5)
        queued = _submit_camera_window(block, "camera-b")
        future = queued.future
        frame_ref = weakref.ref(queued.sampled[0].frame)
        del queued

        if reason == "class_filter":
            _run(block, _make_frame(4, video_id="camera-b"), class_filter=["run"])
        elif reason == "model_id":
            block._model_manager = MagicMock()
            block._model_manager.load_action_recognition_model.return_value = (
                _FakeActionRecognitionModel()
            )
            block.run(
                images=[_make_frame(4, video_id="camera-b")], model_id="other-model"
            )
        else:
            _run(block, _make_frame(4, video_id="camera-a"))
            _run(block, _make_frame(1, video_id="camera-c"))

        assert future.cancelled()
        assert frame_ref() is None
    finally:
        model.gate.set()
        block.close()


def test_close_cancels_waiting_windows_and_releases_their_frames():
    block, model = _make_background_block()
    try:
        running = _submit_camera_window(block, "camera-a").future
        assert model.entered.wait(timeout=5)
        queued = _submit_camera_window(block, "camera-b")
        waiting = queued.future
        frame_refs = [weakref.ref(sample.frame) for sample in queued.sampled]
        del queued

        block.close()
        gc.collect()

        assert running.running()
        assert waiting.cancelled()
        assert all(frame_ref() is None for frame_ref in frame_refs)
        model.gate.set()
        running.result(timeout=5)
        assert len(model.calls) == 1
    finally:
        model.gate.set()
        block.close()


def test_dropping_the_block_cancels_calls_that_have_not_started():
    block, model = _make_background_block()
    for number in (1, 2, 3):
        for video_id in ("camera-a", "camera-b"):
            _run(
                block,
                _make_frame(number, video_id=video_id),
                background_inference=True,
            )
    assert model.entered.wait(timeout=5)
    running = block._video_bookkeeping["camera-a"].pending.future
    queued = block._video_bookkeeping["camera-b"].pending.future
    block_reference = weakref.ref(block)

    del block
    gc.collect()
    model.gate.set()
    running.result(timeout=5)

    assert block_reference() is None
    assert queued.cancelled()
    assert len(model.calls) == 1


def test_a_background_stream_keeps_its_mode_when_the_input_turns_off():
    block, model = _make_background_block(responses=[[_model_segment("walk")]])
    try:
        for number in (1, 2, 3):
            _run_background(block, number)
        assert model.entered.wait(timeout=5)

        # Frame 5 is due while the frame-3 call runs. The stream started in
        # background mode, so it defers instead of waiting for the model.
        for number in (4, 5):
            due = _run(block, _make_frame(number))
        assert model.calls == []

        _finish_pending_call(block, model)
        collected = _run(block, _make_frame(6))
        # Frame 6 is due again: the window goes to the worker, not the step.
        assert block._video_bookkeeping["stream-0"].pending is not None
    finally:
        model.gate.set()
        block.close()

    assert due.analyzed_until_frame is None
    assert collected.error_status == ""
    assert collected.analyzed_until_frame == 3
    assert [entry.class_name for entry in collected.timeline] == ["walk"]


def test_a_synchronous_stream_keeps_its_mode_when_the_input_turns_on():
    block, _ = _make_block(responses=[[_model_segment("run")]])

    for number in (1, 2):
        _run(block, _make_frame(number))
    fired = _run(block, _make_frame(3), background_inference=True)

    # The result is on the frame that fired: the call ran inside the step.
    assert fired.analyzed_until_frame == 3
    assert [entry.class_name for entry in fired.timeline] == ["run"]


def test_late_call_gap_ends_at_the_first_retained_source_frame():
    # 4 FPS sampled at 3 FPS: the late window's first sample is intended
    # frame 7.67 and comes from source frame 8, where its ranges start.
    run_kwargs = {
        "stride_seconds": 1.0,
        "sample_fps": 3.0,
        "requires_regular_sampling": True,
    }
    block, model = _make_background_block()
    try:
        for number in range(1, 11):
            _run_background(block, number, **run_kwargs)

        _finish_pending_call(block, model)
        late = _run_background(block, 11, **run_kwargs)
    finally:
        model.gate.set()
        block.close()

    assert "No window covers source frames 6-7." in late.error_status


def test_late_call_gap_matches_continuous_prediction_start_frame():
    # 5 FPS sampled at 2 FPS (stride 2.5): the late window on frame 12 keeps
    # intended sample 8.5 (source frame 9). Continuous spans floor 8.5 to
    # frame 8, so the timeline covers 3-6 and 8-12 and only frame 7 is missing.
    full_window = ModelActionRecognitionPrediction(
        start_frame_idx=0.0,
        end_frame_idx=2.0,
        class_name="walk",
        end_exclusive=True,
    )
    block, model = _make_background_block(responses=[[full_window], [full_window]])
    model.supports_observed_duration = True
    run_kwargs = {
        "stride_seconds": 1.0,
        "sample_fps": 2.0,
        "requires_regular_sampling": True,
        "background_inference": True,
    }
    try:
        for number in range(1, 12):
            _run(block, _make_frame(number, fps=5.0), **run_kwargs)

        _finish_pending_call(block, model)
        late = _run(block, _make_frame(12, fps=5.0), **run_kwargs)
        _finish_pending_call(block, model)
        after = _run(block, _make_frame(13, fps=5.0), **run_kwargs)
    finally:
        model.gate.set()
        block.close()

    ranges = [(entry.start_frame_idx, entry.end_frame_idx) for entry in after.timeline]
    assert ranges == [(3, 6), (8, 12)]
    assert after.analyzed_from_frame == 8
    assert after.analyzed_until_frame == 12
    assert "No window covers source frames 7-7." in late.error_status


@pytest.mark.parametrize(
    "supports_observed_duration, expected_from, expected_until",
    [(False, 4, 6), (True, 3, 6)],
)
def test_analyzed_from_frame_matches_model_window_origin(
    supports_observed_duration, expected_from, expected_until
):
    block, model = _make_block(responses=[[]])
    model.supports_observed_duration = supports_observed_duration
    run_kwargs = {
        "stride_seconds": 1.0,
        "sample_fps": 2.0,
        "requires_regular_sampling": True,
    }

    for number in range(1, 6):
        before = _run(block, _make_frame(number, fps=5.0), **run_kwargs)
    assert before.analyzed_from_frame is None
    assert before.analyzed_until_frame is None

    # Frame 6 fires with samples at intended 3.5 (source 4) and 6.0 (source 6).
    first = _run(block, _make_frame(6, fps=5.0), **run_kwargs)
    assert first.analyzed_from_frame == expected_from
    assert first.analyzed_until_frame == expected_until
