"""Action Recognition Visualization: live strip and synchronized playback."""

from datetime import datetime

import numpy as np
import pytest
import torch
from PIL import ImageFont
from roboflow_workflows.core_steps.visualizations.action_recognition import (
    v1 as visualization_module,
)
from roboflow_workflows.core_steps.visualizations.action_recognition.v1 import (
    MAX_STREAMS,
    TYPE,
    ActionRecognitionVisualizationBlockV1,
    BlockManifest,
)
from roboflow_workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
    ActionRecognitionResult,
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

FPS = 10


@pytest.fixture(autouse=True)
def no_font_download(monkeypatch):
    # The approved font may need a download. Tests draw with the fallback.
    def unavailable(font_family):
        raise RuntimeError("font unavailable in tests")

    monkeypatch.setattr(visualization_module, "resolve_font_path", unavailable)


def _image(
    number, *, video="demo", fps=FPS, size=(180, 320), tensor=False, noise=False
):
    # A flat shade survives JPEG, so a frame is recognizable by its top row.
    # Noise does not compress, so it fills the frame budget quickly.
    pixels = (
        np.random.default_rng(number).integers(0, 256, (*size, 3), dtype=np.uint8)
        if noise
        else np.full((*size, 3), number % 256, dtype=np.uint8)
    )
    representation = (
        {"tensor_image": torch.from_numpy(pixels).permute(2, 0, 1)}
        if tensor
        else {"numpy_image": pixels}
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"{video}-{number}"),
        workflow_root_ancestor_metadata=ImageParentMetadata(
            parent_id=f"{video}-root-{number}"
        ),
        video_metadata=VideoMetadata(
            video_identifier=video,
            frame_number=number,
            fps=fps,
            frame_timestamp=datetime(2026, 1, 1),
        ),
        **representation,
    )


def _action(name, start, end, class_id=0):
    return ActionRecognitionPrediction(
        class_name=name,
        class_id=class_id,
        start_frame_idx=start,
        end_frame_idx=end,
    )


def _predictions(timeline=(), until=None, *, start=1, error_status=""):
    # The model's latest result bundle, covering the stream since frame 1 by default.
    result = ActionRecognitionResult(
        timeline=list(timeline),
        analyzed_from_frame=None if until is None else start,
        analyzed_until_frame=until,
        error_status=error_status,
    )
    return result


@pytest.fixture
def drawn_labels(monkeypatch):
    labels = []
    draw = visualization_module._fitted_text

    def recording(label, **kwargs):
        labels.append(label)
        mask = draw(label, **kwargs)
        return mask

    monkeypatch.setattr(visualization_module, "_fitted_text", recording)
    return labels


def _shade(result):
    # Top-right corner: outside the badges and above the strip.
    return int(result["image"].numpy_image[1, -2, 0])


def _draws_the_timeline(result, number, *, analyzed_until_frame):
    # The strip darkens the frame even with no actions; compare with none.
    empty = ActionRecognitionVisualizationBlockV1().run(
        _image(number), _predictions(until=analyzed_until_frame)
    )
    drawn, blank = result["image"].numpy_image, empty["image"].numpy_image
    lower_half = slice(drawn.shape[0] // 2, None)
    differs = not np.array_equal(drawn[lower_half], blank[lower_half])
    return differs


def _synchronized(
    block,
    number,
    *,
    image=None,
    timeline=(),
    analyzed_until_frame=None,
    analyzed_from_frame=1,
    **kwargs,
):
    result = block.run(
        _image(number, **(image or {})),
        _predictions(timeline, analyzed_until_frame, start=analyzed_from_frame),
        display_mode="synchronized",
        **kwargs,
    )
    return result


class _Model:
    """Reports what a model with this stride and call time would know."""

    def __init__(self, *, stride, latency, timeline=()):
        self.stride, self.latency, self.timeline = stride, latency, list(timeline)

    def at(self, number):
        landed = [
            fire
            for fire in range(1 + self.stride, number + 1, self.stride)
            if fire + self.latency <= number
        ]
        if not landed:
            return None, []

        mark = landed[-1]
        timeline = [entry for entry in self.timeline if entry.start_frame_idx <= mark]
        return mark, timeline


def _play(block, model, frames, **kwargs):
    results = []
    for number in frames:
        mark, timeline = model.at(number)
        results.append(
            block.run(
                _image(number),
                _predictions(timeline, mark),
                **kwargs,
            )
        )
    return results


def test_live_shows_the_current_frame():
    block = ActionRecognitionVisualizationBlockV1()
    model = _Model(stride=20, latency=5, timeline=[_action("wave", 5, 15)])

    results = _play(block, model, range(1, 41))

    assert [result["displayed_frame_number"] for result in results] == list(
        range(1, 41)
    )
    assert all(_shade(result) == number for number, result in enumerate(results, 1))
    # The first result lands on frame 26 and covers frames up to 21.
    assert results[24]["status"] == "analyzing"
    assert results[24]["actions"] == []
    assert results[24]["actions_frame_number"] is None
    assert results[25]["status"] == "live"
    assert results[25]["actions"] == ["wave"]
    assert results[25]["actions_frame_number"] == 15


def test_live_names_the_most_recent_action_on_the_strip():
    block = ActionRecognitionVisualizationBlockV1()
    timeline = [_action("wave", 5, 21), _action("jump", 18, 30, class_id=1)]

    during = block.run(_image(40), _predictions(timeline, 21))
    after = block.run(_image(41), _predictions(timeline, 40))
    # The 30-second strip at 10 FPS starts at frame 101 here: jump left it.
    gone = block.run(_image(401), _predictions(timeline, 400))

    assert during["actions"] == ["jump", "wave"]
    assert during["actions_frame_number"] == 21
    # A quiet latest window does not hide the jump that ended at frame 30.
    assert after["status"] == "live"
    assert after["actions"] == ["jump"]
    assert after["actions_frame_number"] == 30
    assert gone["actions"] == []
    assert gone["actions_frame_number"] == 400


def test_live_names_an_earlier_action_when_the_latest_frame_is_not_analyzed(
    drawn_labels,
):
    block = ActionRecognitionVisualizationBlockV1()

    result = block.run(
        _image(41),
        # Frame 41 lies before the covered window 50-60, so live reports not_analyzed.
        _predictions([_action("punch", 5, 10)], 60, start=50),
    )

    assert result["status"] == "not_analyzed"
    assert result["actions"] == ["punch"]
    assert result["actions_frame_number"] == 10
    assert drawn_labels[:2] == ["punch", "3 s ago"]


@pytest.mark.parametrize("mark", [None, 45])
def test_live_draws_a_strip_on_a_copy_with_or_without_the_mark(mark):
    block = ActionRecognitionVisualizationBlockV1()
    source = _image(50)

    result = block.run(source, _predictions([_action("wave", 20, 40)], mark))

    assert result["status"] == ("analyzing" if mark is None else "live")
    assert result["displayed_frame_number"] == 50
    assert _draws_the_timeline(result, 50, analyzed_until_frame=mark)
    assert np.all(source.numpy_image == 50)
    assert result["image"].numpy_image.shape == source.numpy_image.shape


def test_synchronized_holds_the_first_frame_then_plays_labelled_frames_in_order():
    block = ActionRecognitionVisualizationBlockV1()
    model = _Model(
        stride=20,
        latency=5,
        timeline=[_action("jump", 1, 1, class_id=1), _action("wave", 3, 8)],
    )

    # A repeated frame number must be neither stored nor shown twice.
    results = _play(block, model, [1, 1, *range(2, 121)], display_mode="synchronized")

    held = results[:26]
    assert {result["status"] for result in held} == {"analyzing"}
    assert {result["displayed_frame_number"] for result in held} == {1}
    assert {tuple(result["actions"]) for result in held} == {()}
    playing = results[26:]
    assert {result["status"] for result in playing} == {"synchronized"}
    shown = [result["displayed_frame_number"] for result in playing]
    # Playback shows the held first frame with its result once, then advances
    # frame by frame: no skip, no repeat.
    assert shown == list(range(1, len(playing) + 1))
    # The held first frame shows its own result before playback advances.
    assert playing[0]["actions"] == ["jump"]
    assert playing[1]["actions"] == []
    assert all(
        result["displayed_frame_number"] <= model.at(number)[0]
        for number, result in enumerate(playing, 26)
    )
    # Each label sits on the frame it describes.
    labelled = [
        result["displayed_frame_number"]
        for result in results
        if result["actions"] == ["wave"]
    ]
    assert labelled == [3, 4, 5, 6, 7, 8]
    for result in playing:
        assert result["actions_frame_number"] == result["displayed_frame_number"]
        assert abs(_shade(result) - result["displayed_frame_number"]) <= 2
        displayed = result["displayed_frame_number"]
        assert result["image"].video_metadata.frame_number == displayed
        assert result["image"].parent_metadata.parent_id == f"demo-{displayed}"
        assert (
            result["image"].workflow_root_ancestor_metadata.parent_id
            == f"demo-root-{displayed}"
        )


def test_synchronized_forgets_a_slow_start_once_a_second_result_lands():
    block = ActionRecognitionVisualizationBlockV1()
    # The stream is blocked while the model loads: frames 2-99 never arrive
    # and the first result lands at frame 102. Later results land every 10
    # frames with a lag of 2.
    delays = {}
    for number in [1, *range(100, 160)]:
        mark = (number - 2) // 10 * 10 if number >= 102 else None
        result = _synchronized(block, number, analyzed_until_frame=mark)
        delays[number] = number - result["displayed_frame_number"]

    assert delays[105] > 90
    assert max(delays[number] for number in range(112, 160)) <= 12


def test_synchronized_recovers_from_one_slow_call_without_skipping_an_action():
    block = ActionRecognitionVisualizationBlockV1()
    wave = [_action("wave", 48, 51)]
    delays, wave_frames = [], []
    # Results land every 10 frames with a lag of 2, except one that is 30 late.
    for number in range(1, 200):
        landed = (number - 2) // 10 * 10
        if 50 <= number < 82:
            landed = 40
        mark = landed if landed >= 10 else None
        result = _synchronized(
            block,
            number,
            timeline=wave if mark is not None and mark >= 51 else [],
            analyzed_until_frame=mark,
        )
        delays.append(number - result["displayed_frame_number"])
        if "wave" in result["actions"]:
            wave_frames.append(result["displayed_frame_number"])

    assert max(delays[49:90]) >= 40
    assert max(delays[-20:]) <= 12
    # Catching up after the slow call still shows the action.
    assert any(48 <= frame <= 51 for frame in wave_frames)


def test_synchronized_stops_waiting_at_the_delay_limit():
    block = ActionRecognitionVisualizationBlockV1()

    results = [
        _synchronized(block, number, max_delay_seconds=2) for number in range(1, 61)
    ]

    assert results[20]["status"] == "analyzing"
    assert results[20]["displayed_frame_number"] == 1
    # After the 2-second limit the video moves on without results.
    assert results[-1]["status"] == "behind"
    assert results[-1]["displayed_frame_number"] == 60 - 2 * FPS
    assert results[-1]["actions"] == []
    # A camera that skips frames does not keep frames past the limit either.
    jumped = _synchronized(block, 160, max_delay_seconds=2)
    assert jumped["displayed_frame_number"] == 160
    assert jumped["status"] == "behind"
    assert jumped["actions_frame_number"] is None


def test_memory_budget_is_shared_and_shortens_the_delay_instead_of_failing():
    alone = ActionRecognitionVisualizationBlockV1()
    shared = ActionRecognitionVisualizationBlockV1()
    noisy = {"size": (360, 640), "noise": True}

    for number in range(1, 30):
        last_alone = _synchronized(alone, number, image=noisy, max_buffer_mb=1)
        for video in ("a", "b"):
            last_shared = _synchronized(
                shared, number, image={**noisy, "video": video}, max_buffer_mb=1
            )

    # Without the budget the first frame would still be held.
    assert last_alone["status"] == "behind"
    delay_alone = 29 - last_alone["displayed_frame_number"]
    delay_shared = 29 - last_shared["displayed_frame_number"]
    assert 0 <= delay_shared < delay_alone <= 6


def test_stopped_stream_gives_its_frames_back_to_a_running_stream():
    block = ActionRecognitionVisualizationBlockV1()
    noisy = {"size": (360, 640), "noise": True}
    model = _Model(stride=10, latency=1)

    # A camera fills the budget while it waits for a first result, then stops.
    for number in range(1, 40):
        _synchronized(
            block, number, image={**noisy, "video": "stopped"}, max_buffer_mb=2
        )
    for number in range(1, 40):
        mark, _ = model.at(number)
        last = _synchronized(
            block,
            number,
            image={**noisy, "video": "running"},
            analyzed_until_frame=mark,
            max_buffer_mb=2,
        )

    assert last["status"] == "synchronized"
    assert last["displayed_frame_number"] < 39


@pytest.mark.parametrize("tensor", [False, True])
def test_synchronized_frames_are_downscaled_before_they_are_kept(tensor):
    block = ActionRecognitionVisualizationBlockV1()
    source = _image(1, size=(360, 640), tensor=tensor)

    result = block.run(
        source,
        _predictions(),
        display_mode="synchronized",
        max_frame_side=160,
    )

    assert result["image"].numpy_image.shape == (90, 160, 3)
    assert source.numpy_image.shape == (360, 640, 3)


def test_streams_are_independent_and_a_restart_resets_only_its_stream():
    block = ActionRecognitionVisualizationBlockV1()
    for number in range(1, 11):
        for video in ("a", "b"):
            _synchronized(block, number, image={"video": video}, analyzed_until_frame=5)

    restarted = _synchronized(block, 1, image={"video": "a"})
    other = _synchronized(block, 11, image={"video": "b"}, analyzed_until_frame=5)

    assert restarted["status"] == "analyzing"
    assert restarted["displayed_frame_number"] == 1
    assert other["displayed_frame_number"] > 1


def test_least_recently_used_stream_is_dropped_at_the_limit():
    block = ActionRecognitionVisualizationBlockV1()
    for index in range(MAX_STREAMS):
        _synchronized(block, 1, image={"video": str(index)})
    _synchronized(block, 2, image={"video": "0"})
    _synchronized(block, 1, image={"video": "extra"})

    # A kept stream still holds its first frame; a dropped one starts over.
    kept = _synchronized(block, 3, image={"video": "0"})
    dropped = _synchronized(block, 3, image={"video": "1"})

    assert kept["displayed_frame_number"] == 1
    assert dropped["displayed_frame_number"] == 3


@pytest.mark.parametrize(
    "first_fps, expected_fps",
    [(None, 30), (0, 30), (-1, 30), (float("nan"), 30), (float("inf"), 30), (10, 10)],
)
def test_source_fps_falls_back_like_the_model_and_is_pinned_per_stream(
    first_fps, expected_fps
):
    block = ActionRecognitionVisualizationBlockV1()
    _synchronized(block, 1, image={"fps": first_fps}, max_delay_seconds=1)

    # Later frames declaring another FPS do not change the stream's clock.
    for number in range(2, 61):
        result = _synchronized(block, number, image={"fps": 60}, max_delay_seconds=1)

    assert result["displayed_frame_number"] == 60 - expected_fps


def test_class_names_outside_ascii_draw_with_and_without_the_font(monkeypatch):
    timeline = [_action("Saída 走る", 1, 48, class_id=-1)]
    fallback = ActionRecognitionVisualizationBlockV1().run(
        _image(50), _predictions(timeline, 45)
    )
    # Compare before the font patch, so both renders use the fallback font.
    fallback_drawn = _draws_the_timeline(fallback, 50, analyzed_until_frame=45)
    monkeypatch.setattr(
        visualization_module, "resolve_font_path", lambda font_family: "font.ttf"
    )
    packaged_font = ImageFont.load_default(size=12)
    monkeypatch.setattr(
        ImageFont,
        "truetype",
        lambda path, size: packaged_font.font_variant(size=size),
    )

    with_font = ActionRecognitionVisualizationBlockV1().run(
        _image(50), _predictions(timeline, 45)
    )

    assert fallback["actions"] == ["Saída 走る"]
    assert fallback_drawn
    assert _draws_the_timeline(with_font, 50, analyzed_until_frame=45)


def test_frame_too_small_for_a_strip_is_returned_with_its_outputs():
    block = ActionRecognitionVisualizationBlockV1()

    result = block.run(
        _image(50, size=(16, 24)),
        _predictions([_action("wave", 20, 60)], 50),
    )

    assert result["image"].numpy_image.shape == (16, 24, 3)
    assert result["actions"] == ["wave"]


def test_run_rejects_an_unknown_display_mode_and_a_non_bgr_image():
    manifest = BlockManifest(
        type=TYPE,
        name="display",
        image="$inputs.image",
        predictions="$steps.model.predictions",
        display_mode="$inputs.display_mode",
    )
    block = ActionRecognitionVisualizationBlockV1()
    gray = WorkflowImageData.copy_and_replace(
        origin_image_data=_image(1),
        numpy_image=np.zeros((90, 160), dtype=np.uint8),
    )

    # A selector defers the display_mode check to run time.
    assert manifest.display_mode == "$inputs.display_mode"
    with pytest.raises(ValueError, match="display_mode"):
        block.run(_image(1), _predictions(), display_mode="delayed")
    with pytest.raises(ValueError, match="uint8 BGR"):
        block.run(gray, _predictions())


@pytest.mark.parametrize("display_mode", [None, "synchronized"])
def test_real_engine_runs_the_step_while_the_mark_is_still_null(display_mode):
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    outputs = [
        "image",
        "status",
        "actions",
        "actions_frame_number",
        "displayed_frame_number",
    ]
    step = {
        "type": TYPE,
        "name": "display",
        "image": "$inputs.image",
        "predictions": "$inputs.predictions",
    }
    if display_mode:
        step["display_mode"] = display_mode
    engine = ExecutionEngine.init(
        workflow_definition={
            "version": "1.0",
            "inputs": [
                {"type": "WorkflowImage", "name": "image"},
                {
                    "type": "WorkflowBatchInput",
                    "name": "predictions",
                    "kind": ["action_recognition_result"],
                },
            ],
            "steps": [step],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": name,
                    "selector": f"$steps.display.{name}",
                }
                for name in outputs
            ],
        }
    )

    def run(number, predictions, serialize=False):
        image = _image(number)
        return engine.run(
            runtime_parameters={
                "image": [
                    {
                        "type": "numpy_object",
                        "value": image.numpy_image,
                        "video_metadata": image.video_metadata,
                    }
                ],
                "predictions": [predictions],
            },
            serialize_results=serialize,
        )[0]

    warmup = run(1, _predictions())
    timeline = [_action("wave", 1, 5)]
    for number in range(2, 12):
        run(number, _predictions())
    ready = run(12, _predictions(timeline, 10).model_dump(by_alias=True))
    serialized = run(13, _predictions(timeline, 10), serialize=True)

    # Null bounds inside ActionRecognitionResult must not skip the step.
    assert warmup["status"] == "analyzing"
    assert isinstance(warmup["image"], WorkflowImageData)
    if display_mode == "synchronized":
        assert ready["status"] == "synchronized"
        assert ready["displayed_frame_number"] == 1
        assert ready["actions"] == ["wave"]
        assert ready["image"].video_metadata.frame_number == 1
        assert serialized["displayed_frame_number"] == 2
    else:
        # The default display is live.
        assert ready["status"] == "live"
        assert ready["displayed_frame_number"] == 12
        assert ready["actions"] == ["wave"]
        assert ready["actions_frame_number"] == 5
        assert serialized["displayed_frame_number"] == 13
    assert isinstance(serialized["image"], dict)


def test_synchronized_marks_failed_window_frames_not_analyzed_instead_of_no_action():
    # Frames 2-11 failed, then 12-21 succeeded with no actions. When playback
    # shows frame 2, it must not claim "No action".
    block = ActionRecognitionVisualizationBlockV1()
    for number in range(1, 21):
        _synchronized(block, number, analyzed_until_frame=None)

    # Frame 21 lands the first successful window covering 12-21 only.
    first = _synchronized(block, 21, analyzed_from_frame=12, analyzed_until_frame=21)
    second = _synchronized(block, 22, analyzed_from_frame=12, analyzed_until_frame=21)

    assert first["displayed_frame_number"] == 1
    assert first["status"] == "not_analyzed"
    assert first["actions"] == []
    assert first["actions_frame_number"] is None
    assert second["displayed_frame_number"] == 2
    assert second["status"] == "not_analyzed"
    assert second["actions"] == []
    assert second["actions_frame_number"] is None

    # Compare frame 2 ("Not analyzed" + hatched gap) against a frame that was
    # actually covered ("No action" + clean strip).
    covered_block = ActionRecognitionVisualizationBlockV1()
    for number in range(1, 21):
        _synchronized(covered_block, number, analyzed_until_frame=None)
    _synchronized(covered_block, 21, analyzed_from_frame=1, analyzed_until_frame=21)
    covered_second = _synchronized(
        covered_block, 22, analyzed_from_frame=1, analyzed_until_frame=21
    )
    assert covered_second["status"] == "synchronized"
    assert covered_second["actions_frame_number"] == 2
    assert not np.array_equal(
        second["image"].numpy_image, covered_second["image"].numpy_image
    )

    # Once playback reaches frame 12, the covered window takes over.
    for number in range(23, 33):
        result = _synchronized(
            block, number, analyzed_from_frame=12, analyzed_until_frame=21
        )
    assert result["displayed_frame_number"] == 12
    assert result["status"] == "synchronized"
    assert result["actions_frame_number"] == 12


@pytest.mark.parametrize(
    ("ranges", "expected_status", "actions_frame"),
    [
        (((2, 22),), "synchronized", 14),
        (((2, 8), (16, 22)), "not_analyzed", None),
    ],
)
def test_synchronized_recovers_quiet_window_coverage_after_display_skips_frames(
    ranges, expected_status, actions_frame
):
    block = ActionRecognitionVisualizationBlockV1()
    for number in range(1, 15):
        predictions = _predictions() if number < 8 else _predictions(until=8, start=2)
        block.run(_image(number), predictions, display_mode="synchronized")

    # Continue If skips the display on inputs 15–21. At 22 the latest window
    # alone cannot tell us whether the model classified buffered frame 14.
    resumed = block.run(
        _image(22),
        ActionRecognitionResult(
            timeline=[],
            analyzed_from_frame=16,
            analyzed_until_frame=22,
            covered_frame_ranges=ranges,
        ),
        display_mode="synchronized",
    )

    assert resumed["displayed_frame_number"] == 14
    assert resumed["status"] == expected_status
    assert resumed["actions_frame_number"] == actions_frame
    assert resumed["actions"] == []


def test_engine_skips_step_when_upstream_image_is_none():
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    engine = ExecutionEngine.init(
        workflow_definition={
            "version": "1.0",
            "inputs": [
                {"type": "WorkflowImage", "name": "image"},
                {
                    "type": "WorkflowBatchInput",
                    "name": "predictions",
                    "kind": ["action_recognition_result"],
                },
            ],
            "steps": [
                {
                    "type": TYPE,
                    "name": "display",
                    "image": "$inputs.image",
                    "predictions": "$inputs.predictions",
                    "display_mode": "synchronized",
                }
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "image",
                    "selector": "$steps.display.image",
                },
                {
                    "type": "JsonField",
                    "name": "status",
                    "selector": "$steps.display.status",
                },
            ],
        }
    )

    result = engine.run(
        runtime_parameters={
            "image": [None],
            "predictions": [_predictions()],
        }
    )[0]

    assert result == {"image": None, "status": None}


def test_engine_wires_model_predictions_into_visualization():
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    from inference_models import (
        ActionRecognitionPrediction as ModelActionRecognitionPrediction,
    )
    from inference_models import VideoSampling

    class _StubModel:
        class_names = ["wave"]
        video_sampling = VideoSampling(window_seconds=0.5, sample_fps=2.0, min_frames=1)
        supports_confidence = False
        supports_observed_duration = False

        def infer(self, frames, class_names=None, fps=None, **kwargs):
            return [
                ModelActionRecognitionPrediction(
                    start_frame_idx=0, end_frame_idx=len(frames) - 1, class_name="wave"
                )
            ]

    class _StubModelsProvider:
        def load_action_recognition_model(self, model_id, api_key=None):
            return _StubModel()

    engine = ExecutionEngine.init(
        workflow_definition={
            "version": "1.0",
            "inputs": [{"type": "WorkflowImage", "name": "image"}],
            "steps": [
                {
                    "type": "roboflow_core/roboflow_action_recognition_model@v1",
                    "name": "actions",
                    "images": "$inputs.image",
                    "model_id": "stub-model/1",
                    "stride_seconds": 0.2,
                },
                {
                    "type": TYPE,
                    "name": "display",
                    "image": "$inputs.image",
                    "predictions": "$steps.actions.predictions",
                    "display_mode": "live",
                },
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "status",
                    "selector": "$steps.display.status",
                },
                {
                    "type": "JsonField",
                    "name": "actions",
                    "selector": "$steps.display.actions",
                },
                {
                    "type": "JsonField",
                    "name": "predictions",
                    "selector": "$steps.actions.predictions",
                },
            ],
        },
        init_parameters={"workflows_core.model_manager": _StubModelsProvider()},
    )

    try:
        results = []
        for number in range(1, 5):
            frame = _image(number, fps=10)
            results.append(
                engine.run(
                    runtime_parameters={
                        "image": [
                            {
                                "type": "numpy_object",
                                "value": frame.numpy_image,
                                "video_metadata": frame.video_metadata,
                            }
                        ]
                    },
                    serialize_results=True,
                )[0]
            )
    finally:
        engine.close()

    assert results[0]["status"] == "analyzing"
    assert results[0]["predictions"]["analyzed_until_frame"] is None
    assert results[2]["status"] == "live"
    assert results[2]["actions"] == ["wave"]
    assert results[2]["predictions"]["analyzed_from_frame"] == 1
    assert results[2]["predictions"]["analyzed_until_frame"] == 1


def test_single_jpeg_larger_than_budget_is_released_after_run():
    block = ActionRecognitionVisualizationBlockV1()
    huge_noise = _image(1, size=(1440, 2560), noise=True)

    result = block.run(
        huge_noise,
        _predictions(),
        display_mode="synchronized",
        max_frame_side=2560,
        max_buffer_mb=1,
    )

    total_bytes = sum(stream.buffered_bytes for stream in block._streams.values())
    assert result["displayed_frame_number"] == 1
    assert total_bytes <= 1024**2
    assert total_bytes == 0


def test_synchronized_resets_playback_when_model_resets_timeline():
    block = ActionRecognitionVisualizationBlockV1()
    old_timeline = [_action("wave", 1, 22)]

    for number in range(1, 23):
        mark = 10 if number >= 12 else None
        if number >= 22:
            mark = 20
        block.run(
            _image(number),
            _predictions(old_timeline if mark is not None else [], mark),
            display_mode="synchronized",
        )

    # Frame 23: the model reset (for example after changing class_filter) and
    # reports analyzed_until_frame=None.
    reset_frame = block.run(
        _image(23),
        _predictions(),
        display_mode="synchronized",
    )
    assert reset_frame["status"] == "analyzing"
    assert reset_frame["displayed_frame_number"] == 23
    assert reset_frame["actions"] == []

    for number in range(24, 30):
        waiting = block.run(
            _image(number),
            _predictions(),
            display_mode="synchronized",
        )
        assert waiting["displayed_frame_number"] == 23

    new_timeline = [_action("jump", 23, 28)]
    resumed = block.run(
        _image(30),
        _predictions(new_timeline, 28, start=23),
        display_mode="synchronized",
    )
    assert resumed["displayed_frame_number"] == 23
    assert resumed["status"] == "synchronized"
    assert resumed["actions"] == ["jump"]


def test_close_releases_all_buffered_synchronized_frames():
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    engine = ExecutionEngine.init(
        workflow_definition={
            "version": "1.0",
            "inputs": [
                {"type": "WorkflowImage", "name": "image"},
                {
                    "type": "WorkflowBatchInput",
                    "name": "predictions",
                    "kind": ["action_recognition_result"],
                },
            ],
            "steps": [
                {
                    "type": TYPE,
                    "name": "display",
                    "image": "$inputs.image",
                    "predictions": "$inputs.predictions",
                    "display_mode": "synchronized",
                }
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "status",
                    "selector": "$steps.display.status",
                }
            ],
        }
    )

    for number in range(1, 10):
        frame = _image(number, size=(360, 640), noise=True)
        engine.run(
            runtime_parameters={
                "image": [
                    {
                        "type": "numpy_object",
                        "value": frame.numpy_image,
                        "video_metadata": frame.video_metadata,
                    }
                ],
                "predictions": [_predictions()],
            }
        )

    step = engine._engine._compiled_workflow.steps["display"].step
    assert sum(stream.buffered_bytes for stream in step._streams.values()) > 0

    engine.close()

    assert sum(stream.buffered_bytes for stream in step._streams.values()) == 0
    assert step._streams == {}


def test_live_badge_shows_sub_second_lag_when_displayed_frame_is_ahead():
    timeline = [_action("wave", 1, 20)]
    current_result = ActionRecognitionVisualizationBlockV1().run(
        _image(20), _predictions(timeline, 20)
    )
    lagged_result = ActionRecognitionVisualizationBlockV1().run(
        _image(20), _predictions(timeline, 14)
    )

    top_band = slice(0, 40)
    assert lagged_result["actions"] == ["wave"]
    assert lagged_result["actions_frame_number"] == 14
    assert not np.array_equal(
        current_result["image"].numpy_image[top_band],
        lagged_result["image"].numpy_image[top_band],
    )


def test_synchronized_badge_shows_the_current_delay(drawn_labels):
    block = ActionRecognitionVisualizationBlockV1()
    for number in range(1, 12):
        _synchronized(block, number)
    held_labels = list(drawn_labels)
    drawn_labels.clear()

    result = _synchronized(
        block, 12, timeline=[_action("wave", 1, 5)], analyzed_until_frame=10
    )

    # While the first frame is held its badge already counts the wait.
    assert not any(label.startswith("Delay") for label in held_labels)
    assert result["displayed_frame_number"] == 1
    assert drawn_labels[:2] == ["wave", "Delay 1.1 s"]


def test_live_badge_never_shows_a_delay(drawn_labels):
    ActionRecognitionVisualizationBlockV1().run(
        _image(40), _predictions([_action("wave", 5, 21)], 30)
    )

    assert not any(label.startswith("Delay") for label in drawn_labels)


def _strip_column(result, frame_number, *, displayed, history_seconds):
    # The lower band at the strip x of ``frame_number``, from the same geometry.
    pixels = result["image"].numpy_image
    width = pixels.shape[1]
    _, pad, _ = visualization_module._sizes(pixels)
    x0, x1 = pad * 2, width - pad * 2
    history_frames = round(history_seconds * FPS)
    start = displayed - history_frames
    x = round(x0 + (frame_number - start) / history_frames * (x1 - x0))
    column = pixels[pixels.shape[0] // 2 :, x, 0]
    return column


@pytest.mark.parametrize("display_mode", ["live", "synchronized"])
def test_strip_does_not_hatch_frames_before_the_stream_started(display_mode):
    # Joining mid-stream at frame 100: frames 90-99 predate the stream, so they
    # are not failed windows. Frames 106-110 after the mark still are.
    block = ActionRecognitionVisualizationBlockV1()
    hatched_lane = int(55 * 0.6 + 40)
    for number in range(100, 111):
        until = 105 if number >= 105 else None
        result = block.run(
            _image(number),
            _predictions(until=until, start=100),
            display_mode=display_mode,
            history_seconds=2.0,
        )

    displayed = result["displayed_frame_number"]
    pre_stream = _strip_column(result, 95, displayed=displayed, history_seconds=2.0)
    assert hatched_lane not in pre_stream
    if display_mode == "live":
        gap = _strip_column(result, 108, displayed=displayed, history_seconds=2.0)
        assert hatched_lane in gap


def test_uncovered_spans_start_at_the_first_frame_of_the_stream():
    spans = visualization_module._uncovered_spans(
        [(100, 105)], visible=[], start=90, first_frame=100, displayed=110
    )
    assert spans == [(106, 110)]

    before_any_result = visualization_module._uncovered_spans(
        [], visible=[], start=-290, first_frame=1, displayed=10
    )
    assert before_any_result == [(1, 10)]

    displayed_before_stream = visualization_module._uncovered_spans(
        [], visible=[], start=90, first_frame=100, displayed=95
    )
    assert displayed_before_stream == []
