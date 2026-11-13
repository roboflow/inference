"""Action Recognition Visualization: live strip and synchronized playback."""

from datetime import datetime

import numpy as np
import pytest
import torch
from PIL import ImageFont
from pydantic import ValidationError
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


def _shade(result):
    # Top-right corner: outside the badges and above the strip.
    return int(result["image"].numpy_image[1, -2, 0])


def _strip_changed(result, number):
    return not np.all(result["image"].numpy_image[-12:] == number % 256)


def _synchronized(block, number, *, image=None, analyzed_until_frame=None, **kwargs):
    result = block.run(
        _image(number, **(image or {})),
        [],
        analyzed_until_frame=analyzed_until_frame,
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
                timeline,
                analyzed_until_frame=mark,
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
    assert results[25]["actions_frame_number"] == 21


def test_live_names_the_actions_at_the_last_analyzed_frame():
    block = ActionRecognitionVisualizationBlockV1()
    timeline = [_action("wave", 5, 21), _action("jump", 18, 30, class_id=1)]

    during = block.run(_image(40), timeline, analyzed_until_frame=21)
    after = block.run(_image(41), timeline, analyzed_until_frame=40)
    quiet = block.run(_image(42), timeline, analyzed_until_frame=41)

    assert during["actions"] == ["jump", "wave"]
    assert during["actions_frame_number"] == 21
    assert after["actions"] == []
    assert quiet["actions_frame_number"] == 41


@pytest.mark.parametrize("mark", [None, 45])
def test_live_draws_a_strip_on_a_copy_with_or_without_the_mark(mark):
    block = ActionRecognitionVisualizationBlockV1()
    source = _image(50)

    result = block.run(source, [_action("wave", 20, 40)], analyzed_until_frame=mark)

    assert result["status"] == ("analyzing" if mark is None else "live")
    assert result["displayed_frame_number"] == 50
    assert _strip_changed(result, 50)
    assert np.all(source.numpy_image == 50)
    assert result["image"].numpy_image.shape == source.numpy_image.shape


def test_synchronized_holds_the_first_frame_then_plays_from_it_without_gaps():
    block = ActionRecognitionVisualizationBlockV1()
    model = _Model(stride=20, latency=5, timeline=[_action("wave", 3, 8)])

    # A repeated frame number must be neither stored nor shown twice.
    results = _play(block, model, [1, 1, *range(2, 121)], display_mode="synchronized")

    held = results[:26]
    assert {result["status"] for result in held} == {"analyzing"}
    assert {result["displayed_frame_number"] for result in held} == {1}
    assert {tuple(result["actions"]) for result in held} == {()}
    playing = results[26:]
    assert {result["status"] for result in playing} == {"synchronized"}
    shown = [result["displayed_frame_number"] for result in playing]
    # Playback continues from the held first frame: no skip, no repeat.
    assert shown == list(range(2, len(playing) + 2))
    assert all(
        result["displayed_frame_number"] <= model.at(number)[0]
        for number, result in enumerate(playing, 26)
    )


def test_synchronized_puts_each_label_on_the_frame_it_describes():
    block = ActionRecognitionVisualizationBlockV1()
    model = _Model(stride=20, latency=5, timeline=[_action("wave", 3, 8)])

    results = _play(block, model, range(1, 60), display_mode="synchronized")

    labelled = [
        result["displayed_frame_number"]
        for result in results
        if result["actions"] == ["wave"]
    ]
    assert labelled == [3, 4, 5, 6, 7, 8]
    for result in results[25:]:
        assert result["actions_frame_number"] == result["displayed_frame_number"]
        assert abs(_shade(result) - result["displayed_frame_number"]) <= 2
        assert (
            result["image"].video_metadata.frame_number
            == result["displayed_frame_number"]
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


def test_synchronized_recovers_its_delay_after_one_slow_call():
    block = ActionRecognitionVisualizationBlockV1()
    marks = {}
    # Results land every 10 frames with a lag of 2, except one that is 30 late.
    for number in range(1, 200):
        landed = (number - 2) // 10 * 10
        if 50 <= number < 82:
            landed = 40
        marks[number] = landed if landed >= 10 else None

    delays = []
    for number in range(1, 200):
        result = _synchronized(block, number, analyzed_until_frame=marks[number])
        delays.append(number - result["displayed_frame_number"])

    assert max(delays[49:90]) >= 40
    assert max(delays[-20:]) <= 12


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


@pytest.mark.parametrize("tensor", [False, True])
def test_synchronized_frames_are_downscaled_before_they_are_kept(tensor):
    block = ActionRecognitionVisualizationBlockV1()
    source = _image(1, size=(360, 640), tensor=tensor)

    result = block.run(
        source,
        [],
        analyzed_until_frame=None,
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
        _image(50), timeline, analyzed_until_frame=45
    )
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
        _image(50), timeline, analyzed_until_frame=45
    )

    assert fallback["actions"] == ["Saída 走る"]
    assert _strip_changed(fallback, 50)
    assert _strip_changed(with_font, 50)


def test_frame_too_small_for_a_strip_is_returned_with_its_outputs():
    block = ActionRecognitionVisualizationBlockV1()

    result = block.run(
        _image(50, size=(16, 24)),
        [_action("wave", 20, 60)],
        analyzed_until_frame=50,
    )

    assert result["image"].numpy_image.shape == (16, 24, 3)
    assert result["actions"] == ["wave"]


def test_run_rejects_an_unknown_display_mode_and_a_non_bgr_image():
    manifest = BlockManifest(
        type=TYPE,
        name="display",
        image="$inputs.image",
        timeline="$steps.model.timeline",
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
        block.run(_image(1), [], display_mode="delayed")
    with pytest.raises(ValueError, match="uint8 BGR"):
        block.run(gray, [])


def test_manifest_requires_the_analyzed_mark_for_synchronized_display():
    with pytest.raises(ValidationError, match="analyzed_until_frame"):
        BlockManifest(
            type=TYPE,
            name="display",
            image="$inputs.image",
            timeline="$steps.model.timeline",
            display_mode="synchronized",
        )


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
        "timeline": "$inputs.timeline",
        "analyzed_until_frame": "$inputs.mark",
    }
    if display_mode:
        step["display_mode"] = display_mode
    engine = ExecutionEngine.init(
        workflow_definition={
            "version": "1.0",
            "inputs": [
                {"type": "WorkflowImage", "name": "image"},
                {"type": "WorkflowBatchInput", "name": "timeline"},
                {"type": "WorkflowBatchInput", "name": "mark"},
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

    def run(number, timeline, mark, serialize=False):
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
                "timeline": [timeline],
                "mark": [mark],
            },
            serialize_results=serialize,
        )[0]

    warmup = run(1, [], None)
    timeline = [_action("wave", 1, 5)]
    for number in range(2, 12):
        run(number, [], None)
    ready = run(12, timeline, 10)
    serialized = run(13, timeline, 10, serialize=True)

    # A null mark must not skip the step: the video output would go missing.
    assert warmup["status"] == "analyzing"
    assert isinstance(warmup["image"], WorkflowImageData)
    if display_mode == "synchronized":
        assert ready["status"] == "synchronized"
        assert ready["displayed_frame_number"] == 2
        assert ready["actions"] == ["wave"]
        assert ready["image"].video_metadata.frame_number == 2
        assert serialized["displayed_frame_number"] == 3
    else:
        # The default display is live.
        assert ready["status"] == "live"
        assert ready["displayed_frame_number"] == 12
        assert ready["actions_frame_number"] == 10
        assert serialized["displayed_frame_number"] == 13
    assert isinstance(serialized["image"], dict)
