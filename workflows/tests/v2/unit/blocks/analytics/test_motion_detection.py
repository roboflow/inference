"""``roboflow_core/motion_detection@v1`` called directly on generated scenes.

Scene: a static gradient with a 32 px white square at ``x = 12 * t``. With
``history=5`` the oracle is analytic: frames 0-4 are warmup, then every
frame has one region, the strip the square newly covers::

    t        0..4      5                 6                 ...
    motion   False     True              True
    alarm    False     True              False
    box      -         [80, 32, 92, 64]  [92, 32, 104, 64]  [12t+20, 32, 12t+32, 64]

A static scene never has motion. Interleaved sources must give the results
of the same sources run alone. Zones: ``None`` watches the whole image, ``[]``
nothing, and a list of polygons their union.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.analytics.background_subtraction.v1 import BackgroundModelCapacityError
from roboflow_workflows.v2.blocks.analytics.motion_detection.v1 import DEFAULT_MINIMUM_CONTOUR_AREA, DEFAULT_MORPHOLOGICAL_KERNEL_SIZE, MotionDetectionBlock, _moving_contours, _simplified_polygons
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.prediction_ops import class_names_by_id
from roboflow_workflows.v2.data.row_identity import row_record

T0 = datetime(2026, 10, 13)
HEIGHT, WIDTH, SIDE, STEP, TOP = 96, 192, 32, 12, 32
HISTORY = 5


def scene(x: Optional[int]) -> torch.Tensor:
    pixels = (torch.arange(WIDTH) // 3).to(torch.uint8).expand(3, HEIGHT, WIDTH)
    pixels = pixels.clone()
    if x is not None:
        pixels[:, TOP : TOP + SIDE, x : x + SIDE] = 255

    return pixels


def frame(
    t: int,
    *,
    video: str = "cam-a",
    number: Optional[int] = None,
    x: Any = "moving",
    width: int = WIDTH,
) -> ImageData:
    metadata = VideoMetadata(
        video_identifier=video,
        frame_number=t if number is None else number,
        frame_timestamp=T0 + timedelta(seconds=t),
    )
    pixels = scene(STEP * t if x == "moving" else x)[:, :, :width].clone()
    image = ImageData.from_tensor(pixels, video_metadata=metadata)

    return image


def summary(result: Dict[str, Any]) -> tuple:
    boxes = [[int(v) for v in box] for box in result["detections"].xyxy.tolist()]

    return result["motion"], result["alarm"], boxes


def run_all(block: MotionDetectionBlock, images: List[ImageData], **params) -> list:
    params = {"history": HISTORY, **params}

    return [summary(block.run(image=image, **params)) for image in images]


def oracle(t: int) -> tuple:
    if t < HISTORY:
        return False, False, []

    return True, t == HISTORY, [[STEP * t + 20, TOP, STEP * t + SIDE, TOP + SIDE]]


def test_moving_square_matches_the_analytic_oracle():
    results = run_all(MotionDetectionBlock(), [frame(t) for t in range(14)])

    assert results == [oracle(t) for t in range(14)]


def test_static_scene_never_moves_and_returns_empty_aligned_outputs():
    block = MotionDetectionBlock()
    for t in range(12):
        image = frame(t, x=None)
        result = block.run(image=image, history=HISTORY)
        assert result["motion"] is False and result["alarm"] is False
        assert result["motion_zones"] == []
        detections = result["detections"]
        assert tuple(detections.xyxy.shape) == (0, 4)
        assert detections.class_id.shape == detections.confidence.shape == (0,)
        assert detections.bboxes_metadata == []
        assert frame_of(detections) == FrameRef.from_image(image)


def test_regions_are_aligned_across_detections_and_motion_zones():
    block = MotionDetectionBlock()
    result = [block.run(image=frame(t), history=HISTORY) for t in range(7)][-1]
    detections = result["detections"]

    assert len(result["motion_zones"]) == len(detections.xyxy) == 1
    polygon = torch.tensor(result["motion_zones"][0])
    x0, y0 = polygon.min(dim=0).values.tolist()
    x1, y1 = polygon.max(dim=0).values.tolist()
    assert detections.xyxy.tolist() == [[x0, y0, x1 + 1, y1 + 1]]
    assert detections.class_id.tolist() == [0]
    assert detections.confidence.tolist() == [1.0]
    assert class_names_by_id(detections.image_metadata) == {0: "motion"}
    assert detections.image_metadata["prediction_type"] == "object-detection"
    assert detections.bboxes_metadata[0]["detection_id"]
    assert row_record(detections) is not None


def test_without_warmup_suppression_the_first_frame_is_all_motion():
    results = run_all(
        MotionDetectionBlock(), [frame(0), frame(1)], suppress_first_detections=False
    )

    assert results[0] == (True, True, [[0, 0, WIDTH, HEIGHT]])
    assert results[1][1] is False


def test_discarded_fractional_regions_do_not_start_or_preserve_motion():
    # The zone is valid, but its corners truncate to one pixel. On frame 0
    # the foreground covers it; on frame 5 the moving square covers it again.
    fractional_zone = [[90.1, 40.1], [90.4, 40.1], [90.1, 40.4]]
    block = MotionDetectionBlock()
    for t in range(7):
        result = block.run(
            image=frame(t),
            history=HISTORY,
            detection_zone=fractional_zone if t in (0, 5) else None,
            suppress_first_detections=False,
        )
        motion = t not in (0, 5)
        assert (result["motion"], result["alarm"]) == (motion, t in (1, 6))
        assert block._sources._states["cam-a"].previous_motion is motion
        assert len(result["motion_zones"]) == len(result["detections"].xyxy)
        if motion:
            assert result["motion_zones"]
        else:
            assert result["motion_zones"] == []
            assert tuple(result["detections"].xyxy.shape) == (0, 4)


# A 3x110 px bar: its contour area (2 x 109 = 218) passes the default filter
# (> 200), but simplification by 1% of its perimeter leaves 2 points.
THIN_BAR = (slice(40, 43), slice(6, 116))
THIN_FRAME = 8


def thin_bar_frame(t: int) -> ImageData:
    # Gradient, then the bar alone on frame 8, then a square from x = 120.
    pixels = scene(120 + STEP * (t - THIN_FRAME - 1) if t > THIN_FRAME else None)
    if t == THIN_FRAME:
        pixels[:, THIN_BAR[0], THIN_BAR[1]] = 255
    metadata = VideoMetadata(
        video_identifier="cam-a",
        frame_number=t,
        frame_timestamp=T0 + timedelta(seconds=t),
    )

    return ImageData.from_tensor(pixels, video_metadata=metadata)


def test_unzoned_thin_region_is_not_motion_and_the_next_region_alarms():
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    mask[THIN_BAR] = 255
    contours = _moving_contours(
        mask,
        kernel_size=DEFAULT_MORPHOLOGICAL_KERNEL_SIZE,
        minimum_area=DEFAULT_MINIMUM_CONTOUR_AREA,
    )
    assert len(contours) == 1 and _simplified_polygons(contours) == []
    # No zone, suppression off. Frame 0 is all motion, frames 1-7 are quiet.
    # Frame 8 shows only the bar: no motion, and previous motion stays false,
    # so frame 9's square alarms. Frame 10 continues without an alarm.
    # (V1: motion and alarm on frame 8, then no alarm on frame 9.)
    block = MotionDetectionBlock()
    for t in range(THIN_FRAME + 3):
        result = block.run(
            image=thin_bar_frame(t), history=HISTORY, suppress_first_detections=False
        )
        motion = t == 0 or t > THIN_FRAME
        alarm = t in (0, THIN_FRAME + 1)
        assert (result["motion"], result["alarm"]) == (motion, alarm), t
        assert block._sources._states["cam-a"].previous_motion is motion
        assert len(result["motion_zones"]) == len(result["detections"].xyxy)
        assert bool(result["motion_zones"]) is motion


def test_alarm_fires_again_when_motion_resumes_after_stopping():
    # Square moves 0..8, rests at x=96 for 12 frames, then moves on.
    xs = [STEP * t for t in range(9)] + [96] * 12 + [108, 120, 132]
    block = MotionDetectionBlock()
    results = [
        summary(block.run(image=frame(t, x=x), history=HISTORY))
        for t, x in enumerate(xs)
    ]
    motion = [r[0] for r in results]
    alarms = [t for t, r in enumerate(results) if r[1]]

    assert motion[HISTORY:9] == [True] * (9 - HISTORY)
    assert motion[-4] is False and motion[-3:] == [True] * 3
    assert alarms[0] == HISTORY and alarms[-1] == len(xs) - 3
    # The resting square became background: leaving it reveals a region too.
    leading, revealed = [128, TOP, 140, TOP + SIDE], [96, TOP, 108, TOP + SIDE]
    assert sorted(results[-3][2]) == [revealed, leading]


def test_interleaved_sources_match_the_same_sources_run_alone():
    moving = [frame(t, video="moving") for t in range(12)]
    static = [frame(t, video="static", x=None) for t in range(12)]
    late = [frame(t, video="late", number=t + 100) for t in range(8)]
    alone = {
        "moving": run_all(MotionDetectionBlock(), moving),
        "static": run_all(MotionDetectionBlock(), static),
        "late": run_all(MotionDetectionBlock(), late),
    }

    block = MotionDetectionBlock()
    together: Dict[str, list] = {"moving": [], "static": [], "late": []}
    for pulse in range(12):
        members = [moving[pulse], static[pulse]]
        if pulse >= 4:
            members.insert(1, late[pulse - 4])
        for image in members:
            source = image.video_metadata.video_identifier
            together[source] += run_all(block, [image])

    assert together == alone


@pytest.mark.parametrize(
    "again",
    [
        pytest.param({}, id="lower-frame-number"),
        pytest.param({"number_offset": 100, "width": 180}, id="new-size"),
    ],
)
def test_restart_resets_model_warmup_and_alarm_together(again):
    # The source's whole record restarts: warmup runs again and, as the
    # previous motion is forgotten, the alarm fires again.
    offset, width = again.get("number_offset", 0), again.get("width", WIDTH)
    block = MotionDetectionBlock()
    first = run_all(block, [frame(t) for t in range(8)])
    second = run_all(
        block, [frame(t, number=t + offset, width=width) for t in range(8)]
    )

    assert first == second == [oracle(t) for t in range(8)]


def test_plain_run_without_video_metadata_uses_the_default_source():
    # No engine context and no video id: a direct call keys by "default".
    block = MotionDetectionBlock()
    images = [ImageData.from_tensor(scene(STEP * t)) for t in range(8)]

    assert run_all(block, images) == [oracle(t) for t in range(8)]
    assert sorted(block._sources._states) == ["default"]


def test_reset_state_forgets_every_source():
    block = MotionDetectionBlock()
    run_all(block, [frame(t, video=v) for t in range(8) for v in ("a", "b")])
    block.reset_state()

    assert run_all(block, [frame(8, video="a")]) == [(False, False, [])]


def test_capacity_error_is_explicit_and_leaves_known_sources_running():
    block = MotionDetectionBlock()
    for video in ("a", "b"):
        block.run(image=frame(0, video=video), max_sources=2)

    with pytest.raises(BackgroundModelCapacityError, match="max_sources=2"):
        block.run(image=frame(0, video="c"), max_sources=2)
    for t in range(1, 7):
        result = summary(
            block.run(image=frame(t, video="a"), history=HISTORY, max_sources=2)
        )
    assert result == oracle(6)


ZONE = [[0, 0], [110, 0], [110, HEIGHT], [0, HEIGHT]]


@pytest.mark.parametrize("zone", [ZONE, json.dumps(ZONE)], ids=["points", "json"])
def test_detection_zone_clips_regions_and_ignores_motion_outside(zone):
    results = run_all(
        MotionDetectionBlock(), [frame(t) for t in range(10)], detection_zone=zone
    )

    assert results[5:8] == [oracle(5), oracle(6), (True, False, [[104, 32, 111, 64]])]
    assert results[8:] == [(False, False, [])] * 2


def test_no_zone_watches_the_whole_image():
    images = [frame(t) for t in range(10)]

    assert run_all(MotionDetectionBlock(), images, detection_zone=None) == [
        oracle(t) for t in range(10)
    ]


@pytest.mark.parametrize(
    "zone", [[], [None, None], "[]"], ids=["empty", "nones", "json"]
)
def test_empty_zone_watches_nothing(zone):
    # Unlike V1, [] is no region: an empty dynamic zone result never turns
    # into whole-image motion.
    results = run_all(
        MotionDetectionBlock(), [frame(t) for t in range(10)], detection_zone=zone
    )

    assert results == [(False, False, [])] * 10


RIGHT = [[150, 0], [WIDTH, 0], [WIDTH, HEIGHT], [150, HEIGHT]]


def test_zone_list_uses_the_union_of_its_polygons():
    images = [frame(t) for t in range(14)]
    union = run_all(MotionDetectionBlock(), images, detection_zone=[ZONE, None, RIGHT])
    left = run_all(MotionDetectionBlock(), images, detection_zone=ZONE)
    right = run_all(MotionDetectionBlock(), images, detection_zone=RIGHT)

    assert [r[2] for r in union] == [a[2] + b[2] for a, b in zip(left, right)]
    assert [t for t, r in enumerate(union) if r[0]] == [5, 6, 7, 10, 11, 12, 13]
    # Motion stops in the gap between the polygons, so it alarms again.
    assert [t for t, r in enumerate(union) if r[1]] == [5, 10]
    assert union[10][2] == [[150, TOP, 152, TOP + SIDE]]


def test_overlapping_polygons_clip_like_their_merged_polygon():
    halves = [
        [[0, 0], [60, 0], [60, HEIGHT], [0, HEIGHT]],
        [[50, 0], [110, 0], [110, HEIGHT], [50, HEIGHT]],
    ]
    images = [frame(t) for t in range(10)]

    assert run_all(MotionDetectionBlock(), images, detection_zone=halves) == run_all(
        MotionDetectionBlock(), images, detection_zone=ZONE
    )


def test_detections_keep_the_geometry_of_a_cropped_image():
    block = MotionDetectionBlock()
    for t in range(7):
        crop = frame(t).crop((0, 16, WIDTH, 80))
        result = block.run(image=crop, history=HISTORY)

    detections = result["detections"]
    assert detections.xyxy.tolist() == [[92, 16, 104, 48]]
    assert frame_of(detections) == FrameRef.from_image(crop)
    assert crop.root.map_xy(92, 16) == (92, 32)


def test_noncontiguous_input_matches_contiguous_input():
    def view(image: ImageData) -> ImageData:
        pixels = image.tensor_image.permute(1, 2, 0).contiguous().permute(2, 0, 1)

        return ImageData.from_tensor(pixels, video_metadata=image.video_metadata)

    images = [frame(t) for t in range(8)]

    assert run_all(MotionDetectionBlock(), [view(i) for i in images]) == run_all(
        MotionDetectionBlock(), images
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_detections_stay_on_the_input_device():
    image = ImageData.from_tensor(
        scene(0).to("cuda"), video_metadata=frame(0).video_metadata
    )
    detections = MotionDetectionBlock().run(image=image)["detections"]

    assert detections.xyxy.device == image.device


@pytest.mark.parametrize(
    "params, message",
    [
        (
            {"minimum_contour_area": 0},
            "minimum_contour_area must be a positive integer",
        ),
        ({"morphological_kernel_size": 2.0}, "morphological_kernel_size must be"),
        ({"history": None}, "history must be a positive integer"),
        (
            {"suppress_first_detections": "yes"},
            "suppress_first_detections must be a bool",
        ),
        ({"detection_zone": [[0, 0], [1, 1]]}, "at least 3"),
        ({"detection_zone": [[0, 0], [1, float("nan")], [2, 0]]}, "finite numbers"),
        ({"detection_zone": [[0, 0], [9, 9], [9, 0], [0, 9]]}, "simple polygon"),
        ({"detection_zone": "[[0, 0], [1"}, "not valid JSON"),
        ({"detection_zone": {"x": 1}}, "list of at least 3"),
        ({"detection_zone": [ZONE, [[0, 0], [1, 1]]]}, "at least 3"),
        ({"detection_zone": [ZONE, "x"]}, "at least 3"),
    ],
)
def test_selected_values_are_checked_with_clear_errors(params, message):
    with pytest.raises(ValueError, match=message):
        MotionDetectionBlock().run(image=frame(0), **params)


def test_literal_parameters_are_validated_and_selectors_accepted():
    params: Any = MotionDetectionBlock.Params
    with pytest.raises(ValidationError, match="at least 3"):
        params(image="$inputs.image", detection_zone=[[0, 0], [1, 1]])
    with pytest.raises(ValidationError, match="at least 3"):
        params(image="$inputs.image", detection_zone=[ZONE, [[0, 0]]])
    params(image="$inputs.image", detection_zone=[ZONE, None, RIGHT])
    params(image="$inputs.image", detection_zone=[])
    with pytest.raises(ValidationError):
        params(image="$inputs.image", minimum_contour_area=0)
    with pytest.raises(ValidationError):
        params(image="$inputs.image", suppress_first_detections=1)

    params(
        image="$inputs.image",
        detection_zone="$inputs.zone",
        suppress_first_detections="$inputs.warmup",
        history="$inputs.history",
    )
