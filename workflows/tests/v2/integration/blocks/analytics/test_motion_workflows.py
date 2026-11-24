"""Motion blocks in compiled workflows on generated scenes.

Two cameras alternate through one session: ``moving`` shows a 32 px white
square advancing 12 px per frame over a static gradient, ``static`` shows
the gradient alone. With ``history=5`` the moving camera's oracle is::

    processed frame k   0..4    5          k >= 5
    motion              False   True       True
    alarm               False   True       False
    box                 -       newly covered strip [12t+20, 32, 12t+32, 64]

The static camera never has motion. Regions whose simplified outline keeps
fewer than 3 points (a subpixel zone, an unzoned thin bar) are never motion.
Every case runs in both block execution modes; one also runs pipelined with
two pulses in flight. The last three cases feed ``roboflow_core/dynamic_zone@v1`` zones
(regions candidate, a pinned test prerequisite) into ``detection_zone``:
union, empty and two sources.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.analytics.background_subtraction.v1 import BackgroundModelCapacityError, BackgroundSubtractionBlock
from roboflow_workflows.v2.blocks.analytics.motion_detection.v1 import MotionDetectionBlock
from roboflow_workflows.v2.blocks.control.continue_if_boolean.v1 import ContinueIfBlock
from roboflow_workflows.v2.blocks.geometry.dynamic_zone.v1 import DynamicZoneBlock
from roboflow_workflows.v2.blocks.image.dynamic_crop.v1 import DynamicCropBlock
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.integration.blocks.geometry.deterministic import (
    painted_image,
    regions_catalogue,
)

MODES = ["run", "phases"]
T0 = datetime(2026, 10, 13)
HEIGHT, WIDTH, SIDE, STEP, TOP = 96, 192, 32, 12, 32
HISTORY = 5
CATALOGUE = Catalogue(
    [
        BackgroundSubtractionBlock,
        MotionDetectionBlock,
        ContinueIfBlock,
        DynamicCropBlock,
    ],
    kinds=[IMAGE_KIND, BOOLEAN_KIND, INTEGER_KIND, LIST_OF_VALUES_KIND, STRING_KIND],
)
IMAGE_INPUT = {"type": "WorkflowImage", "name": "image"}
MOTION_STEP = {
    "type": MotionDetectionBlock.type,
    "name": "motion",
    "image": "$inputs.image",
    "history": HISTORY,
}
MOTION_OUTPUTS = {
    "motion": "$steps.motion.motion",
    "alarm": "$steps.motion.alarm",
    "detections": "$steps.motion.detections",
    "zones": "$steps.motion.motion_zones",
}


def frame(camera: str, t: int, *, x: Optional[int] = None) -> ImageData:
    pixels = (torch.arange(WIDTH) // 3).to(torch.uint8).expand(3, HEIGHT, WIDTH)
    pixels = pixels.clone()
    if camera == "moving":
        x = STEP * t if x is None else x
        pixels[:, TOP : TOP + SIDE, x : x + SIDE] = 255
    metadata = VideoMetadata(
        video_identifier=camera,
        frame_number=t,
        frame_timestamp=T0 + timedelta(seconds=t),
    )
    image = ImageData.from_tensor(pixels, video_metadata=metadata)

    return image


def oracle(t: int, *, processed: Optional[int] = None) -> tuple:
    processed = t if processed is None else processed
    if processed < HISTORY:
        return False, False, []

    alarm = processed == HISTORY
    return True, alarm, [[STEP * t + 20, TOP, STEP * t + SIDE, TOP + SIDE]]


def compile_plan(mode: str, steps: List[dict], outputs: Dict[str, str], inputs=()):
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT, *inputs],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )

    return plan


def values(result) -> Dict[str, Any]:
    def single(value: Any) -> Any:
        if isinstance(value, Batch):
            members = [member for _, member in value.iter_with_indices()]
            assert len(members) <= 1
            return members[0] if members else None
        return value

    return {name: single(value) for name, value in result.outputs.data.items()}


def summary(row: Dict[str, Any]) -> tuple:
    boxes = [[int(v) for v in box] for box in row["detections"].xyxy.tolist()]

    return row["motion"], row["alarm"], boxes


def flatten(value: Any) -> list:
    if not isinstance(value, Batch):
        return [value]

    return [item for _, member in value.iter_with_indices() for item in flatten(member)]


def interleaved(count: int):
    for t in range(count):
        for camera in ("moving", "static"):
            yield camera, t, {"image": frame(camera, t)}


@pytest.mark.parametrize("mode", MODES)
def test_interleaved_cameras_keep_separate_models(mode):
    steps = [
        MOTION_STEP,
        {
            "type": BackgroundSubtractionBlock.type,
            "name": "subtract",
            "image": "$inputs.image",
            "history": HISTORY,
        },
    ]
    plan = compile_plan(
        mode, steps, {**MOTION_OUTPUTS, "mask": "$steps.subtract.image"}
    )
    rows = {}
    with plan.create_session() as session:
        for camera, t, inputs in interleaved(12):
            row = values(session.run(inputs))
            rows[(camera, t)] = summary(row)
            mask = row["mask"]
            assert mask.size_hw == (HEIGHT, WIDTH) and mask.channels == 3
            assert mask.video_metadata.video_identifier == camera
            if camera == "static" and t >= 2:
                assert int(mask.tensor_image.count_nonzero()) == 0

    assert [rows[("moving", t)] for t in range(12)] == [oracle(t) for t in range(12)]
    assert [rows[("static", t)] for t in range(12)] == [(False, False, [])] * 12


@pytest.mark.parametrize("mode", MODES)
def test_pipelined_run_keeps_each_cameras_frame_order(mode):
    plan = compile_plan(mode, [MOTION_STEP], MOTION_OUTPUTS)
    with plan.create_session() as session:
        with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
            futures = [
                (camera, t, pipeline.submit(inputs))
                for camera, t, inputs in interleaved(10)
            ]
        rows = {(c, t): summary(values(f.result())) for c, t, f in futures}

    assert [rows[("moving", t)] for t in range(10)] == [oracle(t) for t in range(10)]
    assert all(rows[("static", t)][0] is False for t in range(10))


@pytest.mark.parametrize("mode", MODES)
def test_gated_frames_never_reach_the_model(mode):
    # Frames 2 and 8 are gated off. Warmup counts processed frames, so frame 6
    # is the first evaluated one; frame 9's strip starts where frame 7 ended.
    gate = {
        "type": ContinueIfBlock.type,
        "name": "gate",
        "condition": "$inputs.keep",
        "next_steps": ["$steps.motion"],
    }
    inputs = [{"type": "WorkflowParameter", "name": "keep"}]
    plan = compile_plan(mode, [gate, MOTION_STEP], MOTION_OUTPUTS, inputs=inputs)
    rows = {}
    with plan.create_session() as session:
        for t in range(10):
            keep = t not in (2, 8)
            row = values(session.run({"image": frame("moving", t), "keep": keep}))
            rows[t] = summary(row) if keep else row.get("motion")

    assert rows[2] is None and rows[8] is None
    assert [rows[t] for t in (0, 1, 3, 4, 5)] == [(False, False, [])] * 5
    assert rows[6] == oracle(6, processed=HISTORY)
    assert rows[7] == oracle(7)
    assert rows[9] == (
        True,
        False,
        [[STEP * 7 + SIDE, TOP, STEP * 9 + SIDE, TOP + SIDE]],
    )


@pytest.mark.parametrize("mode", MODES)
def test_selected_parameters_are_used_and_bad_values_fail_clearly(mode):
    step = {
        **MOTION_STEP,
        "history": "$inputs.history",
        "detection_zone": "$inputs.zone",
    }
    inputs = [
        {"type": "WorkflowParameter", "name": "history"},
        {"type": "WorkflowParameter", "name": "zone"},
    ]
    plan = compile_plan(mode, [step], MOTION_OUTPUTS, inputs=inputs)
    zone = [[0, 0], [110, 0], [110, HEIGHT], [0, HEIGHT]]
    with plan.create_session() as session:
        rows = [
            summary(
                values(
                    session.run(
                        {"image": frame("moving", t), "history": HISTORY, "zone": zone}
                    )
                )
            )
            for t in range(10)
        ]
        with pytest.raises(StepExecutionError) as failed:
            session.run(
                {"image": frame("moving", 10), "history": HISTORY, "zone": [[0, 0]]}
            )

    assert rows[5:8] == [
        oracle(5),
        oracle(6),
        (True, False, [[104, TOP, 111, TOP + SIDE]]),
    ]
    assert rows[8:] == [(False, False, [])] * 2
    assert isinstance(failed.value.__cause__, ValueError)
    assert "at least 3" in str(failed.value.__cause__)


def test_literal_zone_is_checked_when_compiling():
    step = {**MOTION_STEP, "detection_zone": [[0, 0], [5, 5]]}

    with pytest.raises(ParamsValidationError, match="at least 3"):
        compile_plan("run", [step], MOTION_OUTPUTS)


@pytest.mark.parametrize("mode", MODES)
def test_capacity_is_explicit_and_a_processing_reset_frees_it(mode):
    plan = compile_plan(mode, [{**MOTION_STEP, "max_sources": 1}], MOTION_OUTPUTS)
    with plan.create_session() as session:
        for t in range(7):
            session.run({"image": frame("moving", t)})
        with pytest.raises(StepExecutionError) as failed:
            session.run({"image": frame("static", 0)})
        cause = failed.value.__cause__
        assert isinstance(cause, BackgroundModelCapacityError)
        assert "'static'" in str(cause) and "max_sources=1" in str(cause)
        # The admitted camera keeps its model.
        assert summary(values(session.run({"image": frame("moving", 7)}))) == oracle(7)

        session.update(plan, reset=True)
        assert summary(values(session.run({"image": frame("static", 0)}))) == (
            False,
            False,
            [],
        )
        with pytest.raises(StepExecutionError):
            session.run({"image": frame("moving", 8)})


@pytest.mark.parametrize("mode", MODES)
def test_motion_boxes_crop_the_newly_covered_strip(mode):
    crop = {
        "type": DynamicCropBlock.type,
        "name": "crop",
        "image": "$inputs.image",
        "predictions": "$steps.motion.detections",
    }
    plan = compile_plan(mode, [MOTION_STEP, crop], {"crops": "$steps.crop.crops"})
    with plan.create_session() as session:
        for t in range(HISTORY + 2):
            result = session.run({"image": frame("moving", t)})

    images = flatten(result.outputs.data["crops"])
    assert len(images) == 1
    assert images[0].size_hw == (SIDE, STEP)
    assert bool((images[0].tensor_image == 255).all())
    assert images[0].root.map_xy(0, 0) == (STEP * (HISTORY + 1) + 20, TOP)


@pytest.mark.parametrize("mode", MODES)
def test_literal_fractional_zone_emits_no_motion_regions_or_crops(mode):
    step = {
        **MOTION_STEP,
        "detection_zone": [[90.1, 40.1], [90.4, 40.1], [90.1, 40.4]],
        "suppress_first_detections": False,
    }
    crop = {
        "type": DynamicCropBlock.type,
        "name": "crop",
        "image": "$inputs.image",
        "predictions": "$steps.motion.detections",
    }
    plan = compile_plan(
        mode, [step, crop], {**MOTION_OUTPUTS, "crops": "$steps.crop.crops"}
    )
    with plan.create_session() as session:
        result = session.run({"image": frame("moving", 0)})

    row = values(result)
    assert summary(row) == (False, False, [])
    assert row["zones"] == []
    assert tuple(row["detections"].xyxy.shape) == (0, 4)
    assert flatten(result.outputs.data["crops"]) == []


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("warmup", [False, True], ids=["warmup-off", "default-warmup"])
@pytest.mark.parametrize("reset", ["rewind", "processing"])
def test_fractional_region_transitions_rearm_alarm_and_survive_reset(
    mode, warmup, reset
):
    # Actual MOG2 foreground intersects the subpixel zone on both frame 0
    # and frame 5. Other frames use the whole image and see a moving strip.
    fractional_zone = [[90.1, 40.1], [90.4, 40.1], [90.1, 40.4]]
    step = {**MOTION_STEP, "detection_zone": "$inputs.zone"}
    if not warmup:
        step["suppress_first_detections"] = False
    crop = {
        "type": DynamicCropBlock.type,
        "name": "crop",
        "image": "$inputs.image",
        "predictions": "$steps.motion.detections",
    }
    plan = compile_plan(
        mode,
        [step, crop],
        {**MOTION_OUTPUTS, "crops": "$steps.crop.crops"},
        inputs=[{"type": "WorkflowParameter", "name": "zone"}],
    )
    with plan.create_session() as session:
        for lifetime in range(2):
            if lifetime and reset == "processing":
                session.update(plan, reset=True)
            for t in range(8):
                image = frame("moving", t)
                result = session.run(
                    {
                        "image": image,
                        "zone": fractional_zone if t in (0, 5) else None,
                    }
                )
                row = values(result)
                crops = flatten(result.outputs.data["crops"])
                # Frames 0 and 5 collapse: never motion. With warmup, frames
                # 0-4 are suppressed, so the first alarm is frame 6; without
                # it, frame 1 alarms and frame 6 alarms again after frame 5.
                motion = t not in (0, 5) and (not warmup or t >= HISTORY)
                alarm = t == 6 or (not warmup and t == 1)
                assert (row["motion"], row["alarm"]) == (motion, alarm)
                # Before the background stabilizes, leaving the initial
                # square can reveal a second region as well as its front.
                assert bool(crops) is motion
                assert_crops_show_their_boxes(row, crops, image)


# Unzoned thin region, then a usable one. A 3x110 px bar passes the default
# area filter (contour area 2 x 109 = 218 > 200), but simplification by 1% of
# its perimeter leaves 2 points, so it is not emitted.
THIN_BAR = (slice(40, 43), slice(6, 116))
THIN_FRAME = 8


def thin_bar_frame(t: int) -> ImageData:
    # Gradient, then the bar alone on frame 8, then a square from x = 120.
    pixels = (torch.arange(WIDTH) // 3).to(torch.uint8).expand(3, HEIGHT, WIDTH)
    pixels = pixels.clone()
    if t == THIN_FRAME:
        pixels[:, THIN_BAR[0], THIN_BAR[1]] = 255
    elif t > THIN_FRAME:
        x = 120 + STEP * (t - THIN_FRAME - 1)
        pixels[:, TOP : TOP + SIDE, x : x + SIDE] = 255
    metadata = VideoMetadata(
        video_identifier="thin",
        frame_number=t,
        frame_timestamp=T0 + timedelta(seconds=t),
    )

    return ImageData.from_tensor(pixels, video_metadata=metadata)


@pytest.mark.parametrize("mode", MODES)
def test_unzoned_thin_region_is_quiet_and_the_next_usable_region_alarms(mode):
    subtract = {
        "type": BackgroundSubtractionBlock.type,
        "name": "subtract",
        "image": "$inputs.image",
        "history": HISTORY,
    }
    crop = {
        "type": DynamicCropBlock.type,
        "name": "crop",
        "image": "$inputs.image",
        "predictions": "$steps.motion.detections",
    }
    outputs = {
        **MOTION_OUTPUTS,
        "mask": "$steps.subtract.image",
        "crops": "$steps.crop.crops",
    }
    plan = compile_plan(mode, [MOTION_STEP, subtract, crop], outputs)
    with plan.create_session() as session:
        for t in range(THIN_FRAME + 3):
            image = thin_bar_frame(t)
            result = session.run({"image": image})
            row = values(result)
            crops = flatten(result.outputs.data["crops"])
            # No zone, default warmup: frames 0-4 are suppressed and 5-7 are
            # a static gradient. Frame 8's bar is not motion and leaves
            # previous motion false, so frame 9's square alarms; frame 10
            # continues it. (V1: alarm on frame 8, none on frame 9.)
            motion = t > THIN_FRAME
            assert (row["motion"], row["alarm"]) == (motion, t == THIN_FRAME + 1), t
            assert bool(crops) is motion
            assert_crops_show_their_boxes(row, crops, image)
            if t == THIN_FRAME:
                # MOG2 saw exactly the bar; only simplification dropped it.
                foreground = row["mask"].tensor_image[0] > 0
                assert bool(foreground[THIN_BAR].all())
                assert int(foreground.sum()) == 3 * 110


def assert_crops_show_their_boxes(row: Dict[str, Any], crops: list, image) -> None:
    boxes = row["detections"].xyxy.tolist()
    assert tuple(row["detections"].xyxy.shape) == (len(crops), 4)
    assert len(row["zones"]) == len(boxes)
    for box, crop in zip(boxes, crops):
        x0, y0, x1, y1 = map(int, box)
        assert crop.size_hw == (y1 - y0, x1 - x0)
        assert torch.equal(crop.tensor_image, image.tensor_image[:, y0:y1, x0:x1])
        assert crop.root.map_xy(0, 0) == (x0, y0)


# Segmented area -> dynamic zone -> motion. Markers (deterministic.py) are
# static; a square painted into the blue channel only moves across them, so
# their masks, and the zones, stay the same while MOG2 sees motion.
LEFT_MARKER, RIGHT_MARKER = (0, 8, 110, 88, 1, 0), (150, 8, WIDTH, 88, 2, 0)
# Zone corners run through the outer pixel centres of each marker.
LEFT_ZONE = [[0, 8], [109, 8], [109, 87], [0, 87]]
RIGHT_ZONE = [[150, 8], [WIDTH - 1, 8], [WIDTH - 1, 87], [150, 87]]
ZONE_CATALOGUE = regions_catalogue().with_blocks(
    [DynamicZoneBlock, MotionDetectionBlock]
)


def marked_frame(camera: str, t: int, markers, *, number: Optional[int] = None):
    pixels = torch.from_numpy(painted_image((HEIGHT, WIDTH), markers))
    pixels = pixels.permute(2, 0, 1).contiguous()
    background = (torch.arange(WIDTH) // 3).to(torch.uint8).expand(HEIGHT, WIDTH)
    pixels[:, pixels[0] != 255] = background[pixels[0] != 255]
    pixels[2, TOP : TOP + SIDE, STEP * t : STEP * t + SIDE] = 255
    metadata = VideoMetadata(
        video_identifier=camera,
        frame_number=t if number is None else number,
        frame_timestamp=T0 + timedelta(seconds=t),
    )

    return ImageData.from_tensor(pixels, video_metadata=metadata)


def zone_motion_plan(mode: str, *, dynamic: bool):
    motion = {**MOTION_STEP, "detection_zone": "$inputs.zone"}
    steps = [motion]
    outputs = dict(MOTION_OUTPUTS)
    if dynamic:
        motion["detection_zone"] = "$steps.zone.zones"
        steps = [
            {
                "type": "test/masked_marker_detector@v1",
                "name": "segment",
                "image": "$inputs.image",
            },
            {
                "type": DynamicZoneBlock.type,
                "name": "zone",
                "predictions": "$steps.segment.predictions",
                "required_number_of_vertices": 4,
            },
            motion,
        ]
        outputs["zone"] = "$steps.zone.zones"
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT, {"type": "WorkflowParameter", "name": "zone"}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    plan = compile_workflow(
        definition,
        catalogue=ZONE_CATALOGUE,
        options=CompileOptions(block_execution=mode),
    )

    return plan


def run_frames(plan, frames, *, zone=None) -> List[Dict[str, Any]]:
    with plan.create_session() as session:
        rows = [values(session.run({"image": image, "zone": zone})) for image in frames]

    return rows


def zone_corners(zone: List[List[int]]) -> List[List[int]]:
    return sorted(map(list, zone))


@pytest.mark.parametrize("mode", MODES)
def test_dynamic_zones_feed_motion_as_the_union_of_their_polygons(mode):
    frames = [marked_frame("court", t, [LEFT_MARKER, RIGHT_MARKER]) for t in range(14)]
    dynamic = run_frames(zone_motion_plan(mode, dynamic=True), frames)
    literal = run_frames(
        zone_motion_plan(mode, dynamic=False), frames, zone=[LEFT_ZONE, RIGHT_ZONE]
    )

    assert [[zone_corners(z) for z in row["zone"]] for row in dynamic] == [
        [zone_corners(LEFT_ZONE), zone_corners(RIGHT_ZONE)]
    ] * 14
    assert [summary(row) for row in dynamic] == [summary(row) for row in literal]
    motion = [t for t, row in enumerate(dynamic) if row["motion"]]
    alarms = [t for t, row in enumerate(dynamic) if row["alarm"]]
    # Moving in the left zone, then in the gap (no motion), then the right zone.
    assert motion[0] == HISTORY and 8 not in motion and motion[-1] == 13
    assert alarms == [HISTORY, min(t for t in motion if t > 8)]
    boxes = [box for row in dynamic for box in summary(row)[2]]
    assert boxes and all(
        x1 <= 110 or x0 >= 150 for x0, _, x1, _ in boxes
    ), "a box outside both zones"


@pytest.mark.parametrize("mode", MODES)
def test_no_segmented_area_means_no_motion_not_whole_frame_motion(mode):
    frames = [marked_frame("court", t, []) for t in range(10)]
    rows = run_frames(zone_motion_plan(mode, dynamic=True), frames)
    whole = run_frames(zone_motion_plan(mode, dynamic=False), frames, zone=None)

    assert [row["zone"] for row in rows] == [[]] * 10
    assert [summary(row) for row in rows] == [(False, False, [])] * 10
    assert any(row["motion"] for row in whole), "the scene itself moves"


@pytest.mark.parametrize("mode", MODES)
def test_two_sources_keep_their_zones_and_restart_separately(mode):
    # "a" sees both zones; "b" only the left one. Then "a" rewinds to frame 0:
    # its whole state restarts (warmup, alarm), "b" goes on undisturbed.
    def a(t, number=None):
        return marked_frame("a", t, [LEFT_MARKER, RIGHT_MARKER], number=number)

    def b(t):
        return marked_frame("b", t, [LEFT_MARKER])

    plan = zone_motion_plan(mode, dynamic=True)
    alone_a = [summary(row) for row in run_frames(plan, [a(t) for t in range(8)])]
    alone_b = [summary(row) for row in run_frames(plan, [b(t) for t in range(14)])]
    together = {"a": [], "a again": [], "b": []}
    with plan.create_session() as session:
        for t in range(14):
            if t < 8:
                together["a"].append(summary(values(session.run({"image": a(t)}))))
            else:
                image = a(t - 8, number=t - 8)
                together["a again"].append(
                    summary(values(session.run({"image": image})))
                )
            together["b"].append(summary(values(session.run({"image": b(t)}))))

    assert together["a"] == alone_a
    assert together["a again"] == alone_a[:6]
    assert together["b"] == alone_b
    assert alone_a[HISTORY][1] is True
