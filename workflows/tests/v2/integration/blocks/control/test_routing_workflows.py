"""Compiled workflows routed and throttled by the V2 routing blocks.

Fake models (``tests/v2/support/models``); not real inference. The fake
classifier's top class depends on mean brightness::

    brightness 20 -> person    100 -> car    200 -> dog

    image ─▶ classification ─▶ property_definition(top_class)
                                 ├─▶ switch_case ─▶ dogs | cars | others
                                 ├─▶ delta_filter ─▶ detector     (class changed)
                                 └─▶ rate_limiter ─▶ detector     (cooldown over)

A context-free value (``_Echo`` of a batch input without metadata) next to an
image with a source still takes the lane and time from the image.
"""

from datetime import datetime
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.control.routing import (
    DeltaFilterBlock,
    RateLimiterBlock,
    SwitchCaseBlock,
)
from roboflow_workflows.v2.blocks.formatters import BLOCKS as FORMATTER_BLOCKS
from roboflow_workflows.v2.blocks.image import DynamicCropBlock
from roboflow_workflows.v2.blocks.models import BLOCKS as MODEL_BLOCKS
from roboflow_workflows.v2.data.batch import (
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.object_detection import Detections
from tests.v2.support.models.fake_task_models import (
    Classifier,
    FakeModel,
    NmsDetector,
)

MODES = ("run", "phases")
DARK, MID, BRIGHT = 20, 100, 200  # person, car, dog
IMAGE_INPUT = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}
TOP_CLASS = [{"type": "ClassificationPropertyExtract", "property_name": "top_class"}]
FRAME_TIMESTAMP = datetime(2026, 1, 1)


class HalvesDetector(FakeModel):
    """Two boxes per image: its left half (class 0) and its right half (class 1)."""

    def post_process(self, raw, sizes, **kwargs: Any) -> List[Detections]:
        self._record(**kwargs)
        predictions = []
        for height, width in sizes:
            xyxy = torch.tensor(
                [[0, 0, width // 2, height], [width // 2, 0, width, height]]
            ).float()
            predictions.append(
                Detections(
                    xyxy=xyxy,
                    class_id=torch.tensor([0, 1]),
                    confidence=torch.tensor([0.9, 0.9]),
                )
            )

        return predictions


class _Echo(Block):
    """Returns its value; a batch input without metadata gives it no context."""

    type = "test/echo@v1"
    outputs = {"out": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value: Any) -> Dict[str, Any]:
        return {"out": value}


def _compile(definition: Dict[str, Any], *, mode: str = "run"):
    catalogue = Catalogue(
        [
            *MODEL_BLOCKS,
            *FORMATTER_BLOCKS,
            DynamicCropBlock,
            SwitchCaseBlock,
            DeltaFilterBlock,
            RateLimiterBlock,
            _Echo,
        ],
        kinds=NATIVE_KINDS,
    )
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )

    return plan


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _detector(name: str, *, image: str = "$inputs.image") -> Dict[str, Any]:
    return {
        "type": "v2/object_detection@v1",
        "name": name,
        "image": image,
        "model_id": name,
        "confidence": 0.0,
    }


def _classified(image: str = "$inputs.image") -> List[Dict[str, Any]]:
    steps = [
        {
            "type": "v2/classification@v1",
            "name": "classifier",
            "image": image,
            "model_id": "cls",
        },
        {
            "type": "v2/property_definition@v1",
            "name": "top_class",
            "data": "$steps.classifier.predictions",
            "operations": TOP_CLASS,
        },
    ]

    return steps


def _image(value: int, *, name: str, video: Optional[str] = None, frame: int = 0):
    metadata = None
    if video is not None:
        metadata = VideoMetadata(
            video_identifier=video,
            frame_number=frame,
            frame_timestamp=FRAME_TIMESTAMP,
            fps=10.0,
        )
    pixels = torch.full((3, 20, 30), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=name, video_metadata=metadata)

    return image


def _ran(rows: List[Dict[str, Any]], output: str) -> List[bool]:
    return [row[output] is not None for row in rows]


# --- switch_case -----------------------------------------------------------


def _routing(**switch: Any) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT],
        "steps": [
            *_classified(),
            {
                "type": "v2/switch_case@v1",
                "name": "route",
                "value": "$steps.top_class.output",
                "cases": {"dog": "$steps.dogs", "car": "$steps.cars"},
                "default_next_steps": ["$steps.others"],
                **switch,
            },
            _detector("dogs"),
            _detector("cars"),
            _detector("others"),
        ],
        "outputs": [
            _output("top_class", "$steps.top_class.output"),
            *(
                _output(name, f"$steps.{name}.predictions")
                for name in ("dogs", "cars", "others")
            ),
        ],
    }

    return definition


def _routing_models() -> Dict[str, Any]:
    models = {"cls": Classifier()}
    models.update({name: NmsDetector() for name in ("dogs", "cars", "others")})

    return models


@pytest.mark.parametrize("mode", MODES)
def test_each_image_runs_only_the_detector_of_its_class(mode: str) -> None:
    models = _routing_models()
    session = _compile(_routing(), mode=mode).create_session({"models": models})
    images = [_image(v, name=n) for v, n in ((DARK, "a"), (MID, "b"), (BRIGHT, "c"))]

    rows = session.run({"image": images}).rows()

    assert [row["top_class"] for row in rows] == ["person", "car", "dog"]
    assert _ran(rows, "dogs") == [False, False, True]
    assert _ran(rows, "cars") == [False, True, False]
    assert _ran(rows, "others") == [True, False, False]
    assert [models[name].batch_sizes for name in ("dogs", "cars", "others")] == [
        [1],
        [1],
        [1],
    ]


def test_case_insensitive_keys_route_and_an_empty_default_routes_nowhere() -> None:
    definition = _routing(
        cases={"DOG": "$steps.dogs", "Car": "$steps.cars"},
        case_insensitive=True,
        default_next_steps=[],
    )
    session = _compile(definition).create_session({"models": _routing_models()})
    images = [_image(v, name=n) for v, n in ((DARK, "a"), (MID, "b"), (BRIGHT, "c"))]

    rows = session.run({"image": images}).rows()

    assert _ran(rows, "dogs") == [False, False, True]
    assert _ran(rows, "cars") == [False, True, False]


def test_a_shared_case_target_fails_workflow_compilation() -> None:
    definition = _routing(default_next_steps=["$steps.dogs"])

    with pytest.raises(Exception, match="Duplicated targets"):
        _compile(definition)


@pytest.mark.parametrize("mode", MODES)
def test_each_crop_is_routed_alone(mode: str) -> None:
    # Left half dark (person), right half bright (dog).
    image = torch.cat(
        [
            torch.full((3, 20, 20), DARK, dtype=torch.uint8),
            torch.full((3, 20, 20), BRIGHT, dtype=torch.uint8),
        ],
        dim=2,
    )
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT],
        "steps": [
            {
                "type": "v2/object_detection@v1",
                "name": "halves",
                "image": "$inputs.image",
                "model_id": "halves",
            },
            {
                "type": "v2/dynamic_crop@v1",
                "name": "crops",
                "image": "$inputs.image",
                "predictions": "$steps.halves.predictions",
            },
            *_classified("$steps.crops.crops"),
            {
                "type": "v2/switch_case@v1",
                "name": "route",
                "value": "$steps.top_class.output",
                "cases": {"dog": "$steps.dogs"},
            },
            _detector("dogs", image="$steps.crops.crops"),
        ],
        "outputs": [
            _output("classes", "$steps.top_class.output"),
            _output("dogs", "$steps.dogs.predictions"),
        ],
    }
    models = {"halves": HalvesDetector(), "cls": Classifier(), "dogs": NmsDetector()}
    session = _compile(definition, mode=mode).create_session({"models": models})

    rows = session.run({"image": [ImageData.from_tensor(image, image_id="x")]}).rows()

    assert rows[0]["classes"] == ["person", "dog"]
    assert [crop is not None for crop in rows[0]["dogs"]] == [False, True]
    assert models["dogs"].batch_sizes == [1]


# --- delta_filter ----------------------------------------------------------


def _class_changes() -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT],
        "steps": [
            *_classified(),
            {
                "type": "v2/delta_filter@v1",
                "name": "changed",
                "image": "$inputs.image",
                "value": "$steps.top_class.output",
                "next_steps": ["$steps.detector"],
            },
            _detector("detector"),
        ],
        "outputs": [_output("detections", "$steps.detector.predictions")],
    }

    return definition


def _cameras(values: Sequence[int], *, frame: int) -> List[ImageData]:
    images = [
        _image(value, name=f"cam{n}-{frame}", video=f"cam{n}", frame=frame)
        for n, value in enumerate(values)
    ]

    return images


@pytest.mark.parametrize("mode", MODES)
def test_delta_filter_follows_each_camera_and_reset_forgets(mode: str) -> None:
    plan = _compile(_class_changes(), mode=mode)
    session = plan.create_session(
        {"models": {"cls": Classifier(), "detector": NmsDetector()}}
    )

    def changed(values: Sequence[int], frame: int) -> List[bool]:
        rows = session.run({"image": _cameras(values, frame=frame)}).rows()

        return _ran(rows, "detections")

    assert changed([DARK, DARK], 0) == [True, True]
    assert changed([DARK, BRIGHT], 1) == [False, True]
    assert changed([MID, BRIGHT], 2) == [True, False]

    session.update(plan, reset=True)

    assert changed([MID, BRIGHT], 3) == [True, True]


def test_delta_filter_without_video_metadata_keys_by_batch_position() -> None:
    session = _compile(_class_changes()).create_session(
        {"models": {"cls": Classifier(), "detector": NmsDetector()}}
    )

    def changed(values: Sequence[int]) -> List[bool]:
        images = [_image(v, name=f"p{n}") for n, v in enumerate(values)]

        return _ran(session.run({"image": images}).rows(), "detections")

    assert changed([DARK, DARK]) == [True, True]
    assert changed([DARK, MID]) == [False, True]


def test_delta_filter_crops_of_a_frame_share_its_video_lane() -> None:
    # Crops keep the frame's video metadata: person then dog in crop order.
    image = torch.cat(
        [
            torch.full((3, 20, 20), DARK, dtype=torch.uint8),
            torch.full((3, 20, 20), BRIGHT, dtype=torch.uint8),
        ],
        dim=2,
    )
    metadata = VideoMetadata(
        video_identifier="cam", frame_number=0, frame_timestamp=FRAME_TIMESTAMP
    )
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT],
        "steps": [
            {
                "type": "v2/object_detection@v1",
                "name": "halves",
                "image": "$inputs.image",
                "model_id": "halves",
            },
            {
                "type": "v2/dynamic_crop@v1",
                "name": "crops",
                "image": "$inputs.image",
                "predictions": "$steps.halves.predictions",
            },
            *_classified("$steps.crops.crops"),
            {
                "type": "v2/delta_filter@v1",
                "name": "changed",
                "image": "$steps.crops.crops",
                "value": "$steps.top_class.output",
                "next_steps": ["$steps.detector"],
            },
            _detector("detector", image="$steps.crops.crops"),
        ],
        "outputs": [_output("detections", "$steps.detector.predictions")],
    }
    models = {
        "halves": HalvesDetector(),
        "cls": Classifier(),
        "detector": NmsDetector(),
    }
    session = _compile(definition).create_session({"models": models})
    frame = ImageData.from_tensor(image, image_id="f", video_metadata=metadata)
    mirrored = ImageData.from_tensor(
        torch.flip(image, dims=[2]), image_id="g", video_metadata=metadata
    )

    # Crops person, dog: both change (none -> person -> dog).
    first = session.run({"image": [frame]}).rows()[0]["detections"]
    # Crops dog, person: the lane last saw dog, so only person changes.
    second = session.run({"image": [mirrored]}).rows()[0]["detections"]

    assert [crop is not None for crop in first] == [True, True]
    assert [crop is not None for crop in second] == [False, True]


# --- rate_limiter ----------------------------------------------------------


def _throttled(**limiter: Any) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT],
        "steps": [
            *_classified(),
            {
                "type": "v2/rate_limiter@v1",
                "name": "limit",
                "depends_on": "$steps.classifier.predictions",
                "next_steps": ["$steps.detector"],
                "cooldown_seconds": 1.0,
                **limiter,
            },
            _detector("detector"),
        ],
        "outputs": [_output("detections", "$steps.detector.predictions")],
    }

    return definition


def _stamped(seconds: Sequence[Tuple[str, float]]) -> InputValue:
    """One frame per (camera, observed seconds) pair."""
    frames = InputValue(
        data=[_image(MID, name=f"{camera}@{t}") for camera, t in seconds],
        metadata=EntryMetadata(
            sample={
                (n,): SampleContext(source_id=camera)
                for n, (camera, _) in enumerate(seconds)
            },
            temporal={
                (n,): TemporalContext(
                    observed_coverage=Timestamp(
                        round(t * 1000), Fraction(1, 1000), "engine.monotonic"
                    )
                )
                for n, (_, t) in enumerate(seconds)
            },
        ),
    )

    return frames


@pytest.mark.parametrize("mode", MODES)
def test_each_camera_has_its_own_cooldown_on_observed_time(mode: str) -> None:
    plan = _compile(_throttled(), mode=mode)
    session = plan.create_session(
        {"models": {"cls": Classifier(), "detector": NmsDetector()}}
    )

    def allowed(stamps: Sequence[Tuple[str, float]]) -> List[bool]:
        return _ran(session.run({"image": _stamped(stamps)}).rows(), "detections")

    assert allowed([("a", 0.0), ("b", 0.6)]) == [True, True]
    assert allowed([("a", 0.7), ("b", 1.2)]) == [False, False]
    assert allowed([("a", 1.0), ("b", 1.6)]) == [True, True]

    session.update(plan, reset=True)

    assert allowed([("a", 1.1), ("b", 1.7)]) == [True, True]


def test_global_scope_shares_one_cooldown_like_v1() -> None:
    session = _compile(_throttled(scope="global")).create_session(
        {"models": {"cls": Classifier(), "detector": NmsDetector()}}
    )

    def allowed(stamps: Sequence[Tuple[str, float]]) -> List[bool]:
        return _ran(session.run({"image": _stamped(stamps)}).rows(), "detections")

    assert allowed([("a", 0.0), ("b", 0.6)]) == [True, False]
    assert allowed([("a", 1.0), ("b", 1.6)]) == [True, False]
    assert allowed([("b", 2.0), ("a", 2.0)]) == [True, False]


@pytest.mark.parametrize("mode", MODES)
def test_video_time_throttles_by_frame_number_not_processing_time(mode: str) -> None:
    session = _compile(
        _throttled(video_reference_image="$inputs.image"), mode=mode
    ).create_session({"models": {"cls": Classifier(), "detector": NmsDetector()}})

    def allowed(frame: int) -> List[bool]:
        images = [
            _image(MID, name=f"{video}-{frame}", video=video, frame=frame)
            for video in ("a", "b")
        ]

        return _ran(session.run({"image": images}).rows(), "detections")

    # 10 fps: frame 10 is exactly the 1 s cooldown after frame 0.
    assert [allowed(frame) for frame in (0, 5, 10, 19)] == [
        [True, True],
        [False, False],
        [True, True],
        [False, False],
    ]


# --- lane field next to a context-free value -------------------------------


def _beside_context_free_value(control: Dict[str, Any], *, mode: str):
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT, {"type": "WorkflowBatchInput", "name": "value"}],
        "steps": [
            {"type": _Echo.type, "name": "echo", "value": "$inputs.value"},
            {**control, "name": "control", "next_steps": ["$steps.detector"]},
            _detector("detector"),
        ],
        "outputs": [_output("detections", "$steps.detector.predictions")],
    }
    session = _compile(definition, mode=mode).create_session(
        {"models": {"detector": NmsDetector()}}
    )

    def passed(cameras: Sequence[Tuple[str, float]], values: List[int]) -> List[bool]:
        rows = session.run({"image": _stamped(cameras), "value": values}).rows()

        return _ran(rows, "detections")

    return passed


@pytest.mark.parametrize("mode", MODES)
def test_delta_filter_keys_by_the_image_source_not_the_value(mode: str) -> None:
    changed = _beside_context_free_value(
        {
            "type": "v2/delta_filter@v1",
            "image": "$inputs.image",
            "value": "$steps.echo.out",
        },
        mode=mode,
    )

    assert changed([("a", 0.0), ("b", 0.0)], [1, 2]) == [True, True]
    # The cameras swap positions; each keeps its own value.
    assert changed([("b", 0.1), ("a", 0.1)], [2, 1]) == [False, False]


@pytest.mark.parametrize("mode", MODES)
def test_rate_limiter_takes_source_and_time_from_the_reference_image(
    mode: str,
) -> None:
    allowed = _beside_context_free_value(
        {
            "type": "v2/rate_limiter@v1",
            "depends_on": "$steps.echo.out",
            "video_reference_image": "$inputs.image",
            "cooldown_seconds": 1.0,
        },
        mode=mode,
    )

    # No video metadata: the image's source and observed time count.
    assert allowed([("a", 0.0), ("b", 0.6)], [1, 2]) == [True, True]
    assert allowed([("b", 1.2), ("a", 0.7)], [1, 2]) == [False, False]
    assert allowed([("a", 1.0), ("b", 1.6)], [1, 2]) == [True, True]
