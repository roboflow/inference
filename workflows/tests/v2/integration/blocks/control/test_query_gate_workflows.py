"""Compiled workflows gated by ``roboflow_core/continue_if@v1`` with product blocks only.

Fake models (``tests/v2/support/models``); not real inference. The fake
classifier's top confidence depends on mean brightness::

    brightness 20 -> person 0.85    100 -> car 0.78    200 -> dog 0.58

    image ─▶ roboflow_core/roboflow_classification_model@v3 ─▶ roboflow_core/continue_if@v1 ─▶ roboflow_core/roboflow_object_detection_model@v3
                                     top_class_confidence >= minimum
"""

from fractions import Fraction
from typing import Any, Dict, List, Sequence

import pytest
import torch
from roboflow_workflows.v2.blocks.control.continue_if.v1 import CONDITION_STATEMENT_EXAMPLE, ContinueIfQueryBlock
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
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.object_detection import Detections
from tests.v2.support.models.fake_task_models import (
    Classifier,
    FakeModel,
    NmsDetector,
)

MODES = ("run", "phases")
DARK, MID, BRIGHT = 20, 100, 200
IMAGE_INPUT = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}
MINIMUM_INPUT = {"type": "WorkflowParameter", "name": "minimum", "default_value": 0.7}
TOP_CONFIDENCE = [
    {"type": "ClassificationPropertyExtract", "property_name": "top_class_confidence"}
]


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


def _compile(definition: Dict[str, Any], *, mode: str = "run"):
    catalogue = Catalogue(
        [*MODEL_BLOCKS, *FORMATTER_BLOCKS, DynamicCropBlock, ContinueIfQueryBlock],
        kinds=NATIVE_KINDS,
    )
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )

    return plan


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _image(value: int, *, name: str, width: int = 30) -> ImageData:
    pixels = torch.full((3, 20, width), value, dtype=torch.uint8)

    return ImageData.from_tensor(pixels, image_id=name)


def _gate(on: str, *, next_steps: Sequence[str], stop_delay: Any = 0) -> dict:
    step = {
        "type": "roboflow_core/continue_if@v1",
        "name": "gate",
        "condition_statement": CONDITION_STATEMENT_EXAMPLE,
        "evaluation_parameters": {"prediction": on, "minimum": "$inputs.minimum"},
        "next_steps": list(next_steps),
        "stop_delay": stop_delay,
    }

    return step


def _classified_detection(*, stop_delay: Any = 0, inputs=()) -> dict:
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT, MINIMUM_INPUT, *inputs],
        "steps": [
            {
                "type": "roboflow_core/roboflow_classification_model@v3",
                "name": "classifier",
                "image": "$inputs.image",
                "model_id": "cls",
            },
            {
                "type": "roboflow_core/property_definition@v1",
                "name": "top_confidence",
                "data": "$steps.classifier.predictions",
                "operations": TOP_CONFIDENCE,
            },
            _gate(
                "$steps.classifier.predictions",
                next_steps=["$steps.detector"],
                stop_delay=stop_delay,
            ),
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "detector",
                "image": "$inputs.image",
                "model_id": "od",
                "confidence": 0.0,
            },
        ],
        "outputs": [
            _output("top_confidence", "$steps.top_confidence.output"),
            _output("detections", "$steps.detector.predictions"),
        ],
    }

    return definition


@pytest.mark.parametrize("mode", MODES)
def test_a_classification_gates_the_detector_with_product_blocks_only(
    mode: str,
) -> None:
    detector = NmsDetector()
    session = _compile(_classified_detection(), mode=mode).create_session(
        {"models": {"cls": Classifier(), "od": detector}}
    )
    images = [_image(BRIGHT, name="x"), _image(DARK, name="y"), _image(MID, name="z")]

    rows = session.run({"image": images, "minimum": 0.7}).rows()

    top = [row["top_confidence"] for row in rows]
    assert top == pytest.approx([0.58, 0.85, 0.78], abs=0.01)
    assert [row["detections"] is not None for row in rows] == [False, True, True]
    assert detector.batch_sizes == [2]


def test_the_gate_minimum_is_a_runtime_input() -> None:
    session = _compile(_classified_detection()).create_session(
        {"models": {"cls": Classifier(), "od": NmsDetector()}}
    )
    images = [_image(BRIGHT, name="x"), _image(DARK, name="y"), _image(MID, name="z")]

    rows = session.run({"image": images, "minimum": 0.8}).rows()

    assert [row["detections"] is not None for row in rows] == [False, True, False]


def test_an_undeclared_operand_fails_workflow_compilation() -> None:
    definition = _classified_detection()
    definition["steps"][2]["evaluation_parameters"] = {
        "prediction": "$steps.classifier.predictions"
    }

    with pytest.raises(Exception, match="operand `minimum` is not declared"):
        _compile(definition)


def test_a_negative_stop_delay_fails_workflow_compilation() -> None:
    with pytest.raises(Exception, match="greater than or equal to 0"):
        _compile(_classified_detection(stop_delay=-1))


def test_property_definition_and_expression_label_each_image() -> None:
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT],
        "steps": [
            {
                "type": "roboflow_core/roboflow_classification_model@v3",
                "name": "classifier",
                "image": "$inputs.image",
                "model_id": "cls",
            },
            {
                "type": "roboflow_core/property_definition@v1",
                "name": "top_class",
                "data": "$steps.classifier.predictions",
                "operations": [
                    {
                        "type": "ClassificationPropertyExtract",
                        "property_name": "top_class",
                    }
                ],
            },
            {
                "type": "roboflow_core/expression@v1",
                "name": "label",
                "data": {"top_class": "$steps.top_class.output"},
                "switch": {
                    "type": "CasesDefinition",
                    "cases": [
                        {
                            "type": "CaseDefinition",
                            "condition": {
                                "type": "StatementGroup",
                                "statements": [
                                    {
                                        "type": "BinaryStatement",
                                        "left_operand": {
                                            "type": "DynamicOperand",
                                            "operand_name": "top_class",
                                        },
                                        "comparator": {"type": "in (Sequence)"},
                                        "right_operand": {
                                            "type": "StaticOperand",
                                            "value": ["car", "dog"],
                                        },
                                    }
                                ],
                            },
                            "result": {
                                "type": "StaticCaseResult",
                                "value": "not a person",
                            },
                        }
                    ],
                    "default": {"type": "StaticCaseResult", "value": "person"},
                },
            },
        ],
        "outputs": [
            _output("top_class", "$steps.top_class.output"),
            _output("label", "$steps.label.output"),
        ],
    }
    session = _compile(definition).create_session({"models": {"cls": Classifier()}})

    rows = session.run(
        {
            "image": [
                _image(DARK, name="a"),
                _image(MID, name="b"),
                _image(BRIGHT, name="c"),
            ]
        }
    ).rows()

    assert [row["top_class"] for row in rows] == ["person", "car", "dog"]
    assert [row["label"] for row in rows] == ["person", "not a person", "not a person"]


@pytest.mark.parametrize("mode", MODES)
def test_each_crop_decides_alone(mode: str) -> None:
    # Left half dark (0.85 passes), right half bright (0.58 stops).
    image = torch.cat(
        [
            torch.full((3, 20, 20), DARK, dtype=torch.uint8),
            torch.full((3, 20, 20), BRIGHT, dtype=torch.uint8),
        ],
        dim=2,
    )
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT, MINIMUM_INPUT],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "halves",
                "image": "$inputs.image",
                "model_id": "halves",
            },
            {
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "crops",
                "image": "$inputs.image",
                "predictions": "$steps.halves.predictions",
            },
            {
                "type": "roboflow_core/roboflow_classification_model@v3",
                "name": "classifier",
                "image": "$steps.crops.crops",
                "model_id": "cls",
            },
            _gate("$steps.classifier.predictions", next_steps=["$steps.detector"]),
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "detector",
                "image": "$steps.crops.crops",
                "model_id": "od",
                "confidence": 0.0,
            },
        ],
        "outputs": [
            _output("classes", "$steps.classifier.predictions"),
            _output("detections", "$steps.detector.predictions"),
        ],
    }
    detector = NmsDetector()
    session = _compile(definition, mode=mode).create_session(
        {"models": {"halves": HalvesDetector(), "cls": Classifier(), "od": detector}}
    )

    rows = session.run(
        {"image": [ImageData.from_tensor(image, image_id="split")], "minimum": 0.7}
    ).rows()

    crops = rows[0]["detections"]
    assert [int(prediction.class_id[0]) for prediction in rows[0]["classes"]] == [0, 2]
    assert [prediction is not None for prediction in crops] == [True, False]
    assert detector.batch_sizes == [1]


def _camera_frames(values: Sequence[int], *, seconds: float) -> InputValue:
    """Frame ``n`` of camera ``cam<n>``, all observed at ``seconds``."""
    stamp = Timestamp(round(seconds * 1000), Fraction(1, 1000), "engine.monotonic")
    frames = InputValue(
        data=[_image(value, name=f"cam{n}") for n, value in enumerate(values)],
        metadata=EntryMetadata(
            sample={
                (n,): SampleContext(source_id=f"cam{n}") for n in range(len(values))
            },
            temporal={
                (n,): TemporalContext(observed_coverage=stamp)
                for n in range(len(values))
            },
        ),
    )

    return frames


def test_stop_delay_holds_each_source_on_its_own_observed_time() -> None:
    session = _compile(_classified_detection(stop_delay=5.0)).create_session(
        {"models": {"cls": Classifier(), "od": NmsDetector()}}
    )

    def opened(values: Sequence[int], seconds: float) -> List[bool]:
        rows = session.run(
            {"image": _camera_frames(values, seconds=seconds), "minimum": 0.7}
        ).rows()

        return [row["detections"] is not None for row in rows]

    assert opened([DARK, BRIGHT], seconds=0.0) == [True, False]
    # cam0's true at 0 s holds it through 5 s; cam1 was never true.
    assert opened([BRIGHT, BRIGHT], seconds=4.0) == [True, False]
    assert opened([BRIGHT, DARK], seconds=5.0) == [True, True]
    assert opened([BRIGHT, BRIGHT], seconds=6.0) == [False, True]
    assert opened([BRIGHT, BRIGHT], seconds=10.5) == [False, False]


def test_a_selector_bound_stop_delay_is_read_per_run() -> None:
    delay_input = {"type": "WorkflowParameter", "name": "delay", "default_value": 0}
    session = _compile(
        _classified_detection(stop_delay="$inputs.delay", inputs=[delay_input])
    ).create_session({"models": {"cls": Classifier(), "od": NmsDetector()}})

    def opened(value: int, seconds: float, delay: float) -> bool:
        rows = session.run(
            {
                "image": _camera_frames([value], seconds=seconds),
                "minimum": 0.7,
                "delay": delay,
            }
        ).rows()

        return rows[0]["detections"] is not None

    assert opened(DARK, seconds=0.0, delay=2.0)
    assert opened(BRIGHT, seconds=1.0, delay=2.0)
    assert not opened(BRIGHT, seconds=1.5, delay=0)
    with pytest.raises(Exception, match="stop_delay must be a finite number"):
        opened(BRIGHT, seconds=2.0, delay=-1.0)


@pytest.mark.parametrize("left, right", [(DARK, BRIGHT), (BRIGHT, DARK)])
def test_crops_of_one_frame_share_a_source_hold_in_any_order(left, right) -> None:
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT, MINIMUM_INPUT],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "halves",
                "image": "$inputs.image",
                "model_id": "halves",
            },
            {
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "crops",
                "image": "$inputs.image",
                "predictions": "$steps.halves.predictions",
            },
            {
                "type": "roboflow_core/roboflow_classification_model@v3",
                "name": "classifier",
                "image": "$steps.crops.crops",
                "model_id": "cls",
            },
            _gate(
                "$steps.classifier.predictions",
                next_steps=["$steps.detector"],
                stop_delay=5.0,
            ),
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "detector",
                "image": "$steps.crops.crops",
                "model_id": "od",
                "confidence": 0.0,
            },
        ],
        "outputs": [_output("detections", "$steps.detector.predictions")],
    }
    session = _compile(definition).create_session(
        {
            "models": {
                "halves": HalvesDetector(),
                "cls": Classifier(),
                "od": NmsDetector(),
            }
        }
    )

    def opened(values: Sequence[int], seconds: float) -> List[bool]:
        halves = [torch.full((3, 20, 20), value, dtype=torch.uint8) for value in values]
        frame = ImageData.from_tensor(torch.cat(halves, dim=2), image_id="cam0")
        stamp = Timestamp(round(seconds * 1000), Fraction(1, 1000), "engine.monotonic")
        frames = InputValue(
            data=[frame],
            metadata=EntryMetadata(
                sample={(0,): SampleContext(source_id="cam0")},
                temporal={(0,): TemporalContext(observed_coverage=stamp)},
            ),
        )
        rows = session.run({"image": frames, "minimum": 0.7}).rows()

        return [crop is not None for crop in rows[0]["detections"]]

    # Same frame: only the confident crop continues, whichever comes first.
    assert opened([left, right], seconds=0.0) == [left == DARK, right == DARK]
    # Next frame: the earlier true of this source holds both crops.
    assert opened([BRIGHT, BRIGHT], seconds=1.0) == [True, True]
