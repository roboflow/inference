"""Model-block contracts that only a compiled workflow can show.

Fake models (``tests/v2/support/models``); not real inference. The grid
producer double proves what the instance-segmentation contract allows, not
the speed of any real compact backend:

    segmenter (grid views) ── segmentation ──▶ roboflow_core/dynamic_crop@v1 (mask_opacity 0)
                           ╰─ predictions  (nobody reads it: never built)
"""

from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.control.continue_if_boolean.v1 import ContinueIfBlock
from roboflow_workflows.v2.blocks.image import DynamicCropBlock
from roboflow_workflows.v2.blocks.models import BLOCKS
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks.grid import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import LineageError, StepExecutionError
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.support.models.classification_gate import TopConfidenceAtLeast
from tests.v2.support.models.fake_task_models import Classifier, NmsDetector
from tests.v2.support.models.grid_segmentation import (
    GridInstanceModel,
    GridInstanceSegmentationBlock,
)

MODES = ("run", "phases")
DARK, MID, BRIGHT = 20, 100, 200
IMAGE_INPUT = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}


@pytest.fixture
def materializations(monkeypatch) -> List[int]:
    """Every full-resolution upscale of any SegmentationView in the test."""
    calls: List[int] = []
    materialize = SegmentationView._materialize

    def counted(view):
        calls.append(len(view))
        return materialize(view)

    monkeypatch.setattr(SegmentationView, "_materialize", counted)

    return calls


def _compile(definition: Dict[str, Any], *, mode: str):
    catalogue = Catalogue(
        [
            *BLOCKS,
            GridInstanceSegmentationBlock,
            DynamicCropBlock,
            ContinueIfBlock,
            TopConfidenceAtLeast,
        ],
        kinds=[*NATIVE_KINDS, INSTANCE_SEGMENTATION_VIEW_KIND],
    )
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )

    return plan


def _images() -> List[ImageData]:
    images = [
        ImageData.from_tensor(
            torch.full((3, height, width), value, dtype=torch.uint8), image_id=name
        )
        for value, name, height, width in [
            (BRIGHT, "x", 20, 30),
            (DARK, "y", 12, 30),
            (MID, "z", 20, 16),
        ]
    ]

    return images


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _segmenter(**params: Any) -> Dict[str, Any]:
    step = {
        "type": GridInstanceSegmentationBlock.type,
        "name": "segmenter",
        "image": "$inputs.image",
        "model_id": "seg",
        "confidence": 0.0,
        **params,
    }

    return step


@pytest.mark.parametrize("mode", MODES)
def test_a_box_only_chain_never_builds_a_full_resolution_mask(
    mode: str, materializations: List[int]
) -> None:
    model = GridInstanceModel()
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT],
        "steps": [
            _segmenter(),
            {
                "type": DynamicCropBlock.type,
                "name": "crop",
                "image": "$inputs.image",
                "predictions": "$steps.segmenter.segmentation",
                "mask_opacity": 0.0,
            },
        ],
        "outputs": [_output("crops", "$steps.crop.crops")],
    }

    session = _compile(definition, mode=mode).create_session({"models": {"seg": model}})
    rows = session.run({"image": _images()}).rows()

    assert model.demand == {"segmentation": True, "predictions": False}
    # Bright, dark and mid images have 3, 0 and 2 instances: one crop each.
    assert [len(row["crops"]) for row in rows] == [3, 0, 2]
    assert [crop.tensor_image.shape[1:] for crop in rows[0]["crops"]] == [
        (20, 30),
        (18, 28),
        (16, 26),
    ]
    assert materializations == []


@pytest.mark.parametrize("mode", MODES)
def test_both_outputs_of_a_grid_producer_describe_the_same_rows(
    mode: str, materializations: List[int]
) -> None:
    model = GridInstanceModel()
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT],
        "steps": [_segmenter(class_filter=["car", "dog"])],
        "outputs": [
            _output("segmentation", "$steps.segmenter.segmentation"),
            _output("predictions", "$steps.segmenter.predictions"),
        ],
    }

    session = _compile(definition, mode=mode).create_session({"models": {"seg": model}})
    rows = session.run({"image": _images()}).rows()

    assert model.demand == {"segmentation": True, "predictions": True}
    for row, class_ids in zip(rows, [[1, 2], [], [1]]):
        view, predictions = row["segmentation"], row["predictions"]
        assert view.detections.class_id.tolist() == class_ids
        assert predictions.class_id.tolist() == class_ids
        assert torch.equal(view.full_res().mask, predictions.mask)
        assert row_record(view.detections) == row_record(predictions)
    assert sorted(materializations) == [0, 1, 2]


def _crop_classifier_definition(model_id: str) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT],
        "steps": [
            _segmenter(),
            {
                "type": DynamicCropBlock.type,
                "name": "crop",
                "image": "$inputs.image",
                "predictions": "$steps.segmenter.segmentation",
                "mask_opacity": 0.0,
            },
            {
                "type": "roboflow_core/roboflow_classification_model@v3",
                "name": "classifier",
                "image": "$steps.crop.crops",
                "model_id": model_id,
            },
        ],
        "outputs": [_output("labels", "$steps.classifier.predictions")],
    }

    return definition


@pytest.mark.parametrize("mode", MODES)
def test_one_classifier_call_serves_every_crop_and_zero_crops_call_nothing(
    mode: str,
) -> None:
    classifier = Classifier()
    session = _compile(_crop_classifier_definition("cls"), mode=mode).create_session(
        {"models": {"seg": GridInstanceModel(), "cls": classifier}}
    )

    rows = session.run({"image": _images()}).rows()
    dark = session.run({"image": _images()[1:2]}).rows()

    # Crops of every image (3, 0 and 2) reach the classifier in one call.
    assert classifier.batch_sizes == [5]
    assert [len(row["labels"]) for row in rows] == [3, 0, 2]
    assert [len(row["labels"]) for row in dark] == [0]


def test_a_crop_step_cannot_take_its_model_from_the_parent_image() -> None:
    # The segmenter's model_id is per image; every crop inherits its parent's
    # value, which still varies over the call.
    definition = _crop_classifier_definition("$steps.segmenter.model_id")

    with pytest.raises(
        LineageError, match="classifier model_id is passed once per call"
    ):
        _compile(definition, mode="run")


def _filter_definition(*, per_image: bool = False) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [
            IMAGE_INPUT,
            {
                "type": "WorkflowBatchInput" if per_image else "WorkflowParameter",
                "name": "classes",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "detector",
                "image": "$inputs.image",
                "model_id": "od",
                "confidence": 0.0,
                "class_filter": "$inputs.classes",
            }
        ],
        "outputs": [_output("predictions", "$steps.detector.predictions")],
    }

    return definition


@pytest.mark.parametrize("mode", MODES)
def test_one_selected_class_filter_applies_to_every_image_in_one_forward(
    mode: str,
) -> None:
    model = NmsDetector()
    session = _compile(_filter_definition(), mode=mode).create_session(
        {"models": {"od": model}}
    )

    by_filter = [
        [
            row["predictions"].class_id.tolist()
            for row in session.run({"image": _images(), "classes": classes}).rows()
        ]
        for classes in (["dog", "car"], None, ["person", "car", "dog"])
    ]

    assert model.batch_sizes == [3, 3, 3]
    assert model.calls.count("forward") == 3
    # A list as long as the batch is still one filter for every image.
    assert by_filter == [
        [[1, 2], [], [1]],
        [[0, 1, 2], [], [0, 1]],
        [[0, 1, 2], [], [0, 1]],
    ]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "classes, message",
    [([1], "got 1"), (["person", None], "got None")],
    ids=["number", "none-item"],
)
def test_a_selected_malformed_class_filter_fails_before_any_load(
    mode: str, classes: List[Any], message: str
) -> None:
    requested = []

    def refuse(request):
        requested.append(request)
        raise AssertionError(f"model {request} was resolved")

    session = _compile(_filter_definition(), mode=mode).create_session(
        {"models": ModelCache(refuse)}
    )

    expected = f"class_filter holds class names, {message}"
    with pytest.raises(StepExecutionError, match=expected):
        session.run({"image": _images(), "classes": classes})

    assert requested == []


def test_a_per_image_class_filter_fails_compilation() -> None:
    with pytest.raises(
        LineageError, match="detector class_filter is passed once per call"
    ):
        _compile(_filter_definition(per_image=True), mode="run")


@pytest.mark.parametrize("mode", MODES)
def test_a_downstream_step_runs_only_for_confident_classifications(
    mode: str,
) -> None:
    detector = NmsDetector()
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
                "type": TopConfidenceAtLeast.type,
                "name": "confident",
                "prediction": "$steps.classifier.predictions",
                "minimum": 0.7,
            },
            {
                "type": ContinueIfBlock.type,
                "name": "gate",
                "condition": "$steps.confident.confident",
                "next_steps": ["$steps.detector"],
            },
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "detector",
                "image": "$inputs.image",
                "model_id": "od",
                "confidence": 0.0,
            },
        ],
        "outputs": [
            _output("classification", "$steps.classifier.predictions"),
            _output("top_confidence", "$steps.confident.top_confidence"),
            _output("detections", "$steps.detector.predictions"),
        ],
    }

    session = _compile(definition, mode=mode).create_session(
        {"models": {"cls": Classifier(), "od": detector}}
    )
    rows = session.run({"image": _images()}).rows()

    # The full distribution stays on every result; only the gate decides.
    assert [row["classification"].confidence.shape for row in rows] == [(1, 3)] * 3
    confident = [row["top_confidence"] >= 0.7 for row in rows]
    assert confident == [False, True, True]  # bright is the least certain
    assert [row["detections"] is not None for row in rows] == confident
    assert detector.batch_sizes == [2]
