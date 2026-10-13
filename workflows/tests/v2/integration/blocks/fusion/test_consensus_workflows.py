"""``v2/detections_consensus@v1`` compiled and run by the real V2 engine.

Three stand-in models predict a batch of two 40x60 images; they arrive as
workflow batch inputs carrying native predictions::

    image  model  rows (class, confidence, box)
    0      a      person 0.9 (10, 10, 30, 30)   car 0.8 (40, 0, 50, 10)
    0      b      person 0.7 (12, 10, 30, 30)   car 0.6 (40, 0, 50, 10)
    0      c      person 0.5 (10, 12, 30, 30)
    1      a      person 0.9 (10, 10, 30, 30)
    1      b      car    0.6 (10, 10, 30, 30)
    1      c      (nothing)

Image 0: the person is seen by 3 models, the car by 2. Image 1: two models see
one object but disagree on its class.
"""

import itertools

import numpy as np
import pycocotools.mask as mask_utils
import pytest
import torch
from roboflow_workflows.v2.blocks import create_catalogue, fusion
from roboflow_workflows.v2.blocks.fusion import consensus
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import FrameMapping
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

SIZE_HW = (40, 60)
NAMES = {0: "person", 1: "car"}
PERSON = (10, 10, 30, 30)
CAR = (40, 0, 50, 10)
IMAGE_ROWS = [
    {
        "a": [(0, 0.9, PERSON), (1, 0.8, CAR)],
        "b": [(0, 0.7, (12, 10, 30, 30)), (1, 0.6, CAR)],
        "c": [(0, 0.5, (10, 12, 30, 30))],
    },
    {
        "a": [(0, 0.9, PERSON)],
        "b": [(1, 0.6, PERSON)],
        "c": [],
    },
]
KNOBS = (
    "required_votes",
    "class_aware",
    "iou_threshold",
    "confidence",
    "classes_to_consider",
    "required_objects",
)
DEFAULTS = {
    "required_votes": 2,
    "class_aware": True,
    "iou_threshold": 0.3,
    "confidence": 0.0,
    "classes_to_consider": None,
    "required_objects": None,
}


def frame(image: int) -> FrameRef:
    image_id = f"image-{image}"
    mapping = FrameMapping.identity(image_id, SIZE_HW)
    return FrameRef(image_id=image_id, size_hw=SIZE_HW, parent=mapping, root=mapping)


def detections(image: int, model: str, *, frame_ref=None) -> Detections:
    rows = IMAGE_ROWS[image][model]
    frame_ref = frame_ref or frame(image)
    prediction = Detections(
        xyxy=torch.tensor([box for _, _, box in rows], dtype=torch.float32).reshape(
            -1, 4
        ),
        class_id=torch.tensor([c for c, _, _ in rows], dtype=torch.int64),
        confidence=torch.tensor([s for _, s, _ in rows], dtype=torch.float32),
        image_metadata={**frame_ref.prediction_metadata(), "class_names": NAMES},
    )
    return with_row_identity(prediction, namespace=model)


def with_masks(prediction: Detections, *, encoded: bool) -> InstanceDetections:
    masks = torch.zeros((len(prediction), *SIZE_HW), dtype=torch.bool)
    for row, (x0, y0, x1, y1) in enumerate(prediction.xyxy.int().tolist()):
        masks[row, y0:y1, x0:x1] = True
    if encoded:
        counts = [
            mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))[
                "counts"
            ]
            for mask in masks
        ]
        masks = InstancesRLEMasks(image_size=SIZE_HW, masks=counts)
    instances = InstanceDetections(
        xyxy=prediction.xyxy,
        class_id=prediction.class_id,
        confidence=prediction.confidence,
        mask=masks,
        image_metadata=prediction.image_metadata,
    )

    return instances


def workflow(step: dict, kinds=None) -> dict:
    kinds = kinds or {model: OBJECT_DETECTION_PREDICTION_KIND for model in "abc"}
    return {
        "version": "2.0",
        "inputs": [
            *(
                {"type": "WorkflowBatchInput", "name": model, "kind": [kind.name]}
                for model, kind in kinds.items()
            ),
            *({"type": "WorkflowParameter", "name": knob} for knob in KNOBS),
        ],
        "steps": [
            {
                "type": "v2/detections_consensus@v1",
                "name": "consensus",
                "predictions": [f"$inputs.{model}" for model in kinds],
                **{knob: f"$inputs.{knob}" for knob in KNOBS},
                **step,
            },
            {
                "type": "v2/per_class_confidence_filter@v1",
                "name": "confident",
                "predictions": "$steps.consensus.predictions",
                "class_thresholds": {"car": 0.75},
                "default_threshold": 0.65,
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.{selector}"}
            for name, selector in (
                ("predictions", "consensus.predictions"),
                ("object_present", "consensus.object_present"),
                ("presence_confidence", "consensus.presence_confidence"),
                ("confident", "confident.predictions"),
            )
        ],
    }


def run(step=None, sources=None, kinds=None, **knobs) -> list:
    plan = compile_workflow(
        workflow(step or {}, kinds),
        catalogue=create_catalogue().with_blocks(fusion.BLOCKS),
    )
    sources = sources or {
        model: [detections(image, model) for image in range(2)] for model in "abc"
    }
    rows = plan.create_session().run({**sources, **DEFAULTS, **knobs}).rows()
    return rows


def test_three_models_agree_per_image_and_feed_a_downstream_filter() -> None:
    # when
    first, second = run()

    # then: image 0 keeps the person (3 votes) and the car (2 votes)
    merged = first["predictions"]
    assert type(merged) is Detections
    assert [row["class"] for row in merged.bboxes_metadata] == ["person", "car"]
    assert merged.confidence.tolist() == pytest.approx([0.7, 0.7])
    expected = torch.tensor([[32 / 3, 32 / 3, 30.0, 30.0], [40.0, 0.0, 50.0, 10.0]])
    assert torch.allclose(merged.xyxy, expected)
    assert frame_of(merged) == frame(0)
    assert row_record(merged).namespace not in ("a", "b", "c")
    assert first["object_present"] is True
    assert first["presence_confidence"] == {
        "person": pytest.approx(0.7),
        "car": pytest.approx(0.7),
    }
    assert [row["class"] for row in first["confident"].bboxes_metadata] == ["person"]
    # image 1: person vs car is no agreement when class aware
    assert len(second["predictions"]) == 0
    assert frame_of(second["predictions"]) == frame(1)
    assert second["object_present"] is False
    assert second["presence_confidence"] == {}


@pytest.mark.parametrize(
    "knobs, image_0_classes, present, presence",
    [
        ({"required_votes": 3}, ["person"], [True, False], {"person": 0.7}),
        ({"iou_threshold": 0.95}, ["car"], [True, False], {"car": 0.7}),
        ({"confidence": 0.75}, [], [False, False], {}),
        ({"classes_to_consider": ["car"]}, ["car"], [True, False], {"car": 0.7}),
        ({"required_objects": 3}, ["person", "car"], [False, False], {}),
        (
            {"required_objects": {"car": 1}},
            ["person", "car"],
            [True, False],
            {"person": 0.7, "car": 0.7},
        ),
        (
            {"class_aware": False},
            ["person", "car"],
            [True, True],
            {"any_object": 0.7},
        ),
    ],
)
def test_selector_knobs_change_the_vote(
    knobs: dict, image_0_classes: list, present: list, presence: dict
) -> None:
    # when
    rows = run(**knobs)

    # then
    merged = rows[0]["predictions"]
    assert [row["class"] for row in merged.bboxes_metadata] == image_0_classes
    assert [row["object_present"] for row in rows] == present
    assert rows[0]["presence_confidence"] == pytest.approx(presence)


def test_class_agnostic_vote_merges_disagreeing_classes() -> None:
    second = run(class_aware=False)[1]

    assert [row["class"] for row in second["predictions"].bboxes_metadata] == ["person"]
    assert second["presence_confidence"] == {"any_object": pytest.approx(0.75)}


def test_literal_options_and_the_full_field_name() -> None:
    # given
    step = {
        "predictions_batches": ["$inputs.a", "$inputs.b", "$inputs.c"],
        "required_votes": 3,
        "detections_merge_coordinates_aggregation": "max",
        "presence_confidence_aggregation": "min",
    }
    definition = workflow(step)
    del definition["steps"][0]["predictions"]
    plan = compile_workflow(
        definition, catalogue=create_catalogue().with_blocks(fusion.BLOCKS)
    )
    sources = {
        model: [detections(image, model) for image in range(2)] for model in "abc"
    }

    # when
    first = plan.create_session().run({**sources, **DEFAULTS}).rows()[0]

    # then: the largest of three 20x20, 18x20, 20x18 boxes is the first
    assert first["predictions"].xyxy.tolist() == [list(map(float, PERSON))]
    assert first["presence_confidence"] == {"person": pytest.approx(0.7)}


@pytest.mark.parametrize(
    "step, message",
    [
        ({"required_votes": 0}, "required_votes"),
        ({"iou_threshold": 1.5}, "iou_threshold"),
        ({"detections_merge_mask_aggregation": "average"}, "mask_aggregation"),
        ({"predictions": []}, "predictions"),
    ],
)
def test_invalid_literals_fail_compilation(step: dict, message: str) -> None:
    with pytest.raises(ParamsValidationError, match=message):
        compile_workflow(
            workflow(step), catalogue=create_catalogue().with_blocks(fusion.BLOCKS)
        )


@pytest.mark.parametrize("field", ["predictions", "predictions_batches"])
def test_one_selector_instead_of_a_list_fails_compilation(field: str) -> None:
    definition = workflow({})
    del definition["steps"][0]["predictions"]
    definition["steps"][0][field] = "$inputs.a"

    with pytest.raises(ParamsValidationError) as raised:
        compile_workflow(
            definition, catalogue=create_catalogue().with_blocks(fusion.BLOCKS)
        )

    assert (
        f'{field}: Input should be a list of selectors, e.g. ["$inputs.a"]; '
        "one selector cannot fill the whole list"
    ) in str(raised.value)


@pytest.mark.parametrize(
    "knobs, message",
    [
        ({"required_votes": 0}, "required_votes is invalid after resolution"),
        ({"iou_threshold": 1.5}, "iou_threshold is invalid after resolution"),
        ({"required_objects": {"car": 0}}, "required_objects must be None"),
    ],
)
def test_invalid_resolved_values_fail_the_step(knobs: dict, message: str) -> None:
    with pytest.raises(StepExecutionError, match=message):
        run(**knobs)


def test_dense_and_rle_masks_vote_into_one_instance_segmentation() -> None:
    # given
    kinds = {
        "a": INSTANCE_SEGMENTATION_PREDICTION_KIND,
        "b": RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        "c": OBJECT_DETECTION_PREDICTION_KIND,
    }
    sources = {
        "a": [with_masks(detections(i, "a"), encoded=False) for i in range(2)],
        "b": [with_masks(detections(i, "b"), encoded=True) for i in range(2)],
        "c": [detections(i, "c") for i in range(2)],
    }

    # when
    first, second = run(
        sources=sources,
        kinds=kinds,
        step={"detections_merge_mask_aggregation": "intersection"},
    )

    # then: box-only row c0 counts as an empty mask, so the person is empty
    merged = first["predictions"]
    assert isinstance(merged, InstanceDetections)
    assert merged.image_metadata["prediction_type"] == "instance-segmentation"
    assert merged.mask.flatten(1).sum(dim=1).tolist() == [0, 100]
    assert merged.xyxy.tolist() == [[0.0, 0.0, 0.0, 0.0], [40.0, 0.0, 49.0, 9.0]]
    assert len(second["predictions"]) == 0


def image_0_named(image_ids, *, masked: bool) -> dict:
    """Image 0 of each model under a local image id that may be unknown."""
    mapping = FrameMapping.identity("image-0", SIZE_HW)
    sources = {}
    for model, image_id in zip("abc", image_ids):
        frame_ref = FrameRef(
            image_id=image_id, size_hw=SIZE_HW, parent=mapping, root=mapping
        )
        prediction = detections(0, model, frame_ref=frame_ref)
        if masked:
            prediction = with_masks(prediction, encoded=False)
        sources[model] = [prediction]

    return sources


def kinds_of(masked: bool) -> dict:
    kind = (
        INSTANCE_SEGMENTATION_PREDICTION_KIND
        if masked
        else OBJECT_DETECTION_PREDICTION_KIND
    )
    return {model: kind for model in "abc"}


@pytest.mark.parametrize("masked", [False, True], ids=["boxes", "dense"])
@pytest.mark.parametrize(
    "image_ids", list(itertools.permutations([None, "A", "B"])), ids=str
)
def test_an_unknown_image_id_does_not_bridge_two_known_ids(
    image_ids, masked: bool, monkeypatch
) -> None:
    # given: the step must fail before it asks for any mask
    def no_masks(payload):
        raise AssertionError("the frame check asked for masks")

    monkeypatch.setattr(consensus, "mask_provider_of", no_masks)

    # when / then
    with pytest.raises(StepExecutionError, match="different images or coordinate"):
        run(sources=image_0_named(image_ids, masked=masked), kinds=kinds_of(masked))


@pytest.mark.parametrize("masked", [False, True], ids=["boxes", "dense"])
@pytest.mark.parametrize(
    "image_ids", [(None, "A", "A"), ("A", "A", None), (None, None, None)], ids=str
)
def test_unknown_image_ids_agree_with_one_known_id(image_ids, masked: bool) -> None:
    (result,) = run(
        sources=image_0_named(image_ids, masked=masked), kinds=kinds_of(masked)
    )

    merged = result["predictions"]
    assert [row["class"] for row in merged.bboxes_metadata] == ["person", "car"]


def test_a_missing_prediction_skips_consensus_for_that_image() -> None:
    # given
    sources = {
        model: [detections(image, model) for image in range(2)] for model in "abc"
    }
    sources["c"][1] = None

    # when
    first, second = run(sources=sources)

    # then
    assert len(first["predictions"]) == 2
    assert second["predictions"] is None
    assert second["object_present"] is None
