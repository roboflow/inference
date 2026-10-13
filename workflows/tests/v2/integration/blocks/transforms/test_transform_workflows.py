"""Transform blocks composed through the real V2 compiler and session.

Street image (44x60), rows in the segmenter's order (object number)::

    row  class   conf  box
    0    person  0.3   (40, 24, 48, 40)
    1    car     0.4   (36, 8, 56, 20)
    2    bus     0.7   (4, 24, 28, 36)
    3    person  0.8   (24, 4, 32, 24)
    4    car     0.9   (4, 4, 20, 16)

The chain under test::

    segmenter ─▶ per-class filter ─▶ detections filter ─▶ transformation ─▶ export
                 person >= 0.5        car or person         sort by confidence
                 others >= 0.35       → rows 1, 3, 4        → rows 4, 3, 1
                 → rows 1, 2, 3, 4
"""

import pytest
from roboflow_workflows.v2.blocks.query.detection_ops import derived_detection_id
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.engine.compilation import compile_workflow

from .synthetic import STREET, painted_image, transforms_catalogue

CLASS_IN = {
    "type": "DetectionsFilter",
    "filter_operation": {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {
                    "type": "DynamicOperand",
                    "operations": [
                        {"type": "ExtractDetectionProperty", "property_name": "class_name"}
                    ],
                },
                "comparator": {"type": "in (Sequence)"},
                "right_operand": {"type": "DynamicOperand", "operand_name": "classes"},
            }
        ],
    },
}  # fmt: skip
INPUTS = ("thresholds", "classes")


def chain_steps(source: str):
    return [
        {"type": "test/synthetic_segmenter@v1", "name": "segment", "image": "$inputs.image"},
        {
            "type": "v2/per_class_confidence_filter@v1",
            "name": "confident",
            "predictions": f"$steps.segment.{source}",
            "class_thresholds": "$inputs.thresholds",
            "default_threshold": 0.35,
        },
        {
            "type": "v2/detections_filter@v1",
            "name": "wanted",
            "predictions": "$steps.confident.predictions",
            "operations": [CLASS_IN],
            "operations_parameters": {"classes": "$inputs.classes"},
        },
        {
            "type": "v2/detections_transformation@v1",
            "name": "ordered",
            "predictions": "$steps.wanted.predictions",
            "operations": [
                {"type": "SortDetections", "mode": "confidence", "ascending": False}
            ],
        },
        {
            "type": "test/detections_to_dictionary@v1",
            "name": "export",
            "predictions": "$steps.ordered.predictions",
        },
    ]  # fmt: skip


def workflow(steps, outputs, extra_inputs=INPUTS):
    return {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            *({"type": "WorkflowParameter", "name": name} for name in extra_inputs),
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }


def run(steps, outputs, inputs, extra_inputs=INPUTS):
    plan = compile_workflow(
        workflow(steps, outputs, extra_inputs), catalogue=transforms_catalogue()
    )
    return plan.create_session().run(inputs)


def street_inputs(images=None):
    return {
        "image": images or [painted_image((44, 60), STREET)],
        "thresholds": {"person": 0.5},
        "classes": ["car", "person"],
    }


OUTPUTS = {
    "segmented": "$steps.segment.predictions",
    "ordered": "$steps.ordered.predictions",
    "serialized": "$steps.export.serialized",
}


def test_dense_chain_keeps_masks_and_exports_ids_of_the_original_rows() -> None:
    # when
    row = run(chain_steps("predictions"), OUTPUTS, street_inputs()).rows()[0]

    # then
    ordered, serialized = row["ordered"], row["serialized"]
    namespace = row_record(row["segmented"]).namespace
    assert row_record(ordered).host_ids() == [4, 3, 1]
    assert ordered.mask[:, 10, 12].tolist() == [True, False, False]
    assert ordered.mask.sum(dim=(1, 2)).tolist() == [192, 160, 240]
    predictions = serialized["predictions"]
    assert [p["detection_id"] for p in predictions] == [
        derived_detection_id(namespace, row_id) for row_id in (4, 3, 1)
    ]
    assert [p["class"] for p in predictions] == ["car", "person", "car"]
    assert [(p["x"], p["y"]) for p in predictions] == [(12, 10), (28, 14), (46, 14)]
    assert all(p["points"] for p in predictions)
    assert row["segmented"].bboxes_metadata is None


def test_view_chain_stays_compact_until_export() -> None:
    # when
    row = run(chain_steps("segmentation"), OUTPUTS, street_inputs()).rows()[0]

    # then
    ordered = row["ordered"]
    assert isinstance(ordered, SegmentationView)
    assert row_record(ordered.detections).host_ids() == [4, 3, 1]
    assert tuple(ordered.scores.shape) == (3, 11, 15)
    namespace = row_record(row["segmented"]).namespace
    assert [p["detection_id"] for p in row["serialized"]["predictions"]] == [
        derived_detection_id(namespace, row_id) for row_id in (4, 3, 1)
    ]
    assert ordered.materialization_count == 1


def test_batch_with_an_image_without_objects() -> None:
    # given
    images = [painted_image((44, 60), []), painted_image((44, 60), STREET)]

    # when
    first, second = run(
        chain_steps("predictions"), OUTPUTS, street_inputs(images)
    ).rows()

    # then
    assert first["ordered"].xyxy.shape == (0, 4)
    assert first["serialized"]["predictions"] == []
    assert row_record(second["ordered"]).host_ids() == [4, 3, 1]


def test_sorted_and_filtered_rows_still_join_their_crop_classifications() -> None:
    # given
    steps = [
        *chain_steps("predictions")[:4],
        {
            "type": "v2/dynamic_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.ordered.predictions",
        },
        {
            "type": "test/class_id_classifier@v1",
            "name": "classify",
            "image": "$steps.crop.crops",
        },
        {
            "type": "v2/detections_classes_replacement@v1",
            "name": "replace",
            "object_detection_predictions": "$steps.ordered.predictions",
            "classification_predictions": "$steps.classify.predictions",
            "fallback_class_name": "unknown",
            "fallback_class_id": 99,
        },
    ]

    # when
    row = run(
        steps, {"replaced": "$steps.replace.predictions"}, street_inputs()
    ).rows()[0]

    # then: object numbers 9, 8, 4 are rows 4, 3, 1
    replaced = row["replaced"]
    assert [m["class"] for m in replaced.bboxes_metadata] == [
        "seen-9",
        "seen-8",
        "seen-4",
    ]
    assert row_record(replaced).host_ids() == [4, 3, 1]


def test_gated_filter_produces_no_output() -> None:
    # given
    steps = [
        *chain_steps("predictions")[:2],
        {
            "type": "v2/continue_if",
            "name": "gate",
            "condition": "$inputs.enabled",
            "next_steps": ["$steps.wanted"],
        },
        *chain_steps("predictions")[2:4],
    ]
    inputs = {**street_inputs(), "enabled": False}

    # when
    result = run(
        steps,
        {"ordered": "$steps.ordered.predictions"},
        inputs,
        extra_inputs=(*INPUTS, "enabled"),
    )

    # then
    assert result.rows()[0]["ordered"] is None


def test_invalid_runtime_threshold_names_the_class() -> None:
    # given
    inputs = {**street_inputs(), "thresholds": {"person": 50}}

    # when
    with pytest.raises(Exception) as error:
        run(chain_steps("predictions"), OUTPUTS, inputs)

    # then
    assert "class_thresholds['person'] must be a number in [0, 1]" in str(error.value)


def test_undeclared_operand_fails_at_compile_time() -> None:
    # given
    steps = chain_steps("predictions")
    steps[2]["operations_parameters"] = {}

    # when
    with pytest.raises(Exception) as error:
        compile_workflow(workflow(steps, OUTPUTS), catalogue=transforms_catalogue())

    # then
    assert "operand `classes` is not declared" in str(error.value)
