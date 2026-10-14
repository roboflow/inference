"""Merge and combine composed through the real V2 compiler and session.

Street image (40x60), as the two stand-in models see it::

    model     row  class   conf  box
    people    0    person  0.9   (4, 4, 12, 20)
    people    1    person  0.6   (32, 8, 40, 32)
    vehicles  0    car     0.8   (12, 16, 28, 28)
    vehicles  1    bus     0.7   (40, 4, 56, 24)
    vehicles  2    car     0.4   (20, 28, 36, 36)

The chain under test::

    people ───┐
              ├─▶ combine ─▶ per-class filter ─▶ merge "scene" ─▶ export
    vehicles ─┘   5 rows      person >= 0.7      union of kept rows
                              others >= 0.5      → (4, 4, 56, 28) at 0.7
                              → rows 0, 2, 3
"""

import pytest
from roboflow_workflows.v2.blocks.query.detection_ops import derived_detection_id
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.engine.compilation import compile_workflow

from .synthetic import STREET, combine_catalogue, painted_image


def detector(name: str, model: str) -> dict:
    return {
        "type": "test/painted_detector@v1",
        "name": name,
        "image": "$inputs.image",
        "model": model,
    }


def chain_steps(output: str = "predictions") -> list:
    return [
        detector("people", "people"),
        detector("vehicles", "vehicles"),
        {
            "type": "roboflow_core/detections_combine@v1",
            "name": "combined",
            "prediction_one": f"$steps.people.{output}",
            "prediction_two": f"$steps.vehicles.{output}",
        },
        {
            "type": "roboflow_core/per_class_confidence_filter@v1",
            "name": "confident",
            "predictions": "$steps.combined.predictions",
            "class_thresholds": {"person": 0.7},
            "default_threshold": 0.5,
        },
        {
            "type": "roboflow_core/detections_merge@v1",
            "name": "scene",
            "predictions": "$steps.confident.predictions",
            "class_name": "$inputs.merged_name",
        },
        {
            "type": "test/detections_to_dictionary@v1",
            "name": "export",
            "predictions": "$steps.combined.predictions",
        },
    ]


OUTPUTS = {
    "people": "$steps.people.predictions",
    "vehicles": "$steps.vehicles.predictions",
    "combined": "$steps.combined.predictions",
    "confident": "$steps.confident.predictions",
    "scene": "$steps.scene.predictions",
    "serialized": "$steps.export.serialized",
}


def run(steps, outputs=OUTPUTS, images=None, parameters=("merged_name",), **values):
    workflow = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            *({"type": "WorkflowParameter", "name": name} for name in parameters),
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    plan = compile_workflow(workflow, catalogue=combine_catalogue())
    defaults = {"merged_name": "scene"}
    inputs = {
        "image": images or [painted_image((40, 60), STREET)],
        **{name: defaults[name] for name in parameters if name in defaults},
        **values,
    }
    return plan.create_session().run(inputs).rows()


def test_two_models_combine_filter_and_merge_into_one_scene_box() -> None:
    # when
    row = run(chain_steps())[0]

    # then
    combined = row_record(row["combined"])
    assert combined.namespace not in (
        row_record(row["people"]).namespace,
        row_record(row["vehicles"]).namespace,
    )
    assert row_record(row["confident"]).host_ids() == [0, 2, 3]
    exported = row["serialized"]["predictions"]
    assert [p["class"] for p in exported] == ["person", "person", "car", "bus", "car"]
    assert [p["class_id"] for p in exported] == [0, 0, 0, 1, 0]
    assert [p["detection_id"] for p in exported] == [
        derived_detection_id(combined.namespace, row_id) for row_id in range(5)
    ]
    scene = row["scene"]
    assert scene.xyxy.tolist() == [[4.0, 4.0, 56.0, 28.0]]
    assert scene.confidence.tolist() == pytest.approx([0.7])
    assert scene.image_metadata["class_names"] == {0: "scene"}
    assert row_record(scene).namespace != combined.namespace


def test_views_stay_compact_through_combine_filter_and_merge() -> None:
    # when
    row = run(chain_steps("segmentation"), outputs={
        "combined": "$steps.combined.predictions",
        "scene": "$steps.scene.predictions",
        "serialized": "$steps.export.serialized",
    })[0]  # fmt: skip

    # then
    combined = row["combined"]
    assert isinstance(combined, SegmentationView)
    assert tuple(combined.scores.shape) == (5, 10, 15)
    assert row["scene"].xyxy.tolist() == [[4.0, 4.0, 56.0, 28.0]]
    assert all(p["points"] for p in row["serialized"]["predictions"])


def test_gated_branch_skips_combine_and_merge() -> None:
    # given
    steps = [
        {
            "type": "roboflow_core/continue_if_boolean@v1",
            "name": "gate",
            "condition": "$inputs.use_vehicles",
            "next_steps": ["$steps.vehicles"],
        },
        *chain_steps(),
    ]

    # when
    off = run(steps, parameters=("merged_name", "use_vehicles"), use_vehicles=False)
    on = run(steps, parameters=("merged_name", "use_vehicles"), use_vehicles=True)

    # then
    assert len(off[0]["people"]) == 2
    assert off[0]["vehicles"] is None
    assert off[0]["combined"] is None and off[0]["scene"] is None
    assert len(on[0]["combined"]) == 5


def test_batch_image_without_vehicles_passes_people_through() -> None:
    # given
    people_only = [marker for marker in STREET if marker[5] == 0]
    images = [painted_image((40, 60), people_only), painted_image((40, 60), STREET)]

    # when
    first, second = run(chain_steps(), images=images)

    # then
    assert first["combined"] is first["people"]
    assert first["scene"].xyxy.tolist() == [[4.0, 4.0, 12.0, 20.0]]
    assert len(second["combined"]) == 5


def test_filtered_subsets_of_one_model_keep_its_row_identity() -> None:
    # given
    steps = [
        detector("vehicles", "vehicles"),
        {
            "type": "roboflow_core/per_class_confidence_filter@v1",
            "name": "buses",
            "predictions": "$steps.vehicles.predictions",
            "class_thresholds": {"car": 1.0},
            "default_threshold": 0.0,
        },
        {
            "type": "roboflow_core/per_class_confidence_filter@v1",
            "name": "sure_cars",
            "predictions": "$steps.vehicles.predictions",
            "class_thresholds": {"bus": 1.0},
            "default_threshold": 0.5,
        },
        {
            "type": "roboflow_core/detections_combine@v1",
            "name": "combined",
            "prediction_one": "$steps.buses.predictions",
            "prediction_two": "$steps.sure_cars.predictions",
        },
    ]

    # when
    row = run(
        steps,
        outputs={
            "vehicles": "$steps.vehicles.predictions",
            "combined": "$steps.combined.predictions",
        },
        parameters=(),
    )[0]

    # then
    record = row_record(row["combined"])
    assert record.namespace == row_record(row["vehicles"]).namespace
    assert record.host_ids() == [1, 0]
    assert row["combined"].bboxes_metadata is None
