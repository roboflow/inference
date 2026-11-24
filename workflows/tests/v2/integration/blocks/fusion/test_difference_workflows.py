"""Detection changes through selector-bound run/phases, batches and nesting."""

import pytest
import torch
from roboflow_workflows.v2.blocks.control.continue_if_boolean.v1 import ContinueIfBlock
from roboflow_workflows.v2.blocks.fusion.detections_difference.v1 import DetectionsDifferenceBlock
from roboflow_workflows.v2.data.batch import (
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

CATALOGUE = Catalogue(
    [DetectionsDifferenceBlock, ContinueIfBlock],
    kinds=[OBJECT_DETECTION_PREDICTION_KIND],
)
OUTPUTS = (
    "removed_detections",
    "persisted_detections",
    "new_detections",
    "removed_count",
    "new_count",
    "verified",
)
KNOBS = {
    "spatial_weight": 0.4,
    "class_mismatch_penalty": 0.15,
    "reject_cost": 0.1,
    "class_strict": False,
    "min_removed_to_verify": 1,
}


def _prediction(boxes, *, image, class_id=0):
    source = ImageData.from_tensor(
        torch.zeros((3, 64, 64), dtype=torch.uint8), image_id=image
    )
    rows = Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.full((len(boxes),), class_id, dtype=torch.int64),
        confidence=torch.ones(len(boxes)),
        image_metadata=source.prediction_metadata(),
        bboxes_metadata=[{"detection_id": f"{image}-{i}"} for i in range(len(boxes))],
    )
    rows = with_row_identity(rows)

    return rows


def _definition(*, batch=False):
    prediction_type = "WorkflowBatchInput" if batch else "WorkflowParameter"
    definition = {
        "version": "2.0",
        "inputs": [
            {
                "type": prediction_type,
                "name": name,
                "kind": ["object_detection_prediction"],
            }
            for name in ("before", "after")
        ]
        + [
            {
                "type": "WorkflowParameter",
                "name": name,
                "kind": [
                    (
                        "boolean"
                        if name == "class_strict"
                        else "integer" if name == "min_removed_to_verify" else "float"
                    )
                ],
            }
            for name in KNOBS
        ],
        "steps": [
            {
                "type": DetectionsDifferenceBlock.type,
                "name": "changes",
                "reference_predictions": "$inputs.before",
                "candidate_predictions": "$inputs.after",
                **{name: f"$inputs.{name}" for name in KNOBS},
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.changes.{name}"}
            for name in OUTPUTS
        ],
    }

    return definition


def _inputs():
    before = _prediction([[0, 0, 4, 4], [20, 20, 24, 24]], image="before")
    after = _prediction([[20, 20, 24, 24], [40, 40, 44, 44]], image="after")
    inputs = {"before": before, "after": after, **KNOBS}

    return inputs


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("pipeline", [False, True])
def test_all_selector_knobs_run_and_pipeline_preserve_before_after_sources(
    mode, pipeline
):
    plan = compile_workflow(
        _definition(), catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session()
    inputs = _inputs()
    inputs["before"] = InputValue(
        inputs["before"],
        EntryMetadata(
            sample={(): SampleContext("before")},
            temporal={(): TemporalContext(observed_coverage=Timestamp(1, 1, "qa"))},
        ),
    )
    inputs["after"] = InputValue(
        inputs["after"],
        EntryMetadata(
            sample={(): SampleContext("after")},
            temporal={(): TemporalContext(observed_coverage=Timestamp(2, 1, "qa"))},
        ),
    )
    try:
        if pipeline:
            with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipe:
                futures = [pipe.submit(inputs) for _ in range(2)]
            results = [future.result() for future in futures]
        else:
            results = [session.run(inputs)]
        for result in results:
            row = result.rows()[0]
            assert (row["removed_count"], row["new_count"], row["verified"]) == (
                1,
                1,
                True,
            )
            assert (
                row["removed_detections"].bboxes_metadata[0]["detection_id"]
                == "before-0"
            )
            assert (
                row["persisted_detections"].bboxes_metadata[0]["detection_id"]
                == "after-0"
            )
            assert row["new_detections"].bboxes_metadata[0]["detection_id"] == "after-1"
            assert (
                result.outputs.metadata["removed_detections"].sample_at(()).source_id
                == "before"
            )
            assert (
                result.outputs.metadata["new_detections"]
                .temporal_at(())
                .observed_coverage.ticks
                == 2
            )
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_selected_strict_class_and_verification_knobs_change_the_result(mode):
    session = compile_workflow(
        _definition(), catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    ).create_session()
    inputs = {
        "before": _prediction([[0, 0, 4, 4]], image="before", class_id=1),
        "after": _prediction([[0, 0, 4, 4]], image="after", class_id=2),
        **KNOBS,
    }
    try:
        soft = session.run(
            {
                **inputs,
                "spatial_weight": 0.0,
                "class_mismatch_penalty": 0.2,
                "reject_cost": 0.2,
            }
        ).rows()[0]
        strict = session.run(
            {**inputs, "class_strict": True, "reject_cost": 10.0}
        ).rows()[0]
        not_enough = session.run(
            {**inputs, "class_strict": True, "min_removed_to_verify": 2}
        ).rows()[0]
        assert soft["removed_count"] == 0 and not soft["verified"]
        assert strict["removed_count"] == 1 and strict["verified"]
        assert not not_enough["verified"]
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_batched_pairs_and_nested_outputs_keep_rows_and_identities(mode):
    child = _definition(batch=True)
    definition = {
        "version": "2.0",
        "inputs": child["inputs"],
        "steps": [
            {
                "type": "inner_workflow",
                "name": "nested",
                "workflow_definition": child,
                "parameter_bindings": {
                    field["name"]: f"$inputs.{field['name']}"
                    for field in child["inputs"]
                },
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.nested.{name}"}
            for name in OUTPUTS
        ],
    }
    session = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    ).create_session()
    first = _inputs()
    empty = _prediction([], image="empty")
    try:
        rows = session.run(
            {
                **KNOBS,
                "before": [first["before"], empty],
                "after": [first["after"], first["after"]],
            }
        ).rows()
        assert [(r["removed_count"], r["new_count"], r["verified"]) for r in rows] == [
            (1, 1, True),
            (0, 2, False),
        ]
        assert (
            row_record(rows[0]["removed_detections"]).namespace
            == row_record(first["before"]).namespace
        )
        assert (
            row_record(rows[1]["removed_detections"]).namespace
            == row_record(empty).namespace
        )
        assert len(rows[1]["persisted_detections"]) == 0
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_instance_reference_and_box_candidate_preserve_their_output_families(mode):
    definition = _definition()
    definition["inputs"][0]["kind"] = ["instance_segmentation_prediction"]
    inputs = _inputs()
    before = inputs["before"]
    inputs["before"] = InstanceDetections(
        xyxy=before.xyxy,
        class_id=before.class_id,
        confidence=before.confidence,
        mask=torch.ones((2, 64, 64), dtype=torch.bool),
        image_metadata=before.image_metadata,
        bboxes_metadata=before.bboxes_metadata,
    )
    session = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    ).create_session()
    try:
        row = session.run(inputs).rows()[0]
        assert isinstance(row["removed_detections"], InstanceDetections)
        assert row["removed_detections"].mask.shape == (1, 64, 64)
        assert isinstance(row["persisted_detections"], Detections)
        assert (
            row_record(row["removed_detections"]).namespace
            == row_record(before).namespace
        )
    finally:
        session.close()


@pytest.mark.parametrize(
    "name,value",
    [
        ("spatial_weight", 2.0),
        ("class_mismatch_penalty", -1.0),
        ("reject_cost", -1.0),
        ("min_removed_to_verify", 0),
    ],
)
def test_selected_numeric_bounds_are_enforced(name, value):
    session = compile_workflow(_definition(), catalogue=CATALOGUE).create_session()
    try:
        with pytest.raises(StepExecutionError, match=name):
            session.run({**_inputs(), name: value})
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_filtered_call_and_undemanded_stateless_step(mode):
    definition = _definition()
    definition["inputs"].append(
        {"type": "WorkflowParameter", "name": "enabled", "kind": ["boolean"]}
    )
    definition["steps"].insert(
        0,
        {
            "type": ContinueIfBlock.type,
            "name": "gate",
            "condition": "$inputs.enabled",
            "next_steps": ["$steps.changes"],
        },
    )
    session = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    ).create_session()
    try:
        result = session.run({**_inputs(), "enabled": False})
        assert all(status == "filtered" for status in result.statuses.values())
    finally:
        session.close()
    definition["steps"] = [definition["steps"][1]]
    definition["outputs"] = [
        {"type": "JsonField", "name": "enabled", "selector": "$inputs.enabled"}
    ]
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    assert plan.steps == ()
