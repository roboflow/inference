import pytest
from roboflow_workflows.core_steps.common.query_language.entities.enums import (
    ActionRecognitionProperty,
)
from roboflow_workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from roboflow_workflows.core_steps.common.query_language.operations.action_recognition.base import (
    extract_action_recognition_property,
)
from roboflow_workflows.core_steps.formatters.property_definition.v1 import (
    BlockManifest,
)
from roboflow_workflows.execution_engine.core import ExecutionEngine


def _extract(field: str) -> dict:
    return {"type": "ActionRecognitionPropertyExtract", "property_name": field}


def test_catalog_offers_sequence_length_for_the_timeline() -> None:
    definitions = BlockManifest.model_json_schema()["$defs"]
    extractor = definitions["ActionRecognitionPropertyExtract"]
    field_kinds = extractor["output_kind_by_property"]
    length_inputs = {kind.name for kind in definitions["SequenceLength"]["input_kind"]}

    assert field_kinds.keys() == set(definitions["ActionRecognitionProperty"]["enum"])
    assert {
        field
        for field, kinds in field_kinds.items()
        if length_inputs.intersection(kinds)
    } == {"timeline"}
    assert length_inputs.intersection(kind.name for kind in extractor["output_kind"])


def test_property_definition_extracts_action_recognition_result_fields() -> None:
    fields = ["timeline", "analyzed_from_frame", "analyzed_until_frame", "error_status"]
    workflow = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "results",
                "kind": ["action_recognition_result"],
            }
        ],
        "steps": [
            {
                "type": "roboflow_core/property_definition@v1",
                "name": f"extract_{field}",
                "data": "$inputs.results",
                "operations": [_extract(field)],
            }
            for field in fields
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": field,
                "selector": f"$steps.extract_{field}.output",
            }
            for field in fields
        ],
    }
    walk = {"class": "walk", "class_id": 0, "start_frame_idx": 14, "end_frame_idx": 18}
    completed = {
        "timeline": [walk],
        "analyzed_from_frame": 12,
        "analyzed_until_frame": 21,
        "error_status": "A later window was skipped.",
    }
    warmup = {
        "timeline": [],
        "analyzed_from_frame": None,
        "analyzed_until_frame": None,
        "error_status": "",
    }

    engine = ExecutionEngine.init(workflow_definition=workflow)
    try:
        outputs = engine.run(
            runtime_parameters={"results": [warmup, completed]}, serialize_results=True
        )
    finally:
        engine.close()

    assert outputs == [warmup, completed]


def test_scalar_action_result_gates_counting_a_batch_of_timelines() -> None:
    workflow = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "progress",
                "kind": ["action_recognition_result"],
            },
            {
                "type": "WorkflowBatchInput",
                "name": "results",
                "kind": ["action_recognition_result"],
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/continue_if@v1",
                "name": "gate",
                "evaluation_parameters": {"progress": "$inputs.progress"},
                "condition_statement": {
                    "type": "StatementGroup",
                    "statements": [
                        {
                            "type": "BinaryStatement",
                            "left_operand": {
                                "type": "DynamicOperand",
                                "operand_name": "progress",
                                "operations": [_extract("analyzed_until_frame")],
                            },
                            "comparator": {"type": "(Number) >"},
                            "right_operand": {"type": "StaticOperand", "value": 4},
                        }
                    ],
                },
                "next_steps": ["$steps.count"],
            },
            {
                "type": "roboflow_core/property_definition@v1",
                "name": "count",
                "data": "$inputs.results",
                "operations": [_extract("timeline"), {"type": "SequenceLength"}],
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "count", "selector": "$steps.count.output"}
        ],
    }
    walk = {"class": "walk", "class_id": 0, "start_frame_idx": 2, "end_frame_idx": 4}
    inputs = {
        "progress": {
            "timeline": [],
            "analyzed_from_frame": 1,
            "analyzed_until_frame": 4,
        },
        "results": [{"timeline": []}, {"timeline": [walk]}],
    }
    engine = ExecutionEngine.init(workflow_definition=workflow)
    try:
        assert engine.run(runtime_parameters=inputs) == [
            {"count": None},
            {"count": None},
        ]
        inputs["progress"]["analyzed_until_frame"] = 5
        assert engine.run(runtime_parameters=inputs) == [{"count": 0}, {"count": 1}]
    finally:
        engine.close()


def test_scalar_extraction_validates_the_whole_result() -> None:
    with pytest.raises(InvalidInputTypeError):
        extract_action_recognition_property(
            {"timeline": "invalid", "error_status": ""},
            property_name=ActionRecognitionProperty.ERROR_STATUS,
            execution_context="condition",
        )
