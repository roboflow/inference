from inference.core.workflows.execution_engine.v1.compiler.core import (
    compile_workflow_graph,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)


def _confidence_transformer_dynamic_block() -> dict:
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": "ConfidenceTransformer",
            "inputs": {
                "confidence": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter"],
                    "selector_data_kind": {"input_parameter": ["*"]},
                    "value_types": ["float"],
                    "has_default_value": True,
                    "default_value": 0.5,
                },
            },
            "outputs": {
                "transformed_confidence": {
                    "type": "DynamicOutputDefinition",
                    "kind": [],
                },
            },
        },
        "code": {
            "type": "PythonCode",
            "run_function_code": """
import math

def run(self, confidence=0.5):
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 0.5

    if not math.isfinite(value):
        value = 0.5

    transformed = 0.75 + value * 0.25
    transformed = max(0.0, min(1.0, transformed))
    return {"transformed_confidence": round(transformed, 4)}
""",
        },
    }


def _workflow_with_nested_dynamic_block_definition() -> dict:
    child_workflow = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "confidence"},
        ],
        "dynamic_blocks_definitions": [_confidence_transformer_dynamic_block()],
        "steps": [
            {
                "type": "ConfidenceTransformer",
                "name": "transform_confidence",
                "confidence": "$inputs.confidence",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "transformed_confidence",
                "selector": "$steps.transform_confidence.transformed_confidence",
            },
        ],
    }
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "confidence"},
        ],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "inner",
                "workflow_definition": child_workflow,
                "parameter_bindings": {"confidence": "$inputs.confidence"},
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "transformed_confidence",
                "selector": "$steps.inner.transformed_confidence",
            },
        ],
    }


def test_compiles_dynamic_block_definition_declared_inside_inner_workflow() -> None:
    result = compile_workflow_graph(
        workflow_definition=_workflow_with_nested_dynamic_block_definition(),
        init_parameters={},
    )

    assert [
        (step.name, step.type) for step in result.parsed_workflow_definition.steps
    ] == [("inner__transform_confidence", "ConfidenceTransformer")]
