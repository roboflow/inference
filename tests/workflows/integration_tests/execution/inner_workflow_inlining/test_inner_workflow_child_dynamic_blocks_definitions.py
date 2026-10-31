"""
Equivalence for a child workflow whose ``dynamic_blocks_definitions`` live only on the
nested definition: parent ``inner_workflow`` vs a flat workflow with hoisted definitions
and an inlined step name.
"""

from typing import Any, Dict
from unittest import mock

from inference.core.managers.base import ModelManager
from inference.core.workflows.execution_engine.v1.dynamic_blocks import block_assembler
from tests.workflows.integration_tests.execution.inner_workflow_inlining._common import (
    execution_engine,
)

_SCALAR_ECHO_DYNAMIC_RUN_CODE = """
def run(self, value: str) -> BlockResult:
    return {"output": value}
"""

_DYNAMIC_BLOCK_TYPE = "InnerScalarEcho"


def _dynamic_block_definition() -> Dict[str, Any]:
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": _DYNAMIC_BLOCK_TYPE,
            "inputs": {
                "value": {
                    "type": "DynamicInputDefinition",
                    "selector_types": ["input_parameter"],
                },
            },
            "outputs": {
                "output": {"type": "DynamicOutputDefinition", "kind": []},
            },
        },
        "code": {
            "type": "PythonCode",
            "run_function_code": _SCALAR_ECHO_DYNAMIC_RUN_CODE,
        },
    }


def _child_workflow_with_dynamic_block() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "dynamic_blocks_definitions": [_dynamic_block_definition()],
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "child_msg",
                "default_value": "default-child",
            },
        ],
        "steps": [
            {
                "type": _DYNAMIC_BLOCK_TYPE,
                "name": "pick",
                "value": "$inputs.child_msg",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "echo",
                "selector": "$steps.pick.output",
            },
        ],
    }


def _nested_parent_workflow() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "root_msg",
                "default_value": "unused-root",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested",
                "workflow_definition": _child_workflow_with_dynamic_block(),
                "parameter_bindings": {
                    "child_msg": "$inputs.root_msg",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "final",
                "selector": "$steps.nested.echo",
            },
        ],
    }


def _flat_inlined_equivalent() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "dynamic_blocks_definitions": [_dynamic_block_definition()],
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "root_msg",
                "default_value": "unused-root",
            },
        ],
        "steps": [
            {
                "type": _DYNAMIC_BLOCK_TYPE,
                "name": "nested__pick",
                "value": "$inputs.root_msg",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "final",
                "selector": "$steps.nested__pick.output",
            },
        ],
    }


@mock.patch.object(block_assembler, "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", True)
def test_inlined_dynamic_block_matches_inner_workflow_child_definitions(
    model_manager: ModelManager,
) -> None:
    nested_engine = execution_engine(model_manager, _nested_parent_workflow())
    flat_engine = execution_engine(model_manager, _flat_inlined_equivalent())

    runtime_parameters = {"root_msg": "dynamic-inner-value"}
    nested_result = nested_engine.run(runtime_parameters=runtime_parameters)
    flat_result = flat_engine.run(runtime_parameters=runtime_parameters)

    assert nested_result == flat_result
    assert len(nested_result) == 1
    assert nested_result[0] == {"final": "dynamic-inner-value"}
