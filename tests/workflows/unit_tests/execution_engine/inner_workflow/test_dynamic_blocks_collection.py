import pytest

from inference.core.workflows.execution_engine.v1.inner_workflow.compiler_bridge import (
    collect_dynamic_blocks_definitions_from_raw_workflow_definition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.errors import (
    InnerWorkflowInvalidStepEntryError,
)


def _dynamic_block_definition(block_type: str) -> dict:
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": block_type,
            "inputs": {
                "value": {
                    "type": "DynamicInputDefinition",
                    "value_types": ["float"],
                },
            },
            "outputs": {"output": {"type": "DynamicOutputDefinition", "kind": []}},
        },
        "code": {
            "type": "PythonCode",
            "run_function_code": "def run(self, value):\n    return {'output': value}\n",
        },
    }


def test_collects_dynamic_blocks_from_inner_workflow_definitions() -> None:
    root_block = _dynamic_block_definition("RootDynamicBlock")
    child_block = _dynamic_block_definition("ConfidenceTransformer")
    grandchild_block = _dynamic_block_definition("GrandchildDynamicBlock")
    workflow_definition = {
        "version": "1.0",
        "inputs": [],
        "dynamic_blocks_definitions": [root_block],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "inner",
                "workflow_definition": {
                    "version": "1.0",
                    "inputs": [],
                    "dynamic_blocks_definitions": [child_block],
                    "steps": [
                        {
                            "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                            "name": "grandchild",
                            "workflow_definition": {
                                "version": "1.0",
                                "inputs": [],
                                "dynamic_blocks_definitions": [grandchild_block],
                                "steps": [],
                                "outputs": [],
                            },
                            "parameter_bindings": {},
                        }
                    ],
                    "outputs": [],
                },
                "parameter_bindings": {},
            }
        ],
        "outputs": [],
    }

    result = collect_dynamic_blocks_definitions_from_raw_workflow_definition(
        workflow_definition
    )

    assert result == [root_block, child_block, grandchild_block]


def test_collecting_dynamic_blocks_deduplicates_exact_repeated_definitions() -> None:
    shared_block = _dynamic_block_definition("SharedDynamicBlock")
    workflow_definition = {
        "version": "1.0",
        "inputs": [],
        "dynamic_blocks_definitions": [shared_block],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "inner",
                "workflow_definition": {
                    "version": "1.0",
                    "inputs": [],
                    "dynamic_blocks_definitions": [shared_block],
                    "steps": [],
                    "outputs": [],
                },
                "parameter_bindings": {},
            }
        ],
        "outputs": [],
    }

    result = collect_dynamic_blocks_definitions_from_raw_workflow_definition(
        workflow_definition
    )

    assert result == [shared_block]


def test_collecting_dynamic_blocks_rejects_circular_python_workflow_objects() -> None:
    workflow_definition = {
        "version": "1.0",
        "inputs": [],
        "steps": [],
        "outputs": [],
    }
    workflow_definition["steps"].append(
        {
            "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
            "name": "inner",
            "workflow_definition": workflow_definition,
            "parameter_bindings": {},
        }
    )

    with pytest.raises(InnerWorkflowInvalidStepEntryError, match="Circular"):
        collect_dynamic_blocks_definitions_from_raw_workflow_definition(
            workflow_definition
        )
