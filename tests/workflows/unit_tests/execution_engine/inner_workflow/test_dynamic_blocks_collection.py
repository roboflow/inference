"""Tests for collecting dynamic block definitions from nested inner workflows."""

from typing import Any, Dict
from unittest import mock

from inference.core.workflows.execution_engine.v1.inner_workflow import (
    dynamic_blocks_collection,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.dynamic_blocks_collection import (
    apply_collected_dynamic_blocks_definitions_to_workflow_root,
    collect_dynamic_blocks_definitions_from_workflow_definition,
)


def _dynamic_block_definition(block_type: str) -> Dict[str, Any]:
    return {
        "type": "DynamicBlockDefinition",
        "manifest": {
            "type": "ManifestDescription",
            "block_type": block_type,
            "inputs": {},
            "outputs": {},
        },
        "code": {
            "type": "PythonCode",
            "run_function_code": "def run(self): return {}",
        },
    }


def test_collect_returns_empty_when_no_dynamic_blocks() -> None:
    workflow = {
        "version": "1.0",
        "inputs": [],
        "steps": [],
        "outputs": [],
    }

    collected = collect_dynamic_blocks_definitions_from_workflow_definition(
        workflow_definition=workflow,
    )

    assert collected == []


def test_collect_from_root_only() -> None:
    parent_block = _dynamic_block_definition("ParentBlock")
    workflow = {
        "version": "1.0",
        "dynamic_blocks_definitions": [parent_block],
        "inputs": [],
        "steps": [],
        "outputs": [],
    }

    collected = collect_dynamic_blocks_definitions_from_workflow_definition(
        workflow_definition=workflow,
    )

    assert collected == [parent_block]


def test_collect_from_nested_inner_workflow() -> None:
    child_block = _dynamic_block_definition("ChildBlock")
    child_workflow = {
        "version": "1.0",
        "dynamic_blocks_definitions": [child_block],
        "inputs": [],
        "steps": [
            {
                "type": "ChildBlock",
                "name": "use_child",
            },
        ],
        "outputs": [],
    }
    workflow = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "nested",
                "workflow_definition": child_workflow,
                "parameter_bindings": {},
            },
        ],
        "outputs": [],
    }

    collected = collect_dynamic_blocks_definitions_from_workflow_definition(
        workflow_definition=workflow,
    )

    assert collected == [child_block]


def test_collect_merges_parent_and_child_with_parent_first() -> None:
    parent_block = _dynamic_block_definition("SharedType")
    child_block = _dynamic_block_definition("SharedType")
    child_only = _dynamic_block_definition("ChildOnly")
    child_workflow = {
        "version": "1.0",
        "dynamic_blocks_definitions": [child_block, child_only],
        "inputs": [],
        "steps": [],
        "outputs": [],
    }
    workflow = {
        "version": "1.0",
        "dynamic_blocks_definitions": [parent_block],
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "nested",
                "workflow_definition": child_workflow,
                "parameter_bindings": {},
            },
        ],
        "outputs": [],
    }

    collected = collect_dynamic_blocks_definitions_from_workflow_definition(
        workflow_definition=workflow,
    )

    assert collected == [parent_block, child_only]


def test_apply_hoists_collected_definitions_to_workflow_root() -> None:
    child_block = _dynamic_block_definition("ChildBlock")
    child_workflow = {
        "version": "1.0",
        "dynamic_blocks_definitions": [child_block],
        "inputs": [],
        "steps": [],
        "outputs": [],
    }
    workflow = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "nested",
                "workflow_definition": child_workflow,
                "parameter_bindings": {},
            },
        ],
        "outputs": [],
    }

    merged = apply_collected_dynamic_blocks_definitions_to_workflow_root(
        workflow_definition=workflow,
    )

    assert merged == [child_block]
    assert workflow["dynamic_blocks_definitions"] == [child_block]


def test_collect_deduplicates_same_block_type_from_repeated_inner_child() -> None:
    child_block = _dynamic_block_definition("ChildBlock")
    child_workflow = {
        "version": "1.0",
        "dynamic_blocks_definitions": [child_block],
        "inputs": [],
        "steps": [],
        "outputs": [],
    }
    inner_step = {
        "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
        "name": "nested",
        "workflow_definition": child_workflow,
        "parameter_bindings": {},
    }
    workflow = {
        "version": "1.0",
        "inputs": [],
        "steps": [inner_step, {**inner_step, "name": "nested_copy"}],
        "outputs": [],
    }

    collected = collect_dynamic_blocks_definitions_from_workflow_definition(
        workflow_definition=workflow,
    )

    assert collected == [child_block]


@mock.patch.object(dynamic_blocks_collection.logger, "warning")
def test_collect_logs_warning_for_duplicate_block_type(mock_warning: mock.Mock) -> None:
    parent_block = _dynamic_block_definition("SharedType")
    child_block = _dynamic_block_definition("SharedType")
    child_workflow = {
        "version": "1.0",
        "dynamic_blocks_definitions": [child_block],
        "inputs": [],
        "steps": [],
        "outputs": [],
    }
    workflow = {
        "version": "1.0",
        "dynamic_blocks_definitions": [parent_block],
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "nested",
                "workflow_definition": child_workflow,
                "parameter_bindings": {},
            },
        ],
        "outputs": [],
    }

    collected = collect_dynamic_blocks_definitions_from_workflow_definition(
        workflow_definition=workflow,
    )

    assert collected == [parent_block]
    mock_warning.assert_called_once()
    assert mock_warning.call_args.args[1] == "SharedType"


def test_collect_passes_through_non_dict_entries_for_downstream_validation() -> None:
    invalid_entry = "not-a-dynamic-block-definition"
    workflow = {
        "version": "1.0",
        "dynamic_blocks_definitions": [invalid_entry],
        "inputs": [],
        "steps": [],
        "outputs": [],
    }

    collected = collect_dynamic_blocks_definitions_from_workflow_definition(
        workflow_definition=workflow,
    )

    assert collected == [invalid_entry]


def test_collect_passes_through_non_list_dynamic_blocks_definitions() -> None:
    invalid_definitions = {"type": "DynamicBlockDefinition"}
    workflow = {
        "version": "1.0",
        "dynamic_blocks_definitions": invalid_definitions,
        "inputs": [],
        "steps": [],
        "outputs": [],
    }

    collected = collect_dynamic_blocks_definitions_from_workflow_definition(
        workflow_definition=workflow,
    )

    assert collected == [invalid_definitions]
