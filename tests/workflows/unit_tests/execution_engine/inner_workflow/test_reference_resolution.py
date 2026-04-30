"""Tests for ``roboflow_core/inner_workflow@v1`` reference resolution before compilation."""

from typing import Any, Dict, Optional

import pytest

from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
    WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER,
    normalize_inner_workflow_references_in_definition,
    workflow_definition_contains_unresolved_inner_workflow_reference,
)


def _echo_spec() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "child_msg",
                "default_value": "x",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/first_non_empty_or_default@v1",
                "name": "pick",
                "data": ["$inputs.child_msg"],
                "default": "fallback-inner",
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


def test_normalize_replaces_reference_with_inline_workflow() -> None:
    calls: list[tuple[str, str, Optional[str]]] = []

    def resolver(
        workspace_id: str,
        workflow_id: str,
        workflow_version_id: Optional[str],
        init_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        calls.append((workspace_id, workflow_id, workflow_version_id))
        assert init_parameters["marker"] == 1
        return _echo_spec()

    raw = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "nested",
                "workflow_workspace_id": "my-ws",
                "workflow_id": "wf-1",
                "workflow_version_id": "v9",
                "parameter_bindings": {"child_msg": "$inputs.p"},
            },
        ],
        "outputs": [],
    }
    init_parameters = {
        WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver,
        "marker": 1,
    }
    out = normalize_inner_workflow_references_in_definition(raw, init_parameters)

    assert raw["steps"][0].get("workflow_definition") is None
    assert out["steps"][0]["workflow_definition"] == _echo_spec()
    assert "workflow_workspace_id" not in out["steps"][0]
    assert calls == [("my-ws", "wf-1", "v9")]


def test_normalize_deduplicates_identical_references() -> None:
    calls: list[tuple[str, str, Optional[str]]] = []

    def resolver(
        workspace_id: str,
        workflow_id: str,
        workflow_version_id: Optional[str],
        init_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        calls.append((workspace_id, workflow_id, workflow_version_id))
        return _echo_spec()

    raw = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "a",
                "workflow_workspace_id": "ws",
                "workflow_id": "same",
                "parameter_bindings": {"child_msg": "$inputs.p"},
            },
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "b",
                "workflow_workspace_id": "ws",
                "workflow_id": "same",
                "parameter_bindings": {"child_msg": "$inputs.p"},
            },
        ],
        "outputs": [],
    }
    out = normalize_inner_workflow_references_in_definition(
        raw,
        {WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver},
    )
    assert len(calls) == 1
    assert (
        out["steps"][0]["workflow_definition"] == out["steps"][1]["workflow_definition"]
    )


def test_normalize_resolves_reference_inside_inline_workflow() -> None:
    calls: list[str] = []

    def resolver(
        workspace_id: str,
        workflow_id: str,
        workflow_version_id: Optional[str],
        init_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        calls.append(workflow_id)
        return _echo_spec()

    raw = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "outer",
                "workflow_definition": {
                    "version": "1.0",
                    "inputs": [{"type": "WorkflowParameter", "name": "wrapper_msg"}],
                    "steps": [
                        {
                            "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                            "name": "inner",
                            "workflow_workspace_id": "ws",
                            "workflow_id": "inner-id",
                            "parameter_bindings": {
                                "child_msg": "$inputs.wrapper_msg",
                            },
                        },
                    ],
                    "outputs": [],
                },
                "parameter_bindings": {"wrapper_msg": "$inputs.p"},
            },
        ],
        "outputs": [],
    }
    out = normalize_inner_workflow_references_in_definition(
        raw,
        {WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver},
    )
    assert calls == ["inner-id"]
    inner = out["steps"][0]["workflow_definition"]["steps"][0]
    assert inner["workflow_definition"] == _echo_spec()
    assert "workflow_workspace_id" not in inner


def test_mutual_exclusion_workflow_and_reference() -> None:
    def resolver(
        workspace_id: str,
        workflow_id: str,
        workflow_version_id: Optional[str],
        init_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        return _echo_spec()

    raw = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "bad",
                "workflow_definition": _echo_spec(),
                "workflow_workspace_id": "ws",
                "workflow_id": "id",
                "parameter_bindings": {"child_msg": "$inputs.p"},
            },
        ],
        "outputs": [],
    }
    with pytest.raises(WorkflowDefinitionError):
        normalize_inner_workflow_references_in_definition(
            raw,
            {WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver},
        )


def test_contains_unresolved_reference_false_for_inline_only() -> None:
    raw = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "n",
                "workflow_definition": _echo_spec(),
                "parameter_bindings": {},
            },
        ],
        "outputs": [],
    }
    assert not workflow_definition_contains_unresolved_inner_workflow_reference(raw)


def test_normalize_returns_same_object_when_no_reference() -> None:
    raw = {
        "version": "1.0",
        "inputs": [],
        "steps": [],
        "outputs": [],
    }
    out = normalize_inner_workflow_references_in_definition(raw, {})
    assert out is raw
