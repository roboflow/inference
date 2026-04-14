"""Tests for ``use_subworkflow`` reference resolution before compilation."""

from typing import Any, Dict, Optional

import pytest

from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.v1.subworkflow.constants import (
    USE_SUBWORKFLOW_BLOCK_TYPE,
)
from inference.core.workflows.execution_engine.v1.subworkflow.reference_resolution import (
    WORKFLOWS_CORE_SUBWORKFLOW_SPEC_RESOLVER,
    normalize_use_subworkflow_references_in_definition,
    workflow_definition_contains_unresolved_subworkflow_reference,
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


def test_normalize_replaces_reference_with_embedded() -> None:
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
                "type": USE_SUBWORKFLOW_BLOCK_TYPE,
                "name": "nested",
                "embedded_workflow_workspace_id": "my-ws",
                "embedded_workflow_id": "wf-1",
                "embedded_workflow_version_id": "v9",
                "parameter_bindings": {"child_msg": "$inputs.p"},
            },
        ],
        "outputs": [],
    }
    init_parameters = {
        WORKFLOWS_CORE_SUBWORKFLOW_SPEC_RESOLVER: resolver,
        "marker": 1,
    }
    out = normalize_use_subworkflow_references_in_definition(raw, init_parameters)

    assert raw["steps"][0].get("embedded_workflow") is None
    assert out["steps"][0]["embedded_workflow"] == _echo_spec()
    assert "embedded_workflow_workspace_id" not in out["steps"][0]
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
                "type": USE_SUBWORKFLOW_BLOCK_TYPE,
                "name": "a",
                "embedded_workflow_workspace_id": "ws",
                "embedded_workflow_id": "same",
                "parameter_bindings": {"child_msg": "$inputs.p"},
            },
            {
                "type": USE_SUBWORKFLOW_BLOCK_TYPE,
                "name": "b",
                "embedded_workflow_workspace_id": "ws",
                "embedded_workflow_id": "same",
                "parameter_bindings": {"child_msg": "$inputs.p"},
            },
        ],
        "outputs": [],
    }
    out = normalize_use_subworkflow_references_in_definition(
        raw,
        {WORKFLOWS_CORE_SUBWORKFLOW_SPEC_RESOLVER: resolver},
    )
    assert len(calls) == 1
    assert out["steps"][0]["embedded_workflow"] == out["steps"][1]["embedded_workflow"]


def test_normalize_resolves_reference_inside_inline_embedded_workflow() -> None:
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
                "type": USE_SUBWORKFLOW_BLOCK_TYPE,
                "name": "outer",
                "embedded_workflow": {
                    "version": "1.0",
                    "inputs": [{"type": "WorkflowParameter", "name": "wrapper_msg"}],
                    "steps": [
                        {
                            "type": USE_SUBWORKFLOW_BLOCK_TYPE,
                            "name": "inner",
                            "embedded_workflow_workspace_id": "ws",
                            "embedded_workflow_id": "inner-id",
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
    out = normalize_use_subworkflow_references_in_definition(
        raw,
        {WORKFLOWS_CORE_SUBWORKFLOW_SPEC_RESOLVER: resolver},
    )
    assert calls == ["inner-id"]
    inner = out["steps"][0]["embedded_workflow"]["steps"][0]
    assert inner["embedded_workflow"] == _echo_spec()
    assert "embedded_workflow_workspace_id" not in inner


def test_mutual_exclusion_embedded_and_reference() -> None:
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
                "type": USE_SUBWORKFLOW_BLOCK_TYPE,
                "name": "bad",
                "embedded_workflow": _echo_spec(),
                "embedded_workflow_workspace_id": "ws",
                "embedded_workflow_id": "id",
                "parameter_bindings": {"child_msg": "$inputs.p"},
            },
        ],
        "outputs": [],
    }
    with pytest.raises(WorkflowDefinitionError):
        normalize_use_subworkflow_references_in_definition(
            raw,
            {WORKFLOWS_CORE_SUBWORKFLOW_SPEC_RESOLVER: resolver},
        )


def test_contains_unresolved_reference_false_for_inline_only() -> None:
    raw = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": USE_SUBWORKFLOW_BLOCK_TYPE,
                "name": "n",
                "embedded_workflow": _echo_spec(),
                "parameter_bindings": {},
            },
        ],
        "outputs": [],
    }
    assert not workflow_definition_contains_unresolved_subworkflow_reference(raw)


def test_normalize_returns_same_object_when_no_reference() -> None:
    raw = {
        "version": "1.0",
        "inputs": [],
        "steps": [],
        "outputs": [],
    }
    out = normalize_use_subworkflow_references_in_definition(raw, {})
    assert out is raw
