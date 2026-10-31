"""Regression coverage for bounded saved-workflow reference expansion."""

import copy
from unittest.mock import Mock

import pytest
from roboflow_workflows.execution_engine.v1.inner_workflow import reference_resolution
from roboflow_workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from roboflow_workflows.execution_engine.v1.inner_workflow.errors import (
    InnerWorkflowCompositionCycleError,
    InnerWorkflowNestingDepthError,
    InnerWorkflowTotalCountError,
)
from roboflow_workflows.execution_engine.v1.inner_workflow.reference_resolution import (
    WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER,
    normalize_inner_workflow_references_in_definition,
)


def _reference(workflow_id, *, workspace_id="ws", version_id=None):
    return {
        "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
        "name": "child",
        "workflow_workspace_id": workspace_id,
        "workflow_id": workflow_id,
        "workflow_version_id": version_id,
        "parameter_bindings": {},
    }


def _workflow(*steps):
    return {"version": "1.0", "inputs": [], "steps": list(steps), "outputs": []}


def _inline(child):
    return {
        "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
        "name": "inline",
        "workflow_definition": child,
        "parameter_bindings": {},
    }


def _normalize(definition, *, resolver):
    normalized = normalize_inner_workflow_references_in_definition(
        definition,
        {WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver},
    )
    return normalized


@pytest.mark.parametrize("cycle", [("A",), ("A", "B"), ("A", "B", "C")])
def test_normalize_rejects_reference_cycles_without_mutating_definitions(cycle):
    saved = {
        workflow_id: _workflow(_reference(cycle[(index + 1) % len(cycle)]))
        for index, workflow_id in enumerate(cycle)
    }
    original_saved = copy.deepcopy(saved)
    raw = saved[cycle[0]]
    resolver = Mock(side_effect=lambda ws, wf, version, params: saved[wf])

    with pytest.raises(InnerWorkflowCompositionCycleError, match="cycle") as error:
        _normalize(raw, resolver=resolver)

    assert resolver.call_count == len(cycle)
    assert all(workflow_id in str(error.value) for workflow_id in cycle)
    assert saved == original_saved


def test_normalize_keeps_ancestors_through_inline_wrappers():
    saved = _workflow(_inline(_workflow(_reference(" A ", workspace_id=" ws "))))
    resolver = Mock(return_value=saved)

    with pytest.raises(InnerWorkflowCompositionCycleError):
        _normalize(_workflow(_reference("A")), resolver=resolver)

    resolver.assert_called_once()


@pytest.mark.parametrize(
    "child",
    [
        _reference("A", workspace_id="other", version_id="v1"),
        _reference("A", version_id="v2"),
        _reference("A"),
    ],
)
def test_normalize_distinguishes_workspace_and_version(child):
    resolver = Mock(side_effect=[_workflow(child), _workflow()])

    result = _normalize(_workflow(_reference("A", version_id="v1")), resolver=resolver)

    assert resolver.call_count == 2
    assert (
        result["steps"][0]["workflow_definition"]["steps"][0]["workflow_definition"]
        == _workflow()
    )


def test_normalize_allows_diamond_reuse_and_copies_each_occurrence():
    saved = {
        "A": _workflow(_reference("C")),
        "B": _workflow(_reference("C")),
        "C": _workflow(),
    }
    resolver = Mock(side_effect=lambda ws, wf, version, params: saved[wf])

    result = _normalize(_workflow(_reference("A"), _reference("B")), resolver=resolver)

    assert [call.args[1] for call in resolver.call_args_list] == ["A", "C", "B"]
    left = result["steps"][0]["workflow_definition"]["steps"][0]["workflow_definition"]
    right = result["steps"][1]["workflow_definition"]["steps"][0]["workflow_definition"]
    assert left == right == saved["C"]
    assert left is not right and left is not saved["C"]


@pytest.mark.parametrize("inline_wrapper", [False, True])
def test_normalize_stops_at_depth_limit_before_fetching_child(
    monkeypatch, inline_wrapper
):
    monkeypatch.setattr(reference_resolution, "WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH", 2)
    raw = _workflow(_reference("0"))
    if inline_wrapper:
        raw = _workflow(_inline(raw))
    resolver = Mock(
        side_effect=lambda ws, wf, version, params: _workflow(
            _reference(str(int(wf) + 1))
        )
    )

    with pytest.raises(InnerWorkflowNestingDepthError, match="limit of 2"):
        _normalize(raw, resolver=resolver)

    assert resolver.call_count == (1 if inline_wrapper else 2)


def test_normalize_accepts_exact_depth_and_count_limits(monkeypatch):
    monkeypatch.setattr(reference_resolution, "WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH", 2)
    monkeypatch.setattr(reference_resolution, "WORKFLOWS_MAX_INNER_WORKFLOW_COUNT", 2)
    resolver = Mock(return_value=_workflow())

    result = _normalize(
        _workflow(_inline(_workflow(_reference("A")))), resolver=resolver
    )

    resolver.assert_called_once()
    assert (
        result["steps"][0]["workflow_definition"]["steps"][0]["workflow_definition"]
        == _workflow()
    )


@pytest.mark.parametrize("repeated", [False, True])
def test_normalize_counts_occurrences_before_fetching_over_limit(monkeypatch, repeated):
    monkeypatch.setattr(reference_resolution, "WORKFLOWS_MAX_INNER_WORKFLOW_COUNT", 2)
    raw = _workflow(*(_reference("A" if repeated else str(i)) for i in range(3)))
    resolver = Mock(return_value=_workflow())

    with pytest.raises(InnerWorkflowTotalCountError, match="limit of 2"):
        _normalize(raw, resolver=resolver)

    assert resolver.call_count == (1 if repeated else 2)


def test_normalize_counts_nested_and_inline_occurrences_across_siblings(monkeypatch):
    monkeypatch.setattr(reference_resolution, "WORKFLOWS_MAX_INNER_WORKFLOW_COUNT", 3)
    raw = _workflow(_inline(_workflow(_reference("A"))), _reference("B"))
    resolver = Mock(side_effect=[_workflow(), _workflow(_reference("C"))])

    with pytest.raises(InnerWorkflowTotalCountError, match="limit of 3"):
        _normalize(raw, resolver=resolver)

    assert [call.args[1] for call in resolver.call_args_list] == ["A", "B"]


@pytest.mark.parametrize("inline_dispatch", [False, True])
def test_normalize_keeps_dispatched_cycles_opaque(monkeypatch, inline_dispatch):
    monkeypatch.setattr(reference_resolution, "WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH", 1)
    monkeypatch.setattr(reference_resolution, "WORKFLOWS_MAX_INNER_WORKFLOW_COUNT", 1)
    dispatch = (
        _inline(_workflow(_reference("A"))) if inline_dispatch else _reference("A")
    )
    dispatch["execution_mode"] = "remote_dispatch"
    resolver = Mock(return_value=_workflow(dispatch))

    result = _normalize(_workflow(_reference("A")), resolver=resolver)

    resolver.assert_called_once()
    assert result["steps"][0]["workflow_definition"]["steps"] == [dispatch]


@pytest.mark.parametrize("self_reference", [False, True])
def test_compiler_rejects_cycles_before_loading_blocks(monkeypatch, self_reference):
    from roboflow_workflows.execution_engine.v1.compiler import core

    saved = {
        "A": _workflow(_reference("A" if self_reference else "B")),
        "B": _workflow(_reference("A")),
    }
    resolver = Mock(side_effect=lambda ws, wf, version, params: saved[wf])
    load_blocks = Mock(
        side_effect=AssertionError("cycle must fail before block loading")
    )
    monkeypatch.setattr(core, "load_workflow_blocks", load_blocks)

    with pytest.raises(InnerWorkflowCompositionCycleError):
        core.compile_workflow_graph(
            workflow_definition=saved["A"],
            init_parameters={WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver},
        )

    assert resolver.call_count == (1 if self_reference else 2)
    load_blocks.assert_not_called()
