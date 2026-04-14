import pytest

from inference.core.workflows.execution_engine.v1.inner_workflow import (
    InnerWorkflowCompositionCycleError,
    InnerWorkflowNestingDepthError,
    assert_composition_acyclic,
    build_composition_digraph,
    find_composition_cycles,
    max_nesting_depth_from_root,
    validate_inner_workflow_composition,
)


def test_build_composition_digraph_empty() -> None:
    g = build_composition_digraph([])
    assert g.number_of_nodes() == 0


def test_assert_composition_acyclic_accepts_dag() -> None:
    edges = [("A", "B"), ("A", "C"), ("B", "D")]
    g = build_composition_digraph(edges)
    assert_composition_acyclic(g)


def test_assert_composition_acyclic_rejects_cycle() -> None:
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    g = build_composition_digraph(edges)
    with pytest.raises(InnerWorkflowCompositionCycleError):
        assert_composition_acyclic(g)


def test_find_composition_cycles() -> None:
    cycles = find_composition_cycles([("A", "B"), ("B", "A")])
    assert len(cycles) == 1
    assert set(cycles[0]) == {"A", "B"}


def test_max_nesting_depth_from_root_leaf() -> None:
    g = build_composition_digraph([])
    assert max_nesting_depth_from_root(g, "root") == 0


def test_max_nesting_depth_from_root_single_child() -> None:
    g = build_composition_digraph([("root", "child")])
    assert max_nesting_depth_from_root(g, "root") == 1


def test_max_nesting_depth_from_root_chain() -> None:
    g = build_composition_digraph([("R", "A"), ("A", "B"), ("B", "C")])
    assert max_nesting_depth_from_root(g, "R") == 3


def test_max_nesting_depth_from_root_diamond() -> None:
    g = build_composition_digraph([("R", "A"), ("R", "B"), ("A", "C"), ("B", "C")])
    assert max_nesting_depth_from_root(g, "R") == 2


def test_validate_inner_workflow_composition_ok() -> None:
    validate_inner_workflow_composition(
        containment_edges=[("R", "A"), ("A", "B")],
        root_workflow_id="R",
        max_nesting_depth=2,
    )


def test_validate_inner_workflow_composition_depth_exceeded() -> None:
    with pytest.raises(InnerWorkflowNestingDepthError):
        validate_inner_workflow_composition(
            containment_edges=[("R", "A"), ("A", "B")],
            root_workflow_id="R",
            max_nesting_depth=1,
        )


def test_validate_inner_workflow_composition_cycle() -> None:
    with pytest.raises(InnerWorkflowCompositionCycleError):
        validate_inner_workflow_composition(
            containment_edges=[("R", "A"), ("A", "B"), ("B", "A")],
            root_workflow_id="R",
            max_nesting_depth=10,
        )
