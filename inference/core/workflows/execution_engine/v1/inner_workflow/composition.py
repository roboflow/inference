"""
Compile-time validation of workflow *composition* (which workflow references which).

This is separate from the per-workflow execution DAG: the step graph must remain acyclic,
while this module validates the *meta-graph* of nested workflow references (e.g. ``inner_workflow``).

See docs/workflows/inner_workflow_design.md.
"""

from typing import Collection, Hashable, Iterable, List, Tuple

import networkx as nx

from inference.core.workflows.execution_engine.v1.inner_workflow.errors import (
    InnerWorkflowCompositionCycleError,
    InnerWorkflowNestingDepthError,
)


def build_composition_digraph(
    containment_edges: Iterable[Tuple[Hashable, Hashable]],
) -> nx.DiGraph:
    """
    Build a directed graph where an edge (parent, child) means
    "parent's definition directly embeds or references child as an inner workflow".
    """
    graph = nx.DiGraph()
    for parent, child in containment_edges:
        graph.add_edge(parent, child)
    return graph


def assert_composition_acyclic(graph: nx.DiGraph) -> None:
    """Raise InnerWorkflowCompositionCycleError if the composition graph is not a DAG."""
    if graph.number_of_nodes() == 0:
        return
    if nx.is_directed_acyclic_graph(graph):
        return
    try:
        cycle_edges = nx.find_cycle(graph)
        cycle_nodes = [u for u, _ in cycle_edges] + [cycle_edges[-1][1]]
    except nx.NetworkXNoCycle:
        cycle_nodes = []
    raise InnerWorkflowCompositionCycleError(
        "Inner workflow composition graph contains a cycle. "
        f"Involved nodes (partial): {cycle_nodes!r}"
    )


def max_nesting_depth_from_root(graph: nx.DiGraph, root: Hashable) -> int:
    """
    Maximum number of containment edges on any path starting at ``root``.

    - If ``root`` has no outgoing containment edges, depth is ``0``.
    - If ``root`` references one child and that child references none, depth is ``1``.
    """
    if root not in graph or graph.out_degree(root) == 0:
        return 0

    memo: dict[Hashable, int] = {}

    def depth_from(node: Hashable) -> int:
        if node in memo:
            return memo[node]
        successors = list(graph.successors(node))
        if not successors:
            memo[node] = 0
            return 0
        best = max(1 + depth_from(s) for s in successors)
        memo[node] = best
        return best

    return depth_from(root)


def validate_inner_workflow_composition(
    *,
    containment_edges: Collection[Tuple[Hashable, Hashable]],
    root_workflow_id: Hashable,
    max_nesting_depth: int,
) -> None:
    """
    Validate that the composition graph is acyclic and within max depth from ``root``.

    Args:
        containment_edges: (parent_workflow_id, child_workflow_id) for each direct
            inner-workflow reference.
        root_workflow_id: Identity of the workflow being compiled (opaque string or tuple).
        max_nesting_depth: Maximum allowed value from :func:`max_nesting_depth_from_root`.
    """
    graph = build_composition_digraph(containment_edges)
    assert_composition_acyclic(graph)
    depth = max_nesting_depth_from_root(graph, root_workflow_id)
    if depth > max_nesting_depth:
        raise InnerWorkflowNestingDepthError(
            f"Inner workflow nesting depth from root {root_workflow_id!r} is {depth}, "
            f"which exceeds the limit of {max_nesting_depth}."
        )


def find_composition_cycles(
    containment_edges: Collection[Tuple[Hashable, Hashable]],
) -> List[List[Hashable]]:
    """
    Return a list of simple cycles in the composition graph (for diagnostics / tests).

    Each cycle is a list of node ids in order. Empty if the graph is acyclic.
    """
    graph = build_composition_digraph(containment_edges)
    return [list(c) for c in nx.simple_cycles(graph)]
