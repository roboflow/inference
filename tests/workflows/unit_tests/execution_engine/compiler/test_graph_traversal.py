import networkx as nx

from inference.core.workflows.execution_engine.v1.compiler.graph_traversal import (
    traverse_graph_ensuring_parents_are_reached_first,
)


def test_traverse_graph_ensuring_parents_are_reached_first_when_no_actual_nodes_in_the_graph() -> (
    None
):
    # given
    graph = nx.DiGraph()
    graph.add_node("super-node")

    # when
    result = traverse_graph_ensuring_parents_are_reached_first(
        graph=graph,
        start_node="super-node",
    )

    # then
    assert result == ["super-node"], (
        "Expected nothing apart from super-node to be returned "
        "when no actual nodes in graph"
    )


def test_traverse_graph_ensuring_parents_are_reached_first_when_multiple_nodes_in_the_graph() -> (
    None
):
    # given
    # Graph structure
    #
    #   input-a -> step-a1 -> step-a2 -> step-a3 -> output-1
    #   input-b \                                \- step-c -> output-2
    #   input-c --> step-b-----------------------/
    #
    graph = nx.DiGraph()
    graph.add_node("super-node")
    graph.add_node("input-a")
    graph.add_node("input-b")
    graph.add_node("input-c")
    graph.add_node("step-a1")
    graph.add_node("step-a2")
    graph.add_node("step-a3")
    graph.add_node("step-b")
    graph.add_node("step-c")
    graph.add_node("output-1")
    graph.add_node("output-2")
    graph.add_edge("super-node", "input-a")
    graph.add_edge("super-node", "input-b")
    graph.add_edge("super-node", "input-c")
    graph.add_edge("input-a", "step-a1")
    graph.add_edge("step-a1", "step-a2")
    graph.add_edge("step-a2", "step-a3")
    graph.add_edge("step-a3", "output-1")
    graph.add_edge("step-a3", "step-c")
    graph.add_edge("input-b", "step-b")
    graph.add_edge("input-c", "step-b")
    graph.add_edge("step-b", "step-c")
    graph.add_edge("step-c", "output-2")

    # when
    result = traverse_graph_ensuring_parents_are_reached_first(
        graph=graph,
        start_node="super-node",
    )

    # then
    assert len(result) == 11, "Total number of elements must equal to number of nodes"
    assert result[0] == "super-node", "Expected super-node to come first"
    assert (
        0 < result.index("input-a") < 4
    ), "input-a must come after super node, but before any step"
    assert (
        0 < result.index("input-b") < 4
    ), "input-b must come after super node, but before any step"
    assert (
        0 < result.index("input-c") < 4
    ), "input-c must come after super node, but before any step"
    assert 4 <= result.index("step-a1") < 6, "step-a1 must come after inputs"
    assert 4 <= result.index("step-b") < 6, "step-b must come after inputs"
    assert result.index("step-a2") == 6, "step a-2 must be 6th"
    assert result.index("step-a3") == 7, "step a-2 must be 7th"
    assert 7 < result.index("step-c") < 10, "step-c must be after steps-a* and step-b"
    assert 7 < result.index("output-1") < 10, "output-1 must be after steps-a*"
    assert result.index("output-2") == 10, "output-2 must be last"
