# import networkx as nx
#
# from inference.enterprise.workflows.complier.execution_engine import (
#     construct_response,
#     get_all_nodes_in_execution_path,
# )
#
#
# def test_get_all_nodes_in_execution_path() -> None:
#     # given
#     graph = nx.DiGraph()
#     graph.add_edge("a", "b")
#     graph.add_edge("a", "c")
#     graph.add_edge("c", "d")
#     graph.add_edge("d", "e")
#
#     # when
#     result = get_all_nodes_in_execution_path(execution_graph=graph, source="c")
#
#     # then
#     assert result == {"c", "d", "e"}
