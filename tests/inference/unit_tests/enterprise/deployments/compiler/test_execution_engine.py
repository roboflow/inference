import networkx as nx
import pytest

from inference.enterprise.deployments.complier.execution_engine import (
    construct_response,
    get_all_nodes_in_execution_path,
)
from inference.enterprise.deployments.constants import OUTPUT_NODE_KIND
from inference.enterprise.deployments.entities.outputs import (
    CoordinatesSystem,
    JsonField,
)
from inference.enterprise.deployments.errors import DeploymentCompilerRuntimeError


def test_get_all_nodes_in_execution_path() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    graph.add_edge("a", "c")
    graph.add_edge("c", "d")
    graph.add_edge("d", "e")

    # when
    result = get_all_nodes_in_execution_path(execution_graph=graph, source="c")

    # then
    assert result == {"c", "d", "e"}


def test_construct_response_when_field_needs_to_be_grabbed_from_nested_output_in_own_coordinates() -> (
    None
):
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$outputs.some",
        kind=OUTPUT_NODE_KIND,
        definition=JsonField(
            type="JsonField",
            name="some",
            selector="$steps.a.predictions",
            coordinates_system=CoordinatesSystem.OWN,
        ),
    )
    graph.add_node(
        "$outputs.other",
        kind=OUTPUT_NODE_KIND,
        definition=JsonField(
            type="JsonField",
            name="other",
            selector="$steps.b.predictions",
            coordinates_system=CoordinatesSystem.OWN,
        ),
    )
    outputs_lookup = {
        "$steps.a": [
            {
                "predictions_parent_coordinates": ["a", "b", "c"],
                "predictions": ["a", "b"],
            },
            {
                "predictions_parent_coordinates": ["d", "e", "f"],
                "predictions": ["d", "e"],
            },
        ],
        "$steps.b": [{"predictions": ["g", "h", "i"]}],
    }

    # when
    result = construct_response(execution_graph=graph, outputs_lookup=outputs_lookup)

    # then
    assert result == {"some": [["a", "b"], ["d", "e"]], "other": [["g", "h", "i"]]}


def test_construct_response_when_field_needs_to_be_grabbed_from_nested_output_in_parent_coordinates() -> (
    None
):
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$outputs.some",
        kind=OUTPUT_NODE_KIND,
        definition=JsonField(
            type="JsonField", name="some", selector="$steps.a.predictions"
        ),
    )
    graph.add_node(
        "$outputs.other",
        kind=OUTPUT_NODE_KIND,
        definition=JsonField(
            type="JsonField", name="other", selector="$steps.b.predictions"
        ),
    )
    outputs_lookup = {
        "$steps.a": [
            {
                "predictions_parent_coordinates": ["a", "b", "c"],
                "predictions": ["a", "b"],
            },
            {
                "predictions_parent_coordinates": ["d", "e", "f"],
                "predictions": ["d", "e"],
            },
        ],
        "$steps.b": [{"predictions": ["g", "h", "i"]}],
    }

    # when
    result = construct_response(execution_graph=graph, outputs_lookup=outputs_lookup)

    # then
    assert result == {
        "some": [["a", "b", "c"], ["d", "e", "f"]],
        "other": [["g", "h", "i"]],
    }


def test_construct_response_when_field_needs_to_be_grabbed_from_simple_output_in_own_coordinates() -> (
    None
):
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$outputs.some",
        kind=OUTPUT_NODE_KIND,
        definition=JsonField(
            type="JsonField",
            name="some",
            selector="$steps.a.predictions",
            coordinates_system=CoordinatesSystem.OWN,
        ),
    )
    graph.add_node(
        "$outputs.other",
        kind=OUTPUT_NODE_KIND,
        definition=JsonField(
            type="JsonField",
            name="other",
            selector="$steps.b.predictions",
            coordinates_system=CoordinatesSystem.OWN,
        ),
    )
    outputs_lookup = {
        "$steps.a": {
            "predictions_parent_coordinates": ["a", "b", "c"],
            "predictions": ["a", "b"],
        },
        "$steps.b": {"predictions": ["g", "h", "i"]},
    }

    # when
    result = construct_response(execution_graph=graph, outputs_lookup=outputs_lookup)

    # then
    assert result == {"some": ["a", "b"], "other": ["g", "h", "i"]}


def test_construct_response_when_field_needs_to_be_grabbed_from_simple_output_in_parent_coordinates() -> (
    None
):
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$outputs.some",
        kind=OUTPUT_NODE_KIND,
        definition=JsonField(
            type="JsonField",
            name="some",
            selector="$steps.a.predictions",
        ),
    )
    graph.add_node(
        "$outputs.other",
        kind=OUTPUT_NODE_KIND,
        definition=JsonField(
            type="JsonField",
            name="other",
            selector="$steps.b.predictions",
        ),
    )
    outputs_lookup = {
        "$steps.a": {
            "predictions_parent_coordinates": ["a", "b", "c"],
            "predictions": ["a", "b"],
        },
        "$steps.b": {"predictions": ["g", "h", "i"]},
    }

    # when
    result = construct_response(execution_graph=graph, outputs_lookup=outputs_lookup)

    # then
    assert result == {"some": ["a", "b", "c"], "other": ["g", "h", "i"]}


def test_construct_response_when_step_output_is_missing_due_to_conditional_execution() -> (
    None
):
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$outputs.some",
        kind=OUTPUT_NODE_KIND,
        definition=JsonField(
            type="JsonField",
            name="some",
            selector="$steps.a.predictions",
        ),
    )
    outputs_lookup = {}

    # when
    result = construct_response(execution_graph=graph, outputs_lookup=outputs_lookup)

    # then
    assert result == {"some": None}


def test_construct_response_when_expected_step_property_is_missing() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node(
        "$outputs.some",
        kind=OUTPUT_NODE_KIND,
        definition=JsonField(
            type="JsonField",
            name="some",
            selector="$steps.a.predictions",
        ),
    )
    outputs_lookup = {"$steps.a": {}}

    # when
    with pytest.raises(DeploymentCompilerRuntimeError):
        _ = construct_response(execution_graph=graph, outputs_lookup=outputs_lookup)
