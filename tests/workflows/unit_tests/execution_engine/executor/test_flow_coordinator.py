import networkx as nx
import pytest

from inference.enterprise.workflows.constants import STEP_NODE_KIND
from inference.enterprise.workflows.entities.types import FlowControl
from inference.enterprise.workflows.errors import InvalidBlockBehaviourError
from inference.enterprise.workflows.execution_engine.executor.flow_coordinator import (
    ParallelStepExecutionCoordinator,
    get_all_nodes_in_execution_path,
    handle_execution_branch_selection,
    handle_flow_control,
)


def test_get_all_nodes_in_execution_path_with_source_included() -> None:
    # given
    # 0 -> a ->  b -> d
    #       \-> d -> e
    execution_graph = nx.DiGraph()
    execution_graph.add_edge("$steps.0", "$steps.a")
    execution_graph.add_edge("$steps.a", "$steps.b")
    execution_graph.add_edge("$steps.a", "$steps.c")
    execution_graph.add_edge("$steps.b", "$steps.d")
    execution_graph.add_edge("$steps.c", "$steps.e")

    # when
    result = get_all_nodes_in_execution_path(
        execution_graph=execution_graph,
        source="$steps.a",
        include_source=True,
    )

    # then
    assert result == {"$steps.a", "$steps.b", "$steps.c", "$steps.d", "$steps.e"}


def test_get_all_nodes_in_execution_path_with_source_not_included() -> None:
    # given
    # 0 -> a ->  b -> d
    #       \-> d -> e
    execution_graph = nx.DiGraph()
    execution_graph.add_edge("$steps.0", "$steps.a")
    execution_graph.add_edge("$steps.a", "$steps.b")
    execution_graph.add_edge("$steps.a", "$steps.c")
    execution_graph.add_edge("$steps.b", "$steps.d")
    execution_graph.add_edge("$steps.c", "$steps.e")

    # when
    result = get_all_nodes_in_execution_path(
        execution_graph=execution_graph,
        source="$steps.a",
        include_source=False,
    )

    # then
    assert result == {"$steps.b", "$steps.c", "$steps.d", "$steps.e"}


def test_handle_execution_branch_selection_when_selected_node_is_valid() -> None:
    # given
    #  /->  b1 -> b2
    # a ->  c1 -> c2
    #  \->  d1 -> d2
    execution_graph = nx.DiGraph()
    execution_graph.add_edge("$steps.a", "$steps.b1")
    execution_graph.add_edge("$steps.a", "$steps.c1")
    execution_graph.add_edge("$steps.a", "$steps.d1")
    execution_graph.add_edge("$steps.b1", "$steps.b2")
    execution_graph.add_edge("$steps.c1", "$steps.c2")
    execution_graph.add_edge("$steps.d1", "$steps.d2")
    for node in ["$steps.a", "$steps.b1", "$steps.c1", "$steps.d1"]:
        execution_graph.nodes[node]["kind"] = STEP_NODE_KIND

    # when
    nodes_to_discard = handle_execution_branch_selection(
        current_step="$steps.a",
        execution_graph=execution_graph,
        selected_next_step="$steps.b1",
    )

    # then
    assert nodes_to_discard == {
        "$steps.c1",
        "$steps.c2",
        "$steps.d1",
        "$steps.d2",
    }


def test_handle_execution_branch_selection_when_selected_node_is_invalid() -> None:
    # given
    #  /->  b1 -> b2
    # a ->  c1 -> c2
    execution_graph = nx.DiGraph()
    execution_graph.add_edge("$steps.a", "$steps.b1")
    execution_graph.add_edge("$steps.a", "$steps.c1")
    execution_graph.add_edge("$steps.b1", "$steps.b2")
    execution_graph.add_edge("$steps.c1", "$steps.c2")
    for node in ["$steps.a", "$steps.b1", "$steps.c1"]:
        execution_graph.nodes[node]["kind"] = STEP_NODE_KIND

    # when
    with pytest.raises(InvalidBlockBehaviourError):
        _ = handle_execution_branch_selection(
            current_step="$steps.a",
            execution_graph=execution_graph,
            selected_next_step=None,
        )


def test_handle_flow_control_when_branch_is_to_be_terminated() -> None:
    # given
    #  /->  b1 -> b2
    # a ->  c1 -> c2
    execution_graph = nx.DiGraph()
    execution_graph.add_edge("$steps.a", "$steps.b1")
    execution_graph.add_edge("$steps.a", "$steps.c1")
    execution_graph.add_edge("$steps.b1", "$steps.b2")
    execution_graph.add_edge("$steps.c1", "$steps.c2")
    for node in ["$steps.a", "$steps.b1", "$steps.c1"]:
        execution_graph.nodes[node]["kind"] = STEP_NODE_KIND

    # when
    result = handle_flow_control(
        current_step_selector="$steps.a",
        flow_control=FlowControl(mode="terminate_branch"),
        execution_graph=execution_graph,
    )

    # then
    assert result == {
        "$steps.b1",
        "$steps.b2",
        "$steps.c1",
        "$steps.c2",
    }


def test_handle_flow_control_when_branch_is_to_be_selected() -> None:
    # given
    #  /->  b1 -> b2
    # a ->  c1 -> c2
    execution_graph = nx.DiGraph()
    execution_graph.add_edge("$steps.a", "$steps.b1")
    execution_graph.add_edge("$steps.a", "$steps.c1")
    execution_graph.add_edge("$steps.b1", "$steps.b2")
    execution_graph.add_edge("$steps.c1", "$steps.c2")
    for node in ["$steps.a", "$steps.b1", "$steps.c1"]:
        execution_graph.nodes[node]["kind"] = STEP_NODE_KIND

    # when
    result = handle_flow_control(
        current_step_selector="$steps.a",
        flow_control=FlowControl(mode="select_step", context="$steps.c1"),
        execution_graph=execution_graph,
    )

    # then
    assert result == {
        "$steps.b1",
        "$steps.b2",
    }


def test_parallel_flow_coordinator_when_there_is_simple_execution_path() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node("input_1", kind="INPUT_NODE")
    graph.add_node("step_1", kind="STEP_NODE")
    graph.add_node("step_2", kind="STEP_NODE")
    graph.add_node("output_1", kind="OUTPUT_NODE")
    graph.add_edge("input_1", "step_1")
    graph.add_edge("step_1", "step_2")
    graph.add_edge("step_2", "output_1")

    # when
    coordinator = ParallelStepExecutionCoordinator.init(execution_graph=graph)

    # then
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result == ["step_1"], "At first step_1 must be executed"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result == ["step_2"], "As second - step_2 should be executed"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result is None, "Execution path should end up to this point"


def test_parallel_flow_coordinator_when_there_is_ensemble_of_models() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node("input_1", kind="INPUT_NODE")
    graph.add_node("step_1", kind="STEP_NODE")
    graph.add_node("step_2", kind="STEP_NODE")
    graph.add_node("step_3", kind="STEP_NODE")
    graph.add_node("output_1", kind="OUTPUT_NODE")
    graph.add_edge("input_1", "step_1")
    graph.add_edge("input_1", "step_2")
    graph.add_edge("step_1", "step_3")
    graph.add_edge("step_2", "step_3")
    graph.add_edge("step_3", "output_1")

    # when
    coordinator = ParallelStepExecutionCoordinator.init(execution_graph=graph)

    # then
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert (result == ["step_1", "step_2"]) or (
        result == ["step_2", "step_1"]
    ), "As first two steps - step_1 and step_2 must be taken in any order"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result == ["step_3"], "As third - step_3 should be executed"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result is None, "Execution path should end up to this point"


def test_parallel_flow_coordinator_when_there_is_condition_step() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node("input_1", kind="INPUT_NODE")
    graph.add_node("step_1", kind="STEP_NODE")
    graph.add_node("step_2", kind="STEP_NODE")
    graph.add_node("step_3", kind="STEP_NODE")
    graph.add_node("step_4", kind="STEP_NODE")
    graph.add_node("output_1", kind="OUTPUT_NODE")
    graph.add_node("output_2", kind="OUTPUT_NODE")
    graph.add_edge("input_1", "step_1")
    graph.add_edge("step_1", "step_2")
    graph.add_edge("step_1", "step_3")
    graph.add_edge("step_3", "step_4")
    graph.add_edge("step_2", "output_1")
    graph.add_edge("step_4", "output_2")

    # when
    coordinator = ParallelStepExecutionCoordinator.init(execution_graph=graph)

    # then
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result == ["step_1"], "Step_1 should be executed at first"
    result = coordinator.get_steps_to_execute_next(steps_to_discard={"step_2"})
    assert result == [
        "step_3"
    ], "As third - step_3 should be executed, as step_2 branch is discarded"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result == [
        "step_4"
    ], "step_4 is the only successor of step_3 - hence should be pointed at this step"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result is None, "Execution path should end up to this point"


def test_parallel_flow_coordinator_when_there_are_two_parallel_execution_paths() -> (
    None
):
    # given
    graph = nx.DiGraph()
    graph.add_node("input_1", kind="INPUT_NODE")
    graph.add_node("step_1", kind="STEP_NODE")
    graph.add_node("step_2", kind="STEP_NODE")
    graph.add_node("step_3", kind="STEP_NODE")
    graph.add_node("step_4", kind="STEP_NODE")
    graph.add_node("output_1", kind="OUTPUT_NODE")
    graph.add_node("output_2", kind="OUTPUT_NODE")
    graph.add_edge("input_1", "step_1")
    graph.add_edge("step_1", "step_2")
    graph.add_edge("step_2", "output_1")
    graph.add_edge("input_1", "step_3")
    graph.add_edge("step_3", "step_4")
    graph.add_edge("step_4", "output_2")

    # when
    coordinator = ParallelStepExecutionCoordinator.init(execution_graph=graph)

    # then
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert (result == ["step_1", "step_3"]) or (
        result == ["step_3", "step_1"]
    ), "As first, two steps - step_1 and step_3 must be taken in any order"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert (result == ["step_2", "step_4"]) or (
        result == ["step_4", "step_2"]
    ), "As second, two steps - step_2 and step_4 must be taken in any order"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result is None, "Execution path should end up to this point"
