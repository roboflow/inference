from unittest.mock import MagicMock

import networkx as nx

from inference.core.workflows.execution_engine.v1.compiler.entities import (
    InputNode,
    NodeCategory,
    OutputNode,
    StepNode,
)
from inference.core.workflows.execution_engine.v1.executor.flow_coordinator import (
    ParallelStepExecutionCoordinator,
)


def test_parallel_flow_coordinator_when_there_is_simple_execution_path() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node("input_1", node_compilation_output=assembly_dummy_input("input_1"))
    graph.add_node("step_1", node_compilation_output=assembly_dummy_step("step_1"))
    graph.add_node("step_2", node_compilation_output=assembly_dummy_step("step_2"))
    graph.add_node(
        "output_1", node_compilation_output=assembly_dummy_output("output_1")
    )
    graph.add_edge("input_1", "step_1")
    graph.add_edge("step_1", "step_2")
    graph.add_edge("step_2", "output_1")

    # when
    coordinator = ParallelStepExecutionCoordinator.init(execution_graph=graph)

    # then
    result = coordinator.get_steps_to_execute_next()
    assert set(result) == {"step_1"}, "At first step_1 must be executed"
    result = coordinator.get_steps_to_execute_next()
    assert set(result) == {"step_2"}, "As second - step_2 should be executed"
    result = coordinator.get_steps_to_execute_next()
    assert result is None, "Execution path should end up to this point"


def test_parallel_flow_coordinator_when_there_is_ensemble_of_models() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node("input_1", node_compilation_output=assembly_dummy_input("input_1"))
    graph.add_node("step_1", node_compilation_output=assembly_dummy_step("step_1"))
    graph.add_node("step_2", node_compilation_output=assembly_dummy_step("step_2"))
    graph.add_node("step_3", node_compilation_output=assembly_dummy_step("step_3"))
    graph.add_node(
        "output_1", node_compilation_output=assembly_dummy_output("output_1")
    )
    graph.add_edge("input_1", "step_1")
    graph.add_edge("input_1", "step_2")
    graph.add_edge("step_1", "step_3")
    graph.add_edge("step_2", "step_3")
    graph.add_edge("step_3", "output_1")

    # when
    coordinator = ParallelStepExecutionCoordinator.init(execution_graph=graph)

    # then
    result = coordinator.get_steps_to_execute_next()
    assert set(result) == {
        "step_1",
        "step_2",
    }, "As first two steps - step_1 and step_2 must be taken in any order"
    result = coordinator.get_steps_to_execute_next()
    assert set(result) == {"step_3"}, "As third - step_3 should be executed"
    result = coordinator.get_steps_to_execute_next()
    assert result is None, "Execution path should end up to this point"


def test_parallel_flow_coordinator_when_there_is_condition_step() -> None:
    # given
    graph = nx.DiGraph()
    graph.add_node("input_1", node_compilation_output=assembly_dummy_input("input_1"))
    graph.add_node("step_1", node_compilation_output=assembly_dummy_step("step_1"))
    graph.add_node("step_2", node_compilation_output=assembly_dummy_step("step_2"))
    graph.add_node("step_3", node_compilation_output=assembly_dummy_step("step_3"))
    graph.add_node("step_4", node_compilation_output=assembly_dummy_step("step_4"))
    graph.add_node("step_5", node_compilation_output=assembly_dummy_step("step_5"))
    graph.add_node(
        "output_1", node_compilation_output=assembly_dummy_output("output_1")
    )
    graph.add_node(
        "output_2", node_compilation_output=assembly_dummy_output("output_2")
    )
    graph.add_edge("input_1", "step_1")
    graph.add_edge("step_1", "step_2")
    graph.add_edge("step_2", "step_5")
    graph.add_edge("step_1", "step_3")
    graph.add_edge("step_3", "step_4")
    graph.add_edge("step_5", "output_1")
    graph.add_edge("step_4", "output_2")

    # when
    coordinator = ParallelStepExecutionCoordinator.init(execution_graph=graph)

    # then
    result = coordinator.get_steps_to_execute_next()
    assert set(result) == {"step_1"}, "Step_1 should be executed at first"
    result = coordinator.get_steps_to_execute_next()
    assert set(result) == {
        "step_2",
        "step_3",
    }, "As third - step_2, step_3 should be suggested to be executed, as this is the branching start"
    result = coordinator.get_steps_to_execute_next()
    assert set(result) == {
        "step_4",
        "step_5",
    }, "step_4 is the only successor of step_3 and step_5 is the successor of step_2 - both should be suggested"
    result = coordinator.get_steps_to_execute_next()
    assert result is None, "Execution path should end up to this point"


def test_parallel_flow_coordinator_when_there_are_two_parallel_execution_paths() -> (
    None
):
    # given
    graph = nx.DiGraph()
    graph.add_node("input_1", node_compilation_output=assembly_dummy_input("input_1"))
    graph.add_node("step_1", node_compilation_output=assembly_dummy_step("step_1"))
    graph.add_node("step_2", node_compilation_output=assembly_dummy_step("step_2"))
    graph.add_node("step_3", node_compilation_output=assembly_dummy_step("step_3"))
    graph.add_node("step_4", node_compilation_output=assembly_dummy_step("step_4"))
    graph.add_node(
        "output_1", node_compilation_output=assembly_dummy_output("output_1")
    )
    graph.add_node(
        "output_2", node_compilation_output=assembly_dummy_output("output_2")
    )
    graph.add_edge("input_1", "step_1")
    graph.add_edge("step_1", "step_2")
    graph.add_edge("step_2", "output_1")
    graph.add_edge("input_1", "step_3")
    graph.add_edge("step_3", "step_4")
    graph.add_edge("step_4", "output_2")

    # when
    coordinator = ParallelStepExecutionCoordinator.init(execution_graph=graph)

    # then
    result = coordinator.get_steps_to_execute_next()
    assert set(result) == {
        "step_1",
        "step_3",
    }, "As first, two steps - step_1 and step_3 must be taken in any order"
    result = coordinator.get_steps_to_execute_next()
    assert set(result) == {
        "step_2",
        "step_4",
    }, "As second, two steps - step_2 and step_4 must be taken in any order"
    result = coordinator.get_steps_to_execute_next()
    assert result is None, "Execution path should end up to this point"


def assembly_dummy_input(name: str) -> InputNode:
    return InputNode(
        node_category=NodeCategory.INPUT_NODE,
        name=name,
        selector=f"$inputs.{name}",
        data_lineage=[],
        input_manifest=MagicMock(),
    )


def assembly_dummy_step(name: str) -> StepNode:
    return StepNode(
        node_category=NodeCategory.STEP_NODE,
        name=name,
        selector=f"$steps.{name}",
        data_lineage=[],
        step_manifest=MagicMock(),
    )


def assembly_dummy_output(name: str) -> OutputNode:
    return OutputNode(
        node_category=NodeCategory.OUTPUT_NODE,
        name=name,
        selector=f"$outputs.{name}",
        data_lineage=[],
        output_manifest=MagicMock(),
    )
