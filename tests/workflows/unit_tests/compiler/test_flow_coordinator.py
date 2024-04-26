import networkx as nx

from inference.enterprise.workflows.complier.flow_coordinator import (
    ParallelStepExecutionCoordinator,
    SerialExecutionCoordinator,
)


def test_serial_flow_coordinator_when_there_is_simple_execution_path() -> None:
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
    coordinator = SerialExecutionCoordinator.init(execution_graph=graph)

    # then
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result == ["step_1"], "At first step_1 must be executed"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result == ["step_2"], "As second - step_2 should be executed"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result is None, "Execution path should end up to this point"


def test_serial_flow_coordinator_when_there_is_ensemble_of_models() -> None:
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
    coordinator = SerialExecutionCoordinator.init(execution_graph=graph)

    # then
    first_step = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    second_step = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    result = first_step + second_step
    assert (result == ["step_1", "step_2"]) or (
        result == ["step_2", "step_1"]
    ), "As first two steps - step_1 and step_2 must be taken in any order"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result == ["step_3"], "As third - step_3 should be executed"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result is None, "Execution path should end up to this point"


def test_serial_flow_coordinator_when_there_is_condition_step() -> None:
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
    coordinator = SerialExecutionCoordinator.init(execution_graph=graph)

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


def test_serial_flow_coordinator_when_there_are_two_parallel_execution_paths() -> None:
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
    coordinator = SerialExecutionCoordinator.init(execution_graph=graph)

    # then
    result = []
    result.extend(coordinator.get_steps_to_execute_next(steps_to_discard=set()))
    result.extend(coordinator.get_steps_to_execute_next(steps_to_discard=set()))
    result.extend(coordinator.get_steps_to_execute_next(steps_to_discard=set()))
    result.extend(coordinator.get_steps_to_execute_next(steps_to_discard=set()))

    # then
    assert result.index("step_1") < result.index(
        "step_2"
    ), "Step 2 must be executed after step 1"
    assert result.index("step_3") < result.index(
        "step_4"
    ), "Step 4 must be executed after step 3"
    result = coordinator.get_steps_to_execute_next(steps_to_discard=set())
    assert result is None, "Execution path should end up to this point"


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
