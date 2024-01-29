import abc
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from queue import Queue
from typing import Dict, List, Optional, Set

import networkx as nx

from inference.enterprise.deployments.complier.utils import get_nodes_of_specific_kind
from inference.enterprise.deployments.constants import (
    INPUT_NODE_KIND,
    OUTPUT_NODE_KIND,
    STEP_NODE_KIND,
)


class StepExecutionCoordinator(metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractmethod
    def init(cls, execution_graph: nx.DiGraph) -> "StepExecutionCoordinator":
        pass

    @abc.abstractmethod
    def get_steps_to_execute_next(
        self, steps_to_discard: Set[str]
    ) -> Optional[List[str]]:
        pass


class SerialExecutionCoordinator(StepExecutionCoordinator):

    @classmethod
    def init(cls, execution_graph: nx.DiGraph) -> "StepExecutionCoordinator":
        return cls(execution_graph=execution_graph)

    def __init__(self, execution_graph: nx.DiGraph):
        self._execution_graph = execution_graph.copy()
        self._discarded_steps: Set[str] = set()
        self.__order: Optional[List[str]] = None
        self.__step_pointer = 0

    def get_steps_to_execute_next(
        self, steps_to_discard: Set[str]
    ) -> Optional[List[str]]:
        if self.__order is None:
            self.__establish_execution_order()
        self._discarded_steps.update(steps_to_discard)
        next_step = None
        while self.__step_pointer < len(self.__order):
            candidate_step = self.__order[self.__step_pointer]
            self.__step_pointer += 1
            if candidate_step in self._discarded_steps:
                continue
            return [candidate_step]
        return next_step

    def __establish_execution_order(self) -> None:
        step_nodes = get_nodes_of_specific_kind(
            execution_graph=self._execution_graph, kind=STEP_NODE_KIND
        )
        self.__order = [
            n for n in nx.topological_sort(self._execution_graph) if n in step_nodes
        ]
        self.__step_pointer = 0


class ParallelStepExecutionCoordinator(StepExecutionCoordinator):

    @classmethod
    def init(cls, execution_graph: nx.DiGraph) -> "StepExecutionCoordinator":
        return cls(execution_graph=execution_graph)

    def __init__(self, execution_graph: nx.DiGraph):
        self._execution_graph = execution_graph.copy()
        self._discarded_steps: Set[str] = set()
        self.__execution_groups: Optional[List[List[ExecutionGroup]]] = None
        self.__group_pointers: Dict[str, int] = {}
        self.__completed_groups: Set[str] = set()
        self.__completed_steps: Set[str] = set()

    def get_steps_to_execute_next(
        self, steps_to_discard: Set[str]
    ) -> Optional[List[str]]:
        if self.__execution_groups is None:
            self.__execution_groups = establish_execution_groups(
                execution_graph=self._execution_graph
            )
        self._discarded_steps.update(steps_to_discard)
        next_steps = get_next_steps_to_execute(
            execution_groups=self.__execution_groups,
            completed_steps=self.__completed_steps,
            discarded_steps=self._discarded_steps,
        )
        if len(next_steps) == 0:
            return None
        self.__completed_steps.update(next_steps)
        return next_steps


@dataclass(frozen=True)
class ExecutionGroup:
    group_id: str
    distance: int
    steps: List[str]
    depends_on: Set[str]


def establish_execution_groups(
    execution_graph: nx.DiGraph,
) -> List[List[ExecutionGroup]]:
    step_chains = construct_step_chains(execution_graph=execution_graph)
    super_graph = construct_super_graph(
        execution_graph=execution_graph, step_groups=step_chains
    )
    super_graph = assign_max_distances_from_start(super_graph=super_graph)
    return get_groups_execution_order(super_graph=super_graph)


def construct_step_chains(
    execution_graph: nx.DiGraph,
) -> List[List[str]]:
    step_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph, kind=STEP_NODE_KIND
    )
    steps_topological_order = [
        n for n in nx.topological_sort(execution_graph) if n in step_nodes
    ]
    groups = []
    steps_already_considered = set()
    while len(steps_already_considered) < len(steps_topological_order):
        for step in steps_topological_order:
            if step in steps_already_considered:
                continue
            step_predecessors = list(execution_graph.predecessors(step))
            step_successors = list(execution_graph.successors(step))
            if len(step_predecessors) > 1 or len(step_successors) > 1:
                groups.append([step])
                steps_already_considered.add(step)
                continue
            steps_chain = go_down_the_simple_steps_chain(
                execution_graph=execution_graph, step=step
            )
            groups.append(steps_chain)
            steps_already_considered.update(steps_chain)
    return groups


def go_down_the_simple_steps_chain(execution_graph: nx.DiGraph, step: str) -> List[str]:
    steps_chain = [step]
    successors = list(execution_graph.successors(step))
    if len(successors) == 0:
        return steps_chain
    step = successors[0]
    while True:
        if execution_graph.nodes[step]["kind"] != STEP_NODE_KIND:
            return steps_chain
        step_predecessors = list(execution_graph.predecessors(step))
        step_successors = list(execution_graph.successors(step))
        if len(step_predecessors) > 1 or len(step_successors) > 1:
            return steps_chain
        steps_chain.append(step)
        if len(step_successors) == 0:
            break
        step = step_successors[0]
    return steps_chain


def construct_super_graph(
    execution_graph: nx.DiGraph,
    step_groups: List[List[str]],
) -> nx.DiGraph:
    super_graph = nx.DiGraph()
    super_graph.add_node("start")
    super_graph.add_node("end")
    execution_graph_node2group_id = {
        group_element: str(group_id)
        for group_id, group in enumerate(step_groups)
        for group_element in group
    }
    for group_id, group_content in enumerate(step_groups):
        super_graph.add_node(str(group_id), steps=group_content)
    for group in step_groups:
        group_start = group[0]
        group_end = group[-1]
        in_edges_start = {
            execution_graph_node2group_id.get(p, "start")
            for p in execution_graph.predecessors(group_start)
        }
        out_edges_end = {
            execution_graph_node2group_id.get(p, "end")
            for p in execution_graph.successors(group_end)
        }
        for in_edge_start in in_edges_start:
            super_graph.add_edge(
                in_edge_start, execution_graph_node2group_id[group_start]
            )
        for out_edge_end in out_edges_end:
            super_graph.add_edge(execution_graph_node2group_id[group_end], out_edge_end)
    return super_graph


def assign_max_distances_from_start(super_graph: nx.DiGraph) -> nx.DiGraph:
    nodes_to_consider = Queue()
    nodes_to_consider.put("start")
    while nodes_to_consider.qsize() > 0:
        node_to_consider = nodes_to_consider.get()
        predecessors = list(super_graph.predecessors(node_to_consider))
        if not all(
            super_graph.nodes[p].get("distance") is not None for p in predecessors
        ):
            # we can proceed to establish distance, only if all parents have distances established
            continue
        if len(predecessors) == 0:
            distance_from_start = 0
            depends_on = set()
        else:
            distance_from_start = (
                max(super_graph.nodes[p]["distance"] for p in predecessors) + 1
            )
            depends_on = set(
                chain.from_iterable(
                    super_graph.nodes[p]["depends_on"] for p in predecessors
                )
            ).union([p for p in predecessors if p != "start"])
        super_graph.nodes[node_to_consider]["distance"] = distance_from_start
        super_graph.nodes[node_to_consider]["depends_on"] = depends_on
        for neighbour in super_graph.successors(node_to_consider):
            nodes_to_consider.put(neighbour)
    return super_graph


def get_groups_execution_order(super_graph: nx.DiGraph) -> List[List[ExecutionGroup]]:
    distance2execution_groups = defaultdict(list)
    for node_name, node_data in super_graph.nodes(data=True):
        if node_name in {"start", "end"}:
            continue
        depends_on = {
            step
            for group in node_data["depends_on"]
            for step in super_graph.nodes[group]["steps"]
        }
        execution_group = ExecutionGroup(
            group_id=node_name,
            distance=node_data["distance"],
            steps=node_data["steps"],
            depends_on=depends_on,
        )
        distance2execution_groups[node_data["distance"]].append(execution_group)
    sorted_distances = sorted(list(distance2execution_groups.keys()))
    return [distance2execution_groups[d] for d in sorted_distances]


def get_next_steps_to_execute(
    execution_groups: List[List[ExecutionGroup]],
    completed_steps: Set[str],
    discarded_steps: Set[str],
) -> List[str]:
    result = []
    for distance_group in execution_groups:
        for element in distance_group:
            if not all(e in completed_steps for e in element.depends_on):
                continue
            for step in element.steps:
                if step in discarded_steps or step in completed_steps:
                    continue
                result.append(step)
                break
    return result
