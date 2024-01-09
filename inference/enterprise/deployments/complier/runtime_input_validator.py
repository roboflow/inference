from typing import Any, Dict, Set

from networkx import DiGraph

from inference.enterprise.deployments.complier.utils import get_nodes_of_specific_kind
from inference.enterprise.deployments.constants import INPUT_NODE_KIND
from inference.enterprise.deployments.errors import RuntimeParameterMissingError


def validate_runtime_input(
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
) -> None:
    parameters_without_default_values = get_input_parameters_without_default_values(
        execution_graph=execution_graph,
    )
    missing_parameters = []
    for name in parameters_without_default_values:
        if name not in runtime_parameters:
            missing_parameters.append(name)
    if len(missing_parameters) > 0:
        raise RuntimeParameterMissingError(
            f"Parameters passed to execution runtime do not define required inputs: {missing_parameters}"
        )


def get_input_parameters_without_default_values(execution_graph: DiGraph) -> Set[str]:
    input_nodes = get_nodes_of_specific_kind(
        execution_graph=execution_graph,
        kind=INPUT_NODE_KIND,
    )
    result = set()
    for input_node in input_nodes:
        definition = execution_graph.nodes[input_node]["definition"]
        if definition.type == "InferenceImage":
            result.add(definition.name)
            continue
        if definition.type == "InferenceParameter" and definition.default_value is None:
            result.add(definition.name)
            continue
    return result
