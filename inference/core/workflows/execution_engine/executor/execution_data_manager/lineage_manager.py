from typing import Any, Dict, List, Optional, Tuple, Union

from networkx import DiGraph

from inference.core.workflows.constants import NODE_COMPILATION_OUTPUT_PROPERTY
from inference.core.workflows.execution_engine.compiler.entities import (
    ExecutionGraphNode,
    InputNode,
    NodeCategory,
)


class DataLineageManager:

    @classmethod
    def init(
        cls,
        execution_graph: DiGraph,
        runtime_parameters: Dict[str, Any],
    ) -> "DataLineageManager":
        lineage_chunk2dimension = assembly_lineage_chunk2dimension(
            execution_graph=execution_graph,
            runtime_parameters=runtime_parameters,
        )
        return cls(lineage_chunk2dimension=lineage_chunk2dimension)

    def __init__(
        self,
        lineage_chunk2dimension: Dict[str, Optional[Union[list, int]]],
    ):
        self._lineage_chunk2dimension = lineage_chunk2dimension

    def generate_indices_for_data_lineage(self, lineage: List[str]):
        if len(lineage) == 0:
            raise ValueError("Lineage should declare at least one chunk")
        lineage_indices = self._lineage_chunk2dimension.get(lineage[-1])
        if not lineage_indices:
            raise ValueError("Data lineage not defined")
        return generate_dynamic_batch_indices(dimensions=lineage_indices)


def assembly_lineage_chunk2dimension(
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
) -> Dict[str, Optional[int]]:
    result = {}
    for node in execution_graph.nodes:
        node_data: ExecutionGraphNode = execution_graph.nodes[node][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        if not node_data.data_lineage:
            continue
        dimension_value = None
        if node_data.node_category is NodeCategory.INPUT_NODE:
            input_node_data: InputNode = node_data  # type: ignore
            input_parameter_name = input_node_data.input_manifest.name
            dimension_value = len(runtime_parameters[input_parameter_name])
        result[node_data.data_lineage[-1]] = dimension_value
    return result


def generate_dynamic_batch_indices(
    dimensions: Union[int, list]
) -> List[Tuple[int, ...]]:
    if isinstance(dimensions, int):
        return [(e,) for e in range(dimensions)]
    result = []
    for idx, nested_element in enumerate(dimensions):
        nested_element_index = generate_dynamic_batch_indices(nested_element)
        nested_element_index = [(idx,) + e for e in nested_element_index]
        result.extend(nested_element_index)
    return result
