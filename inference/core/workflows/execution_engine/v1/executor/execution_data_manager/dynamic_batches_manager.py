from typing import Any, Dict, List

from networkx import DiGraph

from inference.core.workflows.errors import AssumptionError, ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    ExecutionGraphNode,
    InputNode,
    NodeCategory,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    identify_lineage,
    node_as,
)

DynamicBatchIndex = tuple


class DynamicBatchesManager:

    @classmethod
    def init(
        cls,
        execution_graph: DiGraph,
        runtime_parameters: Dict[str, Any],
    ) -> "DynamicBatchesManager":
        lineage2indices = assembly_root_batch_indices(
            execution_graph=execution_graph,
            runtime_parameters=runtime_parameters,
        )
        return cls(lineage2indices=lineage2indices)

    def __init__(
        self,
        lineage2indices: Dict[int, List[DynamicBatchIndex]],
    ):
        self._lineage2indices = lineage2indices

    def register_element_indices_for_lineage(
        self,
        lineage: List[str],
        indices: List[DynamicBatchIndex],
    ) -> None:
        lineage_id = identify_lineage(lineage=lineage)
        self._lineage2indices[lineage_id] = indices

    def get_indices_for_data_lineage(
        self, lineage: List[str]
    ) -> List[DynamicBatchIndex]:
        if not self.is_lineage_registered(lineage=lineage):
            raise ExecutionEngineRuntimeError(
                public_message=f"Lineage {lineage} not found. "
                f"This is most likely a bug. Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_input_assembling",
            )
        lineage_id = identify_lineage(lineage=lineage)
        return self._lineage2indices[lineage_id]

    def is_lineage_registered(self, lineage: List[str]) -> bool:
        lineage_id = identify_lineage(lineage=lineage)
        return lineage_id in self._lineage2indices


def assembly_root_batch_indices(
    execution_graph: DiGraph,
    runtime_parameters: Dict[str, Any],
) -> Dict[int, List[DynamicBatchIndex]]:
    result = {}
    for node in execution_graph.nodes:
        node_data = node_as(
            execution_graph=execution_graph,
            node=node,
            expected_type=ExecutionGraphNode,
        )
        if not node_data.data_lineage:
            continue
        if node_data.node_category is NodeCategory.INPUT_NODE:
            input_node_data = node_as(
                execution_graph=execution_graph,
                node=node,
                expected_type=InputNode,
            )
            input_parameter_name = input_node_data.input_manifest.name
            root_lineage_id = identify_lineage(lineage=node_data.data_lineage[:1])
            result[root_lineage_id] = [
                (i,) for i in range(len(runtime_parameters[input_parameter_name]))
            ]
            if input_node_data.input_manifest.dimensionality > 1:
                lineage_id = identify_lineage(lineage=node_data.data_lineage)
                result[lineage_id] = generate_indices_for_input_node(
                    dimensionality=input_node_data.input_manifest.dimensionality,
                    dimension_value=runtime_parameters[input_parameter_name],
                )
    return result


def generate_indices_for_input_node(
    dimensionality: int, dimension_value: list, indices_prefix: DynamicBatchIndex = ()
) -> List[DynamicBatchIndex]:
    if not isinstance(dimension_value, list):
        raise AssumptionError(
            public_message=f"Could not establish input data batch indices. This is most likely the bug. Contact "
            f"Roboflow team through github issues (https://github.com/roboflow/inference/issues) "
            f"providing full context of the problem - including workflow definition you use.",
            context="workflow_execution | step_input_assembling",
        )
    if dimensionality == len(indices_prefix) + 1:
        return [indices_prefix + (i,) for i in range(len(dimension_value))]
    result = []
    for i, value_element in enumerate(dimension_value):
        result.extend(
            generate_indices_for_input_node(
                dimensionality=dimensionality,
                dimension_value=value_element,
                indices_prefix=indices_prefix + (i,),
            )
        )
    return result
