from typing import Any, Dict, List, Optional

import numpy as np
import supervision as sv
from networkx import DiGraph

from inference.core.workflows.core_steps.common.utils import (
    sv_detections_to_root_coordinates,
)
from inference.core.workflows.errors import ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.constants import (
    WORKFLOW_INPUT_BATCH_LINEAGE_ID,
)
from inference.core.workflows.execution_engine.entities.base import (
    CoordinatesSystem,
    JsonField,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import OutputNode
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    construct_output_selector,
    node_as,
)
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.dynamic_batches_manager import (
    DynamicBatchIndex,
)
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.manager import (
    ExecutionDataManager,
)


def construct_workflow_output(
    workflow_outputs: List[JsonField],
    execution_graph: DiGraph,
    execution_data_manager: ExecutionDataManager,
) -> List[Dict[str, Any]]:
    # Maybe we should make blocks to change coordinates systems:
    # https://github.com/roboflow/inference/issues/440
    output_name2indices = {}
    for output in workflow_outputs:
        output_name2indices[output.name] = execution_data_manager.get_selector_indices(
            selector=output.selector
        )
    batch_oriented_outputs = {
        output for output, indices in output_name2indices.items() if indices is not None
    }
    non_batch_outputs = {
        output.name: execution_data_manager.get_non_batch_data(selector=output.selector)
        for output in workflow_outputs
        if output.name not in batch_oriented_outputs
    }
    if not batch_oriented_outputs:
        return [non_batch_outputs]
    dimensionality_for_output_nodes = {
        output.name: node_as(
            execution_graph=execution_graph,
            node=construct_output_selector(name=output.name),
            expected_type=OutputNode,
        ).dimensionality
        for output in workflow_outputs
    }
    outputs_arrays: Dict[str, Optional[list]] = {
        name: create_array(indices=np.array(indices))
        for name, indices in output_name2indices.items()
        if name in batch_oriented_outputs
    }
    name2selector = {output.name: output.selector for output in workflow_outputs}
    outputs_requested_in_parent_coordinates = {
        output.name
        for output in workflow_outputs
        if output.coordinates_system is CoordinatesSystem.PARENT
    }
    major_batch_size = len(
        execution_data_manager.get_lineage_indices(
            lineage=[WORKFLOW_INPUT_BATCH_LINEAGE_ID]
        )
    )
    for name in batch_oriented_outputs:
        array = outputs_arrays[name]
        indices = output_name2indices[name]
        data = execution_data_manager.get_batch_data(
            selector=name2selector[name],
            indices=indices,
        )
        for index, data_piece in zip(indices, data):
            if (
                name in outputs_requested_in_parent_coordinates
                and data_contains_sv_detections(data=data_piece)
            ):
                data_piece = convert_sv_detections_coordinates(data=data_piece)
            try:
                place_data_in_array(
                    array=array,
                    index=index,
                    data=data_piece,
                )
            except (TypeError, IndexError):
                raise ExecutionEngineRuntimeError(
                    public_message=f"Could not produce output {name} die to mismatch in "
                    f"declared output dimensions versus actual ones."
                    f"This is most likely a bug. Contact Roboflow team through github issues "
                    f"(https://github.com/roboflow/inference/issues) providing full context of"
                    f"the problem - including workflow definition you use.",
                    context="workflow_execution | output_construction",
                )
    results = []
    for i in range(major_batch_size):
        single_result = {}
        for name, value in non_batch_outputs.items():
            single_result[name] = value
        for name, array in outputs_arrays.items():
            if array is None or len(array) <= i:
                level = dimensionality_for_output_nodes[name] - 1
                if level > 0:
                    element = create_empty_index_array(
                        level=level,
                        accumulator=[],
                    )
                else:
                    element = None
            else:
                element = array[i]
            single_result[name] = element
        results.append(single_result)
    return results


def create_array(indices: np.ndarray) -> Optional[list]:
    if indices.size == 0:
        return None
    result = []
    max_idx = indices[:, 0].max() + 1
    for idx in range(max_idx):
        idx_selector = indices[:, 0] == idx
        indices_subset = indices[idx_selector][:, 1:]
        inner_array = create_array(indices_subset)
        if (
            inner_array is None
            and sum(indices_subset.shape) > 0
            and indices_subset.shape[0] == 0
        ):
            inner_array = create_empty_index_array(
                level=indices.shape[-1] - 1,
                accumulator=[],
            )
        result.append(inner_array)
    return result


def create_empty_index_array(level: int, accumulator: list) -> list:
    if level <= 1:
        return accumulator
    return create_empty_index_array(level - 1, [accumulator])


def place_data_in_array(array: list, index: DynamicBatchIndex, data: Any) -> None:
    if len(index) == 0:
        raise ExecutionEngineRuntimeError(
            public_message=f"Reached end of index without possibility to place data in result array."
            f"This is most likely a bug. Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | output_construction",
        )
    elif len(index) == 1:
        array[index[0]] = data
    else:
        first_chunk, *remaining_index = index
        place_data_in_array(array=array[first_chunk], index=remaining_index, data=data)


def data_contains_sv_detections(data: Any) -> bool:
    if isinstance(data, sv.Detections):
        return True
    if isinstance(data, dict):
        result = set()
        for value in data.values():
            result.add(data_contains_sv_detections(data=value))
        return True in result
    if isinstance(data, list):
        result = set()
        for value in data:
            result.add(data_contains_sv_detections(data=value))
        return True in result
    return False


def convert_sv_detections_coordinates(data: Any) -> Any:
    if isinstance(data, sv.Detections):
        return sv_detections_to_root_coordinates(detections=data)
    if isinstance(data, dict):
        return {k: convert_sv_detections_coordinates(data=v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_sv_detections_coordinates(data=element) for element in data]
    return data
