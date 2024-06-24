from typing import Any, Dict, List, Optional

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.common.utils import (
    sv_detections_to_root_coordinates,
)
from inference.core.workflows.entities.base import CoordinatesSystem, JsonField
from inference.core.workflows.execution_engine.executor.execution_data_manager.dynamic_batches_manager import (
    DynamicBatchIndex,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.manager import (
    ExecutionDataManager,
)


def construct_workflow_output(
    workflow_outputs: List[JsonField],
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
    outputs_arrays = {
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
    major_batch_size = 0
    for name in batch_oriented_outputs:
        array = outputs_arrays[name]
        major_batch_size = max(len(array), major_batch_size)
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
            place_data_in_array(
                array=array,
                index=index,
                data=data_piece,
            )
    results = []
    for i in range(major_batch_size):
        single_result = {}
        for name, value in non_batch_outputs.items():
            single_result[name] = value
        for name, array in outputs_arrays.items():
            single_result[name] = array[i]
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
        raise ValueError("Error with indexing")
    elif len(index) == 1:
        array[index[0]] = data
    else:
        first_chunk, *remaining_index = index
        place_data_in_array(array=array[first_chunk], index=remaining_index, data=data)


def data_contains_sv_detections(data: Any) -> bool:
    if isinstance(data, sv.Detections):
        return True
    if not isinstance(data, dict):
        return False
    # we do not go recursively, as we need to check wildcards outputs only
    for value in data.values():
        if isinstance(value, sv.Detections):
            return True
    return False


def convert_sv_detections_coordinates(data: Any) -> Any:
    if isinstance(data, sv.Detections):
        return sv_detections_to_root_coordinates(detections=data)
    if not isinstance(data, dict):
        return data
    return {
        k: (
            v
            if not isinstance(v, sv.Detections)
            else sv_detections_to_root_coordinates(detections=v)
        )
        for k, v in data.items()
    }
