from typing import Any, Dict, List, Optional

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.common.utils import (
    sv_detections_to_root_coordinates,
)
from inference.core.workflows.entities.base import CoordinatesSystem, JsonField
from inference.core.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
    get_step_selector_from_its_output,
    is_input_selector,
    is_step_output_selector,
)
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
    for name in batch_oriented_outputs:
        array = outputs_arrays[name]
        indices = output_name2indices[name]
        data = execution_data_manager.get_batch_data(
            selector=name2selector[name],
            indices=indices,
        )
        print(f"retrieved data: {data}")
        for index, data_piece in zip(indices, data):
            place_data_in_array(
                array=array,
                index=index,
                data=data_piece,
            )
    print(outputs_arrays)
    # all_indices_lengths = [
    #     len(index) for indices in output_name2indices.values() for index in indices
    # ]
    # max_dimensionality = 0
    # if all_indices_lengths:
    #     max_dimensionality = max(all_indices_lengths)
    # if max_dimensionality == 0:
    #     return [
    #         {
    #             output.name: (
    #                 runtime_parameters[get_last_chunk_of_selector(output.selector)]
    #                 if is_input_selector(selector_or_value=output.selector)
    #                 else (
    #                     execution_cache.get_non_batch_output(selector=output.selector)
    #                     if execution_cache.is_value_registered(selector=output.selector)
    #                     else None
    #                 )
    #             )
    #             for output in workflow_outputs
    #         }
    #     ]
    # major_batch_size = (
    #     max(
    #         [
    #             index[0]
    #             for indices in output_name2indices.values()
    #             for index in indices
    #             if index
    #         ]
    #     )
    #     + 1
    # )
    # results = []
    # print("output_name2indices", output_name2indices)
    # for i in range(major_batch_size):
    #     single_result = {}
    #     for output in workflow_outputs:
    #         output_indices = output_name2indices[output.name]
    #         print("name", output.name, "indices", output_indices)
    #         output_dimension = 0
    #         if output_indices:
    #             output_dimension = len(output_indices[0])
    #         if output_dimension == 0:
    #             if is_input_selector(output.selector):
    #                 value = runtime_parameters[
    #                     get_last_chunk_of_selector(output.selector)
    #                 ]
    #             else:
    #                 if execution_cache.is_value_registered(selector=output.selector):
    #                     value = execution_cache.get_non_batch_output(
    #                         selector=output.selector
    #                     )
    #                 else:
    #                     value = None
    #             single_result[output.name] = value
    #             continue
    #         if is_input_selector(output.selector):
    #             single_result[output.name] = runtime_parameters[
    #                 get_last_chunk_of_selector(output.selector)
    #             ][i]
    #             continue
    #         major_element_indices = [idx for idx in output_indices if idx[0] == i]
    #         if execution_cache.is_value_registered(output.selector):
    #             value = execution_cache.get_batch_output(
    #                 selector=output.selector,
    #                 batch_elements_indices=major_element_indices,
    #             )
    #             single_result[output.name] = create_array(
    #                 indices=np.array([e[1:] for e in major_element_indices])
    #             )
    #             for idx, v in zip(major_element_indices, value):
    #                 if len(idx) == 1:
    #                     single_result[output.name] = v
    #                     continue
    #                 idx = idx[1:]
    #                 tmp = single_result[output.name]
    #                 for p in range(len(idx) - 1):
    #                     tmp = tmp[p]
    #                 if (
    #                     isinstance(v, sv.Detections)
    #                     and output.coordinates_system is CoordinatesSystem.PARENT
    #                 ):
    #                     v = sv_detections_to_root_coordinates(detections=v)
    #                 tmp[idx[-1]] = v
    #         else:
    #             single_result[output.name] = None
    #     results.append(single_result)
    # return results


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
            inner_array = []
        result.append(inner_array)
    return result


def place_data_in_array(array: list, index: DynamicBatchIndex, data: Any) -> None:
    if len(index) == 0:
        raise ValueError("Error with indexing")
    if len(index) == 1:
        array[index[0]] = data
    if len(index) >= 2:
        first_chunk, remaining_index = index
        place_data_in_array(array=array[first_chunk], index=remaining_index, data=data)
