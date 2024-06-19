from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, TypeVar, Union

from inference.core.workflows.entities.base import Batch
from inference.core.workflows.execution_engine.compiler.entities import (
    CompoundStepInputDefinition,
    DynamicStepInputDefinition,
    StaticStepInputDefinition,
    StepInputDefinition,
    StepNode,
)
from inference.core.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.branching_manager import (
    BranchingManager,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.dynamic_batches_manager import (
    DynamicBatchesManager,
    DynamicBatchIndex,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.execution_cache import (
    ExecutionCache,
)

T = TypeVar("T")


@dataclass(frozen=True)
class BatchModeSIMDStepInput:
    indices: List[DynamicBatchIndex]
    parameters: Dict[str, Any]


@dataclass(frozen=True)
class NonBatchModeSIMDStepInput:
    index: DynamicBatchIndex
    parameters: Dict[str, Any]


def construct_non_simd_step_input(
    step_node: StepNode,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    branching_manager: BranchingManager,
) -> Optional[Dict[str, Any]]:
    if step_node.batch_oriented_parameters:
        raise ValueError("SIMD input detected!")
    masks = set()
    for execution_branch in step_node.execution_branches_impacting_inputs:
        if not branching_manager.is_execution_branch_registered(
            execution_branch=execution_branch
        ):
            masks.add(False)
            continue
        if branching_manager.is_execution_branch_batch_oriented(
            execution_branch=execution_branch
        ):
            raise ValueError(
                "Should not let batch oriented condition statement influence non-simd step"
            )
        else:
            mask = branching_manager.get_mask(execution_branch=execution_branch)
            masks.add(mask)
    if False in masks:
        return None
    result = {}
    for parameter_name, parameter_spec in step_node.input_data.items():
        if parameter_spec.is_batch_oriented():
            raise ValueError("Not expected to spot batch oriented parameter here")
        if parameter_spec.points_to_input():
            input_name = get_last_chunk_of_selector(selector=parameter_spec.selector)
            result[parameter_name] = runtime_parameters[input_name]
        elif parameter_spec.points_to_step_output():
            result[parameter_name] = execution_cache.get_non_batch_output(
                selector=parameter_spec.selector
            )
        else:
            result[parameter_name] = parameter_spec.value
    return result


def iterate_over_simd_step_input(
    step_node: StepNode,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    dynamic_batches_manager: DynamicBatchesManager,
    branching_manager: BranchingManager,
) -> Generator[NonBatchModeSIMDStepInput, None, None]:
    simd_step_input = construct_simd_step_input(
        step_node=step_node,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
        dynamic_batches_manager=dynamic_batches_manager,
        branching_manager=branching_manager,
    )
    parameters_generator = unfold_parameters(parameters=simd_step_input.parameters)
    for index, single_parameters_set in zip(
        simd_step_input.indices, parameters_generator
    ):
        yield NonBatchModeSIMDStepInput(
            index=index,
            parameters=single_parameters_set,
        )


def construct_simd_step_input(
    step_node: StepNode,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    dynamic_batches_manager: DynamicBatchesManager,
    branching_manager: BranchingManager,
) -> BatchModeSIMDStepInput:
    masks = construct_mask_for_all_inputs_dimensionalities(
        step_node=step_node,
        branching_manager=branching_manager,
    )
    print(f"Masks for step {step_node.name}: {masks}")
    return prepare_parameters(
        step_node=step_node,
        dynamic_batches_manager=dynamic_batches_manager,
        masks=masks,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
    )


def construct_mask_for_all_inputs_dimensionalities(
    step_node: StepNode,
    branching_manager: BranchingManager,
) -> Any:
    inputs_dimensionalities = collect_inputs_dimensionalities(step_node=step_node)
    all_dimensionalities = {dim for dim in inputs_dimensionalities.values() if dim > 0}
    batch_masks, non_batch_masks = [], set()
    print(
        "step_node.execution_branches_impacting_inputs",
        step_node.execution_branches_impacting_inputs,
    )
    for execution_branch in step_node.execution_branches_impacting_inputs:
        if not branching_manager.is_execution_branch_registered(
            execution_branch=execution_branch
        ):
            non_batch_masks.add(False)
            continue
        if branching_manager.is_execution_branch_batch_oriented(
            execution_branch=execution_branch
        ):
            mask = branching_manager.get_mask(execution_branch=execution_branch)
            batch_masks.append(mask)
        else:
            mask = branching_manager.get_mask(execution_branch=execution_branch)
            non_batch_masks.add(mask)
    print("batch_masks", batch_masks)
    print("non_batch_masks", non_batch_masks)
    if False in non_batch_masks:
        return {dimension: set() for dimension in all_dimensionalities}
    return {
        dimension: get_masks_intersection_up_to_dimension(
            batch_masks=batch_masks,
            dimension=dimension,
        )
        for dimension in all_dimensionalities
    }


def collect_inputs_dimensionalities(
    step_node: StepNode,
) -> Dict[str, int]:
    result = {}
    for parameter_name, parameter_specs in step_node.input_data.items():
        if parameter_specs.is_compound_input():
            dimensionalities = [
                nested_definition.get_dimensionality()
                for nested_definition in parameter_specs.iterate_through_definitions()
            ]
            # to help for empty values
            dimensionalities.append(0)
            result[parameter_name] = max(dimensionalities)
        else:
            result[parameter_name] = parameter_specs.get_dimensionality()
    return result


def get_masks_intersection_up_to_dimension(
    batch_masks: List[Set[DynamicBatchIndex]],
    dimension: int,
) -> Optional[Set[DynamicBatchIndex]]:
    batch_masks_in_dimension = [
        {mask_element[:dimension] for mask_element in batch_mask}
        for batch_mask in batch_masks
    ]
    if not batch_masks_in_dimension:
        return None
    return set.intersection(*batch_masks_in_dimension)


class GuardForIndicesWrapping:

    def __init__(self):
        self._registered_wrapping: Optional[List[DynamicBatchIndex]] = None

    def register_wrapping(
        self, indices_before_wrapping: List[DynamicBatchIndex]
    ) -> None:
        if self._registered_wrapping is None:
            self._registered_wrapping = indices_before_wrapping
        elif self._registered_wrapping != indices_before_wrapping:
            raise ValueError("Batches missmatch")


def prepare_parameters(
    step_node: StepNode,
    dynamic_batches_manager: DynamicBatchesManager,
    masks: Dict[int, Optional[Set[DynamicBatchIndex]]],
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
) -> BatchModeSIMDStepInput:
    result = {}
    indices_for_parameter = {}
    guard_of_indices_wrapping = GuardForIndicesWrapping()
    for parameter_name, parameter_specs in step_node.input_data.items():
        if parameter_specs.is_compound_input():
            print(f"{parameter_name} is compound")
            result[parameter_name], indices_for_parameter[parameter_name] = (
                get_compound_parameter_value(
                    parameter=parameter_specs,
                    step_execution_dimensionality_offset=step_node.step_execution_dimensionality_offset,
                    masks=masks,
                    dynamic_batches_manager=dynamic_batches_manager,
                    runtime_parameters=runtime_parameters,
                    execution_cache=execution_cache,
                    guard_of_indices_wrapping=guard_of_indices_wrapping,
                )
            )
        else:
            print(f"{parameter_name} is simple")
            result[parameter_name], indices_for_parameter[parameter_name] = (
                get_non_compound_parameter_value(
                    parameter=parameter_specs,
                    step_execution_dimensionality_offset=step_node.step_execution_dimensionality_offset,
                    masks=masks,
                    dynamic_batches_manager=dynamic_batches_manager,
                    runtime_parameters=runtime_parameters,
                    execution_cache=execution_cache,
                    guard_of_indices_wrapping=guard_of_indices_wrapping,
                )
            )
    batch_parameters_indices = [
        i for i in indices_for_parameter.values() if i is not None
    ]
    ensure_compound_input_indices_match(indices=batch_parameters_indices)
    if not batch_parameters_indices:
        raise ValueError("Should be non simd input!")
    indices = batch_parameters_indices[0]
    if not step_node.step_manifest.accepts_empty_values():
        empty_indices = get_empty_indices(value=result)
        indices = [e for e in indices if e not in empty_indices]
        result = filter_parameters(value=result, empty_indices=empty_indices)
    return BatchModeSIMDStepInput(
        indices=indices,
        parameters=result,
    )


def get_compound_parameter_value(
    parameter: CompoundStepInputDefinition,
    step_execution_dimensionality_offset: int,
    masks: Dict[int, Optional[Set[DynamicBatchIndex]]],
    dynamic_batches_manager: DynamicBatchesManager,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    guard_of_indices_wrapping: GuardForIndicesWrapping,
) -> Tuple[Union[list, Dict[str, Any]], Optional[List[DynamicBatchIndex]]]:
    batch_indices = []
    if parameter.represents_list_of_inputs():
        result = []
        for nested_element in parameter.iterate_through_definitions():
            non_compound_parameter_value, non_compound_indices = (
                get_non_compound_parameter_value(
                    parameter=nested_element,
                    step_execution_dimensionality_offset=step_execution_dimensionality_offset,
                    masks=masks,
                    dynamic_batches_manager=dynamic_batches_manager,
                    runtime_parameters=runtime_parameters,
                    execution_cache=execution_cache,
                    guard_of_indices_wrapping=guard_of_indices_wrapping,
                )
            )
            result.append(nested_element)
            if non_compound_indices is not None:
                batch_indices.append(non_compound_indices)
    else:
        result = {}
        for nested_element in parameter.iterate_through_definitions():
            non_compound_parameter_value, non_compound_indices = (
                get_non_compound_parameter_value(
                    parameter=nested_element,
                    step_execution_dimensionality_offset=step_execution_dimensionality_offset,
                    masks=masks,
                    dynamic_batches_manager=dynamic_batches_manager,
                    runtime_parameters=runtime_parameters,
                    execution_cache=execution_cache,
                    guard_of_indices_wrapping=guard_of_indices_wrapping,
                )
            )
            result[nested_element.parameter_specification.nested_element_key] = (
                non_compound_parameter_value
            )
            if non_compound_indices:
                batch_indices.append(non_compound_indices)
    ensure_compound_input_indices_match(indices=batch_indices)
    result_indices = None
    if batch_indices:
        result_indices = batch_indices[0]
    return result, result_indices


def get_non_compound_parameter_value(
    parameter: StepInputDefinition,
    step_execution_dimensionality_offset: int,
    masks: Dict[int, Optional[Set[DynamicBatchIndex]]],
    dynamic_batches_manager: DynamicBatchesManager,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    guard_of_indices_wrapping: GuardForIndicesWrapping,
) -> Union[Any, Optional[List[DynamicBatchIndex]]]:
    if not parameter.is_batch_oriented():
        input_parameter: DynamicStepInputDefinition = parameter  # type: ignore
        if parameter.points_to_input():
            parameter_name = get_last_chunk_of_selector(
                selector=input_parameter.selector
            )
            return runtime_parameters[parameter_name], None
        static_input: StaticStepInputDefinition = parameter  # type: ignore
        return static_input.value, None
    print(f"Parameter: {parameter.parameter_specification} is dynamic")
    dynamic_parameter: DynamicStepInputDefinition = parameter  # type: ignore
    batch_dimensionality = dynamic_parameter.get_dimensionality()
    lineage_indices = dynamic_batches_manager.get_indices_for_data_lineage(
        lineage=dynamic_parameter.data_lineage,
    )
    mask_for_dimension = masks[batch_dimensionality]
    print("lineage_indices", lineage_indices)
    if dynamic_parameter.points_to_input():
        print("Taking input! Mask:", mask_for_dimension)
        input_name = get_last_chunk_of_selector(selector=dynamic_parameter.selector)
        batch_input = runtime_parameters[input_name]
        if mask_for_dimension is not None:
            if len(lineage_indices) != len(batch_input):
                raise ValueError("Input dimensions missmatch")
            batch_input = [
                input_element if element_index in mask_for_dimension else None
                for element_index, input_element in zip(lineage_indices, batch_input)
            ]
    else:
        print("step output input! Mask: ", mask_for_dimension)
        batch_input = execution_cache.get_batch_output(
            selector=dynamic_parameter.selector,
            batch_elements_indices=lineage_indices,
            mask=mask_for_dimension,
        )
    print(
        "param", parameter, step_execution_dimensionality_offset, batch_dimensionality
    )
    if step_execution_dimensionality_offset == 0:
        return Batch(batch_input, lineage_indices), lineage_indices
    if step_execution_dimensionality_offset < 1:
        raise ValueError("Not expected!")
    if step_execution_dimensionality_offset > 1:
        raise ValueError("Unexpected diff in dimension!")
    if batch_dimensionality < 2:
        raise ValueError("Must have some space to decrease")
    result = reduce_batch_dimensionality(
        indices=lineage_indices,
        data=batch_input,
        guard_of_indices_wrapping=guard_of_indices_wrapping,
    )
    return result, result._indices


def reduce_batch_dimensionality(
    indices: List[DynamicBatchIndex],
    data: List[T],
    guard_of_indices_wrapping: GuardForIndicesWrapping,
) -> Batch[Batch[T]]:
    print("reduce_batch_dimensionality()", indices)
    guard_of_indices_wrapping.register_wrapping(indices_before_wrapping=indices)
    already_spotted_downgraded_indices = set()
    wrapped_batch_index, wrapped_batch_content = [], []
    result_index, result_data = [], []
    for index, data in zip(indices, data):
        downgraded_index = index[:-1]
        if downgraded_index in already_spotted_downgraded_indices:
            wrapped_batch_index.append(index)
            wrapped_batch_content.append(data)
        elif not wrapped_batch_index:
            result_index.append(wrapped_batch_index[-1][:-1])
            result_data.append(Batch(wrapped_batch_content, wrapped_batch_index))
            wrapped_batch_content = []
            wrapped_batch_index = []
    if not wrapped_batch_index:
        result_index.append(wrapped_batch_index[-1][:-1])
        result_data.append(Batch(wrapped_batch_content, wrapped_batch_index))
    return Batch(result_data, result_index)


def ensure_compound_input_indices_match(indices: List[List[DynamicBatchIndex]]) -> None:
    if not indices:
        return None
    reference_set = set(indices[0])
    for index in indices[1:]:
        other_set = set(index)
        if reference_set != other_set:
            raise ValueError("Indices missmatch")


def get_empty_indices(value: Any) -> Set[DynamicBatchIndex]:
    result = set()
    if isinstance(value, dict):
        for v in value.values():
            value_result = get_empty_indices(v)
            result = result.union(value_result)
    if isinstance(value, list):
        for v in value:
            value_result = get_empty_indices(v)
            result = result.union(value_result)
    if isinstance(value, Batch):
        indices = value._indices
        for i, v in zip(indices, value):
            if isinstance(v, Batch):
                value_result = get_empty_indices(v)
                result = result.union(value_result)
            elif v is None:
                result.add(i)
    return result


def filter_parameters(value: Any, empty_indices: Set[DynamicBatchIndex]) -> Any:
    if isinstance(value, dict):
        return {
            k: filter_parameters(value=v, empty_indices=empty_indices)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [filter_parameters(value=v, empty_indices=empty_indices) for v in value]
    if isinstance(value, Batch):
        return value.filter_by_indices(indices_to_remove=empty_indices)
    return value


def unfold_parameters(
    parameters: Dict[str, Any]
) -> Generator[Dict[str, Any], None, None]:
    batch_parameters = get_batch_parameters(parameters=parameters)
    non_batch_parameters = {
        k: v for k, v in parameters.items() if k not in batch_parameters
    }
    if not batch_parameters:
        if not non_batch_parameters:
            return None
        yield non_batch_parameters
        return None
    for unfolded_batch_parameters in iterate_over_batches(
        batch_parameters=batch_parameters
    ):
        yield {**unfolded_batch_parameters, **non_batch_parameters}


def get_batch_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for name, value in parameters.items():
        if isinstance(value, Batch):
            result[name] = value
        elif isinstance(value, list) and any(isinstance(v, Batch) for v in value):
            result[name] = value
        elif isinstance(value, dict) and any(
            isinstance(v, Batch) for v in value.values()
        ):
            result[name] = value
    return result


def iterate_over_batches(
    batch_parameters: Dict[str, Any]
) -> Generator[Dict[str, Any], None, None]:
    index = 0
    end = False
    while not end:
        result = {}
        for name, value in batch_parameters.items():
            if isinstance(value, Batch):
                if len(value) <= index:
                    end = True
                    break
                result[name] = value[index]
            elif isinstance(value, list):
                to_yield = []
                for element in value:
                    if len(element) <= index:
                        end = True
                        break
                    to_yield.append(element[index])
                result[name] = to_yield
            elif isinstance(value, dict):
                to_yield = {}
                for key, key_value in value.items():
                    if not isinstance(key_value, Batch):
                        to_yield[key] = key_value
                    else:
                        if len(key_value) <= index:
                            end = True
                            break
                        else:
                            to_yield[key] = key_value[index]
                result[name] = to_yield
        index += 1
        if not end:
            yield result
