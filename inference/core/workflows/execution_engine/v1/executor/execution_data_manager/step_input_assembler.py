from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, TypeVar, Union

from inference.core.workflows.errors import AssumptionError, ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.entities.base import Batch
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    AutoBatchCastingConfig,
    CompoundStepInputDefinition,
    DynamicStepInputDefinition,
    StaticStepInputDefinition,
    StepInputDefinition,
    StepNode,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    get_last_chunk_of_selector,
)
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.branching_manager import (
    BranchingManager,
)
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.dynamic_batches_manager import (
    DynamicBatchesManager,
    DynamicBatchIndex,
)
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.execution_cache import (
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
        raise ExecutionEngineRuntimeError(
            public_message=f"Attempted to get non-SIMD input for SIMD step: {step_node.name}."
            f"This is most likely a bug. Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | step_input_assembling",
        )
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
            raise ExecutionEngineRuntimeError(
                public_message=f"For step: {step_node.name}, detected batch-oriented condition statements "
                f"impacting non-SIMD step, which should be prevented by Workflows Compiler. "
                f"This is most likely a bug. Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_input_assembling",
            )
        else:
            mask = branching_manager.get_mask(execution_branch=execution_branch)
            masks.add(mask)
    if False in masks:
        return None
    result = {}
    detected_empty_step_output_selector = False
    for parameter_name, parameter_spec in step_node.input_data.items():
        if parameter_spec.is_compound_input():
            result[parameter_name], contains_empty_step_output_selector = (
                construct_non_simd_step_compound_input(
                    step_node=step_node,
                    parameter_spec=parameter_spec,
                    runtime_parameters=runtime_parameters,
                    execution_cache=execution_cache,
                )
            )
        else:
            result[parameter_name], contains_empty_step_output_selector = (
                construct_non_simd_step_non_compound_input(
                    step_node=step_node,
                    parameter_spec=parameter_spec,
                    runtime_parameters=runtime_parameters,
                    execution_cache=execution_cache,
                )
            )
        detected_empty_step_output_selector = (
            detected_empty_step_output_selector or contains_empty_step_output_selector
        )
    if (
        detected_empty_step_output_selector
        and not step_node.step_manifest.accepts_empty_values()
    ):
        return None
    return result


def construct_non_simd_step_compound_input(
    step_node: StepNode,
    parameter_spec: CompoundStepInputDefinition,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
) -> Tuple[Any, bool]:
    if parameter_spec.represents_list_of_inputs():
        return construct_non_simd_step_compound_list_input(
            step_node=step_node,
            parameter_spec=parameter_spec,
            runtime_parameters=runtime_parameters,
            execution_cache=execution_cache,
        )
    return construct_non_simd_step_compound_dict_input(
        step_node=step_node,
        parameter_spec=parameter_spec,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
    )


def construct_non_simd_step_compound_list_input(
    step_node: StepNode,
    parameter_spec: CompoundStepInputDefinition,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
) -> Tuple[List[Any], bool]:
    # Validate mixed array patterns before processing elements
    _validate_mixed_array_selectors(step_node, parameter_spec)

    result = []
    contains_empty_step_output_selector = False
    for nested_definition in parameter_spec.iterate_through_definitions():
        nested_value, value_contains_empty_selector = (
            construct_non_simd_step_non_compound_input(
                step_node=step_node,
                parameter_spec=nested_definition,
                runtime_parameters=runtime_parameters,
                execution_cache=execution_cache,
            )
        )
        result.append(nested_value)
        contains_empty_step_output_selector = (
            contains_empty_step_output_selector or value_contains_empty_selector
        )
    return result, contains_empty_step_output_selector


def construct_non_simd_step_compound_dict_input(
    step_node: StepNode,
    parameter_spec: CompoundStepInputDefinition,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
) -> Tuple[Dict[str, Any], bool]:
    result = {}
    contains_empty_step_output_selector = False
    for nested_definition in parameter_spec.iterate_through_definitions():
        (
            result[nested_definition.parameter_specification.nested_element_key],
            value_contains_empty_selector,
        ) = construct_non_simd_step_non_compound_input(
            step_node=step_node,
            parameter_spec=nested_definition,
            runtime_parameters=runtime_parameters,
            execution_cache=execution_cache,
        )
        contains_empty_step_output_selector = (
            contains_empty_step_output_selector or value_contains_empty_selector
        )
    return result, contains_empty_step_output_selector


def construct_non_simd_step_non_compound_input(
    step_node: StepNode,
    parameter_spec: StepInputDefinition,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
) -> Tuple[Any, bool]:
    if parameter_spec.is_compound_input():
        raise AssumptionError(
            public_message=f"Workflows Execution Error encountered unexpected state probably related to the fact "
            f"that Workflows Compiler allowed for multi-level nesting of inputs selectors which "
            f"is not supported. This is most likely the bug. Contact Roboflow team "
            f"through github issues (https://github.com/roboflow/inference/issues) providing full "
            f"context of the problem - including workflow definition you use.",
            context="workflow_execution | step_input_assembling",
        )
    if parameter_spec.is_batch_oriented():
        raise ExecutionEngineRuntimeError(
            public_message=f"Encountered batch-oriented input for non-SIMD step: {step_node.name}."
            f"This is most likely a bug. Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | step_input_assembling",
        )
    if parameter_spec.points_to_input():
        parameter_spec: DynamicStepInputDefinition = parameter_spec  # type: ignore
        input_name = get_last_chunk_of_selector(selector=parameter_spec.selector)
        return runtime_parameters[input_name], False
    if parameter_spec.points_to_step_output():
        parameter_spec: DynamicStepInputDefinition = parameter_spec  # type: ignore
        step_output = execution_cache.get_non_batch_output(
            selector=parameter_spec.selector
        )
        return step_output, step_output is None
    parameter_spec: StaticStepInputDefinition = parameter_spec  # type: ignore
    return parameter_spec.value, False


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
    masks, scalars_discarded = construct_mask_for_all_inputs_dimensionalities(
        step_node=step_node,
        branching_manager=branching_manager,
    )
    return prepare_parameters(
        step_node=step_node,
        dynamic_batches_manager=dynamic_batches_manager,
        masks=masks,
        scalars_discarded=scalars_discarded,
        runtime_parameters=runtime_parameters,
        execution_cache=execution_cache,
    )


def construct_mask_for_all_inputs_dimensionalities(
    step_node: StepNode,
    branching_manager: BranchingManager,
) -> Tuple[Any, bool]:
    inputs_dimensionalities = collect_inputs_dimensionalities(step_node=step_node)
    all_dimensionalities = {dim for dim in inputs_dimensionalities.values() if dim > 0}
    batch_masks, non_batch_masks = [], set()
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
    scalar_mask_contains_false = False in non_batch_masks
    if scalar_mask_contains_false:
        return {
            dimension: set() for dimension in all_dimensionalities
        }, scalar_mask_contains_false
    return {
        dimension: get_masks_intersection_up_to_dimension(
            batch_masks=batch_masks,
            dimension=dimension,
        )
        for dimension in all_dimensionalities
    }, scalar_mask_contains_false


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
    if not batch_masks:
        return None

    # Initialize intersection with the first batch mask transformed up to the given dimension
    intersection = {mask_element[:dimension] for mask_element in batch_masks[0]}

    # Perform the intersection with the transformed sets of subsequent batch masks
    for batch_mask in batch_masks[1:]:
        intersection &= {mask_element[:dimension] for mask_element in batch_mask}

        # Early exit if intersection becomes empty
        if not intersection:
            return set()

    return intersection


class GuardForIndicesWrapping:

    def __init__(self):
        self._registered_wrapping: Optional[List[DynamicBatchIndex]] = None

    def register_wrapping(
        self, indices_before_wrapping: List[DynamicBatchIndex]
    ) -> None:
        if self._registered_wrapping is None:
            self._registered_wrapping = indices_before_wrapping
        elif self._registered_wrapping != indices_before_wrapping:
            raise ExecutionEngineRuntimeError(
                public_message=f"Detected a situation when step requires dimensionality wrapping, but"
                f"different inputs register different elements indices to wrap which is illegal "
                f"and should be detected earlier by Workflows compiler. This is most likely a bug. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_input_assembling",
            )


def prepare_parameters(
    step_node: StepNode,
    dynamic_batches_manager: DynamicBatchesManager,
    masks: Dict[int, Optional[Set[DynamicBatchIndex]]],
    scalars_discarded: bool,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
) -> BatchModeSIMDStepInput:
    step_requests_batch_input = step_node.step_manifest.accepts_batch_input()
    result = {}
    indices_for_parameter = {}
    guard_of_indices_wrapping = GuardForIndicesWrapping()
    compound_inputs = set()
    contains_empty_scalar_step_output_selector = False
    for parameter_name, parameter_specs in step_node.input_data.items():
        if parameter_specs.is_compound_input():
            (
                result[parameter_name],
                indices_for_parameter[parameter_name],
                value_contains_empty_scalar_step_output_selector,
            ) = get_compound_parameter_value(
                parameter=parameter_specs,
                step_execution_dimensionality=step_node.step_execution_dimensionality,
                masks=masks,
                scalars_discarded=scalars_discarded,
                dynamic_batches_manager=dynamic_batches_manager,
                runtime_parameters=runtime_parameters,
                execution_cache=execution_cache,
                guard_of_indices_wrapping=guard_of_indices_wrapping,
                auto_batch_casting_lineage_supports=step_node.auto_batch_casting_lineage_supports,
                step_requests_batch_input=step_requests_batch_input,
            )
            compound_inputs.add(parameter_name)
        else:
            (
                result[parameter_name],
                indices_for_parameter[parameter_name],
                value_contains_empty_scalar_step_output_selector,
            ) = get_non_compound_parameter_value(
                parameter=parameter_specs,
                step_execution_dimensionality=step_node.step_execution_dimensionality,
                masks=masks,
                scalars_discarded=scalars_discarded,
                dynamic_batches_manager=dynamic_batches_manager,
                runtime_parameters=runtime_parameters,
                execution_cache=execution_cache,
                guard_of_indices_wrapping=guard_of_indices_wrapping,
                auto_batch_casting_lineage_supports=step_node.auto_batch_casting_lineage_supports,
                step_requests_batch_input=step_requests_batch_input,
            )
        contains_empty_scalar_step_output_selector = (
            contains_empty_scalar_step_output_selector
            or value_contains_empty_scalar_step_output_selector
        )
    batch_parameters_indices = [
        i for i in indices_for_parameter.values() if i is not None
    ]
    ensure_compound_input_indices_match(indices=batch_parameters_indices)
    if not batch_parameters_indices:
        raise ExecutionEngineRuntimeError(
            public_message=f"For step: {step_node.name} which is assessed by Workflows Compiler as "
            f"SIMD step Execution Engine cannot detect batch inputs. "
            f"This is most likely a bug. Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | step_input_assembling",
        )
    indices = batch_parameters_indices[0]
    if not step_node.step_manifest.accepts_empty_values():
        if contains_empty_scalar_step_output_selector:
            return BatchModeSIMDStepInput(
                indices=[],
                parameters={},
            )
        empty_indices = get_empty_batch_elements_indices(value=result)
        if empty_indices:
            indices = [e for e in indices if e not in empty_indices]
            result = remove_indices(value=result, indices=empty_indices)
    return BatchModeSIMDStepInput(
        indices=indices,
        parameters=result,
    )


def get_compound_parameter_value(
    parameter: CompoundStepInputDefinition,
    step_execution_dimensionality: int,
    masks: Dict[int, Optional[Set[DynamicBatchIndex]]],
    scalars_discarded: bool,
    dynamic_batches_manager: DynamicBatchesManager,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    guard_of_indices_wrapping: GuardForIndicesWrapping,
    auto_batch_casting_lineage_supports: Dict[str, AutoBatchCastingConfig],
    step_requests_batch_input: bool,
) -> Tuple[Union[list, Dict[str, Any]], Optional[List[DynamicBatchIndex]], bool]:
    contains_empty_scalar_step_output_selector = False
    batch_indices = []
    if parameter.represents_list_of_inputs():
        result = []
        for nested_element in parameter.iterate_through_definitions():
            (
                non_compound_parameter_value,
                non_compound_indices,
                value_contains_empty_scalar_step_output_selector,
            ) = get_non_compound_parameter_value(
                parameter=nested_element,
                step_execution_dimensionality=step_execution_dimensionality,
                masks=masks,
                scalars_discarded=scalars_discarded,
                dynamic_batches_manager=dynamic_batches_manager,
                runtime_parameters=runtime_parameters,
                execution_cache=execution_cache,
                guard_of_indices_wrapping=guard_of_indices_wrapping,
                auto_batch_casting_lineage_supports=auto_batch_casting_lineage_supports,
                step_requests_batch_input=step_requests_batch_input,
            )
            result.append(non_compound_parameter_value)
            contains_empty_scalar_step_output_selector = (
                contains_empty_scalar_step_output_selector
                or value_contains_empty_scalar_step_output_selector
            )
            if non_compound_indices is not None:
                batch_indices.append(non_compound_indices)
    else:
        result = {}
        for nested_element in parameter.iterate_through_definitions():
            (
                non_compound_parameter_value,
                non_compound_indices,
                value_contains_empty_scalar_step_output_selector,
            ) = get_non_compound_parameter_value(
                parameter=nested_element,
                step_execution_dimensionality=step_execution_dimensionality,
                masks=masks,
                scalars_discarded=scalars_discarded,
                dynamic_batches_manager=dynamic_batches_manager,
                runtime_parameters=runtime_parameters,
                execution_cache=execution_cache,
                guard_of_indices_wrapping=guard_of_indices_wrapping,
                auto_batch_casting_lineage_supports=auto_batch_casting_lineage_supports,
                step_requests_batch_input=step_requests_batch_input,
            )
            result[nested_element.parameter_specification.nested_element_key] = (
                non_compound_parameter_value
            )
            contains_empty_scalar_step_output_selector = (
                contains_empty_scalar_step_output_selector
                or value_contains_empty_scalar_step_output_selector
            )
            if non_compound_indices is not None:
                batch_indices.append(non_compound_indices)
    ensure_compound_input_indices_match(indices=batch_indices)
    result_indices = None
    if len(batch_indices) > 0:
        result_indices = batch_indices[0]
    return result, result_indices, contains_empty_scalar_step_output_selector


def get_non_compound_parameter_value(
    parameter: StepInputDefinition,
    step_execution_dimensionality: int,
    masks: Dict[int, Optional[Set[DynamicBatchIndex]]],
    scalars_discarded: bool,
    dynamic_batches_manager: DynamicBatchesManager,
    runtime_parameters: Dict[str, Any],
    execution_cache: ExecutionCache,
    guard_of_indices_wrapping: GuardForIndicesWrapping,
    auto_batch_casting_lineage_supports: Dict[str, AutoBatchCastingConfig],
    step_requests_batch_input: bool,
) -> Tuple[Any, Optional[List[DynamicBatchIndex]], bool]:
    if not parameter.is_batch_oriented():
        requested_as_batch = (
            parameter.parameter_specification.parameter_name
            in auto_batch_casting_lineage_supports
        )
        if parameter.points_to_input():
            input_parameter: DynamicStepInputDefinition = parameter  # type: ignore
            parameter_name = get_last_chunk_of_selector(
                selector=input_parameter.selector
            )
            if not requested_as_batch:
                return runtime_parameters[parameter_name], None, False
            else:
                return apply_auto_batch_casting(
                    parameter_name=parameter_name,
                    value=runtime_parameters[parameter_name],
                    auto_batch_casting_config=auto_batch_casting_lineage_supports[
                        parameter.parameter_specification.parameter_name
                    ],
                    contains_empty_scalar_step_output_selector=False,
                    dynamic_batches_manager=dynamic_batches_manager,
                    step_execution_dimensionality=step_execution_dimensionality,
                    guard_of_indices_wrapping=guard_of_indices_wrapping,
                    step_requests_batch_input=step_requests_batch_input,
                    masks=masks,
                    scalars_discarded=False,
                )
        elif parameter.points_to_step_output():
            input_parameter: DynamicStepInputDefinition = parameter  # type: ignore
            value = execution_cache.get_non_batch_output(
                selector=input_parameter.selector
            )
            if not requested_as_batch:
                return value, None, value is None
            else:
                return apply_auto_batch_casting(
                    parameter_name=parameter.parameter_specification.parameter_name,
                    value=value,
                    auto_batch_casting_config=auto_batch_casting_lineage_supports[
                        parameter.parameter_specification.parameter_name
                    ],
                    contains_empty_scalar_step_output_selector=value is None,
                    dynamic_batches_manager=dynamic_batches_manager,
                    step_execution_dimensionality=step_execution_dimensionality,
                    guard_of_indices_wrapping=guard_of_indices_wrapping,
                    step_requests_batch_input=step_requests_batch_input,
                    masks=masks,
                    scalars_discarded=scalars_discarded,
                )
        else:
            static_input: StaticStepInputDefinition = parameter  # type: ignore
            if not requested_as_batch or static_input.value is None:
                # when we have Optional[Selector()] in manifest - we must retain
                # ability to inject None into the run(...) parameters - as
                # if we treat that as actual batch and broadcast Batch[None],
                # we would behave exactly as condition execution does -
                # and the logic executing after this, will filter-out empty
                # elements - so on None, we behave "the old way" regardless of the fact that ABC
                # was requested
                return static_input.value, None, False
            else:
                return apply_auto_batch_casting(
                    parameter_name=parameter.parameter_specification.parameter_name,
                    value=static_input.value,
                    auto_batch_casting_config=auto_batch_casting_lineage_supports[
                        parameter.parameter_specification.parameter_name
                    ],
                    contains_empty_scalar_step_output_selector=False,
                    dynamic_batches_manager=dynamic_batches_manager,
                    step_execution_dimensionality=step_execution_dimensionality,
                    guard_of_indices_wrapping=guard_of_indices_wrapping,
                    step_requests_batch_input=step_requests_batch_input,
                    masks=masks,
                    scalars_discarded=False,
                )
    dynamic_parameter: DynamicStepInputDefinition = parameter  # type: ignore
    parameter_dimensionality = dynamic_parameter.get_dimensionality()
    lineage_indices = dynamic_batches_manager.get_indices_for_data_lineage(
        lineage=dynamic_parameter.data_lineage,
    )
    mask_for_dimension = masks[parameter_dimensionality]
    if dynamic_parameter.points_to_input():
        input_name = get_last_chunk_of_selector(selector=dynamic_parameter.selector)
        batch_input = _flatten_batch_oriented_inputs(
            runtime_parameters[input_name],
            dimensionality=parameter_dimensionality,
        )
        if mask_for_dimension is not None:
            if len(lineage_indices) != len(batch_input):
                raise ExecutionEngineRuntimeError(
                    public_message=f"Detected a situation when input parameter: "
                    f"{input_name}"
                    f"size is mismatched compared to mask requested for that input. "
                    f"Input length: {len(batch_input)}, mask size: {len(lineage_indices)}. "
                    f"This is most likely a bug. "
                    f"Contact Roboflow team through github issues "
                    f"(https://github.com/roboflow/inference/issues) providing full context of"
                    f"the problem - including workflow definition you use.",
                    context="workflow_execution | step_input_assembling",
                )
            batch_input = [
                input_element if element_index in mask_for_dimension else None
                for element_index, input_element in zip(lineage_indices, batch_input)
            ]
    else:
        batch_input = execution_cache.get_batch_output(
            selector=dynamic_parameter.selector,
            batch_elements_indices=lineage_indices,
            mask=mask_for_dimension,
        )
    if step_execution_dimensionality == parameter_dimensionality:
        return Batch(batch_input, lineage_indices), lineage_indices, False
    if step_execution_dimensionality > parameter_dimensionality:
        raise ExecutionEngineRuntimeError(
            public_message=f"Detected a situation when parameter: "
            f"{parameter.parameter_specification.parameter_name}"
            f"has a dimensionality {parameter_dimensionality} larger "
            f"than step execution dimensionality: {step_execution_dimensionality}. "
            f"This is most likely a bug. "
            f"Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | step_input_assembling",
        )
    if abs(parameter_dimensionality - step_execution_dimensionality) > 1:
        raise ExecutionEngineRuntimeError(
            public_message=f"Detected a situation when parameter: "
            f"{parameter.parameter_specification.parameter_name}"
            f"has a dimensionality {parameter_dimensionality} differing more than one level "
            f"from step execution dimensionality: {step_execution_dimensionality}. "
            f"This is most likely a bug. "
            f"Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | step_input_assembling",
        )
    if step_execution_dimensionality == 0 and not step_requests_batch_input:
        return Batch(batch_input, lineage_indices), lineage_indices, False
    upper_lineage = dynamic_parameter.data_lineage[:-1]
    if len(upper_lineage) == 0:
        if not step_requests_batch_input:
            raise AssumptionError(
                public_message=f"Parameter: {parameter.parameter_specification.parameter_name} "
                f"requires dimensionality wrapping, but registered lineage support is incompatible "
                f"which should be detected by the compiler. This is most likely a bug. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_input_assembling",
            )
        upper_level_indices = [()]
    else:
        upper_level_indices = dynamic_batches_manager.get_indices_for_data_lineage(
            lineage=dynamic_parameter.data_lineage[:-1],
        )
    result = reduce_batch_dimensionality(
        indices=lineage_indices,
        upper_level_index=upper_level_indices,
        data=batch_input,
        guard_of_indices_wrapping=guard_of_indices_wrapping,
    )
    return result, result.indices, False


def apply_auto_batch_casting(
    parameter_name: str,
    value: Any,
    auto_batch_casting_config: AutoBatchCastingConfig,
    contains_empty_scalar_step_output_selector: bool,
    dynamic_batches_manager: DynamicBatchesManager,
    step_execution_dimensionality: int,
    guard_of_indices_wrapping: GuardForIndicesWrapping,
    step_requests_batch_input: bool,
    masks: Dict[int, Optional[Set[DynamicBatchIndex]]],
    scalars_discarded: bool,
) -> Tuple[Any, List[DynamicBatchIndex], bool]:
    if auto_batch_casting_config.lineage_support is None:
        indices = [(0,) * auto_batch_casting_config.casted_dimensionality]
    else:
        indices = dynamic_batches_manager.get_indices_for_data_lineage(
            lineage=auto_batch_casting_config.lineage_support,
        )
        missing_dimensions = auto_batch_casting_config.casted_dimensionality - len(
            auto_batch_casting_config.lineage_support
        )
        if missing_dimensions > 0:
            padding = (0,) * missing_dimensions
            indices = [i + padding for i in indices]
    if scalars_discarded:
        batch_content = [None] * len(indices)
    elif auto_batch_casting_config.lineage_support is None:
        batch_content = [value] * len(indices)
    else:
        support_dimensionality = len(auto_batch_casting_config.lineage_support)
        mask_for_support_dimensionality = masks[support_dimensionality]
        if mask_for_support_dimensionality is None:
            batch_content = [value] * len(indices)
        else:
            batch_content = []
            for index in indices:
                if index[:support_dimensionality] in mask_for_support_dimensionality:
                    batch_content.append(value)
                else:
                    batch_content.append(None)
    created_batch = Batch(content=batch_content, indices=indices)
    if step_execution_dimensionality == auto_batch_casting_config.casted_dimensionality:
        return (
            created_batch,
            indices,
            contains_empty_scalar_step_output_selector or scalars_discarded,
        )
    if step_execution_dimensionality > auto_batch_casting_config.casted_dimensionality:
        raise ExecutionEngineRuntimeError(
            public_message=f"Detected a situation when parameter: "
            f"{parameter_name}"
            f"has auto-batch casted dimensionality {auto_batch_casting_config.casted_dimensionality} larger "
            f"than step execution dimensionality: {step_execution_dimensionality}. "
            f"This is most likely a bug. "
            f"Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | step_input_assembling",
        )
    if (
        abs(
            auto_batch_casting_config.casted_dimensionality
            - step_execution_dimensionality
        )
        > 1
    ):
        raise ExecutionEngineRuntimeError(
            public_message=f"Detected a situation when parameter: "
            f"{parameter_name} has auto batch casted "
            f"dimensionality {auto_batch_casting_config.casted_dimensionality} differing more than one level "
            f"from step execution dimensionality: {step_execution_dimensionality}. "
            f"This is most likely a bug. "
            f"Contact Roboflow team through github issues "
            f"(https://github.com/roboflow/inference/issues) providing full context of"
            f"the problem - including workflow definition you use.",
            context="workflow_execution | step_input_assembling",
        )
    upper_level_lineage_dimensionality = (
        auto_batch_casting_config.casted_dimensionality - 1
    )
    if upper_level_lineage_dimensionality == 0 and not step_requests_batch_input:
        # for batch collapse into scalar
        return (
            created_batch,
            indices,
            contains_empty_scalar_step_output_selector or scalars_discarded,
        )
    if auto_batch_casting_config.lineage_support is None:
        upper_level_indices = [indices[0][:-1]]
    else:
        upper_level_lineage = auto_batch_casting_config.lineage_support[
            :upper_level_lineage_dimensionality
        ]
        if (
            upper_level_lineage_dimensionality < 1
            or len(upper_level_lineage) < upper_level_lineage_dimensionality
        ):
            if not step_requests_batch_input:
                raise AssumptionError(
                    public_message=f"Detected a situation when parameter: {parameter_name} requires dimensionality "
                    f"wrapping, but registered lineage support is incompatible which should be detected "
                    f"by the compiler. This is most likely a bug. "
                    f"Contact Roboflow team through github issues "
                    f"(https://github.com/roboflow/inference/issues) providing full context of"
                    f"the problem - including workflow definition you use.",
                    context="workflow_execution | step_input_assembling",
                )
            upper_level_indices = [()]
        else:
            upper_level_indices = dynamic_batches_manager.get_indices_for_data_lineage(
                lineage=upper_level_lineage,
            )
    result = reduce_batch_dimensionality(
        indices=indices,
        upper_level_index=upper_level_indices,
        data=batch_content,
        guard_of_indices_wrapping=guard_of_indices_wrapping,
    )
    return (
        result,
        result.indices,
        contains_empty_scalar_step_output_selector or scalars_discarded,
    )


def _flatten_batch_oriented_inputs(
    inputs: list,
    dimensionality: int,
) -> List[Any]:
    if dimensionality == 0 or not isinstance(inputs, list):
        raise AssumptionError(
            public_message=f"Could not prepare batch-oriented input data. This is most likely the bug. Contact "
            f"Roboflow team through github issues (https://github.com/roboflow/inference/issues) "
            f"providing full context of the problem - including workflow definition you use.",
            context="workflow_execution | step_input_assembling",
        )
    if dimensionality == 1:
        return inputs
    result = []
    for element in inputs:
        result.extend(
            _flatten_batch_oriented_inputs(
                inputs=element, dimensionality=dimensionality - 1
            )
        )
    return result


def reduce_batch_dimensionality(
    indices: List[DynamicBatchIndex],
    upper_level_index: List[DynamicBatchIndex],
    data: List[T],
    guard_of_indices_wrapping: GuardForIndicesWrapping,
) -> Batch[Optional[Batch[T]]]:
    guard_of_indices_wrapping.register_wrapping(indices_before_wrapping=indices)
    already_spotted_downgraded_indices = set()
    wrapped_batch_index, wrapped_batch_content = [], []
    upper_level_indices_data = {index: None for index in upper_level_index}
    for index, data in zip(indices, data):
        downgraded_index = index[:-1]
        if downgraded_index in already_spotted_downgraded_indices:
            wrapped_batch_index.append(index)
            wrapped_batch_content.append(data)
        else:
            if wrapped_batch_index:
                upper_level_indices_data[wrapped_batch_index[-1][:-1]] = Batch(
                    content=wrapped_batch_content, indices=wrapped_batch_index
                )
            already_spotted_downgraded_indices.add(downgraded_index)
            wrapped_batch_index = [index]
            wrapped_batch_content = [data]
    if wrapped_batch_index:
        upper_level_indices_data[wrapped_batch_index[-1][:-1]] = Batch(
            content=wrapped_batch_content, indices=wrapped_batch_index
        )
    return Batch(
        content=[upper_level_indices_data[index] for index in upper_level_index],
        indices=upper_level_index,
    )


def ensure_compound_input_indices_match(indices: List[List[DynamicBatchIndex]]) -> None:
    if len(indices) < 2:
        return None
    reference_set = set(indices[0])
    for index in indices[1:]:
        other_set = set(index)
        if reference_set != other_set:
            raise ExecutionEngineRuntimeError(
                public_message=f"Detected a situation when step input parameters cannot be created "
                f"due to missmatch in batch element indices. This is most likely a bug. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_input_assembling",
            )


def get_empty_batch_elements_indices(value: Any) -> Set[DynamicBatchIndex]:
    result = set()
    if isinstance(value, dict):
        for v in value.values():
            value_result = get_empty_batch_elements_indices(v)
            result = result.union(value_result)
    if isinstance(value, list):
        for v in value:
            value_result = get_empty_batch_elements_indices(v)
            result = result.union(value_result)
    if isinstance(value, Batch):
        for index, value_element in value.iter_with_indices():
            if isinstance(value_element, Batch):
                value_result = get_empty_batch_elements_indices(value=value_element)
                result = result.union(value_result)
            elif value_element is None:
                result.add(index)
    return result


def remove_indices(value: Any, indices: Set[DynamicBatchIndex]) -> Any:
    if isinstance(value, dict):
        return {k: remove_indices(value=v, indices=indices) for k, v in value.items()}
    if isinstance(value, list):
        return [remove_indices(value=v, indices=indices) for v in value]
    if isinstance(value, Batch):
        return value.remove_by_indices(indices_to_remove=indices)
    return value


def unfold_parameters(
    parameters: Dict[str, Any],
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
    batch_parameters: Dict[str, Any],
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
                    if isinstance(element, Batch):
                        if len(element) <= index:
                            end = True
                            break
                        to_yield.append(element[index])
                    else:
                        to_yield.append(element)
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


def _validate_mixed_array_selectors(
    step_node: StepNode,
    parameter_spec: CompoundStepInputDefinition,
) -> None:
    """
    Validate that mixed arrays don't contain LIST_OF_VALUES_KIND selectors with other elements.

    According to the tech specification:
    - STRING_KIND selectors are allowed in mixed arrays
    - LIST_OF_VALUES_KIND selectors in mixed arrays should trigger clear error messages
    - Pure LIST_OF_VALUES_KIND selectors (not mixed) should work normally

    Args:
        step_node: The step node being processed
        parameter_spec: The compound input definition for the array

    Raises:
        ExecutionEngineRuntimeError: If invalid mixed array patterns are detected
    """
    nested_definitions = list(parameter_spec.iterate_through_definitions())

    # If there's only one element, it's not a mixed array - allow it
    if len(nested_definitions) <= 1:
        return

    # Check for LIST_OF_VALUES_KIND selectors in mixed arrays
    has_literal_elements = False
    has_batch_oriented_selectors = False
    list_selectors = []

    for nested_definition in nested_definitions:
        if isinstance(nested_definition, StaticStepInputDefinition):
            has_literal_elements = True
        elif isinstance(nested_definition, DynamicStepInputDefinition):
            if nested_definition.is_batch_oriented():
                has_batch_oriented_selectors = True
                list_selectors.append(nested_definition.selector)

    # If we have a mixed array with batch-oriented selectors (LIST_OF_VALUES_KIND), that's invalid
    if has_literal_elements and has_batch_oriented_selectors:
        selector_list = ", ".join(f"'{s}'" for s in list_selectors)
        raise ExecutionEngineRuntimeError(
            public_message=f"Invalid mixed array in step '{step_node.name}': "
            f"Array elements can only contain string literals or STRING_KIND selectors. "
            f"Found LIST_OF_VALUES_KIND selector(s): {selector_list}. "
            f"Use either all literals with string selectors: ['literal', '$inputs.string_tag'] "
            f"or a pure list selector: '$inputs.list_tags'",
            context="workflow_execution | step_input_assembling",
        )
