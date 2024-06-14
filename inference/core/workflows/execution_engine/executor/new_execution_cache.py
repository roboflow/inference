from collections import defaultdict
from copy import copy
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

from inference.core.workflows.constants import (
    ROOT_BRANCH_NAME,
    WORKFLOW_INPUT_BATCH_LINEAGE_ID,
)
from inference.core.workflows.entities.base import InputType
from inference.core.workflows.errors import (
    ExecutionEngineRuntimeError,
    InvalidBlockBehaviourError,
)
from inference.core.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
    get_step_selector_from_its_output,
    is_step_output_selector,
)


class BatchStepCache:

    @classmethod
    def init(cls, step_name: str) -> "BatchStepCache":
        return cls(
            step_name=step_name,
            cache_content=defaultdict(lambda: defaultdict()),
        )

    def __init__(
        self,
        step_name: str,
        cache_content: DefaultDict[str, DefaultDict[Tuple[int, ...], Any]],
    ):
        self._step_name = step_name
        self._cache_content = cache_content

    def register_outputs(
        self,
        indices: List[Tuple[int, ...]],
        outputs: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    ) -> None:
        if len(indices) != len(outputs):
            raise ValueError("Outputs misaligned with indices")
        indices_to_register, outputs_to_register = indices, outputs
        output_is_nested = any(isinstance(e, list) for e in outputs)
        if output_is_nested:
            indices_to_register, outputs_to_register = [], []
            outputs = [e if isinstance(e, list) else [e] for e in outputs]
            for main_idx, elements in zip(indices, outputs):
                for element_idx, sub_element in enumerate(elements):
                    indices_to_register.append(main_idx + (element_idx,))
                    outputs_to_register.append(sub_element)
        for idx, element in zip(indices_to_register, outputs_to_register):
            for property_name, property_value in element.items():
                self._cache_content[property_name][idx] = property_value

    def get_outputs(
        self,
        property_name: str,
        indices: List[Tuple[int, ...]],
        mask: Optional[Set[Tuple[int, ...]]] = None,
    ) -> List[Any]:
        return [
            (
                self._cache_content[property_name].get(index)
                if mask is None or index[: len(mask)] in mask
                else None
            )
            for index in indices
        ]

    def get_all_outputs(
        self,
        indices: List[Tuple[int, ...]],
        mask: Optional[Set[Tuple[int, ...]]] = None,
    ) -> List[Dict[str, Any]]:
        all_keys = list(self._cache_content.keys())
        empty_value = {k: None for k in all_keys}
        return [
            (
                {k: self._cache_content[k].get(index) for k in all_keys}
                if mask is None or index[: len(mask)] in mask
                else copy(empty_value)
            )
            for index in indices
        ]

    def is_property_defined(self, property_name: str) -> bool:
        return property_name in self._cache_content


class NonBatchStepCache:

    @classmethod
    def init(cls, step_name: str) -> "NonBatchStepCache":
        return cls(
            step_name=step_name,
            cache_content=dict(),
        )

    def __init__(
        self,
        step_name: str,
        cache_content: Dict[str, Any],
    ):
        self._step_name = step_name
        self._cache_content = cache_content

    def register_outputs(self, outputs: Dict[str, Any]):
        self._cache_content = outputs

    def get_outputs(
        self,
        property_name: str,
    ) -> Any:
        return self._cache_content[property_name]

    def get_all_outputs(self) -> Dict[str, Any]:
        return self._cache_content

    def is_property_defined(self, property_name: str) -> bool:
        return property_name in self._cache_content


class ExecutionBranchesManager:

    @classmethod
    def init(
        cls,
        workflow_inputs: List[InputType],
        runtime_parameters: Dict[str, Any],
    ) -> "ExecutionBranchesManager":
        batch_parameters_sizes = {}
        for workflow_input in workflow_inputs:
            if not workflow_input.is_batch_oriented():
                continue
            batch_parameters_sizes[workflow_input.name] = len(
                runtime_parameters[workflow_input.name]
            )
        different_values = set(v for v in batch_parameters_sizes.values() if v >= 1)
        if len(different_values) > 1:
            raise ValueError(
                "Could not run workflow, as batch workflow inputs are feed with different sizes of batches. "
                f"Batch sizes for inputs: {batch_parameters_sizes}"
            )
        if len(different_values) == 0:
            different_values = {0}
        input_batch_size = next(iter(different_values))
        execution_branches_masks = {
            ROOT_BRANCH_NAME: {(i,) for i in range(input_batch_size)}
        }
        return cls(execution_branches_masks=execution_branches_masks)

    def __init__(
        self, execution_branches_masks: Dict[str, Union[bool, Set[Tuple[int, ...]]]]
    ):
        self._execution_branches_masks = execution_branches_masks
        self._batch_compatibility = {
            branch_name: not isinstance(mask, bool)
            for branch_name, mask in execution_branches_masks.items()
        }

    def register_batch_branch_mask(
        self, branch_name: str, mask: List[Tuple[int, ...]]
    ) -> None:
        if branch_name in self._execution_branches_masks:
            raise ValueError(
                f"Attempted to re-declare existing branch execution mask: {branch_name}"
            )
        self._batch_compatibility[branch_name] = True
        self._execution_branches_masks[branch_name] = set(mask)

    def register_non_batch_branch_mask(self, branch_name: str, mask: bool) -> None:
        if branch_name in self._execution_branches_masks:
            raise ValueError(
                f"Attempted to re-declare existing branch execution mask: {branch_name}"
            )
        self._batch_compatibility[branch_name] = False
        self._execution_branches_masks[branch_name] = mask

    def retrieve_branch_mask_for_batch(self, branch_name: str) -> Set[Tuple[int, ...]]:
        if (
            branch_name not in self._execution_branches_masks
            or not self._batch_compatibility.get(branch_name)
        ):
            raise ValueError(
                f"Attempted to reach non existing branch name: {branch_name}"
            )
        return self._execution_branches_masks[branch_name]

    def retrieve_branch_mask_for_non_batch(self, branch_name: str) -> bool:
        if (
            branch_name not in self._execution_branches_masks
            or self._batch_compatibility.get(branch_name)
        ):
            raise ValueError(
                f"Attempted to reach non existing branch name: {branch_name}"
            )
        return self._execution_branches_masks[branch_name]

    def is_batch_compatible_branch(self, branch_name: str) -> bool:
        if branch_name not in self._batch_compatibility:
            raise ValueError(f"Branch {branch_name} not registered")
        return self._batch_compatibility[branch_name]


class DynamicBatchesManager:

    @classmethod
    def init(
        cls,
        workflow_inputs: List[InputType],
        runtime_parameters: Dict[str, Any],
    ) -> "DynamicBatchesManager":
        dynamic_batch_sizes = {}
        batch_parameters_sizes = {}
        for workflow_input in workflow_inputs:
            if not workflow_input.is_batch_oriented():
                continue
            batch_parameters_sizes[workflow_input.name] = len(
                runtime_parameters[workflow_input.name]
            )
        different_values = set(v for v in batch_parameters_sizes.values() if v >= 1)
        if len(different_values) > 1:
            raise ValueError(
                "Could not run workflow, as batch workflow inputs are feed with different sizes of batches. "
                f"Batch sizes for inputs: {batch_parameters_sizes}"
            )
        if len(different_values) == 0:
            different_values = {0}
        input_batch_size = next(iter(different_values))
        dynamic_batch_sizes[WORKFLOW_INPUT_BATCH_LINEAGE_ID] = input_batch_size
        return cls(dynamic_batch_sizes=dynamic_batch_sizes)

    def __init__(self, dynamic_batch_sizes: Dict[str, Union[int, list]]):
        self._dynamic_batch_sizes = dynamic_batch_sizes

    def register_batch_sizes(self, data_lineage: str, sizes: Union[int, list]) -> None:
        if self.is_lineage_registered(data_lineage=data_lineage):
            raise ValueError("data lineage occupied")
        self._dynamic_batch_sizes[data_lineage] = sizes

    def get_batch_element_indices(self, data_lineage: str) -> List[Tuple[int, ...]]:
        if not self.is_lineage_registered(data_lineage=data_lineage):
            raise ValueError("data lineage not registered")
        return generate_dynamic_batch_indices(
            dimensions=self._dynamic_batch_sizes[data_lineage]
        )

    def is_lineage_registered(self, data_lineage: str) -> bool:
        return data_lineage in self._dynamic_batch_sizes


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


class ExecutionCache:

    @classmethod
    def init(cls) -> "ExecutionCache":
        return cls(cache_content={}, batches_compatibility={})

    def __init__(
        self,
        cache_content: Dict[str, Union[BatchStepCache, NonBatchStepCache]],
        batches_compatibility: Dict[str, bool],
    ):
        self._cache_content = cache_content
        self._batches_compatibility = batches_compatibility

    def register_step(
        self,
        step_name: str,
        compatible_with_batches: bool,
    ) -> None:
        if self.contains_step(step_name=step_name):
            return None
        if compatible_with_batches:
            step_cache = BatchStepCache.init(
                step_name=step_name,
            )
        else:
            step_cache = NonBatchStepCache.init(
                step_name=step_name,
            )
        print(
            f"register_step(): {step_name}, compatible_with_batches: {compatible_with_batches}"
        )
        self._cache_content[step_name] = step_cache
        self._batches_compatibility[step_name] = compatible_with_batches

    def register_batch_of_step_outputs(
        self,
        step_name: str,
        indices: List[Tuple[int, ...]],
        outputs: List[Dict[str, Any]],
    ) -> None:
        if not self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to register batch outputs for "
                f"step {step_name} which is not registered as batch-compatible. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        try:
            self._cache_content[step_name].register_outputs(
                indices=indices, outputs=outputs
            )
        except TypeError as e:
            # checking this case defensively as there is no guarantee on block
            # meeting contract, and we want graceful error handling
            raise InvalidBlockBehaviourError(
                public_message=f"Block implementing step {step_name} should return outputs which are lists of "
                f"dicts, but the type of output does not match expectation.",
                context="workflow_execution | step_output_registration",
                inner_error=e,
            ) from e

    def register_non_batch_step_outputs(
        self, step_name: str, outputs: Dict[str, Any]
    ) -> None:
        if self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to register non-batch outputs for "
                f"step {step_name} which was registered in cache as batch compatible. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        self._cache_content[step_name].register_outputs(outputs=outputs)

    def get_batch_output(
        self,
        selector: str,
        batch_elements_indices: List[Tuple[int, ...]],
        mask: Optional[Set[Tuple[int, ...]]] = None,
    ) -> List[Any]:
        if not self.is_value_registered(selector=selector):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get output which is not registered using "
                f"step {selector}. That behavior should be prevented by workflows compiler, so "
                f"this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        step_selector = get_step_selector_from_its_output(step_output_selector=selector)
        step_name = get_last_chunk_of_selector(selector=step_selector)
        if not self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get output in batch mode which is "
                f"not supported for step {selector}. That behavior should be prevented by "
                f"workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        property_name = get_last_chunk_of_selector(selector=selector)
        return self._cache_content[step_name].get_outputs(
            property_name=property_name,
            indices=batch_elements_indices,
            mask=mask,
        )

    def get_non_batch_output(
        self,
        selector: str,
    ) -> Any:
        if not self.is_value_registered(selector=selector):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get output which is not registered using "
                f"step {selector}. That behavior should be prevented by workflows compiler, so "
                f"this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        step_selector = get_step_selector_from_its_output(step_output_selector=selector)
        step_name = get_last_chunk_of_selector(selector=step_selector)
        if self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get output in non-batch mode which is "
                f"not supported for step {selector} registered as batch-compatible. That behavior "
                f"should be prevented by workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        property_name = get_last_chunk_of_selector(selector=selector)
        return self._cache_content[step_name].get_outputs(property_name=property_name)

    def get_all_batch_step_outputs(
        self,
        step_name: str,
        batch_elements_indices: List[Tuple[int, ...]],
        mask: Optional[Set[Tuple[int, ...]]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get all outputs from step {step_name} "
                f"which is not register in cache. That behavior should be prevented by "
                f"workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        return self._cache_content[step_name].get_all_outputs(
            indices=batch_elements_indices,
            mask=mask,
        )

    def get_all_non_batch_step_outputs(
        self,
        step_name: str,
    ) -> List[Dict[str, Any]]:
        if self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get all non-batch outputs from step {step_name} "
                f"which is registered as batch-compatible. That behavior should be prevented by "
                f"workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        return self._cache_content[step_name].get_all_outputs()

    def step_outputs_batches(self, step_name: str) -> bool:
        if not self.contains_step(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to check outputs status from step {step_name} "
                f"which is not register in cache. That behavior should be prevented by "
                f"workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        return self._batches_compatibility[step_name]

    def is_value_registered(self, selector: Any) -> bool:
        if not is_step_output_selector(selector_or_value=selector):
            return False
        step_selector = get_step_selector_from_its_output(step_output_selector=selector)
        step_name = get_last_chunk_of_selector(selector=step_selector)
        if not self.contains_step(step_name=step_name):
            return False
        property_name = get_last_chunk_of_selector(selector=selector)
        return self._cache_content[step_name].is_property_defined(
            property_name=property_name
        )

    def contains_step(self, step_name: str) -> bool:
        return step_name in self._cache_content
