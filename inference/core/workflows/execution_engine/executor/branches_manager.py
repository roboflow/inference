from typing import Any, Dict, List, Literal, Tuple, TypeVar

from inference.core.workflows.constants import ROOT_BRANCH_NAME
from inference.core.workflows.entities.base import Batch, InputType

T = TypeVar("T")


class ExecutionBranchesDataManager:

    @classmethod
    def init(
        cls,
        workflow_inputs: List[InputType],
        runtime_parameters: Dict[str, Any],
    ) -> "ExecutionBranchesDataManager":
        batch_parameters_sizes = {}
        for workflow_input in workflow_inputs:
            if not workflow_input.is_batch_oriented():
                continue
            batch_parameters_sizes[workflow_input.name]: len(
                runtime_parameters[workflow_input.name]
            )
        different_values = set(v for v in batch_parameters_sizes.values() if v > 1)
        if len(different_values) > 1:
            raise ValueError(
                "Could not run workflow, as batch workflow inputs are feed with different sizes of batches. "
                f"Batch sizes for inputs: {batch_parameters_sizes}"
            )
        if len(different_values) == 0:
            different_values = {0}
        input_batch_size = next(iter(different_values))
        execution_branches_masks = {
            ROOT_BRANCH_NAME: [(i,) for i in range(input_batch_size)]
        }
        return cls(execution_branches_masks=execution_branches_masks)

    def __init__(self, execution_branches_masks: Dict[str, List[Tuple[int, ...]]]):
        self._execution_branches_masks = execution_branches_masks

    def register_branch_mask(
        self, branch_name: str, mask: List[Tuple[int, ...]]
    ) -> None:
        if branch_name in self._execution_branches_masks:
            raise ValueError(
                f"Attempted to re-declare existing branch execution mask: {branch_name}"
            )
        self._execution_branches_masks[branch_name] = mask

    def retrieve_data_for_execution_branch(
        self, branch_name: str, data: Batch[Any]
    ) -> Tuple[List[Tuple[int, ...]], Batch[T]]:
        if branch_name not in self._execution_branches_masks:
            raise ValueError(
                f"Attempted to retrieve data for execution branch which was not registered: {branch_name}"
            )
        indices = self._execution_branches_masks[branch_name]
        retrieved_data = [
            e for index in indices for e in get_data_from_index(data, index)
        ]
        retrieved_indices = [e[0] for e in retrieved_data]
        retrieved_batch = Batch([e[1] for e in retrieved_data])
        return retrieved_indices, retrieved_batch

    def realign_step_results(
        self,
        step_results: List[Any],
        input_indices: List[Tuple[int, ...]],
        step_impact_on_data_dimensionality: Literal[
            "decreases", "keeps_the_same", "increases"
        ],
    ) -> Batch[T]:
        if step_impact_on_data_dimensionality == "decreases":
            input_indices = [e[:-1] for e in input_indices]
        for index, result in zip(input_indices, step_results):
            pass
        return Batch([])


def get_data_from_index(
    data: Batch[Any], index: Tuple[int, ...]
) -> List[Tuple[Tuple[int, ...], T]]:
    # TODO: Error handling
    for i in index:
        data = data[i]
    if isinstance(data, Batch):
        batch_flattened = flatten_batch(batch=data)
        return [
            (index + nested_idx, element) for nested_idx, element in batch_flattened
        ]
    return [(index, data)]


def flatten_batch(batch: Batch[Any]) -> List[Tuple[Tuple[int, ...], T]]:
    result = []
    index = 0
    for element in batch:
        if isinstance(element, Batch):
            element_flattened = flatten_batch(element)
            for chunk_idx, chunk in element_flattened:
                result.append((((index,) + chunk_idx), chunk))
        else:
            result.append((index, element))
        index += 1
    return result
