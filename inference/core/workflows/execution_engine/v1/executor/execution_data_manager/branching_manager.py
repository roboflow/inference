from typing import Dict, Set, Union

from inference.core.workflows.errors import ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.dynamic_batches_manager import (
    DynamicBatchIndex,
)


class BranchingManager:

    @classmethod
    def init(cls) -> "BranchingManager":
        return cls(masks={})

    def __init__(self, masks: Dict[str, Union[Set[DynamicBatchIndex], bool]]):
        self._masks = masks
        self._batch_compatibility = {
            branch_name: not isinstance(mask, bool)
            for branch_name, mask in masks.items()
        }

    def register_batch_oriented_mask(
        self,
        execution_branch: str,
        mask: Set[DynamicBatchIndex],
    ) -> None:
        if execution_branch in self._masks:
            raise ExecutionEngineRuntimeError(
                public_message=f"Attempted to re-register maks for execution branch: {execution_branch}. "
                f"This is most likely a bug. Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_input_assembling",
            )
        self._batch_compatibility[execution_branch] = True
        self._masks[execution_branch] = mask

    def register_non_batch_mask(self, execution_branch: str, mask: bool) -> None:
        if execution_branch in self._masks:
            raise ExecutionEngineRuntimeError(
                public_message=f"Attempted to re-register maks for execution branch: {execution_branch}. "
                f"This is most likely a bug. Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_input_assembling",
            )
        self._batch_compatibility[execution_branch] = False
        self._masks[execution_branch] = mask

    def get_mask(self, execution_branch: str) -> Union[Set[DynamicBatchIndex], bool]:
        if execution_branch not in self._masks:
            raise ExecutionEngineRuntimeError(
                public_message=f"Attempted to get mask for not registered execution branch: {execution_branch}. "
                f"This is most likely a bug. Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_input_assembling",
            )
        return self._masks[execution_branch]

    def is_execution_branch_batch_oriented(self, execution_branch: str) -> bool:
        if execution_branch not in self._batch_compatibility:
            raise ExecutionEngineRuntimeError(
                public_message=f"Attempted to get info about not registered execution branch: {execution_branch}. "
                f"This is most likely a bug. Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_input_assembling",
            )
        return self._batch_compatibility[execution_branch]

    def is_execution_branch_registered(self, execution_branch: str) -> bool:
        return execution_branch in self._masks
