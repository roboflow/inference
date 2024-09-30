from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import BlockResult


class BaseOCRModel(ABC):

    def __init__(self, model_manager, api_key):
        self.model_manager = model_manager
        self.api_key = api_key

    @abstractmethod
    def run(
        self,
        images: Batch[WorkflowImageData],
        step_execution_mode: StepExecutionMode,
        post_process_result: Callable[
            [Batch[WorkflowImageData], List[dict]], BlockResult
        ],
    ) -> BlockResult:
        pass
