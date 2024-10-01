from typing import Callable, List

from inference.core.entities.requests.trocr import TrOCRInferenceRequest
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import load_core_model
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import BlockResult

from .base import BaseOCRModel


class TrOCRModel(BaseOCRModel):

    def run(
        self,
        images: Batch[WorkflowImageData],
        step_execution_mode: StepExecutionMode,
        post_process_result: Callable[
            [Batch[WorkflowImageData], List[dict]], BlockResult
        ],
    ) -> BlockResult:
        if step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(images, post_process_result)
        elif step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(images, post_process_result)
        else:
            raise ValueError(f"Unknown step execution mode: {step_execution_mode}")

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        post_process_result: Callable[
            [Batch[WorkflowImageData], List[dict]], BlockResult
        ],
    ) -> BlockResult:
        predictions = []
        for single_image in images:
            inference_request = TrOCRInferenceRequest(
                image=single_image.to_inference_format(numpy_preferred=True),
                api_key=self.api_key,
            )
            trocr_model_id = load_core_model(
                model_manager=self.model_manager,
                inference_request=inference_request,
                core_model="trocr",
            )
            result = self.model_manager.infer_from_request_sync(
                trocr_model_id, inference_request
            )
            predictions.append(result.model_dump())
        return post_process_result(images, predictions)

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        post_process_result: Callable[
            [Batch[WorkflowImageData], List[dict]], BlockResult
        ],
    ) -> BlockResult:
        raise NotImplementedError("Remote execution is not implemented for TrOCRModel.")
