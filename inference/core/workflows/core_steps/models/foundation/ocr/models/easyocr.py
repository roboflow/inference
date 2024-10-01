from .base import BaseOCRModel
from inference.core.workflows.core_steps.common.entities import (
    StepExecutionMode,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import BlockResult
from typing import Callable, List, Optional
import easyocr
import cv2


class EasyOCRModel(BaseOCRModel):
    def __init__(
        self,
        model_manager,
        api_key: Optional[str],
        easyocr_languages: List[str] = ["en"],
    ):
        super().__init__(model_manager, api_key)
        self.reader = easyocr.Reader(easyocr_languages)

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

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        post_process_result: Callable[
            [Batch[WorkflowImageData], List[dict]], BlockResult
        ],
    ) -> BlockResult:
        predictions = []
        for image_data in images:
            # Convert image_data to numpy array
            inference_image = image_data.to_inference_format(
                numpy_preferred=True,
            )
            img = inference_image["value"]
            # Ensure image is in RGB format
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                pass
            else:
                # Unsupported image format
                raise ValueError("Unsupported image format")
            # Run OCR
            result = self.reader.readtext(img, detail=0)
            text = " ".join(result)
            prediction = {"result": text}
            predictions.append(prediction)
        return post_process_result(images, predictions)

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        post_process_result: Callable[
            [Batch[WorkflowImageData], List[dict]], BlockResult
        ],
    ) -> BlockResult:
        raise NotImplementedError(
            "Remote execution is not implemented for EasyOCRModel."
        )
