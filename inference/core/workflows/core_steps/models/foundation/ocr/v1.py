from typing import Callable, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, model_validator

from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    remove_unexpected_keys_from_dictionary,
)
from inference.core.workflows.execution_engine.constants import (
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    STRING_KIND,
    ImageInputField,
    StepOutputImageSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

from .models.base import BaseOCRModel
from .models.doctr import DoctrOCRModel
from .models.trocr import TrOCRModel
from .models.google_cloud_vision import GoogleCloudVisionOCRModel

LONG_DESCRIPTION = """
 Retrieve the characters in an image using Optical Character Recognition (OCR).

This block returns the text within an image.

You may want to use this block in combination with a detections-based block (i.e. 
ObjectDetectionBlock). An object detection model could isolate specific regions from an 
image (i.e. a shipping container ID in a logistics use case) for further processing. 
You can then use a DynamicCropBlock to crop the region of interest before running OCR.

Using a detections model then cropping detections allows you to isolate your analysis 
on particular regions of an image.
"""

EXPECTED_OUTPUT_KEYS = {"result", "parent_id", "root_parent_id", "prediction_type"}

# Registry of available models
MODEL_REGISTRY = {
    "doctr": {
        "class": DoctrOCRModel,
        "description": "DocTR",
        "required_fields": [],
    },
    "trocr": {
        "class": TrOCRModel,
        "description": "TrOCR",
        "required_fields": [],
    },
    "google-cloud-vision": {
        "class": GoogleCloudVisionOCRModel,
        "description": "Google Cloud Vision OCR",
        "required_fields": ["google_cloud_api_key"],
    },
}

ModelLiteral = Literal["doctr", "trocr", "google-cloud-vision"]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Text Recognition (OCR)",
            "version": "v1",
            "short_description": "Extract text from an image using optical character recognition.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        }
    )
    type: Literal["roboflow_core/ocr_model@v1", "OCRModel"]
    name: str = Field(description="Unique name of step in workflows")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    model: ModelLiteral = Field(
        default="doctr",
        description="The OCR model to use.",
    )
    google_cloud_api_key: Optional[str] = Field(
        default=None,
        description="API key for Google Cloud Vision, required if model is 'google-cloud-vision'.",
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="result", kind=[STRING_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class OCRModelBlockV1(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        model: str,
        google_cloud_api_key: Optional[str] = None,
    ) -> BlockResult:
        ocr_model = self._get_model_instance(
            model=model,
            google_cloud_api_key=google_cloud_api_key,
        )
        return ocr_model.run(
            images=images,
            step_execution_mode=self._step_execution_mode,
            post_process_result=self._post_process_result,
        )

    def _get_model_instance(
        self,
        model: str,
        **kwargs,
    ) -> BaseOCRModel:
        model_info = MODEL_REGISTRY.get(model)
        if not model_info:
            raise ValueError(f"Unknown model: {model}")
        model_class = model_info["class"]
        required_fields = {
            field: kwargs.get(field) for field in model_info.get("required_fields", [])
        }
        return model_class(
            model_manager=self._model_manager, api_key=self._api_key, **required_fields
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        for prediction, image in zip(predictions, images):
            prediction[PREDICTION_TYPE_KEY] = "ocr"
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
            _ = remove_unexpected_keys_from_dictionary(
                dictionary=prediction,
                expected_keys=EXPECTED_OUTPUT_KEYS,
            )
        return predictions
