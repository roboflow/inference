import json
from typing import Dict, List, Literal, Optional, Type
from urllib import response
from uuid import uuid4

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
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
    load_core_model,
    post_process_ocr_result,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
Extract text from images using DocTR Optical Character Recognition (OCR), returning both the extracted text and bounding boxes for each detected text region.

## How This Block Works

This block uses the DocTR OCR model to detect and extract all text content from one or more images. The block:

1. Takes images as input (supports batch processing)
2. Uses DocTR to detect text regions and recognize characters in the image
3. Generates bounding boxes around each detected text region
4. Extracts the recognized text content from each region
5. Returns both a concatenated text string (all text found in the image) and structured predictions with bounding box coordinates for each text region

The block outputs both a plain text string containing all extracted text and structured detection predictions that include bounding boxes, allowing you to know not just what text was found, but where it appears in the image. This makes it useful for workflows that need to process or validate specific text locations.

## Common Use Cases

- **Document Processing**: Extract text from scanned documents, PDFs, receipts, invoices, or forms for automated data entry or document digitization
- **License Plate Recognition**: Read vehicle license plates from images for parking, toll, or security applications
- **Product Label Reading**: Extract text from product labels, barcodes, serial numbers, or expiration dates for inventory management
- **Signage and Street Signs**: Read text from street signs, store signs, or directional signage for mapping or navigation applications
- **ID and Certificate Reading**: Extract information from ID cards, certificates, or official documents for verification or record-keeping
- **Logistics and Shipping**: Read shipping labels, container IDs, package tracking numbers, or shipping manifests for logistics automation

## Connecting to Other Blocks

The extracted text and text detections from this block can be connected to:

- **Object detection blocks** (e.g., Object Detection Model) combined with crop blocks (e.g., Dynamic Crop) to first isolate specific regions containing text before running OCR, improving accuracy by focusing on relevant areas
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log extracted text and metadata for record-keeping or analysis
- **Expression blocks** to parse, validate, or transform extracted text using regular expressions or string operations
- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on whether specific text patterns are found or text content matches certain criteria
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send alerts when specific text is detected (e.g., error messages, warning labels, or important identifiers)
- **Webhook blocks** to send extracted text data to external systems or APIs for further processing
"""

EXPECTED_OUTPUT_KEYS = {
    "result",
    "parent_id",
    "root_parent_id",
    "prediction_type",
    "predictions",
}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OCR Model",
            "version": "v1",
            "short_description": "Extract text from an image using DocTR optical character recognition.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-text",
                "blockPriority": 11,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/ocr_model@v1", "OCRModel"]
    name: str = Field(description="Unique name of step in workflows")
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="result", kind=[STRING_KIND]),
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class OCRModelBlockV1(WorkflowBlock):
    # TODO: we need data model for OCR predictions

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
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(images=images)
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(images=images)
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
    ) -> BlockResult:
        predictions = []
        for single_image in images:
            inference_request = DoctrOCRInferenceRequest(
                image=single_image.to_inference_format(numpy_preferred=True),
                api_key=self._api_key,
                generate_bounding_boxes=True,
            )
            doctr_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="doctr",
            )
            result = self._model_manager.infer_from_request_sync(
                doctr_model_id, inference_request
            )
            predictions.append(result.model_dump(by_alias=True, exclude_none=True))
        return post_process_ocr_result(
            predictions=predictions,
            images=images,
            expected_output_keys=EXPECTED_OUTPUT_KEYS,
        )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CORE_MODEL_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        configuration = InferenceConfiguration(
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
        client.configure(configuration)
        non_empty_inference_images = [i.base64_image for i in images]
        predictions = client.ocr_image(
            inference_input=non_empty_inference_images,
            generate_bounding_boxes=True,
        )
        if len(images) == 1:
            predictions = [predictions]
        return post_process_ocr_result(
            predictions=predictions,
            images=images,
            expected_output_keys=EXPECTED_OUTPUT_KEYS,
        )
