import hashlib
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.cache.lru_cache import LRUCache
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
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
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.entities import InferenceConfiguration

# These are the displayed languages in the UI dropdown
LANGUAGES = Literal[
    "English",
    "Japanese",
    "Kannada",
    "Korean",
    "Latin",
    "Telugu",
    "Simplified Chinese",
]

# Dictionary of displayed_language: (model, language_code)
# This is not an extensive list of supported languages, more codes can be added
MODELS: Dict[str, Tuple[str, List[str]]] = {
    "English": ("english_g2", ["en"]),
    "Japanese": ("japanese_g2", ["en", "ja"]),
    "Kannada": ("kannada_g2", ["en", "kn"]),
    "Korean": ("korean_g2", ["en", "ko"]),
    "Latin": ("latin_g2", ["en", "la", "es", "fr", "it", "pt", "de", "pl", "nl"]),
    "Telugu": ("telugu_g2", ["en", "te"]),
    "Simplified Chinese": ("zh_sim_g2", ["en", "ch_sim"]),
}

LONG_DESCRIPTION = """
Extract text from images using EasyOCR Optical Character Recognition (OCR), supporting multiple languages and returning both extracted text and bounding boxes for each detected text region.

## How This Block Works

This block uses the EasyOCR model to detect and extract text content from one or more images, with support for multiple languages. The block:

1. Takes images as input (supports batch processing)
2. Uses EasyOCR with the selected language model to detect text regions and recognize characters
3. Generates bounding boxes around each detected text region
4. Extracts the recognized text content from each region
5. Returns both a concatenated text string (all text found in the image) and structured predictions with bounding box coordinates for each text region

The block supports multiple languages including English, Japanese, Korean, Simplified Chinese, Latin (with support for Spanish, French, Italian, Portuguese, German, Polish, Dutch), Kannada, and Telugu. Each language model is optimized for that specific language, improving accuracy for multilingual applications. You can also enable quantized models for faster inference at the cost of slightly reduced accuracy.

## Common Use Cases

- **Multilingual Document Processing**: Extract text from documents in various languages including English, Japanese, Korean, Chinese, and European languages for international document processing applications
- **International Product Labels**: Read product labels, ingredients, or instructions in multiple languages for global inventory or compliance checking
- **Multilingual Signage**: Extract text from signs, advertisements, or public notices in different languages for translation services or content analysis
- **International ID and Certificates**: Read information from ID cards, passports, or certificates in various languages for identity verification systems
- **Global Logistics**: Read shipping labels, container IDs, or package information in multiple languages for international logistics automation
- **Content Localization**: Extract text from images for translation workflows, helping identify which language content is in before processing

## Connecting to Other Blocks

The extracted text and text detections from this block can be connected to:

- **Object detection blocks** (e.g., Object Detection Model) combined with crop blocks (e.g., Dynamic Crop) to first isolate specific regions containing text before running OCR, improving accuracy by focusing on relevant areas
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log extracted text and metadata for record-keeping or analysis
- **Expression blocks** to parse, validate, or transform extracted text using regular expressions or string operations
- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on whether specific text patterns are found or text content matches certain criteria
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send alerts when specific text is detected (e.g., error messages, warning labels, or important identifiers)
- **Webhook blocks** to send extracted text data to external systems or APIs for further processing

## Requirements

Note that EasyOCR has limitations running within containers on Apple Silicon (M1/M2/M3 Macs). For Apple Silicon deployments, consider using the OCR Model block (DocTR) instead.
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
            "name": "EasyOCR",
            "version": "v1",
            "short_description": "Extract text from an image using EasyOCR optical character recognition.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-text",
                "blockPriority": 11,
                "inDevelopment": False,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/easy_ocr@v1", "EasyOCR"]
    name: str = Field(description="Unique name of step in workflows")
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    language: LANGUAGES = Field(
        title="Language",
        description="Language model to use for OCR. Select from: English, Japanese, Korean, Simplified Chinese, Latin (supports Spanish, French, Italian, Portuguese, German, Polish, Dutch), Kannada, or Telugu. Each language model is optimized for that specific language, improving accuracy for text in that language. The Latin option supports multiple European languages. Defaults to English.",
        default="English",
    )
    quantize: bool = Field(
        title="Use Quantized Model",
        description="Enable quantized model for faster inference and lower memory usage. Quantized models are smaller and faster, but may be less accurate than full precision models. May not work correctly on all hardware configurations. Recommended for production deployments where speed is prioritized over maximum accuracy. Defaults to False (full precision model).",
        default=False,
    )

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


class EasyOCRBlockV1(WorkflowBlock):
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
        language: LANGUAGES = "English",
        quantize: bool = False,
    ) -> BlockResult:

        if language not in MODELS:
            raise ValueError(f"Unsupported language: {language}")

        version, language_codes = MODELS.get(language, "english_g2")
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                language_codes=language_codes,
                version=version,
                quantize=quantize,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                language_codes=language_codes,
                version=version,
                quantize=quantize,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        language_codes: List[str],
        version: str = "english_g2",
        quantize: bool = False,
    ) -> BlockResult:

        predictions = []
        for single_image in images:

            inference_request = EasyOCRInferenceRequest(
                easy_ocr_version_id=version,
                image=single_image.to_inference_format(numpy_preferred=True),
                api_key=self._api_key,
                language_codes=language_codes,
                quantize=quantize,
            )
            model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="easy_ocr",
            )
            result = self._model_manager.infer_from_request_sync(
                model_id, inference_request
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
        language_codes: List[str],
        version: str = "english_g2",
        quantize: bool = False,
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
            model="easy_ocr",
            version=version,
            quantize=quantize,
            language_codes=language_codes,
        )
        if len(images) == 1:
            predictions = [predictions]
        return post_process_ocr_result(
            predictions=predictions,
            images=images,
            expected_output_keys=EXPECTED_OUTPUT_KEYS,
        )
