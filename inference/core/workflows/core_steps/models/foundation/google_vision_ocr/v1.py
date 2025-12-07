from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

import numpy as np
import requests
import supervision as sv
from pydantic import ConfigDict, Field
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.roboflow_api import post_to_roboflow_api
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_MANAGED_KEY,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Detect text in images using Google Vision OCR.

Supported types of text detection:

- `text_detection`: optimized for areas of text within a larger image.
- `ocr_text_detection`: optimized for dense text documents.

Provide your Google Vision API key or set the value to ``rf_key:account`` (or
``rf_key:user:<id>``) to proxy requests through Roboflow's API.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Google Vision OCR",
            "version": "v1",
            "short_description": "Detect text in images using Google Vision API",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "fa-brands fa-google",
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/google_vision_ocr@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Image to run OCR",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    ocr_type: Literal["text_detection", "ocr_text_detection"] = Field(
        description="Type of OCR to use",
        json_schema_extra={
            "values_metadata": {
                "text_detection": {
                    "name": "Any Scene Text Detection",
                    "description": "Detects and extracts text from any image, including photographs that contain blocks of text.",
                },
                "ocr_text_detection": {
                    "name": "Document Text Detection",
                    "description": "Optimized for dense text documents, such as scanned pages or photographs of printed text.",
                },
            },
        },
    )
    api_key: Union[
        Selector(kind=[STRING_KIND, SECRET_KIND, ROBOFLOW_MANAGED_KEY]), str
    ] = Field(
        default="rf_key:account",
        description="Your Google Vision API key",
        examples=["xxx-xxx", "$inputs.google_api_key"],
        private=True,
    )
    language_hints: Optional[List[str]] = Field(
        default=None,
        description="Optional list of language codes to pass to the OCR API. If not provided, the API will attempt to detect the language automatically."
        "If provided, language codes must be supported by the OCR API, visit https://cloud.google.com/vision/docs/languages for list of supported language codes.",
        examples=[["en", "fr"], ["de"]],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="text", kind=[STRING_KIND]),
            OutputDefinition(name="language", kind=[STRING_KIND]),
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class GoogleVisionOCRBlockV1(WorkflowBlock):

    def __init__(
        self,
        api_key: Optional[str],
    ):
        self._roboflow_api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        ocr_type: Literal["text_detection", "ocr_text_detection"],
        language_hints: Optional[List[str]],
        api_key: str = "rf_key:account",
    ) -> BlockResult:
        # Decide which type of OCR to use
        if ocr_type == "text_detection":
            detection_type = "TEXT_DETECTION"
        elif ocr_type == "ocr_text_detection":
            detection_type = "DOCUMENT_TEXT_DETECTION"
        else:
            raise ValueError(f"Invalid ocr_type: {ocr_type}")

        request_json = _build_request_json(
            image=image,
            detection_type=detection_type,
            language_hints=language_hints,
        )

        # Route to proxy or direct API based on api_key format
        if api_key.startswith(("rf_key:account", "rf_key:user:")):
            result = _execute_proxied_google_vision_request(
                roboflow_api_key=self._roboflow_api_key,
                google_vision_api_key=api_key,
                request_json=request_json,
            )
        else:
            result = _execute_google_vision_request(
                api_key=api_key,
                request_json=request_json,
            )

        return _parse_google_vision_response(result=result, image=image)


def _build_request_json(
    image: WorkflowImageData,
    detection_type: str,
    language_hints: Optional[List[str]],
) -> dict:
    ocr_request = {
        "image": {"content": image.base64_image},
        "features": [{"type": detection_type}],
    }

    if language_hints is not None:
        ocr_request["imageContext"] = {"languageHints": language_hints}

    return {"requests": [ocr_request]}


def _execute_proxied_google_vision_request(
    roboflow_api_key: str,
    google_vision_api_key: str,
    request_json: dict,
) -> dict:
    payload = {
        "google_vision_api_key": google_vision_api_key,
        "request_json": request_json,
    }

    try:
        response_data = post_to_roboflow_api(
            endpoint="apiproxy/google_vision_ocr",
            api_key=roboflow_api_key,
            payload=payload,
        )
        return response_data["responses"][0]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to Roboflow proxy: {e}") from e
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Invalid response structure from Roboflow proxy: {e}"
        ) from e


def _execute_google_vision_request(
    api_key: str,
    request_json: dict,
) -> dict:
    response = requests.post(
        "https://vision.googleapis.com/v1/images:annotate",
        params={"key": api_key},
        json=request_json,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Request to Google Cloud Vision API failed: {str(response.json())}"
        )

    return response.json()["responses"][0]


def _parse_google_vision_response(
    result: dict,
    image: WorkflowImageData,
) -> BlockResult:
    # Check for image without text
    if "textAnnotations" not in result or not result["textAnnotations"]:
        return {
            "text": "",
            "language": "",
            "predictions": sv.Detections.empty(),
        }

    # Extract predictions from the response
    text = result["textAnnotations"][0]["description"]
    language = result["textAnnotations"][0]["locale"]

    xyxy = []
    confidence = []
    classes = []
    detections_id = []

    for page in result["fullTextAnnotation"]["pages"]:
        for block in page["blocks"]:
            # Get bounding box coordinates
            box = block["boundingBox"]["vertices"]
            x_min = min(v.get("x", 0) for v in box)
            y_min = min(v.get("y", 0) for v in box)
            x_max = max(v.get("x", 0) for v in box)
            y_max = max(v.get("y", 0) for v in box)
            xyxy.append([x_min, y_min, x_max, y_max])

            # Only DOCUMENT_TEXT_DETECTION provides confidence score, use 1.0 otherwise
            confidence.append(block.get("confidence", 1.0))

            # Get block text
            block_text = []
            for paragraph in block["paragraphs"]:
                for word in paragraph["words"]:
                    word_text = "".join(symbol["text"] for symbol in word["symbols"])
                    block_text.append(word_text)
            classes.append(" ".join(block_text))

            # Create unique detection id for each block
            detections_id.append(uuid4())

    predictions = sv.Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(confidence),
        class_id=np.arange(len(classes)),
        data={CLASS_NAME_DATA_FIELD: np.array(classes)},
    )

    predictions[DETECTION_ID_KEY] = np.array(detections_id)
    predictions[PREDICTION_TYPE_KEY] = np.array(["ocr"] * len(predictions))
    image_height, image_width = image.numpy_image.shape[:2]
    predictions[IMAGE_DIMENSIONS_KEY] = np.array(
        [[image_height, image_width]] * len(predictions)
    )

    predictions = attach_parents_coordinates_to_sv_detections(
        detections=predictions,
        image=image,
    )

    return {
        "text": text,
        "language": language,
        "predictions": predictions,
    }
