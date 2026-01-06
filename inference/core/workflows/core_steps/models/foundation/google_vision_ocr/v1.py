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
Extract text from images using Google Cloud Vision API's OCR service, with support for scene text detection and dense document processing, returning extracted text, language detection, and bounding box predictions.

## How This Block Works

This block uses Google Cloud Vision API to detect and extract text from images. The block:

1. Takes an image as input and selects one of two OCR modes based on the image type
2. Sends the image to Google Cloud Vision API (either directly or proxied through Roboflow)
3. The API detects text regions, recognizes characters, and identifies the language
4. Returns the extracted text, detected language code, and structured predictions with bounding boxes for each text block

The block supports two OCR modes: **text_detection** (optimized for scene text in photographs, signs, or images with text) and **ocr_text_detection** (optimized for dense text documents like scanned pages or printed text). The document mode provides confidence scores for each detected text block, while scene text mode is better suited for natural images containing text. You can optionally provide language hints to improve accuracy for specific languages, or let the API automatically detect the language.

## Common Use Cases

- **Document Digitization**: Extract text from scanned documents, PDFs, receipts, invoices, forms, or printed materials for automated data entry or archival
- **Scene Text Recognition**: Read text from natural images such as street signs, storefronts, billboards, or product packaging in photographs
- **Multi-language Text Extraction**: Extract text from documents in various languages with automatic language detection or language hints for improved accuracy
- **Compliance and Record-Keeping**: Digitize and extract text from legal documents, certificates, IDs, or official records for searchable archives
- **Receipt and Invoice Processing**: Extract structured information from receipts and invoices for expense tracking, accounting, or financial automation
- **Content Analysis**: Extract and analyze text from images for content moderation, translation workflows, or information extraction

## Connecting to Other Blocks

The extracted text and text detections from this block can be connected to:

- **Object detection blocks** combined with crop blocks (e.g., Dynamic Crop) to first isolate specific regions containing text before running OCR, improving accuracy by focusing on relevant areas
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log extracted text, language, and metadata for record-keeping or analysis
- **Expression blocks** to parse, validate, or transform extracted text using regular expressions or string operations
- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on detected language, text content, or specific text patterns
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send alerts when specific text is detected or when language detection occurs
- **Webhook blocks** to send extracted text data to external systems or APIs for further processing or integration

## Requirements

This block requires a Google Cloud Vision API key. You can either provide your own API key directly or use `rf_key:account` (or `rf_key:user:<id>`) to proxy requests through Roboflow's API. The OCR operation is performed remotely via Google Cloud Vision API, so an active internet connection is required.
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
        description="The input image from which to extract text. Can be a natural scene image, scanned document, or any image containing text. The image will be processed by Google Cloud Vision API based on the selected OCR type.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    ocr_type: Literal["text_detection", "ocr_text_detection"] = Field(
        description="Type of OCR mode to use. 'text_detection' (Any Scene Text Detection) is optimized for detecting text in natural images like photographs, signs, or images with text blocks. 'ocr_text_detection' (Document Text Detection) is optimized for dense text documents like scanned pages, printed text, or documents. Document mode provides confidence scores for each text block, while scene text mode is better for natural images.",
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
        description="Your Google Cloud Vision API key. You can provide your own API key directly, or use 'rf_key:account' (or 'rf_key:user:<id>') to proxy requests through Roboflow's API. To obtain a Google Cloud Vision API key, visit https://cloud.google.com/vision/docs/setup. This field is kept private for security.",
        examples=[
            "your-api-key-here",
            "rf_key:account",
            "$inputs.google_api_key",
            "$secrets.google_vision_key",
        ],
        private=True,
    )
    language_hints: Optional[List[str]] = Field(
        default=None,
        description="Optional list of language codes (BCP-47 format) to help improve OCR accuracy for specific languages. If provided, the API will prioritize these languages when detecting and recognizing text. If not provided, the API will automatically detect the language. Common examples include 'en' (English), 'es' (Spanish), 'fr' (French), 'de' (German), 'ja' (Japanese), 'zh' (Chinese). You can provide multiple language hints if the document contains mixed languages. For a complete list of supported language codes, visit https://cloud.google.com/vision/docs/languages.",
        examples=[["en"], ["en", "es"], ["ja"], ["de", "fr"]],
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
