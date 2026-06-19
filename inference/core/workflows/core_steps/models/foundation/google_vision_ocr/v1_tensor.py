"""Tensor-native sibling of `roboflow_core/google_vision_ocr@v1`.

Calls the Google Vision API (directly or via the Roboflow proxy) on every path —
there is no inference_models adapter here — so the ONLY change is the OUTPUT
representation of the `predictions` kind: instead of `sv.Detections` (with the
recognized text stored as the per-box class name in `.data`) this builds an
`inference_models.Detections` that PRESERVES the numpy per-box semantics:

- the recognized TEXT per block is carried PER BOX in
  ``bboxes_metadata[i][CLASS_NAME_KEY]`` (== "class") so the serializer (C1) emits it
  as ``predictions[i]["class"]`` — exactly as the numpy sibling, which stored the text
  as the per-box class name;
- a non-empty ``image_metadata[CLASS_NAMES_KEY] = {0: "text-region"}`` map is still
  attached as the serializer's class-id fallback (every box carries ``class_id == 0``);
- a ``detection_id`` per box;
- image lineage / dimensions via ``build_native_image_metadata``.

Boxes are built on ``WORKFLOWS_IMAGE_TENSOR_DEVICE``. The google-vision-specific
``text`` (full recognized text) and ``language`` outputs are preserved exactly.
The output kind becomes the tensor-native object-detection kind.
"""

from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

import requests
import torch
from pydantic import ConfigDict, Field

from inference.core.env import WORKFLOWS_IMAGE_TENSOR_DEVICE
from inference.core.roboflow_api import post_to_roboflow_api
from inference.core.workflows.core_steps.common.tensor_native import (
    build_native_image_metadata,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    DETECTION_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    ROBOFLOW_MANAGED_KEY,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.object_detection import Detections

# Google Vision returns blocks of recognized text. To preserve numpy parity for the
# serialized output, the recognized text per block is carried PER BOX in
# `bboxes_metadata[i][CLASS_NAME_KEY]` (== "class"); the serializer (C1) emits that
# value as `predictions[i]["class"]` (matching the numpy sibling, which stored the
# text as the per-box class name). A non-empty `CLASS_NAMES_KEY` map is still
# required on image_metadata as the serializer's class-id fallback.
CLASS_NAMES = {0: "text-region"}
PREDICTION_TYPE = "ocr"

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
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="text", kind=[STRING_KIND]),
            OutputDefinition(name="language", kind=[STRING_KIND]),
            OutputDefinition(
                name="predictions",
                kind=[TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND],
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
            "predictions": _build_native_detections(
                image=image, xyxy=[], confidence=[], texts=[]
            ),
        }

    # Extract predictions from the response
    text = result["textAnnotations"][0]["description"]
    language = result["textAnnotations"][0]["locale"]

    xyxy: List[List[float]] = []
    confidence: List[float] = []
    texts: List[str] = []

    for page in result["fullTextAnnotation"]["pages"]:
        for block in page["blocks"]:
            # Get bounding box coordinates
            box = block["boundingBox"]["vertices"]
            x_min = min(v.get("x", 0) for v in box)
            y_min = min(v.get("y", 0) for v in box)
            x_max = max(v.get("x", 0) for v in box)
            y_max = max(v.get("y", 0) for v in box)
            xyxy.append([float(x_min), float(y_min), float(x_max), float(y_max)])

            # Only DOCUMENT_TEXT_DETECTION provides confidence score, use 1.0 otherwise
            confidence.append(float(block.get("confidence", 1.0)))

            # Get block text
            block_text = []
            for paragraph in block["paragraphs"]:
                for word in paragraph["words"]:
                    word_text = "".join(symbol["text"] for symbol in word["symbols"])
                    block_text.append(word_text)
            texts.append(" ".join(block_text))

    return {
        "text": text,
        "language": language,
        "predictions": _build_native_detections(
            image=image, xyxy=xyxy, confidence=confidence, texts=texts
        ),
    }


def _build_native_detections(
    image: WorkflowImageData,
    xyxy: List[List[float]],
    confidence: List[float],
    texts: List[str],
) -> Detections:
    """Build an `inference_models.Detections` from the parsed Google Vision blocks.

    Every box shares ``class_id == 0`` (the image_metadata class map is the
    serializer's fallback); the recognized text per block is carried PER BOX in
    ``bboxes_metadata[i][CLASS_NAME_KEY]`` so the serializer emits it as the box's
    ``class`` (numpy parity), and each box gets a ``detection_id``. Tensors are
    built on ``WORKFLOWS_IMAGE_TENSOR_DEVICE``.
    """
    number_of_detections = len(xyxy)
    detections = Detections(
        xyxy=(
            torch.tensor(
                xyxy, dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
            )
            if number_of_detections
            else torch.zeros(
                (0, 4), dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
            )
        ),
        class_id=torch.zeros(
            (number_of_detections,),
            dtype=torch.int64,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        confidence=torch.tensor(
            confidence, dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
    )
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names=CLASS_NAMES,
        prediction_type=PREDICTION_TYPE,
    )
    if number_of_detections == 0:
        detections.bboxes_metadata = None
        return detections
    detections.bboxes_metadata = [
        {DETECTION_ID_KEY: str(uuid4()), CLASS_NAME_KEY: texts[index]}
        for index in range(number_of_detections)
    ]
    return detections
