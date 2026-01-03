import hashlib
import json
import logging
import re
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field, model_validator
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.core_steps.common.vlms import VLM_TASKS_METADATA
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json([\s\S]*?)```", flags=re.IGNORECASE)

LONG_DESCRIPTION = """
Parse JSON strings from Visual Language Models (VLMs) and Large Language Models (LLMs) into standardized object detection prediction format by extracting bounding boxes, class names, and confidences, converting normalized coordinates to pixel coordinates, mapping class names to class IDs, and handling multiple model types and task formats to enable VLM-based object detection, LLM detection parsing, and text-to-detection conversion workflows.

## How This Block Works

This block converts VLM/LLM text outputs containing object detection predictions into standardized object detection format compatible with workflow detection blocks. The block:

1. Receives image and VLM output string containing detection results in JSON format
2. Parses JSON content from VLM output:

   **Handles Markdown-wrapped JSON:**
   - Searches for JSON wrapped in Markdown code blocks (```json ... ```)
   - This format is common in LLM/VLM responses
   - If multiple markdown JSON blocks are found, only the first block is parsed
   - Extracts JSON content from within markdown tags

   **Handles raw JSON strings:**
   - If no markdown blocks are found, attempts to parse the entire string as JSON
   - Supports standard JSON format strings
3. Selects appropriate parser based on model type and task type:
   - Uses registered parsers that handle different model outputs (google-gemini, anthropic-claude, florence-2, openai)
   - Supports multiple task types: object-detection, open-vocabulary-object-detection, object-detection-and-caption, phrase-grounded-object-detection, region-proposal, ocr-with-text-detection
   - Each model/task combination uses a specialized parser for that format
4. Parses detection data based on model type:

   **For OpenAI/Gemini/Claude models:**
   - Extracts detections array from parsed JSON
   - Converts normalized coordinates (0-1 range) to pixel coordinates using image dimensions
   - Extracts class names, confidence scores, and bounding box coordinates
   - Maps class names to class IDs using provided classes list
   - Creates detection objects with bounding boxes, classes, and confidences

   **For Florence-2 model:**
   - Uses supervision's built-in LMM parser for Florence-2 format
   - Handles different task types with specialized parsing (object detection, open vocabulary, region proposal, OCR, etc.)
   - For region proposal tasks: assigns "roi" as class name
   - For open vocabulary detection: uses provided classes list for class ID mapping
   - For other tasks: uses MD5-based class ID generation or provided classes
   - Sets confidence to 1.0 for Florence-2 detections (model doesn't provide confidence)
5. Converts coordinates and normalizes data:
   - Converts normalized coordinates (0-1) to absolute pixel coordinates (x_min, y_min, x_max, y_max)
   - Scales coordinates using image width and height
   - Normalizes confidence scores to valid range [0.0, 1.0]
   - Clamps confidence values outside the range
6. Creates class name to class ID mapping:
   - For OpenAI/Gemini/Claude: uses provided classes list to create index mapping (class_name → class_id)
   - Classes are mapped in order (first class = ID 0, second = ID 1, etc.)
   - Classes not in the provided list get class_id = -1
   - For Florence-2: uses different mapping strategies based on task type
7. Constructs object detection predictions:
   - Creates supervision Detections objects with bounding boxes (xyxy format)
   - Includes class IDs, class names, and confidence scores
   - Adds metadata: detection IDs, inference IDs, image dimensions, prediction type
   - Attaches parent coordinates for crop-aware detections
   - Formats predictions in standard object detection format
8. Handles errors:
   - Sets `error_status` to True if JSON parsing fails
   - Sets `error_status` to True if detection parsing fails
   - Returns None for predictions when errors occur
   - Always includes inference_id for tracking
9. Returns object detection predictions:
   - Outputs `predictions` in standard object detection format (compatible with detection blocks)
   - Outputs `error_status` indicating parsing success/failure
   - Outputs `inference_id` for tracking and lineage

The block enables using VLMs/LLMs for object detection by converting their text-based JSON outputs into standardized detection predictions that can be used in workflows like any other object detection model output.

## Common Use Cases

- **VLM-Based Object Detection**: Use Visual Language Models for object detection by parsing VLM outputs into detection predictions (e.g., detect objects with GPT-4V, use Claude Vision for detection, parse Gemini detection outputs), enabling VLM detection workflows
- **Open-Vocabulary Detection**: Use VLMs for open-vocabulary object detection with custom classes (e.g., detect custom objects with VLMs, use open-vocabulary detection, detect objects not in training set), enabling open-vocabulary detection workflows
- **Multi-Task Detection**: Use VLMs for various detection tasks (e.g., object detection with captions, phrase-grounded detection, region proposal, OCR with detection), enabling multi-task detection workflows
- **LLM Detection Parsing**: Parse LLM text outputs containing detection results into standardized format (e.g., parse GPT detection outputs, convert LLM predictions to detection format, use LLMs for detection), enabling LLM detection workflows
- **Text-to-Detection Conversion**: Convert text-based detection outputs from models into workflow-compatible detection predictions (e.g., convert text predictions to detection format, parse text-based detections, convert model outputs to detections), enabling text-to-detection workflows
- **VLM Integration**: Integrate VLM outputs into detection workflows (e.g., use VLMs in detection pipelines, integrate VLM predictions with detection blocks, combine VLM and traditional detection), enabling VLM integration workflows

## Connecting to Other Blocks

This block receives images and VLM outputs and produces object detection predictions:

- **After VLM/LLM blocks** to parse detection outputs into standard format (e.g., VLM output to detections, LLM output to detections, parse model outputs), enabling VLM-to-detection workflows
- **Before detection-based blocks** to use parsed detections (e.g., use parsed detections in workflows, provide detections to downstream blocks, use VLM detections with detection blocks), enabling detection-to-workflow workflows
- **Before filtering blocks** to filter VLM detections (e.g., filter by class, filter by confidence, apply filters to VLM predictions), enabling detection-to-filter workflows
- **Before analytics blocks** to analyze VLM detection results (e.g., analyze VLM detections, perform analytics on parsed detections, track VLM detection metrics), enabling detection analytics workflows
- **Before visualization blocks** to display VLM detection results (e.g., visualize VLM detections, display parsed detection predictions, show VLM detection outputs), enabling detection visualization workflows
- **In workflow outputs** to provide VLM detections as final output (e.g., VLM detection outputs, parsed detection results, VLM-based detection outputs), enabling detection output workflows

## Version Differences

This version (v2) includes the following enhancements over v1:

- **Improved Type System**: The `inference_id` output now uses `INFERENCE_ID_KIND` instead of `STRING_KIND`, providing better type safety and semantic meaning for inference tracking identifiers in the workflow system
- **OpenAI Model Support**: Added support for OpenAI models in addition to Google Gemini, Anthropic Claude, and Florence-2 models, expanding the range of VLM/LLM models that can be used for object detection
- **Enhanced Type Safety**: Improved type system ensures better integration with workflow execution engine and provides clearer semantic meaning for inference tracking

## Requirements

This block requires an image input (for metadata and dimensions) and a VLM output string containing JSON detection data. The JSON can be raw JSON or wrapped in Markdown code blocks (```json ... ```). The block supports four model types: "openai", "google-gemini", "anthropic-claude", and "florence-2". It supports multiple task types: "object-detection", "open-vocabulary-object-detection", "object-detection-and-caption", "phrase-grounded-object-detection", "region-proposal", and "ocr-with-text-detection". The `classes` parameter is required for OpenAI, Gemini, and Claude models (to map class names to IDs) but optional for Florence-2 (some tasks don't require it). Classes are mapped to IDs by index (first class = 0, second = 1, etc.). Classes not in the list get class_id = -1. The block outputs object detection predictions in standard format (compatible with detection blocks), error_status (boolean), and inference_id (INFERENCE_ID_KIND) for tracking.
"""

SHORT_DESCRIPTION = "Parses raw string into object-detection prediction."

SUPPORTED_TASKS = {
    "object-detection",
    "object-detection-and-caption",
    "open-vocabulary-object-detection",
    "phrase-grounded-object-detection",
    "region-proposal",
    "ocr-with-text-detection",
}
RELEVANT_TASKS_METADATA = {
    k: v for k, v in VLM_TASKS_METADATA.items() if k in SUPPORTED_TASKS
}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "VLM as Detector",
            "version": "v2",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-object-ungroup",
                "blockPriority": 5,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/vlm_as_detector@v2"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Input image that was used to generate the VLM prediction. Used to extract image dimensions (width, height) for converting normalized coordinates to pixel coordinates and metadata (parent_id) for the detection predictions. The same image that was provided to the VLM/LLM block should be used here to maintain consistency.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    vlm_output: Selector(kind=[LANGUAGE_MODEL_OUTPUT_KIND]) = Field(
        title="VLM Output",
        description="String output from a VLM or LLM block containing object detection prediction in JSON format. Can be raw JSON string or JSON wrapped in Markdown code blocks (e.g., ```json {...} ```). Format depends on model_type and task_type - different models and tasks produce different JSON structures. If multiple markdown blocks exist, only the first is parsed.",
        examples=[["$steps.lmm.output"], ["$steps.vlm.output"], ["$steps.claude.output"]],
    )
    classes: Optional[
        Union[
            Selector(kind=[LIST_OF_VALUES_KIND]),
            Selector(kind=[LIST_OF_VALUES_KIND]),
            List[str],
        ]
    ] = Field(
        description="List of all class names used by the classification model, in order. Required to generate mapping between class names (from VLM output) and class IDs (for detection format). Classes are mapped to IDs by index: first class = ID 0, second = ID 1, etc. Classes from VLM output that are not in this list get class_id = -1. Required for OpenAI, Gemini, and Claude models. Optional for Florence-2 (some tasks don't require it). Should match the classes the VLM was asked to detect.",
        examples=[["$steps.lmm.classes", "$inputs.classes", ["dog", "cat", "bird"], ["class_a", "class_b"]]],
        json_schema_extra={
            "relevant_for": {
                "model_type": {
                    "values": ["openai", "google-gemini", "anthropic-claude"],
                    "required": True,
                },
            }
        },
    )
    model_type: Literal["openai", "google-gemini", "anthropic-claude", "florence-2"] = (
        Field(
            description="Type of the VLM/LLM model that generated the prediction. Determines which parser is used to extract detection data from the JSON output. Supported models: 'openai' (GPT-4V), 'google-gemini' (Gemini Vision), 'anthropic-claude' (Claude Vision), 'florence-2' (Microsoft Florence-2). Each model type has different JSON output formats, so the correct model type must be specified for proper parsing.",
            examples=[["openai"], ["google-gemini"], ["anthropic-claude"], ["florence-2"]],
        )
    )
    task_type: Literal[tuple(SUPPORTED_TASKS)] = Field(
        description="Task type performed by the VLM/LLM model. Determines which parser and format handler is used. Supported task types: 'object-detection' (standard object detection), 'open-vocabulary-object-detection' (detect objects with custom classes), 'object-detection-and-caption' (detection with captions), 'phrase-grounded-object-detection' (ground phrases to detections), 'region-proposal' (propose regions of interest), 'ocr-with-text-detection' (OCR with text region detection). The task type must match what the VLM/LLM was asked to perform.",
        json_schema_extra={
            "values_metadata": RELEVANT_TASKS_METADATA,
        },
    )

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        if (self.model_type, self.task_type) not in REGISTERED_PARSERS:
            raise ValueError(
                f"Could not parse result of task {self.task_type} for model {self.model_type}"
            )
        if self.model_type != "florence-2" and self.classes is None:
            raise ValueError(
                "Must pass list of classes to this block when using gemini or claude"
            )

        return self

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="inference_id", kind=[INFERENCE_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class VLMAsDetectorBlockV2(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        vlm_output: str,
        classes: Optional[List[str]],
        model_type: str,
        task_type: str,
    ) -> BlockResult:
        inference_id = f"{uuid4()}"
        error_status, parsed_data = string2json(
            raw_json=vlm_output,
        )
        if error_status:
            return {
                "error_status": True,
                "predictions": None,
                "inference_id": inference_id,
            }
        try:
            predictions = REGISTERED_PARSERS[(model_type, task_type)](
                image=image,
                parsed_data=parsed_data,
                classes=classes,
                inference_id=inference_id,
            )
            return {
                "error_status": False,
                "predictions": predictions,
                "inference_id": inference_id,
            }
        except Exception as error:
            logging.warning(
                f"Could not parse VLM prediction for model {model_type} and task {task_type} "
                f"in `roboflow_core/vlm_as_detector@v1` block. "
                f"Error type: {error.__class__.__name__}. Details: {error}"
            )
            return {
                "error_status": True,
                "predictions": None,
                "inference_id": inference_id,
            }


def string2json(
    raw_json: str,
) -> Tuple[bool, dict]:
    json_blocks_found = JSON_MARKDOWN_BLOCK_PATTERN.findall(raw_json)
    if len(json_blocks_found) == 0:
        return try_parse_json(raw_json)
    first_block = json_blocks_found[0]
    return try_parse_json(first_block)


def try_parse_json(content: str) -> Tuple[bool, dict]:
    try:
        return False, json.loads(content)
    except Exception as error:
        logging.warning(
            f"Could not parse JSON to dict in `roboflow_core/vlm_as_detector@v1` block. "
            f"Error type: {error.__class__.__name__}. Details: {error}"
        )
        return True, {}


def parse_llm_object_detection_response(
    image: WorkflowImageData,
    parsed_data: dict,
    classes: List[str],
    inference_id: str,
) -> sv.Detections:
    class_name2id = create_classes_index(classes=classes)
    image_height, image_width = image.numpy_image.shape[:2]
    if len(parsed_data["detections"]) == 0:
        return sv.Detections.empty()
    xyxy, class_id, class_name, confidence = [], [], [], []
    for detection in parsed_data["detections"]:
        xyxy.append(
            [
                detection["x_min"] * image_width,
                detection["y_min"] * image_height,
                detection["x_max"] * image_width,
                detection["y_max"] * image_height,
            ]
        )
        class_id.append(class_name2id.get(detection["class_name"], -1))
        class_name.append(detection["class_name"])
        confidence.append(scale_confidence(detection.get("confidence", 1.0)))
    xyxy = np.array(xyxy).round(0) if len(xyxy) > 0 else np.empty((0, 4))
    confidence = np.array(confidence) if len(confidence) > 0 else np.empty(0)
    class_id = np.array(class_id).astype(int) if len(class_id) > 0 else np.empty(0)
    class_name = np.array(class_name) if len(class_name) > 0 else np.empty(0)
    detection_ids = np.array([str(uuid4()) for _ in range(len(xyxy))])
    dimensions = np.array([[image_height, image_width]] * len(xyxy))
    inference_ids = np.array([inference_id] * len(xyxy))
    prediction_type = np.array(["object-detection"] * len(xyxy))
    data = {
        CLASS_NAME_DATA_FIELD: class_name,
        IMAGE_DIMENSIONS_KEY: dimensions,
        INFERENCE_ID_KEY: inference_ids,
        DETECTION_ID_KEY: detection_ids,
        PREDICTION_TYPE_KEY: prediction_type,
    }
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        mask=None,
        tracker_id=None,
        data=data,
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )


def create_classes_index(classes: List[str]) -> Dict[str, int]:
    return {class_name: idx for idx, class_name in enumerate(classes)}


def scale_confidence(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def parse_florence2_object_detection_response(
    image: WorkflowImageData,
    parsed_data: dict,
    classes: Optional[List[str]],
    inference_id: str,
    florence_task_type: str,
):
    image_height, image_width = image.numpy_image.shape[:2]
    detections = sv.Detections.from_lmm(
        "florence_2",
        result={florence_task_type: parsed_data},
        resolution_wh=(image_width, image_height),
    )
    detections.class_id = np.array([0] * len(detections))
    if florence_task_type == "<REGION_PROPOSAL>":
        detections.data["class_name"] = np.array(["roi"] * len(detections))
    if florence_task_type in {"<OD>", "<CAPTION_TO_PHRASE_GROUNDING>"}:
        unique_class_names = set(detections.data.get("class_name", []))
        class_name_to_id = {
            name: get_4digit_from_md5(name) for name in unique_class_names
        }
        class_ids = [
            class_name_to_id.get(name, -1)
            for name in detections.data.get("class_name", ["unknown"] * len(detections))
        ]
        detections.class_id = np.array(class_ids)
    if florence_task_type in "<OPEN_VOCABULARY_DETECTION>":
        class_name_to_id = {name: idx for idx, name in enumerate(classes)}
        class_ids = [
            class_name_to_id.get(name, -1)
            for name in detections.data.get("class_name", ["unknown"] * len(detections))
        ]
        detections.class_id = np.array(class_ids)
    dimensions = np.array([[image_height, image_width]] * len(detections))
    detection_ids = np.array([str(uuid4()) for _ in range(len(detections))])
    inference_ids = np.array([inference_id] * len(detections))
    prediction_type = np.array(["object-detection"] * len(detections))
    detections.data.update(
        {
            INFERENCE_ID_KEY: inference_ids,
            DETECTION_ID_KEY: detection_ids,
            PREDICTION_TYPE_KEY: prediction_type,
            IMAGE_DIMENSIONS_KEY: dimensions,
        }
    )
    detections.confidence = np.array([1.0 for _ in detections])
    return attach_parents_coordinates_to_sv_detections(
        detections=detections, image=image
    )


def get_4digit_from_md5(input_string):
    md5_hash = hashlib.md5(input_string.encode("utf-8"))
    hex_digest = md5_hash.hexdigest()
    integer_value = int(hex_digest[:9], 16)
    return integer_value % 10000


REGISTERED_PARSERS = {
    # LLMs
    ("openai", "object-detection"): parse_llm_object_detection_response,
    ("google-gemini", "object-detection"): parse_llm_object_detection_response,
    ("anthropic-claude", "object-detection"): parse_llm_object_detection_response,
    # Florence 2
    ("florence-2", "object-detection"): partial(
        parse_florence2_object_detection_response, florence_task_type="<OD>"
    ),
    ("florence-2", "open-vocabulary-object-detection"): partial(
        parse_florence2_object_detection_response,
        florence_task_type="<OPEN_VOCABULARY_DETECTION>",
    ),
    ("florence-2", "object-detection-and-caption"): partial(
        parse_florence2_object_detection_response,
        florence_task_type="<DENSE_REGION_CAPTION>",
    ),
    ("florence-2", "phrase-grounded-object-detection"): partial(
        parse_florence2_object_detection_response,
        florence_task_type="<CAPTION_TO_PHRASE_GROUNDING>",
    ),
    ("florence-2", "region-proposal"): partial(
        parse_florence2_object_detection_response,
        florence_task_type="<REGION_PROPOSAL>",
    ),
    ("florence-2", "ocr-with-text-detection"): partial(
        parse_florence2_object_detection_response,
        florence_task_type="<OCR_WITH_REGION>",
    ),
}
