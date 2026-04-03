import json
import logging
import re
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json([\s\S]*?)```", flags=re.IGNORECASE)

LONG_DESCRIPTION = """
Parse JSON strings from Visual Language Models (VLMs) and Large Language Models (LLMs) into standardized classification prediction format by extracting class predictions, mapping class names to class IDs, handling both single-class and multi-label formats, and converting VLM/LLM text outputs into workflow-compatible classification results for VLM-based classification, LLM classification parsing, and text-to-classification conversion workflows.

## How This Block Works

This block converts VLM/LLM text outputs containing classification predictions into standardized classification prediction format. The block:

1. Receives image and VLM output string containing classification results in JSON format
2. Parses JSON content from VLM output:

   **Handles Markdown-wrapped JSON:**
   - Searches for JSON wrapped in Markdown code blocks (```json ... ```)
   - This format is common in LLM/VLM responses
   - If multiple markdown JSON blocks are found, only the first block is parsed
   - Extracts JSON content from within markdown tags

   **Handles raw JSON strings:**
   - If no markdown blocks are found, attempts to parse the entire string as JSON
   - Supports standard JSON format strings
3. Detects classification format and parses accordingly:

   **Single-Class Classification Format:**
   - Detects format containing "class_name" and "confidence" fields
   - Extracts the predicted class name and confidence score
   - Creates classification prediction with single top class
   - Maps class name to class ID using provided classes list

   **Multi-Label Classification Format:**
   - Detects format containing "predicted_classes" array
   - Extracts all predicted classes with their confidence scores
   - Handles duplicate classes by taking maximum confidence
   - Maps all class names to class IDs using provided classes list
4. Creates class name to class ID mapping:
   - Uses the provided classes list to create index mapping (class_name → class_id)
   - Maps classes in order (first class = ID 0, second = ID 1, etc.)
   - Classes not in the provided list get class_id = -1
5. Normalizes confidence scores:
   - Scales confidence values to valid range [0.0, 1.0]
   - Clamps values outside the range to 0.0 or 1.0
6. Constructs classification prediction:
   - Includes image dimensions (width, height) from input image
   - For single-class: includes "top" class, confidence, and predictions array
   - For multi-label: includes "predicted_classes" list and predictions dictionary
   - Includes inference_id and parent_id for tracking
   - Formats prediction in standard classification prediction format
7. Handles errors:
   - Sets `error_status` to True if JSON parsing fails
   - Sets `error_status` to True if classification format cannot be determined
   - Returns None for predictions when errors occur
   - Always includes inference_id for tracking
8. Returns classification prediction:
   - Outputs `predictions` in standard classification format (compatible with classification blocks)
   - Outputs `error_status` indicating parsing success/failure
   - Outputs `inference_id` with specific type for tracking and lineage

The block enables using VLMs/LLMs for classification by converting their text-based JSON outputs into standardized classification predictions that can be used in workflows like any other classification model output.

## Common Use Cases

- **VLM-Based Classification**: Use Visual Language Models for image classification by parsing VLM outputs into classification predictions (e.g., classify images with VLMs, use GPT-4V for classification, parse Claude Vision classifications), enabling VLM classification workflows
- **LLM Classification Parsing**: Parse LLM text outputs containing classification results into standardized format (e.g., parse GPT classification outputs, convert LLM predictions to classification format, use LLMs for classification), enabling LLM classification workflows
- **Text-to-Classification Conversion**: Convert text-based classification outputs from models into workflow-compatible classification predictions (e.g., convert text predictions to classification format, parse text-based classifications, convert model outputs to classifications), enabling text-to-classification workflows
- **Multi-Format Classification Support**: Handle both single-class and multi-label classification formats from VLM/LLM outputs (e.g., support single-label VLM classifications, support multi-label VLM classifications, handle different classification formats), enabling flexible classification workflows
- **VLM Integration**: Integrate VLM outputs into classification workflows (e.g., use VLMs in classification pipelines, integrate VLM predictions with classification blocks, combine VLM and traditional classification), enabling VLM integration workflows
- **Flexible Classification Sources**: Enable classification from various model types that output text/JSON (e.g., use any text-output model for classification, convert model outputs to classifications, parse various classification formats), enabling flexible classification workflows

## Connecting to Other Blocks

This block receives images and VLM outputs and produces classification predictions:

- **After VLM/LLM blocks** to parse classification outputs into standard format (e.g., VLM output to classification, LLM output to classification, parse model outputs), enabling VLM-to-classification workflows
- **Before classification-based blocks** to use parsed classifications (e.g., use parsed classifications in workflows, provide classifications to downstream blocks, use VLM classifications with classification blocks), enabling classification-to-workflow workflows
- **Before filtering blocks** to filter based on VLM classifications (e.g., filter by VLM classification results, use parsed classifications for filtering, apply filters to VLM predictions), enabling classification-to-filter workflows
- **Before analytics blocks** to analyze VLM classification results (e.g., analyze VLM classifications, perform analytics on parsed classifications, track VLM classification metrics), enabling classification analytics workflows
- **Before visualization blocks** to display VLM classification results (e.g., visualize VLM classifications, display parsed classification predictions, show VLM classification outputs), enabling classification visualization workflows
- **In workflow outputs** to provide VLM classifications as final output (e.g., VLM classification outputs, parsed classification results, VLM-based classification outputs), enabling classification output workflows

## Version Differences

This version (v2) includes the following enhancements over v1:

- **Improved Type System**: The `inference_id` output now uses `INFERENCE_ID_KIND` instead of generic `STRING_KIND`, providing better type safety and semantic clarity for inference ID values in the workflow type system

## Requirements

This block requires an image input (for metadata and dimensions) and a VLM output string containing JSON classification data. The JSON can be raw JSON or wrapped in Markdown code blocks (```json ... ```). The block supports two JSON formats: single-class (with "class_name" and "confidence" fields) and multi-label (with "predicted_classes" array). The `classes` parameter must contain a list of all class names used by the model to generate class_id mappings. Classes are mapped to IDs by index (first class = 0, second = 1, etc.). Classes not in the list get class_id = -1. Confidence scores are normalized to [0.0, 1.0] range. The block outputs classification predictions in standard format (compatible with classification blocks), error_status (boolean), and inference_id (INFERENCE_ID_KIND) for tracking.
"""

SHORT_DESCRIPTION = "Parse a raw string into a classification prediction."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "VLM As Classifier",
            "version": "v2",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-tags",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/vlm_as_classifier@v2"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Input image that was used to generate the VLM prediction. Used to extract image dimensions (width, height) and metadata (parent_id) for the classification prediction. The same image that was provided to the VLM/LLM block should be used here to maintain consistency.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    vlm_output: Selector(kind=[LANGUAGE_MODEL_OUTPUT_KIND]) = Field(
        title="VLM Output",
        description="String output from a VLM or LLM block containing classification prediction in JSON format. Can be raw JSON string (e.g., '{\"class_name\": \"dog\", \"confidence\": 0.95}') or JSON wrapped in Markdown code blocks (e.g., ```json {...} ```). Supports two formats: single-class (with 'class_name' and 'confidence' fields) or multi-label (with 'predicted_classes' array). If multiple markdown blocks exist, only the first is parsed.",
        examples=["$steps.lmm.output", "$steps.vlm.output", "$steps.claude.output"],
    )
    classes: Union[
        Selector(kind=[LIST_OF_VALUES_KIND]),
        Selector(kind=[LIST_OF_VALUES_KIND]),
        List[str],
    ] = Field(
        description="List of all class names used by the classification model, in order. Required to generate mapping between class names (from VLM output) and class IDs (for classification format). Classes are mapped to IDs by index: first class = ID 0, second = ID 1, etc. Classes from VLM output that are not in this list get class_id = -1. Should match the classes the VLM was asked to classify.",
        examples=[
            [
                "$steps.lmm.classes",
                "$inputs.classes",
                ["dog", "cat", "bird"],
                ["class_a", "class_b"],
            ]
        ],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
            OutputDefinition(name="inference_id", kind=[INFERENCE_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class VLMAsClassifierBlockV2(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        vlm_output: str,
        classes: List[str],
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
        if "class_name" in parsed_data and "confidence" in parsed_data:
            return parse_multi_class_classification_results(
                image=image,
                results=parsed_data,
                classes=classes,
                inference_id=inference_id,
            )
        if "predicted_classes" in parsed_data:
            return parse_multi_label_classification_results(
                image=image,
                results=parsed_data,
                classes=classes,
                inference_id=inference_id,
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
            f"Could not parse JSON to dict in `roboflow_core/vlm_as_classifier@v1` block. "
            f"Error type: {error.__class__.__name__}. Details: {error}"
        )
        return True, {}


def parse_multi_class_classification_results(
    image: WorkflowImageData,
    results: dict,
    classes: List[str],
    inference_id: str,
) -> dict:
    try:
        class2id_mapping = create_classes_index(classes=classes)
        height, width = image.numpy_image.shape[:2]
        top_class = results["class_name"]
        confidences = {top_class: scale_confidence(results["confidence"])}
        predictions = []
        if top_class not in class2id_mapping:
            predictions.append(
                {
                    "class": top_class,
                    "class_id": -1,
                    "confidence": confidences.get(top_class, 0.0),
                }
            )
        for class_name, class_id in class2id_mapping.items():
            predictions.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": confidences.get(class_name, 0.0),
                }
            )
        parsed_prediction = {
            "image": {"width": width, "height": height},
            "predictions": predictions,
            "top": top_class,
            "confidence": confidences[top_class],
            "inference_id": inference_id,
            "parent_id": image.parent_metadata.parent_id,
        }
        return {
            "error_status": False,
            "predictions": parsed_prediction,
            "inference_id": inference_id,
        }
    except Exception as error:
        logging.warning(
            f"Could not parse multi-class classification results in `roboflow_core/vlm_as_classifier@v1` block. "
            f"Error type: {error.__class__.__name__}. Details: {error}"
        )
        return {"error_status": True, "predictions": None, "inference_id": inference_id}


def parse_multi_label_classification_results(
    image: WorkflowImageData,
    results: dict,
    classes: List[str],
    inference_id: str,
) -> dict:
    try:
        class2id_mapping = create_classes_index(classes=classes)
        height, width = image.numpy_image.shape[:2]
        predicted_classes_confidences = {}
        for prediction in results["predicted_classes"]:
            if prediction["class"] not in class2id_mapping:
                class2id_mapping[prediction["class"]] = -1
            if prediction["class"] in predicted_classes_confidences:
                old_confidence = predicted_classes_confidences[prediction["class"]]
                new_confidence = scale_confidence(value=prediction["confidence"])
                predicted_classes_confidences[prediction["class"]] = max(
                    old_confidence, new_confidence
                )
            else:
                predicted_classes_confidences[prediction["class"]] = scale_confidence(
                    value=prediction["confidence"]
                )
        predictions = {
            class_name: {
                "confidence": predicted_classes_confidences.get(class_name, 0.0),
                "class_id": class_id,
            }
            for class_name, class_id in class2id_mapping.items()
        }
        parsed_prediction = {
            "image": {"width": width, "height": height},
            "predictions": predictions,
            "predicted_classes": list(predicted_classes_confidences.keys()),
            "inference_id": inference_id,
            "parent_id": image.parent_metadata.parent_id,
        }
        return {
            "error_status": False,
            "predictions": parsed_prediction,
            "inference_id": inference_id,
        }
    except Exception as error:
        logging.warning(
            f"Could not parse multi-label classification results in `roboflow_core/vlm_as_classifier@v1` block. "
            f"Error type: {error.__class__.__name__}. Details: {error}"
        )
        return {"error_status": True, "predictions": None, "inference_id": inference_id}


def create_classes_index(classes: List[str]) -> Dict[str, int]:
    return {class_name: idx for idx, class_name in enumerate(classes)}


def scale_confidence(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
