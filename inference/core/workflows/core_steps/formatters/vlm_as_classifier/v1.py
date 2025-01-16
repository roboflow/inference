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
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json([\s\S]*?)```", flags=re.IGNORECASE)

LONG_DESCRIPTION = """
The block expects string input that would be produced by blocks exposing Large Language Models (LLMs) and 
Visual Language Models (VLMs). Input is parsed to classification prediction and returned as block output.

Accepted formats:

- valid JSON strings

- JSON documents wrapped with Markdown tags (very common for GPT responses)

Example:
```
{"my": "json"}
```

**Details regarding block behavior:**

- `error_status` is set `True` whenever parsing cannot be completed

- in case of multiple markdown blocks with raw JSON content - only first will be parsed
"""

SHORT_DESCRIPTION = "Parse a raw string into a classification prediction."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "VLM as Classifier",
            "version": "v1",
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
    type: Literal["roboflow_core/vlm_as_classifier@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="The image which was the base to generate VLM prediction",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    vlm_output: Selector(kind=[LANGUAGE_MODEL_OUTPUT_KIND]) = Field(
        title="VLM Output",
        description="The string with raw classification prediction to parse.",
        examples=[["$steps.lmm.output"]],
    )
    classes: Union[
        Selector(kind=[LIST_OF_VALUES_KIND]),
        Selector(kind=[LIST_OF_VALUES_KIND]),
        List[str],
    ] = Field(
        description="List of all classes used by the model, required to "
        "generate mapping between class name and class id.",
        examples=[["$steps.lmm.classes", "$inputs.classes", ["class_a", "class_b"]]],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="predictions", kind=[CLASSIFICATION_PREDICTION_KIND]),
            OutputDefinition(name="inference_id", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class VLMAsClassifierBlockV1(WorkflowBlock):

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
