import json
import logging
import re
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
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
    BATCH_OF_BOOLEAN_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_STRING_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(
    r"```json\n([\s\S]*?)\n```", flags=re.IGNORECASE
)

LONG_DESCRIPTION = """
The block expects string input that would be produced by blocks exposing Large Language Models (LLMs) and 
Visual Language Models (VLMs). Input is parsed to object-detection prediction and returned as block output.

Accepted formats:

- valid JSON strings

- JSON documents wrapped with Markdown tags

Example
```
```json
{"my": "json"}
```
```

**Details regarding block behaviour:**

- `error_status` is set `True` whenever parsing cannot be completed

- in case of multiple markdown blocks with raw JSON content - only first will be parsed
"""

SHORT_DESCRIPTION = "Parses raw string into object-detection prediction."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "VLM as Detector",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
        }
    )
    type: Literal["roboflow_core/vlm_as_detector@v1"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="The image which was the base to generate VLM prediction",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    vlm_output: StepOutputSelector(kind=[BATCH_OF_STRING_KIND]) = Field(
        title="VLM Output",
        description="The string with raw classification prediction to parse.",
        examples=[["$steps.lmm.output"]],
    )
    classes: Union[
        WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]),
        StepOutputSelector(kind=[LIST_OF_VALUES_KIND]),
        List[str],
    ] = Field(
        description="List of all classes used by the model, required to "
        "generate mapping between class name and class id.",
        examples=[["$steps.lmm.classes", "$inputs.classes", ["class_a", "class_b"]]],
    )
    model_type: Literal["google-gemini"] = Field(
        description="Type of the model that generated prediction",
        examples=[["google-gemini"]],
    )
    task_type: Literal["object-detection"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BATCH_OF_BOOLEAN_KIND]),
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND]
            ),
            OutputDefinition(name="inference_id", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class VLMAsDetectorBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        vlm_output: str,
        classes: List[str],
        model_type: str,
        task_type: str,
    ) -> BlockResult:
        if (model_type, task_type) in REGISTERED_PARSERS:
            raise ValueError(
                f"Could not parse result of task {task_type} for model {model_type}"
            )
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


def parse_gemini_object_detection_response(
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
        class_id.append(class_name2id.get(class_name, -1))
        class_name.append(class_name)
        confidence.append(1.0)
    xyxy = np.array(xyxy) if len(xyxy) > 0 else np.empty((0, 4))
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


REGISTERED_PARSERS = {
    ("google-gemini", "object-detection"): parse_gemini_object_detection_response,
}
