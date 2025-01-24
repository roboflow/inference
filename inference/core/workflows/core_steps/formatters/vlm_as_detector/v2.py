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
The block expects string input that would be produced by blocks exposing Large Language Models (LLMs) and 
Visual Language Models (VLMs). Input is parsed to object-detection prediction and returned as block output.

Accepted formats:

- valid JSON strings

- JSON documents wrapped with Markdown tags

Example
```
{"my": "json"}
```

**Details regarding block behavior:**

- `error_status` is set `True` whenever parsing cannot be completed

- in case of multiple markdown blocks with raw JSON content - only first will be parsed
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
        description="The image which was the base to generate VLM prediction",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    vlm_output: Selector(kind=[LANGUAGE_MODEL_OUTPUT_KIND]) = Field(
        title="VLM Output",
        description="The string with raw classification prediction to parse.",
        examples=[["$steps.lmm.output"]],
    )
    classes: Optional[
        Union[
            Selector(kind=[LIST_OF_VALUES_KIND]),
            Selector(kind=[LIST_OF_VALUES_KIND]),
            List[str],
        ]
    ] = Field(
        description="List of all classes used by the model, required to "
        "generate mapping between class name and class id.",
        examples=[["$steps.lmm.classes", "$inputs.classes", ["class_a", "class_b"]]],
        json_schema_extra={
            "relevant_for": {
                "model_type": {
                    "values": ["google-gemini", "anthropic-claude"],
                    "required": True,
                },
            }
        },
    )
    model_type: Literal["google-gemini", "anthropic-claude", "florence-2"] = Field(
        description="Type of the model that generated prediction",
        examples=[["google-gemini", "anthropic-claude", "florence-2"]],
    )
    task_type: Literal[tuple(SUPPORTED_TASKS)] = Field(
        description="Task type to performed by model.",
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
    ("google-gemini", "object-detection"): parse_gemini_object_detection_response,
    ("anthropic-claude", "object-detection"): parse_gemini_object_detection_response,
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
