import json
from typing import List, Literal, Optional, Type, TypeVar, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field, model_validator

from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.vlms import VLM_TASKS_METADATA
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    ImageInputField,
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

T = TypeVar("T")
K = TypeVar("K")

DETECTIONS_CLASS_NAME_FIELD = "class_name"
DETECTION_ID_FIELD = "detection_id"

SUPPORTED_TASK_TYPES_LIST = [
    {
        "task_type": "open-vocabulary-object-detection",
        "florence_task": "<OPEN_VOCABULARY_DETECTION>",
    },
    {"task_type": "ocr", "florence_task": "<OCR>"},
    {"task_type": "ocr-with-text-detection", "florence_task": "<OCR_WITH_REGION>"},
    {"task_type": "caption", "florence_task": "<CAPTION>"},
    {"task_type": "detailed-caption", "florence_task": "<DETAILED_CAPTION>"},
    {"task_type": "more-detailed-caption", "florence_task": "<MORE_DETAILED_CAPTION>"},
    {
        "task_type": "object-detection-and-caption",
        "florence_task": "<DENSE_REGION_CAPTION>",
    },
    {"task_type": "object-detection", "florence_task": "<OD>"},
    {
        "task_type": "phrase-grounded-object-detection",
        "florence_task": "<CAPTION_TO_PHRASE_GROUNDING>",
    },
    {
        "task_type": "phrase-grounded-instance-segmentation",
        "florence_task": "<REFERRING_EXPRESSION_SEGMENTATION>",
    },
    {
        "task_type": "detection-grounded-instance-segmentation",
        "florence_task": "<REGION_TO_SEGMENTATION>",
    },
    {
        "task_type": "detection-grounded-classification",
        "florence_task": "<REGION_TO_CATEGORY>",
    },
    {
        "task_type": "detection-grounded-caption",
        "florence_task": "<REGION_TO_DESCRIPTION>",
    },
    {"task_type": "detection-grounded-ocr", "florence_task": "<REGION_TO_OCR>"},
    {"task_type": "region-proposal", "florence_task": "<REGION_PROPOSAL>"},
]
TASK_TYPE_TO_FLORENCE_TASK = {
    task["task_type"]: task["florence_task"] for task in SUPPORTED_TASK_TYPES_LIST
}
RELEVANT_TASKS_METADATA = {
    k: v for k, v in VLM_TASKS_METADATA.items() if k in TASK_TYPE_TO_FLORENCE_TASK
}
RELEVANT_TASKS_DOCS_DESCRIPTION = "\n\n".join(
    f"* **{v['name']}** (`{k}`) - {v['description']}"
    for k, v in RELEVANT_TASKS_METADATA.items()
)
LONG_DESCRIPTION = f"""
**Dedicated inference server required (GPU recommended) - you may want to use dedicated deployment**

This Workflow block introduces **Florence 2**, a Visual Language Model (VLM) capable of performing a 
wide range of tasks, including:

* Object Detection

* Instance Segmentation

* Image Captioning

* Optical Character Recognition (OCR)

* and more...


Below is a comprehensive list of tasks supported by the model, along with descriptions on 
how to utilize their outputs within the Workflows ecosystem:

**Task Descriptions:**

{RELEVANT_TASKS_DOCS_DESCRIPTION}
"""


TaskType = Literal[tuple([task["task_type"] for task in SUPPORTED_TASK_TYPES_LIST])]
GroundingSelectionMode = Literal[
    "first",
    "last",
    "biggest",
    "smallest",
    "most-confident",
    "least-confident",
]

TASKS_REQUIRING_PROMPT = {
    "phrase-grounded-object-detection",
    "phrase-grounded-instance-segmentation",
}
TASKS_REQUIRING_CLASSES = {
    "open-vocabulary-object-detection",
}
TASKS_REQUIRING_DETECTION_GROUNDING = {
    "detection-grounded-instance-segmentation",
    "detection-grounded-classification",
    "detection-grounded-caption",
    "detection-grounded-ocr",
}
LOC_BINS = 1000

TASKS_TO_EXTRACT_LABELS_AS_CLASSES = {
    "<OD>",
    "<DENSE_REGION_CAPTION>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<OCR_WITH_REGION>",
}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Florence-2 Model",
            "version": "v1",
            "short_description": "Run Florence-2 on an image",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["Florence", "Florence-2", "Microsoft"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/florence_2@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    model_version: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]),
        Literal["florence-2-base", "florence-2-large"],
    ] = Field(
        default="florence-2-base",
        description="Model to be used",
        examples=["florence-2-base"],
    )
    task_type: TaskType = Field(
        default="open-vocabulary-object-detection",
        description="Task type to be performed by model. "
        "Value determines required parameters and output response.",
        json_schema_extra={
            "values_metadata": RELEVANT_TASKS_METADATA,
            "recommended_parsers": {
                "open-vocabulary-object-detection": "roboflow_core/vlm_as_detector@v1",
                "ocr-with-text-detection": "roboflow_core/vlm_as_detector@v1",
                "object-detection-and-caption": "roboflow_core/vlm_as_detector@v1",
                "object-detection": "roboflow_core/vlm_as_detector@v1",
                "phrase-grounded-object-detection": "roboflow_core/vlm_as_detector@v1",
                "region-proposal": "roboflow_core/vlm_as_detector@v1",
            },
            "always_visible": True,
        },
    )
    prompt: Optional[Union[WorkflowParameterSelector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Text prompt to the Florence-2 model",
        examples=["my prompt", "$inputs.prompt"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": TASKS_REQUIRING_PROMPT, "required": True},
            },
        },
    )
    classes: Optional[
        Union[WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]), List[str]]
    ] = Field(
        default=None,
        description="List of classes to be used",
        examples=[["class-a", "class-b"], "$inputs.classes"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": TASKS_REQUIRING_CLASSES,
                    "required": True,
                },
            },
        },
    )
    grounding_detection: Optional[
        Union[
            List[int],
            List[float],
            StepOutputSelector(
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
                ]
            ),
            WorkflowParameterSelector(kind=[LIST_OF_VALUES_KIND]),
        ]
    ] = Field(
        default=None,
        description="Detection to ground Florence-2 model. May be statically provided bounding box "
        "`[left_top_x, left_top_y, right_bottom_x, right_bottom_y]` or result of object-detection model. "
        "If the latter is true, one box will be selected based on `grounding_selection_mode`.",
        examples=["$steps.detection.predictions", [10, 20, 30, 40]],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": TASKS_REQUIRING_DETECTION_GROUNDING,
                    "required": True,
                },
            },
        },
    )
    grounding_selection_mode: GroundingSelectionMode = Field(
        default="first",
        description="",
        examples=["first", "most-confident"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {
                    "values": TASKS_REQUIRING_DETECTION_GROUNDING,
                    "required": True,
                },
            },
        },
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        if self.task_type in TASKS_REQUIRING_PROMPT and self.prompt is None:
            raise ValueError(
                f"`prompt` parameter required to be set for task `{self.task_type}`"
            )
        if self.task_type in TASKS_REQUIRING_CLASSES and not self.classes:
            raise ValueError(
                f"`classes` parameter required to be set for task `{self.task_type}`"
            )
        if (
            self.task_type in TASKS_REQUIRING_DETECTION_GROUNDING
            and not self.grounding_detection
        ):
            raise ValueError(
                f"`grounding_detection` parameter required to be set for task `{self.task_type}`"
            )
        return self

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="raw_output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]
            ),
            OutputDefinition(name="parsed_output", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="classes", kind=[LIST_OF_VALUES_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class Florence2BlockV1(WorkflowBlock):

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
        model_version: str,
        task_type: TaskType,
        prompt: Optional[str],
        classes: Optional[List[str]],
        grounding_detection: Optional[
            Union[Batch[sv.Detections], List[int], List[float]]
        ],
        grounding_selection_mode: GroundingSelectionMode,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                task_type=task_type,
                model_version=model_version,
                prompt=prompt,
                classes=classes,
                grounding_detection=grounding_detection,
                grounding_selection_mode=grounding_selection_mode,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "Remote execution is not supported for florence2. Run a local or dedicated inference server to use this block (GPU recommended)."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_version: str,
        task_type: TaskType,
        prompt: Optional[str],
        classes: Optional[List[str]],
        grounding_detection: Optional[
            Union[Batch[sv.Detections], List[int], List[float]]
        ],
        grounding_selection_mode: GroundingSelectionMode,
    ) -> BlockResult:
        requires_detection_grounding = task_type in TASKS_REQUIRING_DETECTION_GROUNDING
        task_type = TASK_TYPE_TO_FLORENCE_TASK[task_type]
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]
        prompts = [prompt] * len(images)
        if classes is not None:
            prompts = ["<and>".join(classes)] * len(images)
        else:
            classes = []
        if grounding_detection is not None:
            prompts = prepare_detection_grounding_prompts(
                images=images,
                grounding_detection=grounding_detection,
                grounding_selection_mode=grounding_selection_mode,
            )
        self._model_manager.add_model(
            model_id=model_version,
            api_key=self._api_key,
        )
        predictions = []
        for image, single_prompt in zip(inference_images, prompts):
            if single_prompt is None and requires_detection_grounding:
                # no grounding bbox found - empty result returned
                predictions.append(
                    {"raw_output": None, "parsed_output": None, "classes": None}
                )
                continue
            request = LMMInferenceRequest(
                api_key=self._api_key,
                model_id=model_version,
                image=image,
                source="workflow-execution",
                prompt=task_type + (single_prompt or ""),
            )
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_version, request=request
            )
            prediction_data = prediction.response[task_type]
            if task_type in TASKS_TO_EXTRACT_LABELS_AS_CLASSES:
                classes = prediction_data.get("labels", [])
            predictions.append(
                {
                    "raw_output": json.dumps(prediction_data),
                    "parsed_output": (
                        prediction_data if isinstance(prediction_data, dict) else None
                    ),
                    "classes": classes,
                }
            )
        return predictions


def prepare_detection_grounding_prompts(
    images: Batch[WorkflowImageData],
    grounding_detection: Union[Batch[sv.Detections], List[float], List[int]],
    grounding_selection_mode: GroundingSelectionMode,
) -> List[Optional[str]]:
    if isinstance(grounding_detection, list):
        return _prepare_grounding_bounding_box_from_coordinates(
            images=images,
            bounding_box=grounding_detection,
        )
    return [
        _prepare_grounding_bounding_box_from_detections(
            image=image.numpy_image,
            detections=detections,
            grounding_selection_mode=grounding_selection_mode,
        )
        for image, detections in zip(images, grounding_detection)
    ]


def _prepare_grounding_bounding_box_from_coordinates(
    images: Batch[WorkflowImageData], bounding_box: Union[List[float], List[int]]
) -> List[str]:
    return [
        _extract_bbox_coordinates_as_location_prompt(
            image=image.numpy_image, bounding_box=bounding_box
        )
        for image in images
    ]


def _prepare_grounding_bounding_box_from_detections(
    image: np.ndarray,
    detections: sv.Detections,
    grounding_selection_mode: GroundingSelectionMode,
) -> Optional[str]:
    if len(detections) == 0:
        return None
    height, width = image.shape[:2]
    if grounding_selection_mode not in COORDINATES_EXTRACTION:
        raise ValueError(
            f"Unknown grounding selection mode: {grounding_selection_mode}"
        )
    extraction_function = COORDINATES_EXTRACTION[grounding_selection_mode]
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = extraction_function(
        detections
    )
    left_top_x = _coordinate_to_loc(value=left_top_x / width)
    left_top_y = _coordinate_to_loc(value=left_top_y / height)
    right_bottom_x = _coordinate_to_loc(value=right_bottom_x / width)
    right_bottom_y = _coordinate_to_loc(value=right_bottom_y / height)
    return f"<loc_{left_top_x}><loc_{left_top_y}><loc_{right_bottom_x}><loc_{right_bottom_y}>"


COORDINATES_EXTRACTION = {
    "first": lambda detections: detections.xyxy[0].tolist(),
    "last": lambda detections: detections.xyxy[0].tolist(),
    "biggest": lambda detections: detections.xyxy[np.argmax(detections.area)].tolist(),
    "smallest": lambda detections: detections.xyxy[np.argmin(detections.area)].tolist(),
    "most-confident": lambda detections: detections.xyxy[
        np.argmax(detections.confidence)
    ].tolist(),
    "least-confident": lambda detections: detections.xyxy[
        np.argmin(detections.confidence)
    ].tolist(),
}


def _extract_bbox_coordinates_as_location_prompt(
    image: np.ndarray,
    bounding_box: Union[List[float], List[int]],
) -> str:
    height, width = image.shape[:2]
    coordinates = bounding_box[:4]
    if len(coordinates) != 4:
        raise ValueError(
            "Could not extract 4 coordinates of bounding box to perform detection "
            "grounded Florence 2 prediction."
        )
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = coordinates
    if all(isinstance(c, float) for c in coordinates):
        left_top_x = _coordinate_to_loc(value=left_top_x)
        left_top_y = _coordinate_to_loc(value=left_top_y)
        right_bottom_x = _coordinate_to_loc(value=right_bottom_x)
        right_bottom_y = _coordinate_to_loc(value=right_bottom_y)
        return f"<loc_{left_top_x}><loc_{left_top_y}><loc_{right_bottom_x}><loc_{right_bottom_y}>"
    if all(isinstance(c, int) for c in coordinates):
        left_top_x = _coordinate_to_loc(value=left_top_x / width)
        left_top_y = _coordinate_to_loc(value=left_top_y / height)
        right_bottom_x = _coordinate_to_loc(value=right_bottom_x / width)
        right_bottom_y = _coordinate_to_loc(value=right_bottom_y / height)
        return f"<loc_{left_top_x}><loc_{left_top_y}><loc_{right_bottom_x}><loc_{right_bottom_y}>"
    raise ValueError(
        "Provided coordinates in mixed format - coordinates must be all integers or all floats in range [0.0-1.0]"
    )


def _coordinate_to_loc(value: float) -> int:
    loc_bin = round(_scale_value(value=value, min_value=0.0, max_value=1.0) * LOC_BINS)
    return _scale_value(  # to make sure 0-999 cutting out 1000 on 1.0
        value=loc_bin,
        min_value=0,
        max_value=LOC_BINS - 1,
    )


def _scale_value(
    value: Union[int, float],
    min_value: Union[int, float],
    max_value: Union[int, float],
) -> Union[int, float]:
    return max(min(value, max_value), min_value)
