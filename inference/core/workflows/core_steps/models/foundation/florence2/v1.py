from typing import List, Literal, Optional, Type, TypeVar, Union
import json

from pydantic import ConfigDict, Field, model_validator
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.entities.requests.inference import LMMInferenceRequest
from inference.core.entities.responses.inference import LMMInferenceResponse
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    load_core_model,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
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

LONG_DESCRIPTION = """
Run Florence-2, a large multimodal model, on an image.

** Dedicated inference server required (GPU recomended) **
"""

TaskType = Literal[
    "<OCR>",
    "<OCR_WITH_REGION>",
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<OD>",
    "<DENSE_REGION_CAPTION>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "<REGION_TO_SEGMENTATION>",
    "<OPEN_VOCABULARY_DETECTION>",
    "<REGION_TO_CATEGORY>",
    "<REGION_TO_DESCRIPTION>",
    "<REGION_TO_OCR>",
    "<REGION_PROPOSAL>",
]

TASKS_REQUIRING_PROMPT = [
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<REFERRING_EXPRESSION_SEGMENTATION>",
    "<REGION_TO_SEGMENTATION>",
    "<OPEN_VOCABULARY_DETECTION>",
    "<REGION_TO_CATEGORY>",
    "<REGION_TO_DESCRIPTION>",
    "<REGION_TO_OCR>",
]


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
        },
        protected_namespaces=(),
    )

    model_version: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]),
        Literal["florence-2-base", "florence-2-large"],
    ] = Field(
        default="florence-2-base",
        description="Model to be used",
        examples=["florence-2-base"],
    )
    type: Literal["roboflow_core/florence_2@v1"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    task_type: TaskType = Field(
        description="Task type to be performed by model. Value of parameter determine set of fields "
        "that are required. For `unconstrained`, `visual-question-answering`, "
        " - `prompt` parameter must be provided."
        "For `structured-answering` - `output-structure` must be provided. For "
        "`classification`, `multi-label-classification` and `object-detection` - "
        "`classes` must be filled. `ocr`, `caption`, `detailed-caption` do not"
        "require any additional parameter.",
    )
    prompt: Optional[Union[WorkflowParameterSelector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Text prompt to the Claude model",
        examples=["my prompt", "$inputs.prompt"],
        json_schema_extra={
            "relevant_for": {
                "task_type": {"values": TASKS_REQUIRING_PROMPT, "required": True},
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
        return self

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]
            )
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
        task_type: TaskType,
        prompt: Optional[str],
        model_version: str,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                task_type=task_type,
                model_version=model_version,
                prompt=prompt,
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
        task_type: TaskType,
        prompt: Optional[str],
        model_version: str,
    ) -> BlockResult:
        inference_images = [
            i.to_inference_format(numpy_preferred=False) for i in images
        ]
        self._model_manager.add_model(
            model_id=model_version,
            api_key=self._api_key,
        )
        predictions = []
        for image in inference_images:
            request = LMMInferenceRequest(
                api_key=self._api_key,
                model_id=model_version,
                image=image,
                source="workflow-execution",
                prompt=task_type + (prompt or ""),
            )
            prediction = self._model_manager.infer_from_request_sync(
                model_id=model_version, request=request
            )
            jsonified = json.dumps(prediction.response)
            predictions.append(jsonified)

        return [{"output": prediction} for prediction in predictions]
