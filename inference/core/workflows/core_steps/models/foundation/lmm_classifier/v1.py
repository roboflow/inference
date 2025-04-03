from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.lmm.v1 import (
    GPT_4V_MODEL_TYPE,
    LMMConfig,
    run_gpt_4v_llm_prompting,
    turn_raw_lmm_output_into_structured,
)
from inference.core.workflows.execution_engine.constants import (
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    IMAGE_METADATA_KIND,
    LIST_OF_VALUES_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    SECRET_KIND,
    STRING_KIND,
    TOP_CLASS_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Classify an image into one or more categories using a Large Multimodal Model (LMM).

You can specify arbitrary classes to an LMMBlock.

The LLMBlock supports two LMMs:

- OpenAI's GPT-4 with Vision.

You need to provide your OpenAI API key to use the GPT-4 with Vision model. 
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "LMM For Classification",
            "version": "v1",
            "short_description": "Run a large multimodal model such as ChatGPT-4v for classification.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "deprecated": True,
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
            },
        }
    )
    type: Literal["roboflow_core/lmm_for_classification@v1", "LMMForClassification"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    lmm_type: Union[Selector(kind=[STRING_KIND]), Literal["gpt_4v"]] = Field(
        description="Type of LMM to be used", examples=["gpt_4v", "$inputs.lmm_type"]
    )
    classes: Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])] = Field(
        description="List of classes that LMM shall classify against",
        examples=[["a", "b"], "$inputs.classes"],
    )
    lmm_config: LMMConfig = Field(
        default_factory=lambda: LMMConfig(),
        description="Configuration of LMM",
        examples=[
            {
                "max_tokens": 200,
                "gpt_image_detail": "low",
                "gpt_model_version": "gpt-4o",
            }
        ],
    )
    remote_api_key: Union[Selector(kind=[STRING_KIND, SECRET_KIND]), Optional[str]] = (
        Field(
            default=None,
            description="Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v`.",
            examples=["xxx-xxx", "$inputs.api_key"],
            private=True,
        )
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="raw_output", kind=[STRING_KIND]),
            OutputDefinition(name="top", kind=[TOP_CLASS_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class LMMForClassificationBlockV1(WorkflowBlock):

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
        lmm_type: str,
        classes: List[str],
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                lmm_type=lmm_type,
                classes=classes,
                lmm_config=lmm_config,
                remote_api_key=remote_api_key,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                lmm_type=lmm_type,
                classes=classes,
                lmm_config=lmm_config,
                remote_api_key=remote_api_key,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        lmm_type: str,
        classes: List[str],
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
    ) -> BlockResult:
        prompt = (
            f"You are supposed to perform image classification task. You are given image that should be "
            f"assigned one of the following classes: {classes}. "
            f'Your response must be JSON in format: {{"top": "some_class"}}'
        )
        images_prepared_for_processing = [
            image.to_inference_format(numpy_preferred=True) for image in images
        ]
        if lmm_type == GPT_4V_MODEL_TYPE:
            raw_output = run_gpt_4v_llm_prompting(
                image=images_prepared_for_processing,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
            )
        else:
            raise ValueError(f"CogVLM has been removed from the Roboflow Core Models.")
        structured_output = turn_raw_lmm_output_into_structured(
            raw_output=raw_output,
            expected_output={"top": "name of the class"},
        )
        predictions = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "top": structured["top"],
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        for prediction, image in zip(predictions, images):
            prediction[PREDICTION_TYPE_KEY] = "classification"
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
        return predictions

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        lmm_type: str,
        classes: List[str],
        lmm_config: LMMConfig,
        remote_api_key: Optional[str],
    ) -> BlockResult:
        prompt = (
            f"You are supposed to   image classification task. You are given image that should be "
            f"assigned one of the following classes: {classes}. "
            f'Your response must be JSON in format: {{"top": "some_class"}}'
        )
        images_prepared_for_processing = [
            image.to_inference_format(numpy_preferred=True) for image in images
        ]
        if lmm_type == GPT_4V_MODEL_TYPE:
            raw_output = run_gpt_4v_llm_prompting(
                image=images_prepared_for_processing,
                prompt=prompt,
                remote_api_key=remote_api_key,
                lmm_config=lmm_config,
            )
        else:
            raise ValueError(f"CogVLM has been removed from the Roboflow Core Models.")
        structured_output = turn_raw_lmm_output_into_structured(
            raw_output=raw_output,
            expected_output={"top": "name of the class"},
        )
        predictions = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "top": structured["top"],
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        for prediction, image in zip(predictions, images):
            prediction[PREDICTION_TYPE_KEY] = "classification"
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
        return predictions
