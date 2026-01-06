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
**⚠️ This block is deprecated.** Use the OpenAI GPT-4 Vision blocks (v1-v4) with the classification task type instead for better functionality and ongoing support. This block classifies images into one of several specified categories using OpenAI's GPT-4 with Vision model.

## How This Block Works

This deprecated block uses OpenAI's GPT-4 with Vision model to classify images into predefined categories. The block:

1. Takes images and a list of class names as input (supports batch processing)
2. Automatically constructs a classification prompt asking the model to assign each image to one of the provided classes
3. Encodes images to base64 format and sends them to OpenAI's GPT-4 Vision API
4. Receives the model's response in JSON format with the predicted class
5. Parses the structured output to extract the top predicted class
6. Returns the predicted class name, raw output, and classification predictions

The block is specialized for classification tasks - you provide a list of class names (e.g., ["cat", "dog", "bird"]) and the model selects the best matching category for each image. The model's response is automatically parsed as JSON to extract the predicted class, making it easy to use in workflows that need classification results.

## Common Use Cases

- **Zero-Shot Image Classification**: Classify images into custom categories without training a model (e.g., classify product types, identify scene types, categorize content)
- **Multi-Class Classification**: Classify images into one of several predefined categories using natural language class names
- **Flexible Category Definitions**: Use descriptive class names that don't require training data (e.g., "happy", "sad", "neutral" for emotion classification)
- **Content Categorization**: Automatically categorize images based on visual content (e.g., classify photos by location type, weather conditions, or activity)
- **Quality Assessment**: Classify images based on quality criteria (e.g., "high quality", "low quality", "needs review")

## Connecting to Other Blocks

The classification results from this block can be connected to:

- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on the predicted class (e.g., process images differently based on category)
- **Filter blocks** to filter images based on classification results (e.g., only process images classified as certain categories)
- **Data storage blocks** (e.g., CSV Formatter, Roboflow Dataset Upload) to log classification results for analytics or record-keeping
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send alerts when specific classes are detected
- **Additional classification blocks** or other workflow blocks that need classification predictions as input

## Deprecation Notice

This block is deprecated. For new workflows, use the **OpenAI GPT-4 Vision blocks (v1-v4)** with the `classification` or `multi-label-classification` task types, which provide:

- More comprehensive task type support (classification, multi-label classification, OCR, captioning, VQA, structured answering)
- Better prompt handling and structured outputs
- Ongoing updates and support
- Additional model versions and configuration options
- More flexibility in how classification is performed
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
        description="Type of Large Multimodal Model to use. Currently only 'gpt_4v' (GPT-4 with Vision) is supported. This block is deprecated - consider using OpenAI GPT-4 Vision blocks (v1-v4) with classification task type instead.",
        examples=["gpt_4v", "$inputs.lmm_type"],
    )
    classes: Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])] = Field(
        description="List of class names to classify images into. The model will assign each image to one of these classes. Provide descriptive class names (e.g., ['cat', 'dog', 'bird'] or ['happy', 'sad', 'neutral']). The block returns the top predicted class for each image. At least one class must be provided.",
        examples=[
            ["cat", "dog", "bird"],
            ["happy", "sad", "neutral"],
            ["high quality", "low quality"],
            "$inputs.classes",
        ],
    )
    lmm_config: LMMConfig = Field(
        default_factory=lambda: LMMConfig(),
        description="Configuration options for the LMM. Includes max_tokens (maximum length of response, default 450), gpt_image_detail ('low', 'high', or 'auto' - controls image resolution sent to API, 'auto' uses API defaults), and gpt_model_version (e.g., 'gpt-4o', defaults to 'gpt-4o'). Higher image detail provides better accuracy but uses more tokens.",
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
            description="OpenAI API key required to use GPT-4 with Vision. You can obtain an API key from https://platform.openai.com/api-keys. This field is kept private for security. Required when lmm_type is 'gpt_4v'.",
            examples=["sk-xxx-xxx", "$inputs.openai_api_key", "$secrets.openai_key"],
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
