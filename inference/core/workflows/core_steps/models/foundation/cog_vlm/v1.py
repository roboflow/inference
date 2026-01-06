import re
from typing import Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    IMAGE_KIND,
    IMAGE_METADATA_KIND,
    PARENT_ID_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

NOT_DETECTED_VALUE = "not_detected"

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json\n([\s\S]*?)\n```")

LONG_DESCRIPTION = """
!!! warning "DEPRECATED - CogVLM reached **End Of Life**"

    CogVLM support in `inference` reached End of Life effective since release `0.38.0` due to dependency conflicts with newer models and security vulnerabilities discovered in the `transformers` library (CVE-2024-11393). This block is **no longer functional** and will raise a runtime error. The block will be completely removed in inference release `0.54.0`, at which point workflows using this block will raise compilation errors. Please migrate to alternative vision-language models such as OpenAI GPT-4 Vision, Anthropic Claude, Google Gemini, or other available VLM blocks.

Run CogVLM, an open-source vision-language model (deprecated).

## How This Block Works

**⚠️ This block is deprecated and no longer functional.** 

This block was designed to process images and text prompts using CogVLM, an open-source vision-language model. The block would have taken images and text prompts as input, processed them through the CogVLM model, and returned text responses. The model required a GPU and could only run on self-hosted devices - it was not available on the Roboflow Hosted API.

This model was previously part of the LMM block. Due to security vulnerabilities and dependency conflicts, CogVLM support has been discontinued. Please use alternative VLM blocks instead.

## Common Use Cases

**Note**: This block is deprecated and no longer functional. For similar functionality, consider using:

- **OpenAI GPT-4 Vision** blocks for general-purpose vision-language tasks
- **Anthropic Claude** blocks for advanced vision-language capabilities
- **Google Gemini** blocks for multimodal understanding
- **Other VLM blocks** available in the model section (e.g., SmolVLM2, Qwen2.5-VL, Florence-2)

## Connecting to Other Blocks

Since this block is deprecated and non-functional, workflows using this block will not execute. If you migrate to alternative VLM blocks, their outputs can typically be connected to:

- **Parser blocks** to convert text responses into structured data
- **Conditional logic blocks** to route workflow execution based on model responses
- **Visualization blocks** to display annotations or text overlays
- **Data storage blocks** to log responses for analytics
- **Notification blocks** to send alerts based on model findings
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "CogVLM",
            "version": "v1",
            "short_description": "DEPRECATED! Run a self-hosted vision language model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["LMM"],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 9,
                "needsGPU": True,
                "inference": True,
                "deprecated": True,
            },
        }
    )
    type: Literal["roboflow_core/cog_vlm@v1", "CogVLM"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    prompt: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Text prompt to the CogVLM model. **Note: This block is deprecated and no longer functional.**",
        examples=["my prompt", "$inputs.prompt"],
        json_schema_extra={
            "multiline": True,
        },
    )
    json_output_format: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional dictionary that maps output field names to their descriptions. When provided, the model attempts to return structured JSON output with fields matching the keys in this dictionary. **Note: This block is deprecated and no longer functional.**",
        examples=[
            {"count": "number of cats in the picture"},
            "$inputs.json_output_format",
        ],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="structured_output", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="raw_output", kind=[STRING_KIND]),
            OutputDefinition(name="*", kind=[WILDCARD_KIND]),
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        result = [
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="root_parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="structured_output", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="raw_output", kind=[STRING_KIND]),
        ]
        if self.json_output_format is None:
            return result
        for key in self.json_output_format.keys():
            result.append(OutputDefinition(name=key, kind=[WILDCARD_KIND]))
        return result

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class CogVLMBlockV1(WorkflowBlock):

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
        prompt: str,
        json_output_format: Optional[Dict[str, str]],
    ) -> BlockResult:
        raise ValueError(
            "CogVLM reached End Of Life in `inference` and is no longer supported. "
            "Removal was correlated with changes introduced by maintainers as a result of "
            "the following security issue: https://nvd.nist.gov/vuln/detail/CVE-2024-11393. "
            "This class will be removed in inference 0.54.0."
        )
