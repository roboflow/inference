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

!!! Warning "CogVLM reached **End Of Life**"

    Due to dependencies conflicts with newer models and security vulnerabilities discovered in `transformers`
    library patched in the versions of library incompatible with the model we announced End Of Life for CogVLM
    support in `inference`, effective since release `0.38.0`.
    
    We are leaving this block in ecosystem until release `0.42.0` for clients to get informed about change that 
    was introduced.
    
    Starting as of now, all Workflows using the block stop being functional (runtime error will be raised), 
    after inference release `0.42.0` - this block will be removed and Execution Engine will raise compilation 
    error seeing the block in Workflow definition. 


Ask a question to CogVLM, an open source vision-language model.

This model requires a GPU and can only be run on self-hosted devices, and is not available on the Roboflow Hosted API.

_This model was previously part of the LMM block._
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
        description="Text prompt to the CogVLM model",
        examples=["my prompt", "$inputs.prompt"],
        json_schema_extra={
            "multiline": True,
        },
    )
    json_output_format: Optional[Dict[str, str]] = Field(
        default=None,
        description="Holds dictionary that maps name of requested output field into its description",
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
