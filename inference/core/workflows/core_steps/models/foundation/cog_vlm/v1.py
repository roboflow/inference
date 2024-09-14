import json
import re
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.entities.requests.cogvlm import CogVLMInferenceRequest
from inference.core.env import (
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import load_image
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import load_core_model
from inference.core.workflows.execution_engine.constants import (
    PARENT_ID_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    IMAGE_METADATA_KIND,
    PARENT_ID_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    ImageInputField,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceHTTPClient
from inference_sdk.http.utils.iterables import make_batches

NOT_DETECTED_VALUE = "not_detected"

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json\n([\s\S]*?)\n```")

LONG_DESCRIPTION = """
Ask a question to CogVLM, an open source vision-language model.

This model requires a GPU and can only be run on self-hosted devices, and is not available on the Roboflow Hosted API.

_This model was previously part of the LMM block._
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "CogVLM",
            "version": "v1",
            "short_description": "Run a self-hosted vision language model",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["LMM"],
        }
    )
    type: Literal["roboflow_core/cog_vlm@v1", "CogVLM"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    prompt: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Text prompt to the CogVLM model",
        examples=["my prompt", "$inputs.prompt"],
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
    def accepts_batch_input(cls) -> bool:
        return True

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
        return ">=1.0.0,<2.0.0"


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
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                prompt=prompt,
                json_output_format=json_output_format,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                prompt=prompt,
                json_output_format=json_output_format,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        prompt: str,
        json_output_format: Optional[Dict[str, str]],
    ) -> BlockResult:
        if json_output_format:
            prompt = (
                f"{prompt}\n\nVALID response format is JSON:\n"
                f"{json.dumps(json_output_format, indent=4)}"
            )
        images_prepared_for_processing = [
            image.to_inference_format(numpy_preferred=True) for image in images
        ]
        raw_output = get_cogvlm_generations_locally(
            image=images_prepared_for_processing,
            prompt=prompt,
            model_manager=self._model_manager,
            api_key=self._api_key,
        )
        structured_output = turn_raw_lmm_output_into_structured(
            raw_output=raw_output,
            expected_output=json_output_format,
        )
        predictions = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "structured_output": structured,
                **structured,
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        for prediction, image in zip(predictions, images):
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
        return predictions

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        prompt: str,
        json_output_format: Optional[Dict[str, str]],
    ) -> BlockResult:
        if json_output_format:
            prompt = (
                f"{prompt}\n\nVALID response format is JSON:\n"
                f"{json.dumps(json_output_format, indent=4)}"
            )
        inference_images = [i.to_inference_format() for i in images]
        raw_output = get_cogvlm_generations_from_remote_api(
            image=inference_images,
            prompt=prompt,
            api_key=self._api_key,
        )
        structured_output = turn_raw_lmm_output_into_structured(
            raw_output=raw_output,
            expected_output=json_output_format,
        )
        predictions = [
            {
                "raw_output": raw["content"],
                "image": raw["image"],
                "structured_output": structured,
                **structured,
            }
            for raw, structured in zip(raw_output, structured_output)
        ]
        for prediction, image in zip(predictions, images):
            prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
            prediction[ROOT_PARENT_ID_KEY] = (
                image.workflow_root_ancestor_metadata.parent_id
            )
        return predictions


def get_cogvlm_generations_locally(
    image: List[dict],
    prompt: str,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> List[Dict[str, Any]]:
    serialised_result = []
    for single_image in image:
        loaded_image, _ = load_image(single_image)
        image_metadata = {
            "width": loaded_image.shape[1],
            "height": loaded_image.shape[0],
        }
        inference_request = CogVLMInferenceRequest(
            image=single_image,
            prompt=prompt,
            api_key=api_key,
        )
        model_id = load_core_model(
            model_manager=model_manager,
            inference_request=inference_request,
            core_model="cogvlm",
        )
        result = model_manager.infer_from_request_sync(model_id, inference_request)
        serialised_result.append(
            {
                "content": result.response,
                "image": image_metadata,
            }
        )
    return serialised_result


def get_cogvlm_generations_from_remote_api(
    image: List[dict],
    prompt: str,
    api_key: Optional[str],
) -> List[Dict[str, Any]]:
    if WORKFLOWS_REMOTE_API_TARGET == "hosted":
        raise ValueError(
            f"CogVLM requires a GPU and can only be executed remotely in self-hosted mode. "
            f"It is not available on the Roboflow Hosted API."
        )
    client = InferenceHTTPClient.init(
        api_url=LOCAL_INFERENCE_API_URL,
        api_key=api_key,
    )
    serialised_result = []
    for single_image in image:
        loaded_image, _ = load_image(single_image)
        image_metadata = {
            "width": loaded_image.shape[1],
            "height": loaded_image.shape[0],
        }
        result = client.prompt_cogvlm(
            visual_prompt=single_image["value"],
            text_prompt=prompt,
        )
        serialised_result.append(
            {
                "content": result["response"],
                "image": image_metadata,
            }
        )
    return serialised_result


def turn_raw_lmm_output_into_structured(
    raw_output: List[Dict[str, Any]],
    expected_output: Optional[Dict[str, str]],
) -> List[dict]:
    if expected_output is None:
        return [{} for _ in range(len(raw_output))]
    return [
        try_parse_lmm_output_to_json(
            output=r["content"],
            expected_output=expected_output,
        )
        for r in raw_output
    ]


def try_parse_lmm_output_to_json(
    output: str, expected_output: Dict[str, str]
) -> Union[list, dict]:
    json_blocks_found = JSON_MARKDOWN_BLOCK_PATTERN.findall(output)
    if len(json_blocks_found) == 0:
        return try_parse_json(output, expected_output=expected_output)
    result = []
    for json_block in json_blocks_found:
        result.append(
            try_parse_json(content=json_block, expected_output=expected_output)
        )
    return result if len(result) > 1 else result[0]


def try_parse_json(content: str, expected_output: Dict[str, str]) -> dict:
    try:
        data = json.loads(content)
        return {key: data.get(key, NOT_DETECTED_VALUE) for key in expected_output}
    except Exception:
        return {key: NOT_DETECTED_VALUE for key in expected_output}
