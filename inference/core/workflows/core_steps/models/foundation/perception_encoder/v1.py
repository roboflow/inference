import hashlib
from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.cache.lru_cache import LRUCache
from inference.core.entities.requests.perception_encoder import (
    PerceptionEncoderImageEmbeddingRequest,
    PerceptionEncoderTextEmbeddingRequest,
)
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import load_core_model
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    EMBEDDING_KIND,
    IMAGE_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceHTTPClient

LONG_DESCRIPTION = """
Use the Meta Perception Encoder model to create semantic embeddings of text and images.

This block accepts an image or string and returns an embedding. The embedding can be used to compare
similarity between different images or between images and text.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Perception Encoder Embedding Model",
            "version": "v1",
            "short_description": "Generate an embedding of an image or string.",
            "long_description": LONG_DESCRIPTION,
            "license": "MIT",
            "block_type": "model",
            "access_third_party": False,
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-paperclip",
                "blockPriority": 9.9,
            },
        }
    )
    type: Literal["roboflow_core/perception_encoder@v1"]
    name: str = Field(description="Unique name of step in workflows")
    data: Union[Selector(kind=[IMAGE_KIND, STRING_KIND]), str] = Field(
        title="Data",
        description="The string or image to generate an embedding for.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    version: Union[
        Literal[
            "PE-Core-B16-224",
            "PE-Core-L14-336",
            "PE-Core-G14-448",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="PE-Core-L14-336",
        description="Variant of Perception Encoder model",
        examples=["PE-Core-B16-224", "$inputs.variant"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="embedding", kind=[EMBEDDING_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


text_cache = LRUCache()


class PerceptionEncoderModelBlockV1(WorkflowBlock):
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
        data: Union[WorkflowImageData, str],
        version: str,
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(data=data, version=version)
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(data=data, version=version)
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        data: Union[WorkflowImageData, str],
        version: str,
    ) -> BlockResult:
        if isinstance(data, str):
            hash_key = hashlib.md5((version + data).encode("utf-8")).hexdigest()
            cached_value = text_cache.get(hash_key)
            if cached_value is not None:
                return {"embedding": cached_value}
            inference_request = PerceptionEncoderTextEmbeddingRequest(
                perception_encoder_version_id=version,
                text=[data],
                api_key=self._api_key,
            )
            pe_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="perception_encoder",
            )
            predictions = self._model_manager.infer_from_request_sync(
                pe_model_id, inference_request
            )
            text_cache.set(hash_key, predictions.embeddings[0])
            return {"embedding": predictions.embeddings[0]}
        else:
            inference_request = PerceptionEncoderImageEmbeddingRequest(
                perception_encoder_version_id=version,
                image=[data.to_inference_format(numpy_preferred=True)],
                api_key=self._api_key,
            )
            pe_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="perception_encoder",
            )
            predictions = self._model_manager.infer_from_request_sync(
                pe_model_id, inference_request
            )
            return {"embedding": predictions.embeddings[0]}

    def run_remotely(
        self,
        data: Union[WorkflowImageData, str],
        version: str,
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_CORE_MODEL_URL
        )
        client = InferenceHTTPClient(api_url=api_url, api_key=self._api_key)
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        if isinstance(data, str):
            result = client.get_perception_encoder_text_embeddings(
                text=data,
                perception_encoder_version=version,
            )
        else:
            result = client.get_perception_encoder_image_embeddings(
                inference_input=data.base64_image,
                perception_encoder_version=version,
            )
        return {"embedding": result["embeddings"][0]}
