from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.entities.requests.embeddings import ImageEmbeddingRequest
from inference.core.env import (
    HOSTED_CLASSIFICATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.models.embeddings import IMAGE_EMBEDDINGS, model_cache_key
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    EMBEDDING_KIND,
    IMAGE_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    RoboflowModelField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

COMPATIBILITY_INFO_FIELDS = {
    "model_id",
    "feature_definition",
    "output_type",
    "dimension",
    "normalization",
    "space_id",
}


def workflow_embedding_info(info: dict) -> dict:
    return {
        key: value for key, value in info.items() if key in COMPATIBILITY_INFO_FIELDS
    }


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Embedding Model",
            "version": "v1",
            "short_description": "Generate image embeddings using a classification model.",
            "long_description": "Extract raw features before the final linear classifier or logits before Softmax/Sigmoid "
            "of a workspace ResNet, ViT or DINOv3 model, or a pretrained ResNet. "
            "Uses the model's original preprocessing. Compare vectors only from the same embedding space.",
            "license": "Apache-2.0",
            "block_type": "model",
            "required_model_capabilities": [IMAGE_EMBEDDINGS],
            "compatible_model_architectures": ["resnet", "vit", "dinov3_probe"],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/embedding_model@v1"]
    data: Selector(kind=[IMAGE_KIND]) = Field(
        title="Data",
        description="Image or crop to embed.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    output_type: Literal["feature_vector", "logits"] = Field(
        default="feature_vector",
        title="Output Type",
        description="Select Feature Vector (input to the final linear layer) or Logits (output before Softmax or Sigmoid).",
        examples=["feature_vector", "logits"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="embedding", kind=[EMBEDDING_KIND]),
            OutputDefinition(name="embedding_info", kind=[DICTIONARY_KIND]),
        ]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["data"]

    @classmethod
    def get_compatible_task_types(cls) -> Optional[List[str]]:
        return ["classification"]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class EmbeddingModelBlockV1(WorkflowBlock):
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
        data: Batch[WorkflowImageData],
        model_id: str,
        output_type: str = "feature_vector",
    ) -> BlockResult:
        if not data:
            return []
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            self._model_manager.add_model(
                model_id,
                self._api_key,
                required_capabilities=[IMAGE_EMBEDDINGS],
                output_type=output_type,
            )
            request = ImageEmbeddingRequest(
                model_id=model_id,
                output_type=output_type,
                api_key=self._api_key,
                source="workflow-execution",
                image=[
                    image.to_inference_format(numpy_preferred=True) for image in data
                ],
            )
            response = self._model_manager.infer_from_request_sync(
                model_cache_key(model_id, [IMAGE_EMBEDDINGS], output_type), request
            )
            info = workflow_embedding_info(
                response.embedding_info.model_dump(exclude_none=True)
            )
            results = [
                {"embedding": embedding, "embedding_info": info}
                for embedding in response.embeddings
            ]
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            client = InferenceHTTPClient(
                api_url=(
                    HOSTED_CLASSIFICATION_URL
                    if WORKFLOWS_REMOTE_API_TARGET == "hosted"
                    else LOCAL_INFERENCE_API_URL
                ),
                api_key=self._api_key,
            )
            client.configure(
                InferenceConfiguration(
                    max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
                    max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
                    source="workflow-execution",
                )
            )
            responses = client.get_image_embeddings(
                [image.base64_image for image in data],
                model_id=model_id,
                output_type=output_type,
            )
            if isinstance(responses, dict):
                responses = [responses]
            results = [
                {
                    "embedding": response["embeddings"][0],
                    "embedding_info": workflow_embedding_info(
                        response["embedding_info"]
                    ),
                }
                for response in responses
            ]
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )
        if len(results) != len(data):
            raise ValueError("Embedding response count does not match the image batch.")
        return results
