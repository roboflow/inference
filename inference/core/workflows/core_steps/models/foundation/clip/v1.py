import hashlib
from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.cache.lru_cache import LRUCache
from inference.core.entities.requests.clip import (
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
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
Use a CLIP model to generate semantic embeddings (vector representations) of images and text that can be compared for similarity.

## How This Block Works

This block takes either an image or a text string as input and processes it through OpenAI's CLIP (Contrastive Language-Image Pre-training) model. CLIP is trained to understand the relationship between images and text, creating embeddings (high-dimensional vectors) that capture semantic meaning. The block:

1. Accepts either an image or a text string as input
2. Processes the input through the CLIP model to generate a semantic embedding vector
3. Returns the embedding, which is a numerical representation of the input's semantic content
4. (For text inputs) Caches the embedding for efficiency if the same text is processed again

The key advantage of CLIP embeddings is that images and text are represented in the same embedding space, meaning you can directly compare embeddings from images and text to find semantic similarity. For example, an image of a cat and the text "a cat" will have similar embeddings, even though one is visual and one is textual.

## Common Use Cases

- **Image-Text Similarity Search**: Compare image embeddings with text embeddings to find images that match text descriptions (e.g., "red car", "sunset", "person wearing hat")
- **Image Search**: Generate embeddings for images and compare them to find visually or semantically similar images in a dataset
- **Text-Based Image Filtering**: Use text queries to filter or search through images by comparing text and image embeddings
- **Content Matching**: Find images that match specific text descriptions without training a custom classification model
- **Semantic Clustering**: Group images or text based on their semantic similarity using embedding comparisons
- **Similarity Scoring**: Calculate similarity scores between different images or between images and text for ranking or filtering purposes

## Connecting to Other Blocks

The embedding outputs from this block can be connected to:

- **Clip Comparison blocks** to compare embeddings and find similarity scores between images and text
- **Cosine Similarity blocks** to calculate similarity between embeddings from multiple CLIP blocks
- **Filter blocks** or **Conditional logic blocks** to route workflow execution based on similarity scores or embedding comparisons
- **Analytics blocks** (e.g., Data Aggregator) to analyze embedding similarities over time
- **Data storage blocks** to store embeddings for later similarity search or analysis
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "CLIP Embedding Model",
            "version": "v1",
            "short_description": "Generate an embedding of an image or string.",
            "long_description": LONG_DESCRIPTION,
            "license": "MIT",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-paperclip",
                "blockPriority": 9.9,
            },
        }
    )
    type: Literal["roboflow_core/clip@v1"]
    name: str = Field(description="Unique name of step in workflows")
    data: Union[Selector(kind=[IMAGE_KIND, STRING_KIND]), str] = Field(
        title="Data",
        description="The image or text string to generate an embedding for. CLIP can process both images and text, generating embeddings in the same semantic space so they can be compared for similarity. For text inputs, embeddings are cached for efficiency.",
        examples=["$inputs.image", "$steps.cropping.crops", "a red car", "$inputs.text_query"],
    )

    version: Union[
        Literal[
            "RN101",
            "RN50",
            "RN50x16",
            "RN50x4",
            "RN50x64",
            "ViT-B-16",
            "ViT-B-32",
            "ViT-L-14-336px",
            "ViT-L-14",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="ViT-B-32",
        description="The CLIP model variant to use. Options include ResNet variants (RN50, RN101, RN50x4, RN50x16, RN50x64) and Vision Transformer variants (ViT-B-16, ViT-B-32, ViT-L-14, ViT-L-14-336px). Larger models (e.g., ViT-L-14, RN50x64) are more accurate but slower and require more resources. ViT-B-32 (default) offers a good balance of accuracy and speed.",
        examples=["ViT-B-32", "ViT-B-16", "ViT-L-14", "$inputs.model_version"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="embedding", kind=[EMBEDDING_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


text_cache = LRUCache()


class ClipModelBlockV1(WorkflowBlock):

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

            inference_request = ClipTextEmbeddingRequest(
                clip_version_id=version,
                text=[data],
                api_key=self._api_key,
            )
            clip_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="clip",
            )
            predictions = self._model_manager.infer_from_request_sync(
                clip_model_id, inference_request
            )

            text_cache.set(hash_key, predictions.embeddings[0])

            return {"embedding": predictions.embeddings[0]}
        else:
            inference_request = ClipImageEmbeddingRequest(
                clip_version_id=version,
                image=[data.to_inference_format(numpy_preferred=True)],
                api_key=self._api_key,
            )
            clip_model_id = load_core_model(
                model_manager=self._model_manager,
                inference_request=inference_request,
                core_model="clip",
            )
            predictions = self._model_manager.infer_from_request_sync(
                clip_model_id, inference_request
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
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()

        if isinstance(data, str):
            result = client.get_clip_text_embeddings(
                text=data,
                clip_version=version,
            )
        else:
            result = client.get_clip_image_embeddings(
                inference_input=data.base64_image,
                clip_version=version,
            )

        return {"embedding": result["embeddings"][0]}
