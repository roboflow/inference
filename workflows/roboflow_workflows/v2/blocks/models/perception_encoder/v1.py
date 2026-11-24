"""Embed images or texts with a Meta Perception Encoder model.

Port of V1 ``roboflow_core/perception_encoder@v1``; the same contract as
``roboflow_core/clip@v1`` with another default model. Text embeddings are
cached per step (``TextEmbeddingCache``): fixed labels run the text encoder once.
"""

from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import Field, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models._shared.embeddings import (
    DEFAULT_PERCEPTION_ENCODER_MODEL_ID,
    TASK_TYPE,
    TextEmbeddingCache,
    embed_values,
    embedding_model,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    model_request,
    models_cache,
    require_rgb,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.predictions import EMBEDDING_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)


class PerceptionEncoderEmbeddingBlock(Block):
    """Embed images or texts with a Meta Perception Encoder model.

    Same contract as ``roboflow_core/clip@v1``; only the default model differs.
    """

    type = "roboflow_core/perception_encoder@v1"
    metadata = {
        "name": "Perception Encoder Embedding Model",
        "port_of": "roboflow_core/perception_encoder@v1",
        "section": "model",
    }
    outputs = {
        "embedding": Output(
            EMBEDDING_KIND,
            source="data",
            description=(
                "Unnormalized 1-D embedding, on the model's device and in its dtype."
            ),
        ),
    }

    class Params(BlockParams):
        model_id: StrictStr | Ref(STRING_KIND) = Field(
            default=DEFAULT_PERCEPTION_ENCODER_MODEL_ID,
            description=(
                "Model id (or local path) of the model in the 'models' resource, "
                "one for every image of the step; a selector may choose it per "
                "run, not per image. To serve images with different models, "
                "route them to one step per model."
            ),
            examples=[
                DEFAULT_PERCEPTION_ENCODER_MODEL_ID,
                "perception-encoder/PE-Core-B16-224",
                "$inputs.model_id",
            ],
        )
        model_package_id: Optional[StrictStr] | Ref(
            STRING_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                "Exact model package to load, one for every image of the step; "
                "None lets the loader choose. A selector may choose it per run, "
                "not per image. A model supplied without a package id rejects a "
                "package. List a model's package ids with "
                'inference_models.AutoModel.describe_model("<model_id>").'
            ),
            examples=[None, "$inputs.model_package_id"],
        )
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        data: StrictStr | Ref(IMAGE_KIND, STRING_KIND, batch="always") = (  # noqa: F821
            Field(
                description=(
                    "Image (RGB) or text to embed. A selector delivers all images "
                    "or texts of a run to the model as one batch."
                ),
                examples=["$inputs.image", "$steps.crop.crops", "a photo of a dog"],
            )
        )

        @field_validator("model_id", "model_package_id")
        @classmethod
        def _reject_empty(cls, value: Optional[str]) -> Optional[str]:
            if value == "":
                raise ValueError("must be a non-empty string")

            return value

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)
        self._texts = TextEmbeddingCache()

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the model this step needs from the ``models`` resource."""
        resources = [
            DependentResource(
                resource_type="model",
                identifier=params.model_id,
                details={
                    "resource": "models",
                    "task_type": TASK_TYPE,
                    "model_package_id": params.model_package_id,
                },
            )
        ]

        return resources

    def run(
        self,
        *,
        data: Union[str, Batch],
        model_id: str,
        model_package_id: Optional[str] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Embed each image or text with the call's model.

        Returns:
            ``{"embedding"}`` for one literal text, otherwise one per row, in
            row order. An empty batch loads no model.

        Raises:
            ValueError: On a non-RGB image, or a ``Batch`` of model or package
                ids.
            TypeError: When the model is not a text-image embedding model.
        """
        values = data.content if isinstance(data, Batch) else [data]
        for value in values:
            if isinstance(value, ImageData):
                require_rgb(value, block=self.type)
        request = model_request(model_id, model_package_id, block=self.type)
        if not values:
            return []

        model = embedding_model(
            self._models.get(request), request=request, block=self.type
        )
        embeddings = embed_values(
            values, model=model, request=request, texts=self._texts
        )
        if not isinstance(data, Batch):
            return {"embedding": embeddings[0]}

        results = [{"embedding": embedding} for embedding in embeddings]

        return results
