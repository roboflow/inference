"""Zero-shot image classification by CLIP image-text similarity.

Port of V1 ``roboflow_core/clip_comparison@v2``::

    model     one ModelRequest for the call, resolved and labels checked first
    images    model.embed_images(the call's images, "rgb")              (N, D)
    labels    TextEmbeddingCache.embed(classes)                         (C, D)
    scores    normalize(images) @ normalize(labels).T                   (N, C)
    summary   max / argmax / min / argmin; one device-to-host copy per
              call for the float and class-name outputs

``classes`` is one label list for every image of a call, like a class
filter; a selector may change it between runs, not between images. Scores
are cosine similarities in [-1, 1], not probabilities. Label embeddings are
cached per step (see ``embeddings``): a fixed label list runs the text
encoder once, and labels that change between runs stay correct.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from inference_models.models.base.classification import ClassificationPrediction
from pydantic import Field, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models._shared.embeddings import (
    TextEmbeddingCache,
    checked_embeddings,
    embedding_model,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    model_request,
    models_cache,
    one_per_call,
    prediction_metadata,
    rgb_tensors,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    TENSOR_KIND,
)
from roboflow_workflows.v2.data.row_identity import with_classification_parent
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)


PREDICTION_TYPE = "classification"

DEFAULT_MODEL_ID = "clip/ViT-B-16"

class ClipComparisonBlock(Block):
    """Score each image against text labels with a CLIP model.

    Example: ``classes=["a cat", "a dog"]`` on a dog photo gives
    ``similarities`` like ``[0.19, 0.27]``, ``most_similar_class`` ``"a dog"``
    and ``least_similar_class`` ``"a cat"``. Ties pick the first label.
    """

    type = "roboflow_core/clip_comparison@v2"
    metadata = {
        "name": "Clip Comparison",
        "port_of": "roboflow_core/clip_comparison@v2",
        "section": "model",
    }
    outputs = {
        "similarities": Output(
            TENSOR_KIND,
            source="images",
            description=(
                "(C,) cosine similarity per label, in classes order, on the "
                "model's device."
            ),
        ),
        "max_similarity": Output(
            FLOAT_KIND, source="images", description="Highest similarity."
        ),
        "most_similar_class": Output(
            STRING_KIND, source="images", description="Label of the highest one."
        ),
        "min_similarity": Output(
            FLOAT_KIND, source="images", description="Lowest similarity."
        ),
        "least_similar_class": Output(
            STRING_KIND, source="images", description="Label of the lowest one."
        ),
        "classification_predictions": Output(
            CLASSIFICATION_PREDICTION_KIND,
            source="images",
            description=(
                "One-image ClassificationPrediction: class_id (1,) of the most "
                "similar label, confidence (1, C) the similarities (not "
                "probabilities)."
            ),
        ),
    }

    class Params(BlockParams):
        model_id: StrictStr | Ref(STRING_KIND) = Field(
            default=DEFAULT_MODEL_ID,
            description=(
                "Model id (or local path) of the model in the 'models' resource, "
                "one for every image of the step; a selector may choose it per "
                "run, not per image. To serve images with different models, "
                "route them to one step per model."
            ),
            examples=[DEFAULT_MODEL_ID, "clip/ViT-B-32", "$inputs.model_id"],
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
        images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to score; delivered as one batch.",
            examples=["$inputs.image", "$steps.crop.crops"],
        )
        classes: List[StrictStr] | Ref(LIST_OF_VALUES_KIND) = Field(
            min_length=1,
            description=(
                "Text labels to score every image against, at least one. A "
                "selector may choose the list per run, not per image."
            ),
            examples=[["a cat", "a dog"], "$inputs.classes"],
        )

        @field_validator("model_id", "model_package_id")
        @classmethod
        def _reject_empty(cls, value: Optional[str]) -> Optional[str]:
            if value == "":
                raise ValueError("must be a non-empty string")

            return value

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)
        self._labels = TextEmbeddingCache()

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
                    "task_type": PREDICTION_TYPE,
                    "model_package_id": params.model_package_id,
                },
            )
        ]

        return resources

    def run(
        self,
        *,
        images: Batch,
        classes: Sequence[str],
        model_id: str = DEFAULT_MODEL_ID,
        model_package_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Score every image of the batch against the labels.

        Returns:
            One result per image, in batch order.

        Raises:
            ValueError: On a selected label that is not a string, a ``Batch``
                of labels or model ids, or a non-RGB image.
        """
        labels = _checked_labels(classes, block=self.type)
        tensors = rgb_tensors(images, block=self.type)
        request = model_request(model_id, model_package_id, block=self.type)
        # An empty batch loads no model.
        if not tensors:
            return []

        model = embedding_model(
            self._models.get(request), request=request, block=self.type
        )

        image_embeddings = checked_embeddings(
            model.embed_images(tensors, input_color_format="rgb"),
            count=len(tensors),
            request=request,
        )
        label_embeddings = self._labels.embed(labels, model=model, request=request)
        scores = cosine_scores(image_embeddings, label_embeddings)
        results = comparison_results(scores, labels=labels, images=images.content)

        return results

def cosine_scores(
    image_embeddings: torch.Tensor, label_embeddings: torch.Tensor
) -> torch.Tensor:
    """Cosine similarity of every image to every label.

    Args:
        image_embeddings: ``(N, D)`` image embeddings.
        label_embeddings: ``(C, D)`` text embeddings.

    Returns:
        ``(N, C)`` similarities in [-1, 1]; a zero vector scores 0.
    """
    scores = (
        F.normalize(image_embeddings, dim=1) @ F.normalize(label_embeddings, dim=1).T
    )

    return scores

def comparison_results(
    scores: torch.Tensor, *, labels: Sequence[str], images: Sequence[ImageData]
) -> List[Dict[str, Any]]:
    """Each image's outputs from its row of ``scores``.

    Args:
        scores: ``(N, C)`` similarities of ``images`` to ``labels``.
        labels: The ``C`` labels, in score column order.
        images: The ``N`` scored images, for prediction metadata and parent row.

    Returns:
        One result per image; tensors stay on ``scores``' device.
    """
    best = scores.max(dim=1)
    worst = scores.min(dim=1)
    # The only device-to-host copy: four numbers per image.
    summaries = torch.stack(
        [
            part.to(torch.float64)
            for part in (best.values, worst.values, best.indices, worst.indices)
        ],
        dim=1,
    ).tolist()

    results = []
    for position, (image, summary) in enumerate(zip(images, summaries)):
        max_similarity, min_similarity, most_similar, least_similar = summary
        prediction = ClassificationPrediction(
            class_id=best.indices[position : position + 1],
            confidence=scores[position : position + 1],
            images_metadata=[
                prediction_metadata(
                    None,
                    image,
                    class_names=labels,
                    prediction_type=PREDICTION_TYPE,
                )
            ],
        )
        results.append(
            {
                "similarities": scores[position],
                "max_similarity": max_similarity,
                "most_similar_class": labels[int(most_similar)],
                "min_similarity": min_similarity,
                "least_similar_class": labels[int(least_similar)],
                "classification_predictions": with_classification_parent(
                    prediction, image.parent_row
                ),
            }
        )

    return results

def _checked_labels(labels: Sequence[Any], *, block: str) -> Tuple[str, ...]:
    # Params validation after resolution rejects an empty selected list but
    # not a non-string item, which would fail inside the text encoder.
    one_per_call(labels, name="classes", block=block)
    others = [label for label in labels if not isinstance(label, str)]
    if others:
        raise ValueError(
            f"{block}: classes must be text labels, got {others!r} in {list(labels)!r}"
        )

    return tuple(labels)
