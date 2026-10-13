"""Zero-shot image classification by CLIP image-text similarity.

Port of V1 ``roboflow_core/clip_comparison@v2``::

    model     ModelRequest per image, all resolved and labels checked first
    images    model.embed_images(images of one request, "rgb")          (N, D)
    labels    TextEmbeddingCache.embed(classes) per distinct label list (C, D)
    scores    normalize(images) @ normalize(labels).T                   (N, C)
    summary   max / argmax / min / argmin; one device-to-host copy per
              (request, label list) for the float and class-name outputs

Scores are cosine similarities in [-1, 1], not probabilities. Label
embeddings are cached per step (see ``embeddings``): a fixed label list runs
the text encoder once, and labels that change per image or run stay correct.
"""

from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from pydantic import Field, StrictStr
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    MODEL_ID_DESCRIPTION,
    ModelIdField,
    ModelSelectionParams,
    RequestField,
    model_dependency,
    models_cache,
    per_row,
    prediction_metadata,
    require_rgb,
    rows_by_request,
)
from roboflow_workflows.v2.blocks.models.embeddings import (
    TextEmbeddingCache,
    checked_embeddings,
    embedding_model,
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

from inference_models.models.base.classification import ClassificationPrediction

PREDICTION_TYPE = "classification"
DEFAULT_MODEL_ID = "clip/ViT-B-16"
ClassesField = Union[Sequence[str], Batch]


class ClipComparisonBlock(Block):
    """Score each image against text labels with a CLIP model.

    Example: ``classes=["a cat", "a dog"]`` on a dog photo gives
    ``similarities`` like ``[0.19, 0.27]``, ``most_similar_class`` ``"a dog"``
    and ``least_similar_class`` ``"a cat"``. Ties pick the first label.
    """

    type = "v2/clip_comparison@v1"
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

    class Params(ModelSelectionParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        images: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to score; delivered as one batch.",
            examples=["$inputs.image", "$steps.crop.crops"],
        )
        classes: List[StrictStr] | Ref(
            LIST_OF_VALUES_KIND, batch="if_varying"  # noqa: F821
        ) = Field(
            min_length=1,
            description=(
                "Text labels to score each image against, at least one. A "
                "selector may choose them per run or per image."
            ),
            examples=[["a cat", "a dog"], "$inputs.classes"],
        )
        model_id: ModelIdField = Field(
            default=DEFAULT_MODEL_ID,
            description=MODEL_ID_DESCRIPTION,
            examples=[DEFAULT_MODEL_ID, "clip/ViT-B-32", "$inputs.model_id"],
        )

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)
        self._labels = TextEmbeddingCache()

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the model this step needs from the ``models`` resource."""
        resources = model_dependency(params, task_type=PREDICTION_TYPE)

        return resources

    def run(
        self,
        *,
        images: Batch,
        classes: ClassesField,
        model_id: RequestField = DEFAULT_MODEL_ID,
        model_package_id: RequestField = None,
    ) -> List[Dict[str, Any]]:
        """Score every image of the batch against its labels.

        Returns:
            One result per image, in batch order.

        Raises:
            ValueError: On a selected label that is not a string, or a non-RGB
                image.
        """
        members = images.content
        row_count = len(members)
        row_classes = [
            _checked_labels(labels, block=self.type)
            for labels in per_row(classes, row_count=row_count, name="classes")
        ]
        for member in members:
            require_rgb(member, block=self.type)
        request_rows = rows_by_request(
            row_count, model_id=model_id, model_package_id=model_package_id
        )
        models = {
            request: embedding_model(
                self._models.get(request), request=request, block=self.type
            )
            for request in request_rows
        }

        results: List[Any] = [None] * row_count
        for request, rows in request_rows.items():
            model = models[request]
            image_embeddings = checked_embeddings(
                model.embed_images(
                    [members[row].tensor_image for row in rows],
                    input_color_format="rgb",
                ),
                count=len(rows),
                request=request,
            )
            positions_by_classes: Dict[Tuple[str, ...], List[int]] = {}
            for position, row in enumerate(rows):
                positions_by_classes.setdefault(row_classes[row], []).append(position)
            for labels, positions in positions_by_classes.items():
                label_embeddings = self._labels.embed(
                    labels, model=model, request=request
                )
                scores = cosine_scores(image_embeddings[positions], label_embeddings)
                group_images = [members[rows[position]] for position in positions]
                group_results = comparison_results(
                    scores, labels=labels, images=group_images
                )
                for position, result in zip(positions, group_results):
                    results[rows[position]] = result

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
    others = [label for label in labels if not isinstance(label, str)]
    if others:
        raise ValueError(
            f"{block}: classes must be text labels, got {others!r} in {list(labels)!r}"
        )

    return tuple(labels)
