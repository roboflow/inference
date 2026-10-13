"""Image and text embeddings with host-owned models: CLIP and Perception Encoder.

Ports of V1 ``roboflow_core/clip@v1`` and ``roboflow_core/perception_encoder@v1``.
Both blocks share one implementation and differ only in their default model::

    data      one image or text per row; a literal text is one call
    model     ModelRequest(model_id, model_package_id) per row, all resolved first
    images    model.embed_images(images of one request, "rgb")   -> (N, D)
    texts     TextEmbeddingCache.embed(texts of one request)     -> (N, D)
    result    row i -> embedding (D,), on the model's device and in its dtype

Text embeddings are cached per step: up to ``TEXT_CACHE_SIZE`` (model request,
text) pairs, least recently used dropped first. Fixed labels therefore run the
text encoder once; texts that change every frame stay correct and only cost
an ``embed_text`` call. The cache lives as long as the step instance; a
processing reset starts empty. The models themselves stay in the host's
``ModelCache``.
"""

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import torch
from pydantic import Field, StrictStr
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import (
    MODEL_ID_DESCRIPTION,
    ModelIdField,
    ModelSelectionParams,
    RequestField,
    model_dependency,
    models_cache,
    per_row,
    require_rgb,
    rows_by_request,
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

TASK_TYPE = "embedding"
TEXT_CACHE_SIZE = 1024
DEFAULT_CLIP_MODEL_ID = "clip/ViT-B-32"
DEFAULT_PERCEPTION_ENCODER_MODEL_ID = "perception-encoder/PE-Core-L14-336"


class TextEmbeddingCache:
    """Recently used text embeddings of one step, keyed by request and text.

    Entries are private copies, so a caller mutating a returned tensor cannot
    change a later result. Two concurrent misses of one text may both run the
    text encoder; the result is the same.

    Args:
        capacity: Most (request, text) pairs kept; 0 disables caching.

    Raises:
        ValueError: When ``capacity`` is negative.
    """

    def __init__(self, capacity: int = TEXT_CACHE_SIZE):
        if capacity < 0:
            raise ValueError(f"capacity must be at least 0, got {capacity}")

        self._capacity = capacity
        self._lock = threading.Lock()
        self._entries: "OrderedDict[Tuple[ModelRequest, str], torch.Tensor]" = (
            OrderedDict()
        )

    def __len__(self) -> int:
        return len(self._entries)

    def embed(
        self, texts: Sequence[str], *, model: Any, request: ModelRequest
    ) -> torch.Tensor:
        """Embed ``texts`` with ``model``, running it only for uncached texts.

        Args:
            texts: Texts to embed; repeats are allowed.
            model: The loaded model of ``request``, with ``embed_text``.
            request: Key of the model, part of every cache key.

        Returns:
            A new ``(len(texts), D)`` tensor, rows in ``texts`` order.

        Raises:
            ValueError: When the model returns another number of embeddings.
        """
        with self._lock:
            found = {}
            for text in texts:
                entry = self._entries.get((request, text))
                if entry is not None:
                    self._entries.move_to_end((request, text))
                    found[text] = entry
        missing = list(dict.fromkeys(text for text in texts if text not in found))

        if missing:
            computed = checked_embeddings(
                model.embed_text(missing), count=len(missing), request=request
            )
            # One copy per row: an evicted entry frees its own memory only.
            fresh = {text: row.clone() for text, row in zip(missing, computed)}
            found.update(fresh)
            self._store(fresh, request=request)
        embeddings = torch.stack([found[text] for text in texts])

        return embeddings

    def _store(
        self, fresh: Mapping[str, torch.Tensor], *, request: ModelRequest
    ) -> None:
        with self._lock:
            for text, embedding in fresh.items():
                self._entries[(request, text)] = embedding
                self._entries.move_to_end((request, text))
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)


def checked_embeddings(
    embeddings: Any, *, count: int, request: ModelRequest
) -> torch.Tensor:
    """Check that a model returned ``(count, D)`` floating embeddings.

    Args:
        embeddings: What ``embed_images`` or ``embed_text`` returned.
        count: Number of inputs given to the model.
        request: The model's request, for error messages.

    Returns:
        ``embeddings`` unchanged.

    Raises:
        ValueError: On another type, rank, row count or a non-floating dtype.
    """
    valid = (
        isinstance(embeddings, torch.Tensor)
        and embeddings.dim() == 2
        and embeddings.shape[0] == count
        and embeddings.is_floating_point()
    )
    if not valid:
        found = (
            f"{tuple(embeddings.shape)} {embeddings.dtype}"
            if isinstance(embeddings, torch.Tensor)
            else type(embeddings).__name__
        )
        raise ValueError(
            f"model {str(request)!r} returned {found} for {count} inputs; "
            f"expected a floating ({count}, D) tensor"
        )

    return embeddings


def embedding_model(model: Any, *, request: ModelRequest, block: str) -> Any:
    """Return ``model`` when it has ``embed_images`` and ``embed_text``.

    Args:
        model: The model the cache returned for ``request``.
        request: The request, for error messages.
        block: Block type, for error messages.

    Returns:
        ``model`` unchanged.

    Raises:
        TypeError: When the model is not a text-image embedding model.
    """
    missing = [
        method
        for method in ("embed_images", "embed_text")
        if not callable(getattr(model, method, None))
    ]
    if missing:
        raise TypeError(
            f"{block}: model {str(request)!r} ({type(model).__name__}) has no "
            f"{' or '.join(missing)}; choose a text-image embedding model such "
            f"as {DEFAULT_CLIP_MODEL_ID!r} or {DEFAULT_PERCEPTION_ENCODER_MODEL_ID!r}"
        )

    return model


class _EmbeddingParams(ModelSelectionParams):
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


class _EmbeddingBlock(Block):
    """Shared implementation of the CLIP and Perception Encoder blocks."""

    outputs = {
        "embedding": Output(
            EMBEDDING_KIND,
            source="data",
            description=(
                "Unnormalized 1-D embedding, on the model's device and in its dtype."
            ),
        ),
    }

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)
        self._texts = TextEmbeddingCache()

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the model this step needs from the ``models`` resource."""
        resources = model_dependency(params, task_type=TASK_TYPE)

        return resources

    def run(
        self,
        *,
        data: Union[str, Batch],
        model_id: RequestField,
        model_package_id: RequestField = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Embed each image or text with its row's model.

        Returns:
            ``{"embedding"}`` for one literal text, otherwise one per row, in
            row order.
        """
        batches = [
            field
            for field in (data, model_id, model_package_id)
            if isinstance(field, Batch)
        ]
        if not batches:
            embeddings = self._embed_rows(
                [data], model_id=model_id, model_package_id=model_package_id
            )
            return {"embedding": embeddings[0]}

        values = per_row(data, row_count=len(batches[0]), name="data")
        embeddings = self._embed_rows(
            values, model_id=model_id, model_package_id=model_package_id
        )
        results = [{"embedding": embedding} for embedding in embeddings]

        return results

    def _embed_rows(
        self,
        values: Sequence[Union[str, ImageData]],
        *,
        model_id: RequestField,
        model_package_id: RequestField,
    ) -> List[torch.Tensor]:
        # One model call per (request, images or texts); every model is
        # resolved and every image checked before the first call.
        for value in values:
            if isinstance(value, ImageData):
                require_rgb(value, block=self.type)
        request_rows = rows_by_request(
            len(values), model_id=model_id, model_package_id=model_package_id
        )
        models = {
            request: embedding_model(
                self._models.get(request), request=request, block=self.type
            )
            for request in request_rows
        }

        embeddings: List[Any] = [None] * len(values)
        for request, rows in request_rows.items():
            model = models[request]
            text_rows = [row for row in rows if isinstance(values[row], str)]
            image_rows = [row for row in rows if not isinstance(values[row], str)]
            if text_rows:
                texts = self._texts.embed(
                    [values[row] for row in text_rows], model=model, request=request
                )
                for row, embedding in zip(text_rows, texts):
                    embeddings[row] = embedding
            if image_rows:
                images = checked_embeddings(
                    model.embed_images(
                        [values[row].tensor_image for row in image_rows],
                        input_color_format="rgb",
                    ),
                    count=len(image_rows),
                    request=request,
                )
                for row, embedding in zip(image_rows, images):
                    embeddings[row] = embedding

        return embeddings


class ClipEmbeddingBlock(_EmbeddingBlock):
    """Embed images or texts with a CLIP model from the host's model cache.

    Compare the embeddings with ``v2/cosine_similarity@v1`` or follow their
    change with ``v2/identify_changes@v1``.
    """

    type = "v2/clip@v1"
    metadata = {
        "name": "CLIP Embedding Model",
        "port_of": "roboflow_core/clip@v1",
        "section": "model",
    }

    class Params(_EmbeddingParams):
        model_id: ModelIdField = Field(
            default=DEFAULT_CLIP_MODEL_ID,
            description=MODEL_ID_DESCRIPTION,
            examples=[DEFAULT_CLIP_MODEL_ID, "clip/ViT-B-16", "$inputs.model_id"],
        )


class PerceptionEncoderEmbeddingBlock(_EmbeddingBlock):
    """Embed images or texts with a Meta Perception Encoder model.

    Same contract as ``v2/clip@v1``; only the default model differs.
    """

    type = "v2/perception_encoder@v1"
    metadata = {
        "name": "Perception Encoder Embedding Model",
        "port_of": "roboflow_core/perception_encoder@v1",
        "section": "model",
    }

    class Params(_EmbeddingParams):
        model_id: ModelIdField = Field(
            default=DEFAULT_PERCEPTION_ENCODER_MODEL_ID,
            description=MODEL_ID_DESCRIPTION,
            examples=[
                DEFAULT_PERCEPTION_ENCODER_MODEL_ID,
                "perception-encoder/PE-Core-B16-224",
                "$inputs.model_id",
            ],
        )
