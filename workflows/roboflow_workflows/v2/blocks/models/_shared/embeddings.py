"""Image and text embedding algorithms of the CLIP-style blocks.

Used by ``roboflow_core/clip@v1``, ``roboflow_core/perception_encoder@v1`` and
``roboflow_core/clip_comparison@v2``::

    images    model.embed_images(the call's images, "rgb")       -> (N, D)
    texts     TextEmbeddingCache.embed(the call's texts)         -> (N, D)
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
from typing import Any, List, Mapping, Sequence, Tuple, Union

import torch
from roboflow_workflows.v2.blocks.models.cache import ModelRequest
from roboflow_workflows.v2.data.image import ImageData


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


def embed_values(
    values: Sequence[Union[str, ImageData]],
    *,
    model: Any,
    request: ModelRequest,
    texts: TextEmbeddingCache,
) -> List[torch.Tensor]:
    """One embedding per value, in order: at most one text and one image call.

    Args:
        values: Texts and RGB images of one call.
        model: The call's text-image embedding model.
        request: The model's request, for the text cache and error messages.
        texts: The step's text embedding cache.

    Returns:
        One ``(D,)`` embedding per value, on the model's device and in its dtype.

    Raises:
        ValueError: When the model returns another number or shape of
            embeddings.
    """
    embeddings: List[Any] = [None] * len(values)
    text_rows = [row for row, value in enumerate(values) if isinstance(value, str)]
    image_rows = [row for row, value in enumerate(values) if not isinstance(value, str)]
    if text_rows:
        text_embeddings = texts.embed(
            [values[row] for row in text_rows], model=model, request=request
        )
        for row, embedding in zip(text_rows, text_embeddings):
            embeddings[row] = embedding
    if image_rows:
        image_embeddings = checked_embeddings(
            model.embed_images(
                [values[row].tensor_image for row in image_rows],
                input_color_format="rgb",
            ),
            count=len(image_rows),
            request=request,
        )
        for row, embedding in zip(image_rows, image_embeddings):
            embeddings[row] = embedding

    return embeddings
