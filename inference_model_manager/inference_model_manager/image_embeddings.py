"""Envelope pairing classifier embeddings with the metadata of their space."""

from __future__ import annotations

from typing import Any, Dict


def describe_image_embeddings(
    model: Any, embeddings: Any, kwargs: Dict[str, Any]
) -> dict:
    """Describe the space of embeddings a classifier has just produced.

    Args:
        model: Classifier that generated the embeddings.
        embeddings: Batched embedding tensor returned by the model.
        kwargs: Keyword arguments the model method was invoked with.

    Returns:
        Mapping with the tensor under ``embeddings`` and, under
        ``embedding_info``, the feature definition, vector dimension, the
        model's preprocessing configuration, backend class name and output
        dtype.
    """
    config = model._inference_config
    info = dict(model.get_embedding_info(kwargs.get("output_type")))
    info.update(
        dimension=int(embeddings.shape[1]),
        preprocessing={
            "image_pre_processing": config.image_pre_processing.model_dump(mode="json"),
            "network_input": config.network_input.model_dump(mode="json"),
        },
        backend=type(model).__name__,
        precision=str(embeddings.dtype),
    )
    result = {"embeddings": embeddings, "embedding_info": info}

    return result
