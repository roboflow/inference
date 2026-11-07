"""Envelope pairing classifier embeddings with the metadata of their space."""

from __future__ import annotations

from typing import Any, Dict


def describe_image_embeddings(
    model: Any, embeddings: Any, kwargs: Dict[str, Any]
) -> dict:
    """Describe the space of embeddings a model has just produced.

    Args:
        model: Classifier or pretrained embedding model that generated the
            embeddings.
        embeddings: Batched embedding tensor returned by the model.
        kwargs: Keyword arguments the model method was invoked with.

    Returns:
        Mapping with the tensor under ``embeddings`` and, under
        ``embedding_info``, the feature definition, vector dimension, the
        model's preprocessing configuration, backend class name and output
        dtype.
    """
    if hasattr(model, "embedding_preprocessing"):
        preprocessing = model.embedding_preprocessing
    else:
        config = model._inference_config
        preprocessing = {
            "image_pre_processing": config.image_pre_processing.model_dump(mode="json"),
            "network_input": config.network_input.model_dump(mode="json"),
        }
    info = dict(model.get_embedding_info(kwargs.get("output_type")))
    info.update(
        dimension=int(embeddings.shape[1]),
        preprocessing=preprocessing,
        backend=type(model).__name__,
        precision=str(embeddings.dtype),
    )
    result = {"embeddings": embeddings, "embedding_info": info}

    return result
