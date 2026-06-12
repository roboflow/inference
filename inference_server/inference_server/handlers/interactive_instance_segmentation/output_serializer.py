from __future__ import annotations

from typing import Any, Optional

from fastapi import Response

from inference_model_manager.serializers_typed import (
    serialize_embeddings,
    serialize_passthrough,
    serialize_text,
)
from inference_server.framework.entities import CommonRequestParams
from inference_server.serializers import serialize_json


class _ModelProxy:
    __slots__ = ("class_names",)

    def __init__(self, class_names: Optional[list]):
        self.class_names = class_names


def _envelope(predictions: list, common: CommonRequestParams) -> Response:
    envelope = {
        "type": "roboflow-inference-server-response-v1",
        "model_info": {
            "model_id": common.model_id,
            "task": "interactive-instance-segmentation",
        },
        "usage": {},
        "predictions": predictions,
    }
    return Response(
        content=serialize_json(envelope),
        media_type="application/json",
    )


def _embeddings_or_passthrough(prediction: Any, proxy: _ModelProxy) -> Any:
    try:
        return serialize_embeddings(prediction, proxy)
    except (AttributeError, TypeError, KeyError):
        return serialize_passthrough(prediction, proxy)


def serialize_sam_embeddings(prediction: Any, common: CommonRequestParams) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    proxy = _ModelProxy(class_names=None)
    typed = [_embeddings_or_passthrough(p, proxy) for p in items]
    return _envelope(typed, common)


def serialize_sam_segmentation(
    prediction: Any, common: CommonRequestParams
) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    proxy = _ModelProxy(class_names=None)
    typed = [serialize_passthrough(p, proxy) for p in items]
    return _envelope(typed, common)


def serialize_sam_text(prediction: Any, common: CommonRequestParams) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    proxy = _ModelProxy(class_names=None)
    typed = [serialize_text(p, proxy) for p in items]
    return _envelope(typed, common)
