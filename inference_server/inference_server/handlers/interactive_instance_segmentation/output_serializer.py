from __future__ import annotations

import dataclasses
from typing import Any, Optional

from fastapi import Response

from inference_model_manager.hash_namespacing import strip_tenant_namespace
from inference_model_manager.serializers_typed import (
    serialize_embeddings,
    serialize_passthrough,
    serialize_sam_segmentation_compact,
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


def _embeddings_or_passthrough(
    prediction: Any, proxy: _ModelProxy, common: CommonRequestParams
) -> Any:
    if hasattr(prediction, "image_hash"):
        stripped = strip_tenant_namespace(prediction.image_hash, common.api_key)
        if stripped != prediction.image_hash and dataclasses.is_dataclass(prediction):
            prediction = dataclasses.replace(prediction, image_hash=stripped)
        typed = serialize_passthrough(prediction, proxy)
        typed["type"] = "roboflow-sam-embeddings-v1"
        return typed
    return serialize_embeddings(prediction, proxy)


def serialize_sam_embeddings(prediction: Any, common: CommonRequestParams) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    proxy = _ModelProxy(class_names=None)
    typed = [_embeddings_or_passthrough(p, proxy, common) for p in items]
    return _envelope(typed, common)


def serialize_sam_masks(prediction: Any, common: CommonRequestParams) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    proxy = _ModelProxy(class_names=None)
    typed = [serialize_sam_segmentation_compact(p, proxy) for p in items]
    return _envelope(typed, common)


def serialize_sam_segmentation(
    prediction: Any, common: CommonRequestParams
) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    proxy = _ModelProxy(class_names=None)
    typed = [serialize_passthrough(p, proxy) for p in items]
    return _envelope(typed, common)
