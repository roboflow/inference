from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import Response

from inference_model_manager.serializers_typed import (
    serialize_detections_compact,
    serialize_detections_rich,
    serialize_embeddings,
    serialize_text,
)
from inference_server.framework.entities import CommonRequestParams


class _ModelProxy:
    __slots__ = ("class_names",)

    def __init__(self, class_names: Optional[list]):
        self.class_names = class_names


def _serialize_text_one(prediction: Any) -> Any:
    proxy = _ModelProxy(class_names=None)
    return serialize_text(prediction, proxy)


def _serialize_detections_one(prediction: Any, style: str) -> Any:
    proxy = _ModelProxy(class_names=None)
    serializer = (
        serialize_detections_rich if style == "rich" else serialize_detections_compact
    )
    return serializer(prediction, proxy)


def _envelope(predictions: list, common: CommonRequestParams) -> Response:
    envelope = {
        "type": "roboflow-inference-server-response-v1",
        "model_info": {"model_id": common.model_id, "task": "vlm"},
        "usage": {},
        "predictions": predictions,
    }
    return Response(
        content=json.dumps(envelope, default=str).encode(),
        media_type="application/json",
    )


def serialize_vlm_text(prediction: Any, common: CommonRequestParams) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    typed = [_serialize_text_one(p) for p in items]
    return _envelope(typed, common)


def serialize_vlm_detections(prediction: Any, common: CommonRequestParams) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    style = common.response_style
    typed = [_serialize_detections_one(p, style) for p in items]
    return _envelope(typed, common)


def serialize_vlm_embeddings(prediction: Any, common: CommonRequestParams) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    proxy = _ModelProxy(class_names=None)
    typed = [serialize_embeddings(p, proxy) for p in items]
    return _envelope(typed, common)


serialize_vlm = serialize_vlm_text
