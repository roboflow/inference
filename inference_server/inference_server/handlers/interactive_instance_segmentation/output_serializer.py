from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import Response

from inference_model_manager.serializers_typed import (
    serialize_embeddings,
    serialize_text,
)

from inference_server.framework.entities import CommonRequestParams


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
        content=json.dumps(envelope, default=str).encode(),
        media_type="application/json",
    )


def serialize_sam_embeddings(
    prediction: Any, common: CommonRequestParams
) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    proxy = _ModelProxy(class_names=None)
    typed = [serialize_embeddings(p, proxy) for p in items]
    return _envelope(typed, common)


def serialize_sam_text(
    prediction: Any, common: CommonRequestParams
) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    proxy = _ModelProxy(class_names=None)
    typed = [serialize_text(p, proxy) for p in items]
    return _envelope(typed, common)
