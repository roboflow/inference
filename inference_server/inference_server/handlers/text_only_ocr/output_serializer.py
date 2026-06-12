from __future__ import annotations

from typing import Any, Optional

from fastapi import Response

from inference_model_manager.serializers_typed import serialize_text
from inference_server.framework.entities import CommonRequestParams
from inference_server.serializers import serialize_json


class _ModelProxy:
    __slots__ = ("class_names",)

    def __init__(self, class_names: Optional[list]):
        self.class_names = class_names


def _serialize_one(prediction: Any) -> Any:
    proxy = _ModelProxy(class_names=None)
    return serialize_text(prediction, proxy)


def serialize_text_only_ocr(prediction: Any, common: CommonRequestParams) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    typed = [_serialize_one(p) for p in items]
    envelope = {
        "type": "roboflow-inference-server-response-v1",
        "model_info": {"model_id": common.model_id, "task": "text-only-ocr"},
        "usage": {},
        "predictions": typed,
    }
    return Response(
        content=serialize_json(envelope),
        media_type="application/json",
    )
