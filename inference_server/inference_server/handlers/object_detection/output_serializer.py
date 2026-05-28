from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import Response

from inference_model_manager.serializers_typed import (
    serialize_detections_compact,
    serialize_detections_rich,
)

from inference_server.framework.entities import CommonRequestParams


class _ModelProxy:
    __slots__ = ("class_names",)

    def __init__(self, class_names: Optional[list]):
        self.class_names = class_names


def _serialize_one(prediction: Any, style: str) -> Any:
    proxy = _ModelProxy(class_names=None)
    serializer = (
        serialize_detections_rich if style == "rich" else serialize_detections_compact
    )
    return serializer(prediction, proxy)


def serialize_object_detection(
    prediction: Any, common: CommonRequestParams
) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    style = common.response_style
    typed = [_serialize_one(p, style) for p in items]
    envelope = {
        "type": "roboflow-inference-server-response-v1",
        "model_info": {"model_id": common.model_id, "task": "object-detection"},
        "usage": {},
        "predictions": typed,
    }
    return Response(
        content=json.dumps(envelope, default=str).encode(),
        media_type="application/json",
    )
