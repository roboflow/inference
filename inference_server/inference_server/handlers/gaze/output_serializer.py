from __future__ import annotations

from typing import Any

from fastapi import Response

from inference_model_manager.serializers_typed import serialize_gaze_compact
from inference_server.framework.entities import CommonRequestParams
from inference_server.serializers import serialize_json


def serialize_gaze(prediction: Any, common: CommonRequestParams) -> Response:
    items = prediction if isinstance(prediction, list) else [prediction]
    typed = [serialize_gaze_compact(p, None) for p in items]
    envelope = {
        "type": "roboflow-inference-server-response-v1",
        "model_info": {"model_id": common.model_id, "task": "gaze-detection"},
        "usage": {},
        "predictions": typed,
    }
    return Response(
        content=serialize_json(envelope),
        media_type="application/json",
    )
