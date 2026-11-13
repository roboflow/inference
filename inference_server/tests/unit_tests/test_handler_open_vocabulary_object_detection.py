from __future__ import annotations

import json

import pytest
import torch
from fastapi import Request

from inference_models import Detections
from inference_server.framework.dispatch import decode_common_request_params
from inference_server.handlers.open_vocabulary_object_detection.input_parser import (
    parse_open_vocabulary_object_detection_input,
)
from inference_server.handlers.open_vocabulary_object_detection.output_serializer import (
    serialize_open_vocabulary_object_detection,
)

_JPEG = bytes(
    [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46]
    + [0x00] * 12
    + [0xFF, 0xD9]
)


def _request(query: bytes, content_type: bytes, body: bytes) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v2/models/infer",
        "query_string": query,
        "headers": [(b"content-type", content_type)],
    }
    state = {"sent": False}

    async def recv():
        if state["sent"]:
            return {"type": "http.disconnect"}
        state["sent"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, recv)


def _detections() -> Detections:
    return Detections(
        xyxy=torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        class_id=torch.tensor([0, 1]),
        confidence=torch.tensor([0.9, 0.8]),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, content_type, body",
    [
        (
            b"model_id=m&classes=cat&classes=dog",
            b"application/octet-stream",
            _JPEG,
        ),
        (
            b"model_id=m",
            b"application/json",
            json.dumps(
                {
                    "inputs": {
                        "image": {
                            "type": "base64",
                            "value": "/9j/4AAQSkZJRgAAAAAAAAAAAAAAAP/Z",
                        },
                        "classes": ["cat", "dog"],
                    }
                }
            ).encode(),
        ),
    ],
)
async def test_response_labels_follow_the_classes_the_model_received(
    query, content_type, body
):
    request = _request(query, content_type, body)
    common = decode_common_request_params(request)

    parsed = await parse_open_vocabulary_object_detection_input(request, common)
    response = serialize_open_vocabulary_object_detection([_detections()], common)

    assert parsed["params"]["classes"] == ["cat", "dog"]
    assert json.loads(response.body)["predictions"][0]["class_names"] == ["cat", "dog"]
