"""Outbound-request parity between the numpy and tensor Vision Events sinks.

Both variants post the same event to the same Roboflow API endpoints; only the
prediction carrier they read differs. These tests pin that the two send the
same prepared requests (URL, headers, body bytes) for the same input, and that
both redact credentials from the error messages they return.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest import mock

import numpy as np
import pytest
import requests
import supervision as sv
import torch
from roboflow_workflows.core_steps.sinks.roboflow.vision_events import (
    v1 as vision_events_v1,
)
from roboflow_workflows.core_steps.sinks.roboflow.vision_events import (
    v1_tensor as vision_events_v1_tensor,
)
from roboflow_workflows.execution_engine.constants import CLASS_NAMES_KEY
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

from inference_models.models.base.object_detection import Detections

VISION_EVENTS_MODULES = [vision_events_v1, vision_events_v1_tensor]

_FIXED_EVENT_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")
_FIXED_NOW = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
_FIXED_MULTIPART_BOUNDARY = "vision-events-parity-boundary"


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return _FIXED_NOW if tz is not None else _FIXED_NOW.replace(tzinfo=None)


def _platform_client() -> mock.MagicMock:
    """A client that adds deployment headers and proxies URLs, like the server's."""
    platform_client = mock.MagicMock()
    platform_client.build_api_headers.side_effect = lambda explicit_headers=None: {
        "x-roboflow-inference-version": "0.0.0-test",
        "x-temporary-auth-token": "batch-token",
        **(explicit_headers or {}),
    }
    platform_client.wrap_url.side_effect = lambda url: f"https://gateway.test/?u={url}"
    return platform_client


def _image() -> WorkflowImageData:
    numpy_image = np.zeros((16, 24, 3), dtype=np.uint8)
    numpy_image[4:12, 6:18] = (10, 200, 30)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=numpy_image,
    )
    return image


def _prediction_for(module) -> Any:
    """The same single detection in the carrier each variant consumes."""
    if module is vision_events_v1:
        detections = sv.Detections(
            xyxy=np.array([[2.0, 4.0, 10.0, 12.0]]),
            class_id=np.array([0]),
            confidence=np.array([0.5]),
            data={"class_name": np.array(["cat"])},
        )
        return detections

    detections = Detections(
        xyxy=torch.tensor([[2.0, 4.0, 10.0, 12.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.5]),
        image_metadata={CLASS_NAMES_KEY: {0: "cat"}},
    )
    return detections


def _ok_response(*args, **kwargs) -> mock.MagicMock:
    response = mock.MagicMock(status_code=200)
    response.json.return_value = {"sourceId": "src-1", "url": ""}
    return response


def _prepared_requests(module) -> List[requests.PreparedRequest]:
    """Run the cloud path of `module` and prepare every request it sent."""
    with mock.patch.object(
        module.requests, "post", side_effect=_ok_response
    ) as post_mock, mock.patch.object(
        module, "uuid4", return_value=_FIXED_EVENT_ID
    ), mock.patch.object(
        module, "datetime", _FrozenDatetime
    ):
        error_status, message, event_id = module._execute_vision_event(
            api_base_url="https://api.roboflow.test",
            api_key="my-secret-key",
            input_image=_image(),
            output_image=_image(),
            prediction=_prediction_for(module),
            event_type="quality_check",
            solution="my-solution",
            event_data={"result": "pass", "externalId": "ext-1"},
            custom_metadata={"camera": "cam-1"},
            platform_client=_platform_client(),
            value_base_units=1999,
        )

    assert (error_status, message) == (False, "Vision event sent successfully")
    assert event_id == str(_FIXED_EVENT_ID)
    prepared = []
    with mock.patch(
        "urllib3.filepost.choose_boundary", return_value=_FIXED_MULTIPART_BOUNDARY
    ):
        for call in post_mock.call_args_list:
            request = requests.Request(
                method="POST",
                url=call.args[0],
                headers=call.kwargs.get("headers"),
                files=call.kwargs.get("files"),
                json=call.kwargs.get("json"),
            )
            prepared.append(request.prepare())
    return prepared


def _as_comparable(request: requests.PreparedRequest) -> Dict[str, Any]:
    comparable = {
        "method": request.method,
        "url": request.url,
        "headers": dict(request.headers),
        "body": request.body,
    }
    return comparable


def test_both_variants_send_identical_prepared_requests() -> None:
    # when
    numpy_requests = _prepared_requests(vision_events_v1)
    tensor_requests = _prepared_requests(vision_events_v1_tensor)

    # then
    assert [_as_comparable(r) for r in tensor_requests] == [
        _as_comparable(r) for r in numpy_requests
    ]
    assert [r.url for r in numpy_requests] == [
        "https://gateway.test/?u=https://api.roboflow.test/vision-events/upload",
        "https://gateway.test/?u=https://api.roboflow.test/vision-events/upload",
        "https://gateway.test/?u=https://api.roboflow.test/vision-events",
    ]
    event_body = numpy_requests[-1].body
    assert b'"objectDetections": [{"class": "cat"' in event_body
    assert b'"valueBaseUnits": 1999' in event_body


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_every_request_carries_the_shared_roboflow_api_headers(module) -> None:
    # when
    prepared = _prepared_requests(module)

    # then
    for request in prepared:
        assert request.headers["x-temporary-auth-token"] == "batch-token"
        assert request.headers["x-roboflow-inference-version"] == "0.0.0-test"
        assert request.headers["Authorization"] == "Bearer my-secret-key"
        assert "my-secret-key" not in request.url
    assert prepared[-1].headers["Content-Type"] == "application/json"


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_configured_authorization_header_wins_over_the_api_key(module) -> None:
    # given
    platform_client = mock.MagicMock()
    platform_client.build_api_headers.side_effect = lambda explicit_headers=None: {
        "Authorization": "Bearer configured-credential",
        **(explicit_headers or {}),
    }
    platform_client.wrap_url.side_effect = lambda url: url

    # when
    with mock.patch.object(
        module.requests, "post", side_effect=_ok_response
    ) as post_mock:
        module._upload_image(
            "https://api.roboflow.test",
            "dummy-workspace",
            _image(),
            platform_client=platform_client,
        )
        module._send_event(
            "https://api.roboflow.test",
            "dummy-workspace",
            {"eventId": "e"},
            platform_client=platform_client,
        )

    # then
    sent_authorization = [
        call.kwargs["headers"]["Authorization"] for call in post_mock.call_args_list
    ]
    assert sent_authorization == ["Bearer configured-credential"] * 2


def _failed_response(*args, **kwargs) -> requests.Response:
    response = requests.Response()
    response.status_code = 500
    response.reason = "Internal Server Error"
    response.url = (
        "https://api.roboflow.test/vision-events/upload"
        "?api_key=my-secret-key&service_secret=my-service-secret"
    )
    response._content = b"boom"
    return response


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_upload_failure_message_redacts_credentials(module) -> None:
    # when
    with mock.patch.object(module.requests, "post", side_effect=_failed_response):
        error_status, message, event_id = module._execute_vision_event(
            api_base_url="https://api.roboflow.test",
            api_key="my-secret-key",
            input_image=_image(),
            output_image=None,
            prediction=None,
            event_type="custom",
            solution="my-solution",
            event_data={},
            custom_metadata={},
            platform_client=_platform_client(),
        )

    # then
    assert error_status is True
    assert event_id == ""
    assert "500 Server Error" in message
    assert "api_key=my***ey" in message
    assert "service_secret=***" in message
    assert "my-secret-key" not in message
    assert "my-service-secret" not in message


@pytest.mark.parametrize("module", VISION_EVENTS_MODULES)
def test_send_event_failure_keeps_the_response_details(module) -> None:
    # when
    with mock.patch.object(module.requests, "post", side_effect=_failed_response):
        error_status, message = module._send_event(
            "https://api.roboflow.test",
            "my-secret-key",
            {"eventId": "e"},
            platform_client=_platform_client(),
        )

    # then
    assert error_status is True
    assert message == "Failed to send vision event. Status: 500. Details: boom"
