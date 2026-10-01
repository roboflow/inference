import json
from concurrent.futures import ThreadPoolExecutor
from email import policy
from email.parser import BytesParser
from types import ModuleType

import numpy as np
import pytest
import requests
import torch
from requests_mock import Mocker

from inference.core import roboflow_api
from inference.core.active_learning.cache_operations import (
    get_current_strategy_limit_usage,
)
from inference.core.active_learning.entities import StrategyLimitType
from inference.core.cache import MemoryCache
from inference.core.exceptions import (
    RoboflowAPIIAnnotationRejectionError,
    RoboflowAPIImageUploadRejectionError,
    RoboflowAPIRequestError,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.roboflow_workflows_plugin.sinks.dataset_upload import v1, v1_tensor
from inference_models.models.base.classification import ClassificationPrediction

_API_URL = "https://dataset-upload.example.test"
_UPLOAD_URL = f"{_API_URL}/dataset/project/upload"
_ANNOTATION_URL = f"{_API_URL}/dataset/project/annotate/image-id"
_SUCCESS = {"success": True, "id": "image-id"}


@pytest.fixture(params=[v1, v1_tensor], ids=["numpy", "tensor"])
def implementation(request: pytest.FixtureRequest) -> ModuleType:
    return request.param


@pytest.fixture(autouse=True)
def configure_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(roboflow_api, "API_BASE_URL", _API_URL)
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", False)


@pytest.fixture
def registration(implementation: ModuleType) -> dict:
    prediction = {"top": "cat", "inference_id": "prediction-id"}
    if implementation is v1_tensor:
        prediction = ClassificationPrediction(
            class_id=torch.tensor([0]),
            confidence=torch.tensor([[1.0]]),
            images_metadata=[
                {"class_names": {0: "cat"}, "inference_id": "prediction-id"}
            ],
        )

    return {
        "target_project": "project",
        "encoded_image": b"jpeg image",
        "local_image_id": "local-id",
        "prediction": prediction,
        "api_key": "test-key",
        "batch_name": "batch",
        "tags": ["tag"],
        "metadata": {"camera": "test-camera"},
    }


@pytest.mark.parametrize("stage", ["upload", "annotation"])
@pytest.mark.parametrize(
    "failure",
    [
        pytest.param({"status_code": status}, id=f"http-{status}")
        for status in [500, 502, 503, 504]
    ]
    + [
        pytest.param({"exc": requests.exceptions.ConnectionError}, id="connection"),
        pytest.param({"exc": requests.exceptions.Timeout}, id="timeout"),
    ],
)
def test_retries_only_the_failed_request(
    implementation: ModuleType,
    registration: dict,
    requests_mock: Mocker,
    stage: str,
    failure: dict,
) -> None:
    responses = [failure, {"json": _SUCCESS}]
    upload = requests_mock.post(
        _UPLOAD_URL, responses if stage == "upload" else [{"json": _SUCCESS}]
    )
    annotation = requests_mock.post(
        _ANNOTATION_URL, responses if stage == "annotation" else [{"json": _SUCCESS}]
    )

    result = implementation.register_datapoint(**registration)

    assert result == "Successfully registered image and annotation"
    assert upload.call_count == (2 if stage == "upload" else 1)
    assert annotation.call_count == (2 if stage == "annotation" else 1)
    assert all(request.text == "cat" for request in annotation.request_history)
    assert all(
        request.qs["prediction"] == ["true"] for request in annotation.request_history
    )


def test_upload_retry_preserves_image_and_metadata(
    implementation: ModuleType,
    registration: dict,
    requests_mock: Mocker,
) -> None:
    uploaded_fields = []

    def consume_upload(request, context):
        body = request.body
        if hasattr(body, "read"):
            body = body.read()
        if isinstance(body, str):
            body = body.encode("utf-8")

        content_type = request.headers["Content-Type"]
        multipart = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii") + body
        )
        uploaded_fields.append(
            {
                part.get_param("name", header="Content-Disposition"): part.get_payload(
                    decode=True
                )
                for part in multipart.iter_parts()
            }
        )
        context.status_code = 503 if len(uploaded_fields) == 1 else 200

        return _SUCCESS

    requests_mock.post(_UPLOAD_URL, json=consume_upload)
    requests_mock.post(_ANNOTATION_URL, json=_SUCCESS)

    result = implementation.register_datapoint(**registration)

    assert result == "Successfully registered image and annotation"
    assert len(uploaded_fields) == 2
    for fields in uploaded_fields:
        assert fields["file"] == b"jpeg image"
        assert fields["name"] == b"local-id.jpg"
        assert json.loads(fields["metadata"]) == {"camera": "test-camera"}
    assert all(
        request.qs["inference_id"] == ["prediction-id"]
        for request in requests_mock.request_history
        if request.path.endswith("/upload")
    )


@pytest.mark.parametrize("stage", ["upload", "annotation"])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 501])
def test_permanent_http_errors_are_not_retried(
    implementation: ModuleType,
    registration: dict,
    requests_mock: Mocker,
    stage: str,
    status: int,
) -> None:
    upload = requests_mock.post(
        _UPLOAD_URL, status_code=status if stage == "upload" else 200, json=_SUCCESS
    )
    annotation = requests_mock.post(_ANNOTATION_URL, status_code=status)

    with pytest.raises(RoboflowAPIRequestError):
        implementation.register_datapoint(**registration)

    assert upload.call_count == 1
    assert annotation.call_count == (0 if stage == "upload" else 1)


@pytest.mark.parametrize(
    "stage,expected_error",
    [
        ("upload", RoboflowAPIImageUploadRejectionError),
        ("annotation", RoboflowAPIIAnnotationRejectionError),
    ],
)
def test_application_rejections_are_not_retried(
    implementation: ModuleType,
    registration: dict,
    requests_mock: Mocker,
    stage: str,
    expected_error: type[Exception],
) -> None:
    upload = requests_mock.post(
        _UPLOAD_URL,
        json={"success": False} if stage == "upload" else _SUCCESS,
    )
    annotation = requests_mock.post(_ANNOTATION_URL, json={"success": False})

    with pytest.raises(expected_error):
        implementation.register_datapoint(**registration)

    assert upload.call_count == 1
    assert annotation.call_count == (0 if stage == "upload" else 1)


@pytest.mark.parametrize("after_timeout", [False, True])
def test_annotation_conflict_reports_failure_and_returns_quota(
    implementation: ModuleType,
    execution: dict,
    requests_mock: Mocker,
    after_timeout: bool,
) -> None:
    upload = requests_mock.post(_UPLOAD_URL, json=_SUCCESS)
    responses = [{"exc": requests.exceptions.Timeout}] if after_timeout else []
    responses.append({"status_code": 409})
    annotation = requests_mock.post(
        _ANNOTATION_URL,
        responses,
    )

    error_status, message = implementation.execute_registration(**execution)

    assert error_status is True
    assert "RoboflowAPIIAlreadyAnnotatedError" in message
    assert upload.call_count == 1
    assert annotation.call_count == (2 if after_timeout else 1)
    assert all("overwrite" not in request.qs for request in annotation.request_history)
    for limit_type in StrategyLimitType:
        assert (
            get_current_strategy_limit_usage(
                cache=execution["cache"],
                workspace="workspace",
                project="project",
                strategy_name="quota",
                limit_type=limit_type,
            )
            == 0
        )


def test_duplicate_after_upload_timeout_does_not_reannotate(
    implementation: ModuleType,
    registration: dict,
    requests_mock: Mocker,
) -> None:
    upload = requests_mock.post(
        _UPLOAD_URL,
        [
            {"exc": requests.exceptions.Timeout},
            {"json": {"duplicate": True, "id": "image-id"}},
        ],
    )
    annotation = requests_mock.post(_ANNOTATION_URL, json=_SUCCESS)

    result = implementation.register_datapoint(**registration)

    assert result == "Duplicated image"
    assert upload.call_count == 2
    assert annotation.call_count == 0


@pytest.fixture
def execution(registration: dict, requests_mock: Mocker) -> dict:
    requests_mock.get(_API_URL, json={"workspace": "workspace"})

    return {
        "image": WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="parent"),
            numpy_image=np.zeros((16, 16, 3), dtype=np.uint8),
        ),
        "prediction": registration["prediction"],
        "target_project": "project",
        "usage_quota_name": "quota",
        "persist_predictions": True,
        "minutely_usage_limit": 10,
        "hourly_usage_limit": 100,
        "daily_usage_limit": 1000,
        "max_image_size": (16, 16),
        "compression_level": 75,
        "registration_tags": ["tag"],
        "labeling_batch_prefix": "batch",
        "new_labeling_batch_frequency": "never",
        "cache": MemoryCache(),
        "api_key": "test-key",
    }


@pytest.mark.parametrize("stage", ["upload", "annotation"])
@pytest.mark.parametrize("exhausted", [False, True])
def test_retry_outcome_preserves_quota_accounting(
    implementation: ModuleType,
    execution: dict,
    requests_mock: Mocker,
    stage: str,
    exhausted: bool,
) -> None:
    responses = [{"status_code": 503}] * 3
    if not exhausted:
        responses[-1] = {"json": _SUCCESS}
    upload = requests_mock.post(
        _UPLOAD_URL, responses if stage == "upload" else [{"json": _SUCCESS}]
    )
    annotation = requests_mock.post(
        _ANNOTATION_URL, responses if stage == "annotation" else [{"json": _SUCCESS}]
    )

    error_status, message = implementation.execute_registration(**execution)

    assert error_status is exhausted
    if exhausted:
        assert "RoboflowAPIUnsuccessfulRequestError" in message
        assert "503" in message
    else:
        assert message == "Successfully registered image and annotation"
    assert upload.call_count == (3 if stage == "upload" else 1)
    expected_annotations = 0 if stage == "upload" and exhausted else 1
    assert annotation.call_count == (
        3 if stage == "annotation" else expected_annotations
    )
    for limit_type in StrategyLimitType:
        assert get_current_strategy_limit_usage(
            cache=execution["cache"],
            workspace="workspace",
            project="project",
            strategy_name="quota",
            limit_type=limit_type,
        ) == (0 if exhausted else 1)


def test_fire_and_forget_retries_inside_the_background_task(
    implementation: ModuleType,
    execution: dict,
    requests_mock: Mocker,
) -> None:
    upload = requests_mock.post(_UPLOAD_URL, json=_SUCCESS)
    annotation = requests_mock.post(
        _ANNOTATION_URL, [{"status_code": 503}, {"json": _SUCCESS}]
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = implementation.register_datapoint_at_roboflow(
            **execution,
            fire_and_forget=True,
            background_tasks=None,
            thread_pool_executor=executor,
        )

    assert result == (False, "Element registration happens in the background task")
    assert upload.call_count == 1
    assert annotation.call_count == 2
