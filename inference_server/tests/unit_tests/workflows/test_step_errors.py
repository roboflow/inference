import base64
import io
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import requests_mock
from inference_sdk.http.errors import HTTPCallErrorError
from PIL import Image
from roboflow_workflows.errors import (
    ClientCausedStepExecutionError,
    RuntimeLimitsCausedStepExecutionError,
)
from roboflow_workflows.prototypes.platform_errors import (
    RoboflowAPIConnectionError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
)

from inference_models.errors import (
    ForbiddenModelAccessError,
    ModelNotFoundError,
    ModelPackageAlternativesExhaustedError,
    ModelPackageRestrictedError,
    ModelRetrievalError,
    PaymentRequiredModelAccessError,
    RetryError,
    UnauthorizedModelAccessError,
)
from inference_server.errors import ServerBusyError
from inference_server.gateway import (
    ModelManagerGateway,
    ReloadAfterEvictionError,
    _redact_secrets,
)
from inference_server.legacy.bridge import Route
from inference_server.legacy.errors import (
    ImageFetchError,
    LegacyHTTPError,
    ModelNotReadyError,
)
from inference_server.workflows import host
from inference_server.workflows import router as router_mod
from inference_server.workflows.errors import (
    PaymentRequiredError,
    RoboflowAPIUsagePausedError,
)
from inference_server.workflows.models_provider import GatewayModelsProvider
from tests.unit_tests.legacy.conftest import (  # noqa: F401
    EvictedModelManager,
    FakeGateway,
    _reset_model_stat_cache,
)

STEP_CONTEXT = "workflow_execution | step_execution"
RESTRICTED = (
    "Model loading failed due to restrictions of server configuration - usually due "
    "to excessive runtime memory requirement of the model (for instance caused by "
    "large input size)."
)
NOT_READY = {"message": "Model is temporarily not ready - retry request."}
OD_WF = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "det",
            "image": "$inputs.image",
            "model_id": "ds/1",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.det.predictions",
        }
    ],
}


def _exhausted(alternatives_errors):
    return ModelPackageAlternativesExhaustedError(
        "none loaded", alternatives_errors=alternatives_errors
    )


@pytest.mark.parametrize(
    "error,raised_class,status_code,message",
    [
        (
            RoboflowAPINotAuthorizedError("denied"),
            ClientCausedStepExecutionError,
            401,
            "Unauthorized error occurred while execution of step det - details of "
            "error: denied. This error usually mean the problem with Roboflow API "
            "key.",
        ),
        (
            PaymentRequiredError("no credits"),
            ClientCausedStepExecutionError,
            402,
            "Not enough credits to execute step det. Verify your workspace billing "
            "page. Details: no credits",
        ),
        (
            RoboflowAPIForbiddenError("denied"),
            ClientCausedStepExecutionError,
            403,
            "Forbidden error occurred while execution of step det - details of "
            "error: denied. This error usually mean the problem with Roboflow API "
            "key.",
        ),
        (
            RoboflowAPIUsagePausedError("paused"),
            ClientCausedStepExecutionError,
            423,
            "Roboflow API usage is paused while executing step det. Contact your "
            "workspace administrator to re-enable API keys. Details: paused",
        ),
        (
            RoboflowAPINotNotFoundError("missing"),
            ClientCausedStepExecutionError,
            404,
            "Could not find requested Roboflow resource while execution of step det "
            "- details of error: missing. This error usually mean the problem with "
            "not existing model.",
        ),
        (
            _exhausted([ValueError("x"), ModelPackageRestrictedError("too big")]),
            RuntimeLimitsCausedStepExecutionError,
            507,
            RESTRICTED,
        ),
        (
            ModelPackageRestrictedError("too big"),
            RuntimeLimitsCausedStepExecutionError,
            507,
            RESTRICTED,
        ),
        (
            LegacyHTTPError(400, "At least one image is required"),
            ClientCausedStepExecutionError,
            400,
            "At least one image is required",
        ),
        (
            LegacyHTTPError(501, "quantize is not supported."),
            ClientCausedStepExecutionError,
            501,
            "quantize is not supported.",
        ),
        (
            ImageFetchError(502, "Could not fetch image from URL."),
            ClientCausedStepExecutionError,
            502,
            "Could not fetch image from URL.",
        ),
        (
            LegacyHTTPError(507, "too big"),
            RuntimeLimitsCausedStepExecutionError,
            507,
            "too big",
        ),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else None,
)
def test_step_error_handler_converts(error, raised_class, status_code, message):
    with pytest.raises(raised_class) as exc:
        host.step_error_handler("det", error)

    assert type(exc.value) is raised_class
    assert exc.value.status_code == status_code
    assert exc.value.public_message == message
    assert exc.value.context == STEP_CONTEXT
    assert exc.value.inner_error is error
    assert exc.value.inner_error.__class__.__name__ == type(error).__name__
    assert exc.value.__cause__ is error


def test_step_error_handler_reraises_server_busy_untouched():
    try:
        raise ServerBusyError("No free SHM slots") from TimeoutError(
            "No free SHM slots"
        )
    except ServerBusyError as error:
        busy = error
    try:
        raise RuntimeError("not busy") from TimeoutError("No free SHM slots")
    except RuntimeError as error:
        not_busy = error

    with pytest.raises(ServerBusyError) as exc:
        host.step_error_handler("det", busy)

    assert exc.value is busy
    assert host.step_error_handler("det", not_busy) is None


def test_step_error_handler_converts_malformed_pipeline_id_to_client_error():
    from inference_model_manager.pipelines import InvalidPipelineIdError

    cause = InvalidPipelineIdError("bad pipeline id")
    error = _wrapped(LookupError("bad pipeline id"), cause)

    with pytest.raises(ClientCausedStepExecutionError) as exc:
        host.step_error_handler("det", error)

    assert type(exc.value) is ClientCausedStepExecutionError
    assert exc.value.status_code == 400
    assert exc.value.public_message == (
        "Problem with Workflow Block configuration - bad pipeline id"
    )
    assert exc.value.context == STEP_CONTEXT
    assert exc.value.inner_error is cause
    assert exc.value.__cause__ is cause


@pytest.mark.parametrize(
    "api_message,message",
    [
        ("custom text", "custom text"),
        (None, "Remote execution of step det is not supported on this deployment."),
    ],
)
def test_step_error_handler_converts_remote_501(api_message, message):
    error = HTTPCallErrorError(
        description="d", status_code=501, api_message=api_message
    )

    with pytest.raises(ClientCausedStepExecutionError) as exc:
        host.step_error_handler("det", error)

    assert exc.value.status_code == 501
    assert exc.value.public_message == message
    assert exc.value.context == (
        "workflow_execution | step_execution | deployment_not_supported"
    )
    assert exc.value.inner_error.__class__.__name__ == (
        "ModelDeploymentNotSupportedError"
    )
    assert str(exc.value.inner_error) == message
    assert exc.value.__cause__ is error


def test_step_error_handler_reraises_model_not_ready_untouched():
    error = ModelNotReadyError()

    with pytest.raises(ModelNotReadyError) as exc:
        host.step_error_handler("det", error)

    assert exc.value is error
    assert exc.value.status_code == 503
    assert exc.value.message == NOT_READY["message"]
    assert exc.value.headers == {"Retry-After": "1"}


def _wrapped(wrapper, cause):
    try:
        raise wrapper from cause
    except Exception as error:
        return error


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("nope"),
        LookupError("ds/1"),
        _wrapped(PermissionError("nope"), UnauthorizedModelAccessError("nope")),
        _wrapped(LookupError("ds/1"), ModelNotFoundError("ds/1")),
        RetryError("down"),
        ModelRetrievalError("down"),
        _wrapped(RuntimeError("down"), RetryError("down")),
        _wrapped(RuntimeError("down"), OSError("down")),
        LegacyHTTPError(500, "Model package is broken."),
        LegacyHTTPError(502, "bad gateway"),
        LegacyHTTPError(503, "Model is temporarily not ready - retry request."),
        RoboflowAPIUnsuccessfulRequestError("code: 500"),
        RoboflowAPIConnectionError("down"),
        RoboflowAPITimeoutError("slow"),
        _exhausted([]),
        _exhausted(None),
        _exhausted([ValueError("x")]),
        ReloadAfterEvictionError("reload after eviction failed"),
    ],
    ids=lambda error: f"{type(error).__name__}-{error}",
)
def test_step_error_handler_leaves_unmatched(error):
    assert host.step_error_handler("det", error) is None


class _RaisingBridge:
    accepts_ndarray = True

    def __init__(self, error):
        self.error = error

    def resolve(self, model_id, api_key, **kwargs):
        raise self.error

    def record_request(self, *args, **kwargs):
        raise AssertionError("not reached")


@pytest.mark.parametrize(
    "wrapper,cause",
    [
        (PermissionError("denied"), UnauthorizedModelAccessError("denied")),
        (LookupError("missing"), ModelNotFoundError("missing")),
        (RuntimeError("no credits"), PaymentRequiredModelAccessError("no credits")),
        (RuntimeError("down"), RetryError("down")),
        (RuntimeError("down"), ModelRetrievalError("down")),
    ],
)
@pytest.mark.parametrize("entry", ["add_model", "get_class_names"])
def test_provider_raises_the_models_error_behind_a_lookup_wrapper(
    wrapper, cause, entry
):
    provider = GatewayModelsProvider(_RaisingBridge(_wrapped(wrapper, cause)), "k")

    with pytest.raises(type(cause)) as exc:
        if entry == "add_model":
            provider.add_model("ds/1", "k")
        else:
            provider.get_class_names("ds/1")

    assert exc.value is not cause
    assert type(exc.value) is type(cause)
    assert str(exc.value) == str(cause)
    assert exc.value.help_url == cause.help_url
    assert getattr(exc.value, "status_code", None) == getattr(
        cause, "status_code", None
    )


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("denied"),
        LookupError("missing"),
        RuntimeError("empty taskType"),
        _wrapped(RuntimeError("down"), OSError("down")),
        _wrapped(LookupError("bad id"), ValueError("bad id")),
        _wrapped(ServerBusyError("busy"), TimeoutError("busy")),
        ModelNotReadyError(),
    ],
    ids=lambda error: f"{type(error).__name__}-{error}",
)
def test_provider_keeps_an_error_without_a_models_error_cause(error):
    provider = GatewayModelsProvider(_RaisingBridge(error), "k")

    with pytest.raises(type(error)) as exc:
        provider.add_model("ds/1", "k")

    assert exc.value is error


LEAKY_MESSAGE = (
    "registry said api_key=SECRETKEY123 and "
    "https://storage.example/x?X-Goog-Signature=abc end"
)
LEAKY_ROOT = "payload https://storage.example/y?X-Goog-Signature=def"


def _leaky_chain():
    try:
        try:
            raise ValueError(LEAKY_ROOT)
        except ValueError as root:
            raise ModelRetrievalError(LEAKY_MESSAGE) from root
    except ModelRetrievalError as cause:
        return _wrapped(RuntimeError("lookup failed"), cause)


def test_provider_rebuilds_the_models_error_without_chain_or_secrets():
    provider = GatewayModelsProvider(_RaisingBridge(_leaky_chain()), "k")

    with pytest.raises(ModelRetrievalError) as exc:
        provider.add_model("ds/1", "k")

    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert exc.value.__suppress_context__ is True
    assert str(exc.value) == _redact_secrets(LEAKY_MESSAGE)
    assert "SECRETKEY123" not in str(exc.value)
    assert "abc" not in str(exc.value)


@pytest.mark.parametrize(
    "cause",
    [
        UnauthorizedModelAccessError("denied", help_url="https://h/1"),
        ModelNotFoundError("missing", help_url=None),
        ForbiddenModelAccessError("forbidden", help_url="https://h/3"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_provider_rebuild_keeps_class_status_code_and_help_url(cause):
    provider = GatewayModelsProvider(
        _RaisingBridge(_wrapped(RuntimeError("lookup failed"), cause)), "k"
    )

    with pytest.raises(type(cause)) as exc:
        provider.add_model("ds/1", "k")

    assert type(exc.value) is type(cause)
    assert getattr(exc.value, "status_code", None) == getattr(
        cause, "status_code", None
    )
    assert exc.value.help_url == cause.help_url
    assert str(exc.value) == str(cause)


def _jpeg_b64():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode()


def _run(client):
    response = client.post(
        "/workflows/run",
        json={
            "specification": OD_WF,
            "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
            "api_key": "k",
        },
    )

    return response


def _gateway_failing_inference(error):
    gateway = FakeGateway(model_info={"ds/1": {"class_names": ["cat"]}})

    async def _raise(**kwargs):
        raise error

    gateway.infer = _raise

    return gateway


class _FailingLoadManager:
    def __init__(self, error):
        self.error = error
        self.executor = None

    def __contains__(self, key):
        return False

    def load(self, key, api_key, **kwargs):
        raise self.error

    def stats(self):
        return {"models": []}

    def shutdown(self):
        pass


def _step_calls(monkeypatch, call):
    class _Provider(GatewayModelsProvider):
        def add_model(self, model_id, api_key, model_id_alias=None, **kwargs):
            self._routes[model_id] = Route(
                model_id=model_id,
                registry_id=model_id,
                task_type="object-detection",
                action="infer",
            )

        def run_object_detection(self, model_id, images, api_key=None, **kwargs):
            return call(self, model_id, api_key)

    monkeypatch.setattr(router_mod, "GatewayModelsProvider", _Provider)


def test_model_not_ready_in_a_step_answers_the_plain_not_ready_response(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _gateway_failing_inference(
        ReloadAfterEvictionError("reload after eviction failed")
    )

    response = _run(legacy_client(gateway))

    assert response.status_code == 503
    assert response.json() == NOT_READY
    assert response.headers["retry-after"] == "1"


def test_failed_reload_in_a_step_is_answered_by_the_recorded_cause(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    manager = EvictedModelManager(reload_error=UnauthorizedModelAccessError("denied"))

    response = _run(legacy_client(ModelManagerGateway(manager)))

    body = response.json()
    assert response.status_code == 401
    assert body["error_type"] == "ClientCausedStepExecutionError"
    assert body["inner_error_type"] == "UnauthorizedModelAccessError"
    assert body["inner_error_message"] == "denied"
    assert manager.load_calls == 2 and manager.process_calls == 1


def test_restricted_alternative_in_a_step_answers_507(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    manager = _FailingLoadManager(_exhausted([ModelPackageRestrictedError("too big")]))

    response = _run(legacy_client(ModelManagerGateway(manager)))

    body = response.json()
    assert response.status_code == 507
    assert body["message"] == RESTRICTED
    assert body["error_type"] == "RuntimeLimitsCausedStepExecutionError"
    assert body["context"] == STEP_CONTEXT
    assert body["inner_error_type"] == "ModelPackageAlternativesExhaustedError"
    assert body["blocks_errors"][0]["block_id"] == "det"
    assert "retry-after" not in response.headers


def test_alternatives_without_a_restricted_one_in_a_step_answer_500(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    manager = _FailingLoadManager(_exhausted([ValueError("x")]))

    response = _run(legacy_client(ModelManagerGateway(manager)))

    body = response.json()
    assert response.status_code == 500
    assert body["error_type"] == "StepExecutionError"
    assert body["inner_error_type"] == "ModelPackageAlternativesExhaustedError"


def test_unauthorised_model_lookup_in_a_step_answers_401_with_the_models_error(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = UnauthorizedModelAccessError("denied")
    gateway = FakeGateway()

    response = _run(legacy_client(gateway))

    body = response.json()
    assert response.status_code == 401
    assert body["message"] == (
        "Unauthorized error occurred while execution of step det - details of "
        "error: denied. This error usually mean the problem with Roboflow API key."
    )
    assert body["error_type"] == "ClientCausedStepExecutionError"
    assert body["context"] == STEP_CONTEXT
    assert body["inner_error_type"] == "UnauthorizedModelAccessError"
    assert body["inner_error_message"] == "denied"
    assert gateway.calls == []


def test_model_not_found_on_lookup_in_a_step_answers_404_with_the_models_error(
    legacy_client, fake_stat
):
    response = _run(legacy_client(FakeGateway()))

    body = response.json()
    assert response.status_code == 404
    assert body["error_type"] == "ClientCausedStepExecutionError"
    assert body["inner_error_type"] == "ModelNotFoundError"
    assert body["inner_error_message"] == "ds/1 - VISIT  FOR FURTHER SUPPORT"


def test_payment_required_on_lookup_in_a_step_answers_402_with_the_models_error(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = PaymentRequiredModelAccessError("no credits")

    response = _run(legacy_client(FakeGateway()))

    body = response.json()
    assert response.status_code == 402
    assert body["message"] == (
        "Not enough credits to execute step det. Verify your workspace billing "
        "page. Details: no credits"
    )
    assert body["error_type"] == "ClientCausedStepExecutionError"
    assert body["inner_error_type"] == "PaymentRequiredModelAccessError"


def test_unreachable_registry_in_a_step_answers_500_with_the_models_error(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = RetryError("registry down")

    response = _run(legacy_client(FakeGateway()))

    body = response.json()
    assert response.status_code == 500
    assert body["message"] == "registry down"
    assert body["error_type"] == "StepExecutionError"
    assert body["context"] == STEP_CONTEXT
    assert body["inner_error_type"] == "RetryError"
    assert body["inner_error_message"] == "registry down"
    assert body["blocks_errors"][0]["block_id"] == "det"
    assert "retry-after" not in response.headers


def test_registry_failure_chain_in_a_step_stays_out_of_body_and_log(
    legacy_client, fake_stat, caplog
):
    try:
        raise ValueError(LEAKY_ROOT)
    except ValueError as root:
        try:
            raise ModelRetrievalError(LEAKY_MESSAGE) from root
        except ModelRetrievalError as error:
            fake_stat["ds/1"] = error
    client = legacy_client(FakeGateway())

    with caplog.at_level(logging.DEBUG):
        response = _run(client)

    logged = "\n".join(logging.Formatter().format(record) for record in caplog.records)
    assert response.status_code == 500
    assert response.json()["inner_error_type"] == "ModelRetrievalError"
    for text in (response.text, logged):
        assert "SECRETKEY123" not in text
        assert "X-Goog-Signature=def" not in text
        assert "storage.example/y" not in text


def test_server_busy_in_a_step_answers_the_plain_busy_response_with_retry_after(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    busy = _wrapped(
        ServerBusyError("No free SHM slots after 5s"), TimeoutError("No free SHM slots")
    )

    response = _run(legacy_client(_gateway_failing_inference(busy)))

    assert response.status_code == 503
    assert response.json() == {"message": "No free SHM slots after 5s"}
    assert response.headers["retry-after"] == "1"


def _post_to_platform(provider, model_id, api_key):
    return host.PLATFORM_CLIENT.post("x/y", api_key=api_key)


def test_platform_post_403_without_block_handler_in_a_step_answers_403(
    legacy_client, monkeypatch
):
    _step_calls(monkeypatch, _post_to_platform)
    client = legacy_client(FakeGateway())

    with requests_mock.Mocker() as mocker:
        mocker.post(
            "https://api.roboflow.com/x/y?api_key=k",
            status_code=403,
            json={"message": "platform text"},
        )
        response = _run(client)

    body = response.json()
    assert response.status_code == 403
    assert body["message"] == (
        "Forbidden error occurred while execution of step det - details of error: "
        "Unauthorized access to roboflow API - check API key regarding correctness "
        "and required scopes. Visit "
        "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
        "to learn how to retrieve one.. This error usually mean the problem with "
        "Roboflow API key."
    )
    assert body["error_type"] == "ClientCausedStepExecutionError"
    assert body["context"] == STEP_CONTEXT
    assert body["inner_error_type"] == "RoboflowAPIForbiddenError"


def test_platform_post_500_without_block_handler_in_a_step_answers_500(
    legacy_client, monkeypatch
):
    _step_calls(monkeypatch, _post_to_platform)
    client = legacy_client(FakeGateway())

    with requests_mock.Mocker() as mocker:
        mocker.post(
            "https://api.roboflow.com/x/y?api_key=k",
            status_code=500,
            json={"message": "platform text"},
        )
        response = _run(client)

    body = response.json()
    assert response.status_code == 500
    assert body["message"] == (
        "Unsuccessful request to Roboflow API with response code: 500"
    )
    assert body["error_type"] == "StepExecutionError"
    assert body["inner_error_type"] == "RoboflowAPIUnsuccessfulRequestError"
    assert "platform text" not in response.text


def test_remote_501_in_a_step_names_the_deployment_error(legacy_client, monkeypatch):
    def _remote_call(provider, model_id, api_key):
        raise HTTPCallErrorError(
            description="d", status_code=501, api_message="not here"
        )

    _step_calls(monkeypatch, _remote_call)

    response = _run(legacy_client(FakeGateway()))

    body = response.json()
    assert response.status_code == 501
    assert body["message"] == "not here"
    assert body["error_type"] == "ClientCausedStepExecutionError"
    assert body["context"] == (
        "workflow_execution | step_execution | deployment_not_supported"
    )
    assert body["inner_error_type"] == "ModelDeploymentNotSupportedError"
    assert body["inner_error_message"] == "not here"


def test_bad_base64_image_in_a_step_stays_client_caused_400(
    legacy_client, fake_stat, monkeypatch
):
    fake_stat["ds/1"] = ("object-detection", "infer")

    class _Provider(GatewayModelsProvider):
        def run_object_detection(self, model_id, images, api_key=None, **kwargs):
            return super().run_object_detection(
                model_id=model_id,
                images=[{"type": "base64", "value": "!!not base64!!"}],
                api_key=api_key,
                **kwargs,
            )

    monkeypatch.setattr(router_mod, "GatewayModelsProvider", _Provider)
    gateway = FakeGateway(model_info={"ds/1": {"class_names": ["cat"]}})

    response = _run(legacy_client(gateway))

    body = response.json()
    assert response.status_code == 400
    assert body["message"] == (
        "Could not load input image. Cause: Malformed base64 input image."
    )
    assert body["error_type"] == "ClientCausedStepExecutionError"
    assert body["inner_error_type"] == "LegacyHTTPError"
    assert [call for call in gateway.calls if call[0] == "infer"] == []


def test_failed_image_fetch_in_a_step_stays_client_caused_with_its_status(
    legacy_client, fake_stat, monkeypatch
):
    from fastapi import Response

    fake_stat["ds/1"] = ("object-detection", "infer")

    async def _fetch(urls):
        return None, Response(status_code=502)

    monkeypatch.setattr("inference_server.legacy.bridge.fetch_url_images", _fetch)

    class _Provider(GatewayModelsProvider):
        def run_object_detection(self, model_id, images, api_key=None, **kwargs):
            return super().run_object_detection(
                model_id=model_id,
                images=[{"type": "url", "value": "https://example.com/a.jpg"}],
                api_key=api_key,
                **kwargs,
            )

    monkeypatch.setattr(router_mod, "GatewayModelsProvider", _Provider)
    gateway = FakeGateway(model_info={"ds/1": {"class_names": ["cat"]}})

    response = _run(legacy_client(gateway))

    body = response.json()
    assert response.status_code == 502
    assert body["message"] == "Could not fetch image from URL."
    assert body["error_type"] == "ClientCausedStepExecutionError"
    assert body["inner_error_type"] == "ImageFetchError"


def test_malformed_pipeline_model_id_in_a_step_answers_400(legacy_client, fake_stat):
    from inference_model_manager.pipelines import InvalidPipelineIdError

    from inference_server.framework import model_stat

    def _refuse(model_id):
        raise InvalidPipelineIdError("bad pipeline id")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model_stat, "resolve_pipeline_request", _refuse)
        response = _run(legacy_client(FakeGateway()))

    body = response.json()
    assert response.status_code == 400
    assert body["message"] == (
        "Problem with Workflow Block configuration - bad pipeline id"
    )
    assert body["error_type"] == "ClientCausedStepExecutionError"
    assert body["context"] == STEP_CONTEXT
    assert body["inner_error_type"] == "InvalidPipelineIdError"


def _run_with_key(client, api_key):
    return client.post(
        "/workflows/run",
        json={
            "specification": OD_WF,
            "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
            "api_key": api_key,
        },
    )


def test_loaded_model_in_a_step_is_refused_to_a_key_without_access(
    legacy_client, key_gated_stat
):
    key_gated_stat.denied_keys = {"key-b"}
    detections = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    gateway = FakeGateway(
        predictions={("ds/1", "infer"): detections},
        model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
    )
    client = legacy_client(gateway)

    assert _run_with_key(client, "key-a").status_code == 200
    refused = _run_with_key(client, "key-b")

    body = refused.json()
    assert refused.status_code == 401
    assert body["error_type"] == "ClientCausedStepExecutionError"
    assert body["inner_error_type"] == "UnauthorizedModelAccessError"
    assert len([c for c in gateway.calls if c[0] == "infer"]) == 1
    assert _run_with_key(client, "key-a").status_code == 200
