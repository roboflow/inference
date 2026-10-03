import asyncio
import base64
import importlib
import io
import json
import logging
import sys

import pydantic
import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient
from inference_model_manager.errors import INPUT_ERROR_PREFIX
from inference_model_manager.pipelines import InvalidPipelineIdError
from inference_models.errors import (
    AssumptionError,
    EnvironmentConfigurationError,
    FileHashSumMissmatch,
    ForbiddenModelAccessError,
    InvalidEnvVariable,
    InvalidParameterError,
    JetsonTypeResolutionError,
    MissingDependencyError,
    ModelInputError,
    ModelLoadingError,
    ModelMetadataConsistencyError,
    ModelNotFoundError,
    ModelPackageAlternativesExhaustedError,
    ModelPackageNegotiationError,
    ModelPackageRestrictedError,
    ModelRetrievalError,
    ModelRuntimeError,
    PaymentRequiredModelAccessError,
    RetryError,
    UnauthorizedModelAccessError,
    UntrustedFileError,
    UsagePausedModelAccessError,
)
from inference_models.models.vllm_proxy.errors import (
    AdapterNotServableError,
    NotServableOnVLLMError,
    VLLMConnectionError,
    VLLMHTTPError,
    VLLMProxyError,
)

from PIL import Image

from inference_server.errors import PayloadTooLargeError, ServerBusyError
from inference_server.legacy.errors import (
    LegacyHTTPError,
    legacy_error_response,
    with_legacy_errors,
)

HELP_URL = "https://help.example/errors"
HELP_SUFFIX = f" - VISIT {HELP_URL} FOR FURTHER SUPPORT"
UNAUTHORIZED = (
    "Unauthorized access to roboflow API - check API key and make sure the key is "
    "valid for workspace you use. Visit "
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
    "to learn how to retrieve one."
)
FORBIDDEN = (
    "Unauthorized access to roboflow API - check API key and make sure the key is "
    "valid and have required scopes. Visit "
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
    "to learn how to retrieve one."
)
PAYMENT_REQUIRED = (
    "Not enough credits to perform this request. Verify your workspace billing page."
)
USAGE_PAUSED = (
    "Roboflow API usage is paused. Please contact your workspace administrator to "
    "re-enable api keys."
)
NOT_FOUND = (
    "Requested Roboflow resource not found. Make sure that workspace, project or "
    "model you referred in request exists."
)
RESTRICTED = (
    "Model loading failed due to restrictions of server configuration - usually due "
    "to excessive runtime memory requirement of the model (for instance caused by "
    "large input size)."
)
MISCONFIGURATION = {"message": "Service misconfiguration."}
INTERNAL_ERROR = {"message": "Internal error."}
REGISTRY_UNREACHABLE = {"message": "Internal error. Could not connect to Roboflow API."}
REGISTRY_REQUEST_FAILED = {"message": "Internal error. Request to Roboflow API failed."}
REGISTRY_TIMEOUT = {"message": "Timeout when attempting to connect to Roboflow API."}


def _jpeg_b64():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def _raised_from(error, cause):
    try:
        raise error from cause
    except type(error) as raised:
        return raised


def _registry_runtime_error(cause):
    return _raised_from(
        RuntimeError(str(cause) or "Roboflow registry unreachable"), cause
    )


def _retrieval_error(status_code):
    error = ModelRetrievalError("denied")
    error.status_code = status_code
    return error


def _validation_error():
    class _Model(pydantic.BaseModel):
        value: int

    try:
        _Model(value="not a number")
    except pydantic.ValidationError as error:
        return error


LEGACY_MATRIX = [
    pytest.param(LegacyHTTPError(418, "tea"), 418, {"message": "tea"}, id="own-error"),
    pytest.param(
        LegacyHTTPError(400, "bad", extra={"error_type": "X"}),
        400,
        {"message": "bad", "error_type": "X"},
        id="own-error-extra",
    ),
    pytest.param(
        PayloadTooLargeError("big"),
        413,
        {"message": "Request payload too large."},
        id="payload-too-large",
    ),
    pytest.param(
        ModelInputError("bad prompt", help_url=HELP_URL),
        400,
        {
            "message": f"Error with model input. Cause: bad prompt{HELP_SUFFIX}",
            "help_url": HELP_URL,
        },
        id="model-input",
    ),
    pytest.param(
        ModelInputError("bad prompt"),
        400,
        {"message": "Error with model input. Cause: bad prompt", "help_url": None},
        id="model-input-no-help-url",
    ),
    pytest.param(
        _raised_from(
            ValueError(f"bad prompt{HELP_SUFFIX}"),
            ModelInputError("bad prompt", help_url=HELP_URL),
        ),
        500,
        INTERNAL_ERROR,
        id="value-error-caused-by-model-input",
    ),
    pytest.param(
        _raised_from(
            ValueError("bad prompt"), RuntimeError(f"{INPUT_ERROR_PREFIX}bad prompt")
        ),
        500,
        INTERNAL_ERROR,
        id="value-error-caused-by-worker-input-report",
    ),
    pytest.param(
        UnauthorizedModelAccessError("nope"),
        401,
        {"message": UNAUTHORIZED},
        id="unauthorised",
    ),
    pytest.param(
        _raised_from(PermissionError("nope"), UnauthorizedModelAccessError("nope")),
        401,
        {"message": UNAUTHORIZED},
        id="registry-unauthorised",
    ),
    pytest.param(PermissionError("nope"), 500, INTERNAL_ERROR, id="permission-error"),
    pytest.param(
        ModelNotFoundError("missing"), 404, {"message": NOT_FOUND}, id="not-found"
    ),
    pytest.param(
        _raised_from(LookupError("missing"), ModelNotFoundError("missing")),
        404,
        {"message": NOT_FOUND},
        id="registry-not-found",
    ),
    pytest.param(
        _raised_from(LookupError("bad id"), InvalidPipelineIdError("bad id")),
        400,
        {"message": "Invalid Model ID sent in request."},
        id="invalid-pipeline-id",
    ),
    pytest.param(LookupError("missing"), 500, INTERNAL_ERROR, id="lookup-error"),
    pytest.param(KeyError("missing"), 500, INTERNAL_ERROR, id="key-error"),
    pytest.param(
        ModelPackageNegotiationError("no package", help_url=HELP_URL),
        500,
        {
            "message": f"Could not negotiate model package - no package{HELP_SUFFIX}",
            "help_url": HELP_URL,
        },
        id="negotiation",
    ),
    pytest.param(
        JetsonTypeResolutionError("unknown jetson"),
        500,
        {
            "message": "Could not negotiate model package - unknown jetson",
            "help_url": None,
        },
        id="negotiation-subclass-before-misconfiguration",
    ),
    pytest.param(
        EnvironmentConfigurationError("x"), 500, MISCONFIGURATION, id="environment"
    ),
    pytest.param(InvalidEnvVariable("x"), 500, MISCONFIGURATION, id="env-variable"),
    pytest.param(
        MissingDependencyError("x"), 500, MISCONFIGURATION, id="missing-dependency"
    ),
    pytest.param(
        InvalidParameterError("x"), 500, MISCONFIGURATION, id="invalid-parameter"
    ),
    pytest.param(
        ModelPackageRestrictedError("too big", help_url=HELP_URL),
        507,
        {"message": RESTRICTED},
        id="restricted-before-loading",
    ),
    pytest.param(
        ModelPackageAlternativesExhaustedError(
            "exhausted",
            help_url=HELP_URL,
            alternatives_errors=[
                ModelLoadingError("broken"),
                ModelPackageRestrictedError("too big"),
            ],
        ),
        507,
        {"message": RESTRICTED, "help_url": HELP_URL},
        id="alternatives-exhausted-restricted",
    ),
    pytest.param(
        ModelPackageAlternativesExhaustedError(
            "exhausted",
            help_url=HELP_URL,
            alternatives_errors=[ModelLoadingError("broken")],
        ),
        500,
        {
            "message": f"Model loading failed: exhausted{HELP_SUFFIX}",
            "help_url": HELP_URL,
        },
        id="alternatives-exhausted",
    ),
    pytest.param(
        ModelPackageAlternativesExhaustedError("exhausted"),
        500,
        {"message": "Model loading failed: exhausted", "help_url": None},
        id="alternatives-exhausted-without-alternatives",
    ),
    pytest.param(
        ModelLoadingError("broken", help_url=HELP_URL),
        500,
        {
            "message": f"Model loading failed: broken{HELP_SUFFIX}",
            "help_url": HELP_URL,
        },
        id="loading",
    ),
    pytest.param(
        UntrustedFileError("untrusted", help_url=HELP_URL),
        500,
        {
            "message": f"Issue with model package file: untrusted{HELP_SUFFIX}",
            "help_url": HELP_URL,
        },
        id="untrusted-file",
    ),
    pytest.param(
        FileHashSumMissmatch("hash"),
        500,
        {"message": "Issue with model package file: hash", "help_url": None},
        id="hash-mismatch",
    ),
    pytest.param(
        PaymentRequiredModelAccessError("x"),
        402,
        {"message": PAYMENT_REQUIRED},
        id="retrieval-402",
    ),
    pytest.param(
        ForbiddenModelAccessError("x"), 403, {"message": FORBIDDEN}, id="retrieval-403"
    ),
    pytest.param(
        UsagePausedModelAccessError("x"),
        423,
        {"message": USAGE_PAUSED},
        id="retrieval-423",
    ),
    pytest.param(
        ModelRetrievalError("gone", help_url=HELP_URL),
        500,
        {
            "message": f"Could not retrieve model gone{HELP_SUFFIX}",
            "help_url": HELP_URL,
        },
        id="retrieval",
    ),
    pytest.param(
        ModelMetadataConsistencyError("odd"),
        500,
        {"message": "Could not retrieve model odd", "help_url": None},
        id="retrieval-subclass",
    ),
    pytest.param(
        _registry_runtime_error(_retrieval_error(402)),
        402,
        {"message": PAYMENT_REQUIRED},
        id="registry-402",
    ),
    pytest.param(
        _registry_runtime_error(_retrieval_error(403)),
        403,
        {"message": FORBIDDEN},
        id="registry-403",
    ),
    pytest.param(
        _registry_runtime_error(_retrieval_error(423)),
        423,
        {"message": USAGE_PAUSED},
        id="registry-423",
    ),
    pytest.param(
        _registry_runtime_error(OSError("down")),
        500,
        INTERNAL_ERROR,
        id="registry-plain-os-error",
    ),
    pytest.param(
        _registry_runtime_error(requests.exceptions.ConnectionError("down")),
        503,
        REGISTRY_UNREACHABLE,
        id="registry-unreachable",
    ),
    pytest.param(
        _registry_runtime_error(ConnectionResetError("reset")),
        503,
        REGISTRY_UNREACHABLE,
        id="registry-builtin-connection-error",
    ),
    pytest.param(
        _registry_runtime_error(requests.exceptions.ConnectTimeout("slow")),
        504,
        REGISTRY_TIMEOUT,
        id="registry-connect-timeout",
    ),
    pytest.param(
        _registry_runtime_error(requests.exceptions.Timeout("slow")),
        504,
        REGISTRY_TIMEOUT,
        id="registry-direct-timeout",
    ),
    pytest.param(
        _registry_runtime_error(
            _raised_from(RetryError("Connectivity error"), OSError("x"))
        ),
        500,
        INTERNAL_ERROR,
        id="registry-retry-plain-os-error",
    ),
    pytest.param(
        _registry_runtime_error(
            _raised_from(RetryError("Connectivity error"), ConnectionResetError("x"))
        ),
        503,
        REGISTRY_UNREACHABLE,
        id="registry-retry-builtin-connection-error",
    ),
    pytest.param(
        _registry_runtime_error(
            _raised_from(
                RetryError("Connectivity error"),
                requests.exceptions.ConnectTimeout("slow"),
            )
        ),
        504,
        REGISTRY_TIMEOUT,
        id="registry-retry-connect-timeout",
    ),
    pytest.param(
        _registry_runtime_error(
            _raised_from(ModelRetrievalError("undecodable"), KeyError("modelMetadata"))
        ),
        500,
        INTERNAL_ERROR,
        id="registry-response-without-expected-key",
    ),
    pytest.param(
        _registry_runtime_error(
            _raised_from(ModelRetrievalError("invalid"), _validation_error())
        ),
        500,
        INTERNAL_ERROR,
        id="registry-response-with-missing-fields",
    ),
    pytest.param(
        _registry_runtime_error(
            _raised_from(
                ModelRetrievalError("undecodable"),
                requests.exceptions.JSONDecodeError("Expecting value", "", 0),
            )
        ),
        502,
        REGISTRY_REQUEST_FAILED,
        id="registry-response-not-json",
    ),
    pytest.param(
        _registry_runtime_error(
            _raised_from(
                RetryError("Connectivity error"),
                requests.exceptions.ConnectionError("down"),
            )
        ),
        503,
        REGISTRY_UNREACHABLE,
        id="registry-connection",
    ),
    pytest.param(
        _registry_runtime_error(
            _raised_from(
                RetryError("Connectivity error"), requests.exceptions.Timeout("slow")
            )
        ),
        504,
        REGISTRY_TIMEOUT,
        id="registry-timeout",
    ),
    pytest.param(
        _registry_runtime_error(RetryError("invalid response code 503")),
        502,
        REGISTRY_REQUEST_FAILED,
        id="registry-retryable-status",
    ),
    pytest.param(
        _registry_runtime_error(ModelRetrievalError("invalid response code 418")),
        502,
        REGISTRY_REQUEST_FAILED,
        id="registry-unsuccessful-request",
    ),
    pytest.param(
        asyncio.TimeoutError(),
        504,
        {"message": "Timed out waiting for inference result."},
        id="inference-timeout",
    ),
    pytest.param(ValueError("bad"), 500, INTERNAL_ERROR, id="value-error"),
    pytest.param(_validation_error(), 500, INTERNAL_ERROR, id="validation-error"),
    pytest.param(RuntimeError("boom"), 500, INTERNAL_ERROR, id="runtime-error"),
    pytest.param(AssumptionError("x"), 500, INTERNAL_ERROR, id="assumption"),
    pytest.param(ModelRuntimeError("x"), 500, INTERNAL_ERROR, id="model-runtime"),
    pytest.param(RetryError("x"), 500, INTERNAL_ERROR, id="retry"),
    pytest.param(
        NotServableOnVLLMError("base variant mismatch"),
        501,
        {"message": "base variant mismatch"},
        id="vllm-not-servable",
    ),
    pytest.param(
        AdapterNotServableError("adapter rejected"),
        501,
        {"message": "adapter rejected"},
        id="vllm-adapter-not-servable",
    ),
    pytest.param(VLLMProxyError("x"), 500, INTERNAL_ERROR, id="vllm-proxy"),
    pytest.param(
        VLLMConnectionError("refused"), 500, INTERNAL_ERROR, id="vllm-connection"
    ),
    pytest.param(
        VLLMHTTPError("bad gateway", 502, "body"),
        500,
        INTERNAL_ERROR,
        id="vllm-http",
    ),
]


@pytest.mark.parametrize("error,status,body", LEGACY_MATRIX)
def test_legacy_answer(error, status, body):
    response = legacy_error_response(error)

    assert response.status_code == status
    assert json.loads(response.body) == body
    assert "retry-after" not in response.headers


def test_own_error_answer_carries_its_headers():
    response = legacy_error_response(
        LegacyHTTPError(503, "not ready", headers={"Retry-After": "1"})
    )

    assert response.status_code == 503
    assert json.loads(response.body) == {"message": "not ready"}
    assert response.headers["retry-after"] == "1"


def test_server_busy_answer_carries_retry_after():
    response = legacy_error_response(ServerBusyError("busy"))

    assert response.status_code == 503
    assert json.loads(response.body) == {"message": "busy"}
    assert response.headers["retry-after"] == "1"


@pytest.mark.parametrize(
    "error,level,has_traceback",
    [
        (ModelInputError("bad"), logging.ERROR, True),
        (ModelLoadingError("broken"), logging.ERROR, True),
        (PaymentRequiredModelAccessError("x"), logging.ERROR, True),
        (_registry_runtime_error(_retrieval_error(403)), logging.ERROR, True),
        (_registry_runtime_error(_retrieval_error(402)), logging.WARNING, False),
        (ValueError("bad"), logging.ERROR, True),
    ],
)
def test_handled_error_is_logged_like_legacy(error, level, has_traceback, caplog):
    with caplog.at_level(logging.DEBUG, logger="inference_server.legacy.errors"):
        legacy_error_response(error)

    assert [record.levelno for record in caplog.records] == [level]
    assert (caplog.records[0].exc_info is not None) == has_traceback


_CAUSE_TEXT = "fetch https://h.example/x?api_key=SECRET-ONE failed"
_ERROR_TEXT = "wrapped api_key=SECRET-TWO"
_BARE_TEXT = "bare-key-value and short"


def _formatted_records(caplog) -> str:
    formatter = logging.Formatter()

    return "\n".join(formatter.format(record) for record in caplog.records)


def test_logged_traceback_carries_no_raw_credentials_of_the_error_or_its_causes(
    caplog,
):
    try:
        try:
            raise OSError(_CAUSE_TEXT)
        except OSError as cause:
            raise RuntimeError(_ERROR_TEXT) from cause
    except RuntimeError as error:
        with caplog.at_level(logging.DEBUG, logger="inference_server.legacy.errors"):
            response = legacy_error_response(error)

    assert response.status_code == 500
    text = _formatted_records(caplog)
    assert "SECRET-ONE" not in text and "SECRET-TWO" not in text
    assert "Traceback" in text
    assert "RuntimeError: wrapped" in text
    assert "OSError: fetch" in text


@pytest.mark.asyncio
async def test_registered_values_are_removed_from_logged_errors(caplog):
    from inference_server.legacy.errors import add_redaction_values

    @with_legacy_errors
    async def route():
        add_redaction_values("bare-key-value", "short")
        raise RuntimeError(_BARE_TEXT)

    with caplog.at_level(logging.DEBUG, logger="inference_server.legacy.errors"):
        response = await route()

    assert response.status_code == 500
    text = _formatted_records(caplog)
    assert "bare-key-value" not in text
    assert "*** and short" in text


def test_unprintable_error_is_still_answered_and_logged(caplog):
    class Unprintable(Exception):
        def __str__(self):
            raise ValueError("no text")

    with caplog.at_level(logging.DEBUG, logger="inference_server.legacy.errors"):
        response = legacy_error_response(Unprintable())

    assert response.status_code == 500
    assert "<unprintable>" in _formatted_records(caplog)


@pytest.mark.asyncio
async def test_decorator_turns_exception_into_response():
    @with_legacy_errors
    async def route():
        raise LookupError("x")

    response = await route()

    assert response.status_code == 500
    assert json.loads(response.body) == INTERNAL_ERROR


def test_legacy_route_reaches_the_mapping_through_the_decorator():
    app = FastAPI()

    @app.get("/infer")
    @with_legacy_errors
    async def _infer():
        raise ModelInputError("bad prompt", help_url=HELP_URL)

    response = TestClient(app).get("/infer")

    assert response.status_code == 400
    assert response.json() == {
        "message": f"Error with model input. Cause: bad prompt{HELP_SUFFIX}",
        "help_url": HELP_URL,
    }


def test_builder_route_reaches_the_mapping_through_the_decorator(tmp_path, monkeypatch):
    from inference_server import configuration
    from inference_server.legacy.router import get_bridge

    module_name = "inference_server.builder.routes"
    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(tmp_path / "cache"))
    sys.modules.pop(module_name, None)
    try:
        routes = importlib.import_module(module_name)

        async def _broken(bridge):
            raise ModelLoadingError("broken")

        monkeypatch.setattr(routes.models, "list_models_with_status", _broken)
        app = FastAPI()
        app.include_router(routes.router, prefix="/build")
        app.dependency_overrides[get_bridge] = lambda: object()

        response = TestClient(app).get(
            "/build/api/models", headers={"X-CSRF": routes.csrf}
        )
    finally:
        sys.modules.pop(module_name, None)

    assert response.status_code == 500
    assert response.json() == {
        "message": "Model loading failed: broken",
        "help_url": None,
    }


@pytest.mark.parametrize(
    "outcome,status,body",
    [
        (UnauthorizedModelAccessError("denied"), 401, {"message": UNAUTHORIZED}),
        (ModelNotFoundError("missing"), 404, {"message": NOT_FOUND}),
        (ForbiddenModelAccessError("denied"), 403, {"message": FORBIDDEN}),
        (RetryError("invalid response code 503"), 502, REGISTRY_REQUEST_FAILED),
        (requests.exceptions.ConnectionError("down"), 503, REGISTRY_UNREACHABLE),
    ],
)
def test_registry_failure_on_a_legacy_route(
    legacy_client, fake_stat, outcome, status, body
):
    from tests.unit_tests.legacy.conftest import FakeGateway

    fake_stat["ws/model/1"] = outcome
    client = legacy_client(FakeGateway())

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ws/model/1", "image": {"type": "base64", "value": "AA=="}},
    )

    assert response.status_code == status
    assert response.json() == body


def test_value_error_from_the_gateway_on_a_legacy_route_is_a_model_input_error(
    legacy_client, fake_stat, caplog
):
    from tests.unit_tests.legacy.conftest import FakeGateway

    def _reject(image, params):
        raise ValueError("bad shape")

    fake_stat["ws/model/1"] = ("object-detection", "infer")
    client = legacy_client(FakeGateway(predictions={("ws/model/1", "infer"): _reject}))

    with caplog.at_level(logging.DEBUG, logger="inference_server.legacy.errors"):
        response = client.post(
            "/infer/object_detection",
            json={
                "model_id": "ws/model/1",
                "image": {"type": "base64", "value": _jpeg_b64()},
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "message": "Error with model input. Cause: bad shape",
        "help_url": None,
    }
    records = [r for r in caplog.records if r.name == "inference_server.legacy.errors"]
    assert [record.levelno for record in records] == [logging.ERROR]
    assert records[0].exc_info is not None


def test_value_error_raised_by_the_route_before_inference_is_an_internal_error(
    legacy_client, fake_stat
):
    from tests.unit_tests.legacy.conftest import FakeGateway

    fake_stat["ws/model/1"] = ValueError("bad id")
    client = legacy_client(FakeGateway())

    response = client.post(
        "/infer/object_detection",
        json={"model_id": "ws/model/1", "image": {"type": "base64", "value": "AA=="}},
    )

    assert response.status_code == 500
    assert response.json() == INTERNAL_ERROR


def test_body_limit_middleware_rejects_oversized_declared_and_streamed_bodies(
    monkeypatch,
):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    from inference_server.legacy.errors import (
        _BodyLimitMiddleware,
        install_legacy_exception_handlers,
    )

    monkeypatch.setattr("inference_server.configuration.MAX_BODY_BYTES", 10)
    app = FastAPI()

    @app.post("/echo")
    async def _echo(request: Request):
        return {"n": len(await request.body())}

    @app.post("/v2/models/infer")
    async def _v2(request: Request):
        return {"n": len(await request.body())}

    install_legacy_exception_handlers(app)
    app.add_middleware(_BodyLimitMiddleware)
    client = TestClient(app)
    assert client.post("/echo", content=b"x" * 11).status_code == 413
    assert client.post("/echo", content=b"x" * 5).json() == {"n": 5}
    assert (
        client.post(
            "/echo",
            content=iter([b"x" * 6, b"x" * 6]),
            headers={"Transfer-Encoding": "chunked"},
        ).status_code
        == 413
    )
    assert client.post("/v2/models/infer", content=b"x" * 11).status_code == 200


@pytest.mark.asyncio
async def test_body_limit_middleware_rejects_streamed_typed_body(monkeypatch):
    from fastapi import FastAPI
    from pydantic import BaseModel

    from inference_server.legacy.errors import (
        _BodyLimitMiddleware,
        install_legacy_exception_handlers,
    )

    monkeypatch.setattr("inference_server.configuration.MAX_BODY_BYTES", 10)

    class _Payload(BaseModel):
        value: str

    app = FastAPI()

    @app.post("/typed")
    async def _typed(payload: _Payload):
        return {"n": len(payload.value)}

    install_legacy_exception_handlers(app)
    app.add_middleware(_BodyLimitMiddleware)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/typed",
        "raw_path": b"/typed",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    chunks = [
        {"type": "http.request", "body": b'{"valu', "more_body": True},
        {"type": "http.request", "body": b'e":"x"}', "more_body": False},
    ]
    sent = []

    async def receive():
        return chunks.pop(0)

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"]) == {"message": "Request payload too large."}
    assert not chunks
