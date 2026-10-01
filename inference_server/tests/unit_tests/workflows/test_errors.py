import json
import logging

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from roboflow_workflows.errors import (
    StepExecutionError,
    WorkflowDefinitionError,
    WorkflowEnvironmentConfigurationError,
    WorkflowsInvalidEnvironmentValueError,
    WorkflowSyntaxError,
)
from roboflow_workflows.prototypes.platform_errors import (
    FeatureDeprecatedError,
    RoboflowAPIConnectionError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPIRequestError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
)

from inference_models.errors import ModelInputError, ModelLoadingError

from inference_server.workflows.errors import with_workflow_errors


@pytest.mark.asyncio
async def test_mapped_workflow_error_uses_workflow_payload():
    @with_workflow_errors
    async def handler():
        raise WorkflowDefinitionError(
            public_message="bad definition",
            context="workflow_compilation",
        )

    response = await handler()

    assert response.status_code == 400
    assert b"bad definition" in response.body
    assert b"WorkflowDefinitionError" in response.body


@pytest.mark.asyncio
async def test_environment_configuration_error_is_500():
    @with_workflow_errors
    async def handler():
        raise WorkflowEnvironmentConfigurationError(
            public_message="misconfigured",
            context="workflow_compilation",
        )

    response = await handler()

    assert response.status_code == 500
    assert b"misconfigured" in response.body


@pytest.mark.asyncio
async def test_unrelated_error_falls_through_to_legacy_response():
    @with_workflow_errors
    async def handler():
        raise LookupError("no such model")

    response = await handler()

    assert response.status_code == 500
    assert json.loads(response.body) == {"message": "Internal error."}


@pytest.mark.asyncio
async def test_http_exception_passes_through():
    @with_workflow_errors
    async def handler():
        raise HTTPException(status_code=413, detail="too large")

    with pytest.raises(HTTPException):
        await handler()


@pytest.mark.asyncio
async def test_successful_call_is_returned_unchanged():
    @with_workflow_errors
    async def handler(value):
        return value

    assert await handler(7) == 7


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
NOT_FOUND = (
    "Requested Roboflow resource not found. Make sure that workspace, project or "
    "model you referred in request exists."
)


@pytest.mark.parametrize(
    "error,status,body",
    [
        (RoboflowAPINotAuthorizedError("x"), 401, {"message": UNAUTHORIZED}),
        (RoboflowAPIForbiddenError("x"), 403, {"message": FORBIDDEN}),
        (RoboflowAPINotNotFoundError("x"), 404, {"message": NOT_FOUND}),
        (
            WorkflowsInvalidEnvironmentValueError("x"),
            500,
            {"message": "Service misconfiguration."},
        ),
        (
            RoboflowAPIUnsuccessfulRequestError("x"),
            502,
            {"message": "Internal error. Request to Roboflow API failed."},
        ),
        (
            RoboflowAPIConnectionError("x"),
            503,
            {"message": "Internal error. Could not connect to Roboflow API."},
        ),
        (
            RoboflowAPITimeoutError("x"),
            504,
            {"message": "Timeout when attempting to connect to Roboflow API."},
        ),
        (RoboflowAPIRequestError("x"), 500, {"message": "Internal error."}),
        (
            ModelLoadingError("broken"),
            500,
            {"message": "Model loading failed: broken", "help_url": None},
        ),
        (
            WorkflowDefinitionError(public_message="bad", context="ctx"),
            400,
            {
                "message": "bad",
                "error_type": "WorkflowDefinitionError",
                "context": "ctx",
                "inner_error_type": None,
                "inner_error_message": "None",
            },
        ),
        (
            WorkflowSyntaxError(public_message="bad", context="ctx"),
            400,
            {
                "message": "bad",
                "error_type": "WorkflowSyntaxError",
                "context": "ctx",
                "inner_error_type": "None",
                "inner_error_message": "None",
                "blocks_errors": None,
                "python_blocks_output_streams": None,
                "python_blocks_debug_traces": None,
            },
        ),
        (
            WorkflowEnvironmentConfigurationError(public_message="bad", context="ctx"),
            500,
            {
                "message": "bad",
                "error_type": "WorkflowEnvironmentConfigurationError",
                "context": "ctx",
                "inner_error_type": None,
                "inner_error_message": "None",
                "python_blocks_output_streams": None,
                "python_blocks_debug_traces": None,
            },
        ),
        (
            FeatureDeprecatedError(feature="/old"),
            410,
            {
                "message": "Feature '/old' has been removed from inference. No "
                "drop-in replacement is provided; contact Roboflow if you require "
                "this capability.",
                "error_type": "FeatureDeprecatedError",
                "feature": "/old",
                "removal_release": None,
                "replacement": None,
                "reason": None,
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_workflow_route_answer(error, status, body):
    @with_workflow_errors
    async def handler():
        raise error

    response = await handler()

    assert response.status_code == status
    assert json.loads(response.body) == body
    assert "retry-after" not in response.headers


@pytest.mark.asyncio
async def test_step_execution_error_keeps_the_block_error_shape():
    @with_workflow_errors
    async def handler():
        raise StepExecutionError(
            block_id="step",
            block_type="some/block@v1",
            public_message="failed",
            context="ctx",
            inner_error=RuntimeError("boom"),
        )

    response = await handler()

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error_type"] == "StepExecutionError"
    assert body["inner_error_message"] == "boom"
    assert body["blocks_errors"][0]["block_id"] == "step"
    assert body["blocks_errors"][0]["property_details"] == "boom"


@pytest.mark.asyncio
async def test_step_execution_error_from_a_model_input_error_answers_500():
    @with_workflow_errors
    async def handler():
        raise StepExecutionError(
            block_id="step",
            block_type="some/block@v1",
            public_message="failed",
            context="ctx",
            inner_error=ModelInputError("bad shape"),
        )

    response = await handler()

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error_type"] == "StepExecutionError"
    assert body["inner_error_message"] == "bad shape"
    assert body["inner_error_type"] == "ModelInputError"
    assert body["blocks_errors"][0]["block_id"] == "step"


@pytest.mark.parametrize(
    "error,level,has_traceback",
    [
        (RoboflowAPINotAuthorizedError("x"), logging.ERROR, True),
        (
            WorkflowDefinitionError(public_message="bad", context="ctx"),
            logging.ERROR,
            True,
        ),
        (FeatureDeprecatedError(feature="/old"), logging.WARNING, False),
    ],
)
@pytest.mark.asyncio
async def test_workflow_route_error_is_logged_like_legacy(
    error, level, has_traceback, caplog
):
    @with_workflow_errors
    async def handler():
        raise error

    with caplog.at_level(logging.DEBUG, logger="inference_server.workflows.errors"):
        await handler()

    assert [record.levelno for record in caplog.records] == [level]
    assert (caplog.records[0].exc_info is not None) == has_traceback


def test_workflow_route_reaches_the_mapping_through_the_decorator():
    app = FastAPI()

    @app.get("/workflows/blocks/describe")
    @with_workflow_errors
    async def _describe():
        raise RoboflowAPINotAuthorizedError("denied")

    response = TestClient(app).get("/workflows/blocks/describe")

    assert response.status_code == 401
    assert response.json() == {"message": UNAUTHORIZED}
