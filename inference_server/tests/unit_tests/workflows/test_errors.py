import pytest
from fastapi import HTTPException
from roboflow_workflows.errors import (
    WorkflowDefinitionError,
    WorkflowEnvironmentConfigurationError,
)

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

    assert response.status_code == 404


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
