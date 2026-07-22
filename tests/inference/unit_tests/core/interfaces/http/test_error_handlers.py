import pytest

from inference.core.exceptions import (
    CannotInitialiseModelDueToInputSizeError,
    ModelDeploymentNotSupportedError,
    RoboflowAPIUsagePausedError,
)
from inference.core.interfaces.http.error_handlers import (
    with_route_exceptions,
    with_route_exceptions_async,
)
from inference.core.workflows.errors import (
    ClientCausedStepExecutionError,
    RuntimeLimitsCausedStepExecutionError,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.errors import (
    InnerWorkflowCompositionCycleError,
    InnerWorkflowInliningStructureError,
    InnerWorkflowParameterBindingsUnknownInputError,
)
from inference_models.errors import (
    FileHashSumMissmatch,
    ForbiddenModelAccessError,
    ModelInputError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelPackageAlternativesExhaustedError,
    ModelPackageNegotiationError,
    ModelPackageRestrictedError,
    ModelRetrievalError,
    PaymentRequiredModelAccessError,
    UnauthorizedModelAccessError,
    UntrustedFileError,
    UsagePausedModelAccessError,
)


def test_with_route_exceptions_when_usage_paused_error_raised():
    @with_route_exceptions
    def my_route():
        raise RoboflowAPIUsagePausedError("usage paused")

    # when
    resp = my_route()

    # then
    assert resp.status_code == 423
    assert "paused" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_usage_paused_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise RoboflowAPIUsagePausedError("usage paused")

    # when
    resp = await my_route()

    # then
    assert resp.status_code == 423
    assert "paused" in resp.body.decode().lower()


def test_with_route_exceptions_when_model_not_found_error_raised():
    @with_route_exceptions
    def my_route():
        raise ModelNotFoundError("model not found")

    resp = my_route()

    assert resp.status_code == 404
    assert "not found" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_model_not_found_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise ModelNotFoundError("model not found")

    resp = await my_route()

    assert resp.status_code == 404
    assert "not found" in resp.body.decode().lower()


def test_with_route_exceptions_when_unauthorized_model_access_error_raised():
    @with_route_exceptions
    def my_route():
        raise UnauthorizedModelAccessError("unauthorized")

    resp = my_route()

    assert resp.status_code == 401
    assert "unauthorized" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_unauthorized_model_access_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise UnauthorizedModelAccessError("unauthorized")

    resp = await my_route()

    assert resp.status_code == 401
    assert "unauthorized" in resp.body.decode().lower()


@pytest.mark.parametrize(
    ("error", "expected_status_code"),
    [
        (PaymentRequiredModelAccessError("payment required"), 402),
        (ForbiddenModelAccessError("forbidden"), 403),
        (UsagePausedModelAccessError("usage paused"), 423),
    ],
)
def test_with_route_exceptions_when_model_access_is_denied(
    error: Exception, expected_status_code: int
):
    @with_route_exceptions
    def my_route():
        raise error

    resp = my_route()

    assert resp.status_code == expected_status_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status_code"),
    [
        (PaymentRequiredModelAccessError("payment required"), 402),
        (ForbiddenModelAccessError("forbidden"), 403),
        (UsagePausedModelAccessError("usage paused"), 423),
    ],
)
async def test_with_route_exceptions_async_when_model_access_is_denied(
    error: Exception, expected_status_code: int
):
    @with_route_exceptions_async
    async def my_route():
        raise error

    resp = await my_route()

    assert resp.status_code == expected_status_code


def test_with_route_exceptions_when_model_input_error_raised():
    @with_route_exceptions
    def my_route():
        raise ModelInputError("bad input")

    resp = my_route()

    assert resp.status_code == 400
    assert "model input" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_model_input_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise ModelInputError("bad input")

    resp = await my_route()

    assert resp.status_code == 400
    assert "model input" in resp.body.decode().lower()


def test_with_route_exceptions_when_inner_workflow_parameter_bindings_error_raised():
    @with_route_exceptions
    def my_route():
        raise InnerWorkflowParameterBindingsUnknownInputError(
            public_message="inner workflow parameter bindings reference unknown input"
        )

    resp = my_route()

    assert resp.status_code == 400
    assert "unknown input" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_inner_workflow_parameter_bindings_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise InnerWorkflowParameterBindingsUnknownInputError(
            public_message="inner workflow parameter bindings reference unknown input"
        )

    resp = await my_route()

    assert resp.status_code == 400
    assert "unknown input" in resp.body.decode().lower()


def test_with_route_exceptions_when_inner_workflow_composition_error_raised():
    @with_route_exceptions
    def my_route():
        raise InnerWorkflowCompositionCycleError(
            public_message="inner workflow composition contains a cycle"
        )

    resp = my_route()

    assert resp.status_code == 400
    assert "cycle" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_inner_workflow_composition_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise InnerWorkflowCompositionCycleError(
            public_message="inner workflow composition contains a cycle"
        )

    resp = await my_route()

    assert resp.status_code == 400
    assert "cycle" in resp.body.decode().lower()


def test_with_route_exceptions_when_inner_workflow_inlining_structure_error_raised():
    @with_route_exceptions
    def my_route():
        raise InnerWorkflowInliningStructureError(
            public_message="inner workflow inlining made no progress"
        )

    resp = my_route()

    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_inner_workflow_inlining_structure_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise InnerWorkflowInliningStructureError(
            public_message="inner workflow inlining made no progress"
        )

    resp = await my_route()

    assert resp.status_code == 500


def test_with_route_exceptions_when_cannot_initialise_model_due_to_input_size_error_raised():
    @with_route_exceptions
    def my_route():
        raise CannotInitialiseModelDueToInputSizeError("input too large")

    resp = my_route()

    assert resp.status_code == 507
    assert "restrictions" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_cannot_initialise_model_due_to_input_size_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise CannotInitialiseModelDueToInputSizeError("input too large")

    resp = await my_route()

    assert resp.status_code == 507
    assert "restrictions" in resp.body.decode().lower()


def test_with_route_exceptions_when_model_deployment_not_supported_error_raised():
    @with_route_exceptions
    def my_route():
        raise ModelDeploymentNotSupportedError("fine-tuned SAM3 disabled")

    resp = my_route()

    assert resp.status_code == 501
    assert "fine-tuned sam3 disabled" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_model_deployment_not_supported_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise ModelDeploymentNotSupportedError("fine-tuned SAM3 disabled")

    resp = await my_route()

    assert resp.status_code == 501
    assert "fine-tuned sam3 disabled" in resp.body.decode().lower()


def test_with_route_exceptions_when_model_restricted_error_raised():
    @with_route_exceptions
    def my_route():
        raise ModelPackageRestrictedError("input too large")

    resp = my_route()

    assert resp.status_code == 507
    assert "restrictions" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_model_restricted_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise ModelPackageRestrictedError("input too large")

    resp = await my_route()

    assert resp.status_code == 507
    assert "restrictions" in resp.body.decode().lower()


def test_with_route_exceptions_when_model_package_alternatives_exhausted_with_restricted_error():
    @with_route_exceptions
    def my_route():
        restricted = ModelPackageRestrictedError("input size too large")
        raise ModelPackageAlternativesExhaustedError(
            "all alternatives exhausted",
            alternatives_errors=[restricted],
        )

    resp = my_route()

    assert resp.status_code == 507
    assert "restrictions" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_model_package_alternatives_exhausted_with_restricted_error():
    @with_route_exceptions_async
    async def my_route():
        restricted = ModelPackageRestrictedError("input size too large")
        raise ModelPackageAlternativesExhaustedError(
            "all alternatives exhausted",
            alternatives_errors=[restricted],
        )

    resp = await my_route()

    assert resp.status_code == 507
    assert "restrictions" in resp.body.decode().lower()


def test_with_route_exceptions_when_model_package_alternatives_exhausted_without_restricted_error():
    @with_route_exceptions
    def my_route():
        raise ModelPackageAlternativesExhaustedError(
            "all alternatives exhausted",
            alternatives_errors=[Exception("generic failure")],
        )

    resp = my_route()

    assert resp.status_code == 500
    assert "failed" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_model_package_alternatives_exhausted_without_restricted_error():
    @with_route_exceptions_async
    async def my_route():
        raise ModelPackageAlternativesExhaustedError(
            "all alternatives exhausted",
            alternatives_errors=[Exception("generic failure")],
        )

    resp = await my_route()

    assert resp.status_code == 500
    assert "failed" in resp.body.decode().lower()


def test_with_route_exceptions_when_model_package_negotiation_error_raised():
    @with_route_exceptions
    def my_route():
        raise ModelPackageNegotiationError("negotiation failed")

    resp = my_route()

    assert resp.status_code == 500
    assert "negotiate" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_model_package_negotiation_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise ModelPackageNegotiationError("negotiation failed")

    resp = await my_route()

    assert resp.status_code == 500
    assert "negotiate" in resp.body.decode().lower()


def test_with_route_exceptions_when_model_loading_error_raised():
    @with_route_exceptions
    def my_route():
        raise ModelLoadingError("loading failed")

    resp = my_route()

    assert resp.status_code == 500
    assert "loading failed" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_model_loading_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise ModelLoadingError("loading failed")

    resp = await my_route()

    assert resp.status_code == 500
    assert "loading failed" in resp.body.decode().lower()


def test_with_route_exceptions_when_model_retrieval_error_raised():
    @with_route_exceptions
    def my_route():
        raise ModelRetrievalError("retrieval failed")

    resp = my_route()

    assert resp.status_code == 500
    assert "retrieve" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_model_retrieval_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise ModelRetrievalError("retrieval failed")

    resp = await my_route()

    assert resp.status_code == 500
    assert "retrieve" in resp.body.decode().lower()


def test_with_route_exceptions_when_untrusted_file_error_raised():
    @with_route_exceptions
    def my_route():
        raise UntrustedFileError("file not trusted")

    resp = my_route()

    assert resp.status_code == 500
    assert "model package file" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_untrusted_file_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise UntrustedFileError("file not trusted")

    resp = await my_route()

    assert resp.status_code == 500
    assert "model package file" in resp.body.decode().lower()


def test_with_route_exceptions_when_file_hash_sum_missmatch_raised():
    @with_route_exceptions
    def my_route():
        raise FileHashSumMissmatch("hash mismatch")

    resp = my_route()

    assert resp.status_code == 500
    assert "model package file" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_file_hash_sum_missmatch_raised():
    @with_route_exceptions_async
    async def my_route():
        raise FileHashSumMissmatch("hash mismatch")

    resp = await my_route()

    assert resp.status_code == 500
    assert "model package file" in resp.body.decode().lower()


def test_with_route_exceptions_when_client_caused_step_execution_error_raised():
    @with_route_exceptions
    def my_route():
        raise ClientCausedStepExecutionError(
            block_id="step_1",
            status_code=400,
            public_message="Client provided invalid input",
            context="workflow_execution | step_execution",
            inner_error=ValueError("bad value"),
        )

    resp = my_route()

    assert resp.status_code == 400
    assert "invalid input" in resp.body.decode().lower()


def test_with_route_exceptions_when_client_caused_step_execution_error_with_507():
    @with_route_exceptions
    def my_route():
        raise ClientCausedStepExecutionError(
            block_id="step_1",
            status_code=507,
            public_message="Runtime constraints exceeded",
            context="workflow_execution | step_execution",
            inner_error=ValueError("too large"),
        )

    resp = my_route()

    assert resp.status_code == 507
    assert "constraints" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_client_caused_step_execution_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise ClientCausedStepExecutionError(
            block_id="step_1",
            status_code=400,
            public_message="Client provided invalid input",
            context="workflow_execution | step_execution",
            inner_error=ValueError("bad value"),
        )

    resp = await my_route()

    assert resp.status_code == 400
    assert "invalid input" in resp.body.decode().lower()


def test_with_route_exceptions_when_runtime_limits_step_execution_error_raised():
    @with_route_exceptions
    def my_route():
        raise RuntimeLimitsCausedStepExecutionError(
            block_id="step_1",
            status_code=507,
            public_message="Model input size exceeds runtime limit",
            context="workflow_execution | step_execution",
            inner_error=CannotInitialiseModelDueToInputSizeError("too large"),
        )

    resp = my_route()

    assert resp.status_code == 507
    assert "runtime limit" in resp.body.decode().lower()


@pytest.mark.asyncio
async def test_with_route_exceptions_async_when_runtime_limits_step_execution_error_raised():
    @with_route_exceptions_async
    async def my_route():
        raise RuntimeLimitsCausedStepExecutionError(
            block_id="step_1",
            status_code=507,
            public_message="Model input size exceeds runtime limit",
            context="workflow_execution | step_execution",
            inner_error=CannotInitialiseModelDueToInputSizeError("too large"),
        )

    resp = await my_route()

    assert resp.status_code == 507
    assert "runtime limit" in resp.body.decode().lower()
