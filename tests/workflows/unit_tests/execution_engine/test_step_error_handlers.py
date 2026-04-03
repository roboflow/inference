import pytest

from inference.core.exceptions import (
    CannotInitialiseModelDueToInputSizeError,
    InferenceModelNotFound,
    InvalidModelIDError,
    ModelManagerLockAcquisitionError,
    PaymentRequiredError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
)
from inference.core.workflows.errors import (
    ClientCausedStepExecutionError,
    RuntimeLimitsCausedStepExecutionError,
)
from inference.core.workflows.execution_engine.v1.step_error_handlers import (
    extended_roboflow_errors_handler,
    legacy_step_error_handler,
)
from inference_models.errors import (
    ModelPackageAlternativesExhaustedError,
    ModelPackageRestrictedError,
)
from inference_sdk.http.errors import HTTPCallErrorError


@pytest.mark.parametrize(
    "error", (ModelManagerLockAcquisitionError(), InferenceModelNotFound())
)
def test_legacy_step_error_handler_when_error_should_be_raised(
    error: Exception,
) -> None:
    # when
    with pytest.raises(type(error)):
        legacy_step_error_handler("some", error)


def test_legacy_step_error_handler_when_error_should_not_be_raised() -> None:
    # when
    legacy_step_error_handler("some", Exception())

    # then - no error


@pytest.mark.parametrize(
    "error", (ModelManagerLockAcquisitionError(), InferenceModelNotFound())
)
def test_extended_roboflow_errors_handler_when_error_should_be_handled_as_in_legacy_case(
    error: Exception,
) -> None:
    # when
    with pytest.raises(type(error)):
        extended_roboflow_errors_handler("some", error)


def test_extended_roboflow_errors_handler_when_invalid_model_id_defined() -> None:
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", InvalidModelIDError())

    # then
    assert error.value.status_code == 400


def test_extended_roboflow_errors_handler_when_invalid_model_id_defined_while_remote_execution() -> (
    None
):
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", HTTPCallErrorError("", 400, None))

    # then
    assert error.value.status_code == 400


def test_extended_roboflow_errors_handler_when_not_authorised_error_occurs() -> None:
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", RoboflowAPINotAuthorizedError())

    # then
    assert error.value.status_code == 401


def test_extended_roboflow_errors_handler_when_not_authorised_error_occurs_while_remote_execution() -> (
    None
):
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", HTTPCallErrorError("", 401, None))

    # then
    assert error.value.status_code == 401


def test_extended_roboflow_errors_handler_when_forbidden_error_occurs() -> None:
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", RoboflowAPIForbiddenError())

    # then
    assert error.value.status_code == 403


def test_extended_roboflow_errors_handler_when_forbidden_error_occurs_while_remote_execution() -> (
    None
):
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", HTTPCallErrorError("", 403, None))

    # then
    assert error.value.status_code == 403


def test_extended_roboflow_errors_handler_when_payment_required_error_occurs() -> None:
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", PaymentRequiredError())

    # then
    assert error.value.status_code == 402


def test_extended_roboflow_errors_handler_when_payment_required_error_occurs_while_remote_execution() -> (
    None
):
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", HTTPCallErrorError("", 402, None))

    # then
    assert error.value.status_code == 402


def test_extended_roboflow_errors_handler_when_not_found_error_occurs() -> None:
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", RoboflowAPINotNotFoundError())

    # then
    assert error.value.status_code == 404


def test_extended_roboflow_errors_handler_when_not_found_error_occurs_while_remote_execution() -> (
    None
):
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", HTTPCallErrorError("", 404, None))

    # then
    assert error.value.status_code == 404


def test_extended_roboflow_errors_handler_when_http_507_occurs() -> None:
    # when
    with pytest.raises(RuntimeLimitsCausedStepExecutionError) as error:
        extended_roboflow_errors_handler(
            "some",
            HTTPCallErrorError(
                "", 507, "Could not load model due to input resolution constraint"
            ),
        )

    # then
    assert error.value.status_code == 507


def test_extended_roboflow_errors_handler_when_old_inference_input_size_error_occurs() -> (
    None
):
    # when
    with pytest.raises(RuntimeLimitsCausedStepExecutionError) as error:
        extended_roboflow_errors_handler(
            "some",
            CannotInitialiseModelDueToInputSizeError(
                "Could not load model due to input resolution constraint"
            ),
        )

    # then
    assert error.value.status_code == 507


def test_extended_roboflow_errors_handler_when_new_inference_model_loading_failed_without_package_restriction_errors() -> (
    None
):
    # when
    extended_roboflow_errors_handler(
        "some",
        ModelPackageAlternativesExhaustedError(
            "Could not load model due to input resolution constraint"
        ),
    )

    # then
    # expected not to raise


def test_extended_roboflow_errors_handler_when_new_inference_model_loading_failed_with_package_restriction_errors() -> (
    None
):
    # when
    with pytest.raises(RuntimeLimitsCausedStepExecutionError) as error:
        extended_roboflow_errors_handler(
            "some",
            ModelPackageAlternativesExhaustedError(
                "Could not load model due to input resolution constraint",
                alternatives_errors=[
                    ModelPackageRestrictedError(
                        "Could not load model due to input resolution constraint"
                    )
                ],
            ),
        )

    # then
    assert error.value.status_code == 507


def test_extended_roboflow_errors_handler_when_new_inference_model_loading_failed_with_package_restriction_errors_directly() -> (
    None
):
    # when
    with pytest.raises(RuntimeLimitsCausedStepExecutionError) as error:
        extended_roboflow_errors_handler(
            "some",
            ModelPackageRestrictedError(
                "Could not load model due to input resolution constraint"
            ),
        )

    # then
    assert error.value.status_code == 507
