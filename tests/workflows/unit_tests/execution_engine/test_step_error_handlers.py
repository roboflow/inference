import pytest

from inference.core.exceptions import ModelManagerLockAcquisitionError, InferenceModelNotFound, InvalidModelIDError, \
    RoboflowAPINotAuthorizedError, RoboflowAPINotNotFoundError, RoboflowAPIForbiddenError
from inference.core.workflows.errors import ClientCausedStepExecutionError
from inference.core.workflows.execution_engine.v1.step_error_handlers import legacy_step_error_handler, \
    extended_roboflow_errors_handler
from inference_sdk.http.errors import HTTPCallErrorError


@pytest.mark.parametrize("error", (ModelManagerLockAcquisitionError(), InferenceModelNotFound()))
def test_legacy_step_error_handler_when_error_should_be_raised(error: Exception) -> None:
    # when
    with pytest.raises(type(error)):
        legacy_step_error_handler("some", error)


def test_legacy_step_error_handler_when_error_should_not_be_raised() -> None:
    # when
    legacy_step_error_handler("some", Exception())

    # then - no error


@pytest.mark.parametrize("error", (ModelManagerLockAcquisitionError(), InferenceModelNotFound()))
def test_extended_roboflow_errors_handler_when_error_should_be_handled_as_in_legacy_case(error: Exception) -> None:
    # when
    with pytest.raises(type(error)):
        extended_roboflow_errors_handler("some", error)


def test_extended_roboflow_errors_handler_when_invalid_model_id_defined() -> None:
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", InvalidModelIDError())

    # then
    assert error.value.status_code == 400


def test_extended_roboflow_errors_handler_when_invalid_model_id_defined_while_remote_execution() -> None:
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


def test_extended_roboflow_errors_handler_when_not_authorised_error_occurs_while_remote_execution() -> None:
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


def test_extended_roboflow_errors_handler_when_forbidden_error_occurs_while_remote_execution() -> None:
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", HTTPCallErrorError("", 403, None))

    # then
    assert error.value.status_code == 403


def test_extended_roboflow_errors_handler_when_not_found_error_occurs() -> None:
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", RoboflowAPINotNotFoundError())

    # then
    assert error.value.status_code == 404


def test_extended_roboflow_errors_handler_when_not_found_error_occurs_while_remote_execution() -> None:
    # when
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler("some", HTTPCallErrorError("", 404, None))

    # then
    assert error.value.status_code == 404
