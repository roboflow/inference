"""WP-A04 characterization of the exceptions and warning the streams raise.

Pins, per historical class: its name (including the RoboflowAPINotNotFoundError
typo), exact base chain, constructor/message behavior, pickling and the error
payload the stream manager builds from it. After WP-A04 the historical names in
`inference.core.exceptions` / `inference.core.warnings` are the very classes
defined by the stream package or the Workflows platform-error module, so every
existing `except` clause and `isinstance` check keeps matching.
"""

import pickle
from typing import List

import pytest

from inference.core import exceptions as core_exceptions
from inference.core import warnings as core_warnings
from inference.core.interfaces.stream import exceptions as stream_exceptions
from inference.core.interfaces.stream import warnings as stream_warnings
from inference.core.interfaces.stream_manager.manager_app.entities import (
    ErrorType,
    OperationStatus,
)
from inference.core.interfaces.stream_manager.manager_app.serialisation import (
    describe_error,
)

# name -> exact MRO names below the class itself
EXPECTED_BASES = {
    "MissingApiKeyError": ["Exception", "BaseException", "object"],
    "CannotInitialiseModelError": ["Exception", "BaseException", "object"],
    "WebRTCConfigurationError": ["Exception", "BaseException", "object"],
    "InvalidEnvironmentVariableError": ["Exception", "BaseException", "object"],
    "RoboflowAPINotAuthorizedError": [
        "RoboflowAPIUnsuccessfulRequestError",
        "RoboflowAPIRequestError",
        "Exception",
        "BaseException",
        "object",
    ],
    "RoboflowAPINotNotFoundError": [
        "RoboflowAPIUnsuccessfulRequestError",
        "RoboflowAPIRequestError",
        "Exception",
        "BaseException",
        "object",
    ],
    "RoboflowAPITimeoutError": [
        "RoboflowAPIRequestError",
        "Exception",
        "BaseException",
        "object",
    ],
    "RoboflowAPIConnectionError": [
        "RoboflowAPIRequestError",
        "Exception",
        "BaseException",
        "object",
    ],
}

STREAM_OWNED_EXCEPTIONS = [
    "MissingApiKeyError",
    "CannotInitialiseModelError",
    "WebRTCConfigurationError",
    "InvalidEnvironmentVariableError",
]
PLATFORM_EXCEPTIONS = [
    "RoboflowAPINotAuthorizedError",
    "RoboflowAPINotNotFoundError",
    "RoboflowAPITimeoutError",
    "RoboflowAPIConnectionError",
]


def _mro_names(cls: type) -> List[str]:
    return [base.__name__ for base in cls.__mro__[1:]]


@pytest.mark.parametrize("name", STREAM_OWNED_EXCEPTIONS)
def test_legacy_name_is_the_stream_owned_class(name: str) -> None:
    cls = getattr(stream_exceptions, name)

    assert getattr(core_exceptions, name) is cls
    assert cls.__module__ == "inference.core.interfaces.stream.exceptions"


@pytest.mark.parametrize("name", PLATFORM_EXCEPTIONS)
def test_legacy_name_is_the_workflows_platform_class(name: str) -> None:
    from roboflow_workflows.prototypes import platform_errors

    cls = getattr(platform_errors, name)

    assert getattr(core_exceptions, name) is cls
    assert cls.__module__ == "roboflow_workflows.prototypes.platform_errors"
    assert (
        core_exceptions.RoboflowAPIUnsuccessfulRequestError
        is platform_errors.RoboflowAPIUnsuccessfulRequestError
    )


def test_legacy_warning_name_is_the_stream_owned_category() -> None:
    assert (
        core_warnings.InferenceExperimentalFeatureWarning
        is stream_warnings.InferenceExperimentalFeatureWarning
    )


def test_stream_owned_definitions_import_nothing() -> None:
    import ast

    for module in (stream_exceptions, stream_warnings):
        source = open(module.__file__, encoding="utf-8").read()
        assert not [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ], module.__name__


def test_workflows_platform_errors_do_not_replace_legacy_status_mapping() -> None:
    # Adding the classes to Workflows must not change which legacy classes are
    # related: the host keeps translating each one to its own ErrorType.
    from roboflow_workflows.prototypes import platform_errors

    assert not issubclass(
        platform_errors.RoboflowAPINotNotFoundError,
        platform_errors.RoboflowAPINotAuthorizedError,
    )
    assert not issubclass(
        platform_errors.RoboflowAPITimeoutError,
        platform_errors.RoboflowAPIConnectionError,
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_BASES))
def test_historical_exception_names_and_base_chains(name: str) -> None:
    cls = getattr(core_exceptions, name)

    assert cls.__name__ == name
    assert cls.__qualname__ == name
    assert _mro_names(cls) == EXPECTED_BASES[name]


def test_not_authorized_is_not_forbidden() -> None:
    assert not issubclass(
        core_exceptions.RoboflowAPINotAuthorizedError,
        core_exceptions.RoboflowAPIForbiddenError,
    )
    assert not issubclass(
        core_exceptions.RoboflowAPIForbiddenError,
        core_exceptions.RoboflowAPINotAuthorizedError,
    )
    assert issubclass(
        core_exceptions.RoboflowAPINotAuthorizedError,
        core_exceptions.RoboflowAPIUnsuccessfulRequestError,
    )


def test_timeout_and_connection_errors_are_not_unsuccessful_requests() -> None:
    for name in ("RoboflowAPITimeoutError", "RoboflowAPIConnectionError"):
        cls = getattr(core_exceptions, name)

        assert issubclass(cls, core_exceptions.RoboflowAPIRequestError)
        assert not issubclass(cls, core_exceptions.RoboflowAPIUnsuccessfulRequestError)


def test_dependent_legacy_subclasses_keep_their_parents() -> None:
    assert issubclass(
        core_exceptions.CannotInitialiseModelDueToInputSizeError,
        core_exceptions.CannotInitialiseModelError,
    )
    input_size_error = core_exceptions.CannotInitialiseModelDueToInputSizeError
    assert _mro_names(input_size_error)[0] == "CannotInitialiseModelError"


@pytest.mark.parametrize("name", sorted(EXPECTED_BASES))
def test_messages_arguments_and_pickling(name: str) -> None:
    cls = getattr(core_exceptions, name)

    error = cls("something failed", 42)
    restored = pickle.loads(pickle.dumps(error))

    assert str(error) == "('something failed', 42)"
    assert str(cls("single message")) == "single message"
    assert str(cls()) == ""
    assert type(restored) is cls
    assert restored.args == ("something failed", 42)


@pytest.mark.parametrize("name", sorted(EXPECTED_BASES))
def test_manager_error_payload_names_the_historical_class(name: str) -> None:
    error = getattr(core_exceptions, name)("details")

    payload = describe_error(
        error,
        error_type=ErrorType.OPERATION_ERROR,
        public_error_message="public",
    )

    assert payload == {
        "status": OperationStatus.FAILURE,
        "error_type": ErrorType.OPERATION_ERROR,
        "error_class": name,
        "error_message": "details",
        "public_error_message": "public",
    }


def test_experimental_warning_category() -> None:
    category = core_warnings.InferenceExperimentalFeatureWarning

    assert category.__name__ == "InferenceExperimentalFeatureWarning"
    assert _mro_names(category) == ["Warning", "Exception", "BaseException", "object"]
    assert not issubclass(category, core_warnings.InferenceDeprecationWarning)
