from unittest import mock

import pytest
from packaging.version import Version

from inference.core.workflows.errors import (
    NotSupportedExecutionEngineError,
    WorkflowEnvironmentConfigurationError,
)
from inference.core.workflows.execution_engine import core
from inference.core.workflows.execution_engine.core import (
    _select_execution_engine,
    retrieve_requested_execution_engine_version, ExecutionEngine,
)
from inference.core.workflows.execution_engine.v1.core import (
    EXECUTION_ENGINE_V1_VERSION,
)


def test_retrieve_requested_execution_engine_version_when_version_not_given_in_manifest() -> (
    None
):
    # when
    result = retrieve_requested_execution_engine_version(workflow_definition={})

    # then
    assert (
        result == EXECUTION_ENGINE_V1_VERSION
    ), "Expected latest v1 version of EE to be returned"


@mock.patch.dict(core.REGISTERED_ENGINES, {}, clear=True)
def test_retrieve_requested_execution_engine_version_when_version_not_given_in_manifest_and_no_default_registered() -> (
    None
):
    # when
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        _ = retrieve_requested_execution_engine_version(workflow_definition={})


def test_retrieve_requested_execution_engine_version_when_matching_version_specified_in_manifest() -> (
    None
):
    # when
    result = retrieve_requested_execution_engine_version(
        workflow_definition={
            "version": "1.0.0",
        }
    )

    # then
    assert result == Version("1.0.0"), "Expected pointed version to be returned"


@mock.patch.dict(core.REGISTERED_ENGINES, {Version("1.3.0"): "a"}, clear=True)
def test_select_execution_engine_when_requested_older_version_with_matching_major() -> (
    None
):
    # when
    result = _select_execution_engine(requested_engine_version=Version("1.1.2"))

    # then
    assert result == "a", "Expected version 1.3.0 to be selected"


@mock.patch.dict(
    core.REGISTERED_ENGINES, {Version("1.3.0"): "a", Version("1.4.0"): "b"}, clear=True
)
def test_select_execution_engine_when_multiple_versions_match() -> None:
    # when
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        _ = _select_execution_engine(requested_engine_version=Version("1.1.2"))


@mock.patch.dict(
    core.REGISTERED_ENGINES, {Version("1.3.0"): "a", Version("1.4.0"): "b"}, clear=True
)
def test_select_execution_engine_when_no_version_match() -> None:
    # when
    with pytest.raises(NotSupportedExecutionEngineError):
        _ = _select_execution_engine(requested_engine_version=Version("2.0.0"))


def test_execution_engine_init_with_invalid_step_error_handler() -> None:
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        _ = ExecutionEngine.init(
            workflow_definition={},
            step_error_handler="invalid",
        )
