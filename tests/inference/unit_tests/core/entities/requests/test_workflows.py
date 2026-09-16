import pytest
from pydantic import ValidationError

from inference.core.entities.requests.workflows import (
    PredefinedWorkflowInferenceRequest,
    WorkflowSpecificationInferenceRequest,
)


def test_workflow_request_runs_sinks_by_default() -> None:
    request = WorkflowSpecificationInferenceRequest(
        inputs={},
        specification={},
    )

    assert request.disable_sinks is False


def test_workflow_request_accepts_sink_disabling_mode() -> None:
    request = WorkflowSpecificationInferenceRequest(
        inputs={},
        specification={},
        disable_sinks=True,
    )

    assert request.disable_sinks is True


def test_workflow_requests_validate_dispatch_depth() -> None:
    for request_type in [
        WorkflowSpecificationInferenceRequest,
        PredefinedWorkflowInferenceRequest,
    ]:
        arguments = {"inputs": {}, "specification": {}}
        assert request_type(**arguments).inner_workflow_dispatch_depth == 0
        request = request_type(**arguments, inner_workflow_dispatch_depth=3)
        assert request.inner_workflow_dispatch_depth == 3
        for invalid_depth in [-1, 1.5, True, "0"]:
            with pytest.raises(ValidationError):
                request_type(**arguments, inner_workflow_dispatch_depth=invalid_depth)
