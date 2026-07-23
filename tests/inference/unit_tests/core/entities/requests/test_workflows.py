from inference.core.entities.requests.workflows import (
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
