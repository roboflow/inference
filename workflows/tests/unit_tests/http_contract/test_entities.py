def test_entities_importable_without_server_packages():
    import sys

    assert not any(
        name == "inference" or name.startswith("inference.") for name in sys.modules
    ), "legacy inference must not be imported by the workflows package"
    from roboflow_workflows.http_contract.entities import (
        WorkflowErrorResponse,
        WorkflowInferenceRequest,
        WorkflowsBlocksDescription,
    )

    request = WorkflowInferenceRequest(inputs={"x": 1})
    assert request.inner_workflow_dispatch_depth == 0 and request.disable_sinks is False
    assert (
        WorkflowErrorResponse(message="m", error_type="t", context="c").blocks_errors
        is None
    )
