def test_client_caused_step_error_maps_to_its_status():
    from roboflow_workflows.errors import ClientCausedStepExecutionError
    from roboflow_workflows.http_contract.errors import workflow_error_payload

    error = ClientCausedStepExecutionError(
        block_id="det",
        status_code=402,
        public_message="no credits",
        context="ctx",
        inner_error=ValueError("x"),
    )
    status, payload = workflow_error_payload(error)
    assert status == 402
    assert payload["blocks_errors"] == [
        {
            "block_id": "det",
            "block_type": None,
            "block_details": None,
            "property_name": None,
            "property_details": None,
            "block_traceback": None,
        }
    ]
    assert payload["error_type"] == "ClientCausedStepExecutionError"


def test_unrelated_error_is_not_mapped():
    from roboflow_workflows.http_contract.errors import workflow_error_payload

    assert workflow_error_payload(ValueError("x")) is None
