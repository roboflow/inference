from unittest import mock

import numpy as np
import pytest
import requests

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.core import ExecutionEngine

# The event value is computed upstream from a measured size and wired into the
# Vision Events block through a selector. The image width stands in for the
# measurement so the test needs no model; the rate makes the result fractional.
VALUE_PER_PIXEL = 0.37

WORKFLOW_WITH_COMPUTED_EVENT_VALUE = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "write_to_event_store"},
    ],
    "steps": [
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "event_value",
            "data": "$inputs.image",
            "operations": [
                {"type": "ExtractImageProperty", "property_name": "width"},
                {"type": "Multiply", "other": VALUE_PER_PIXEL},
            ],
        },
        {
            "type": "roboflow_core/roboflow_vision_events@v1",
            "name": "vision_events",
            "input_image": "$inputs.image",
            "event_type": "quality_check",
            "solution": "surface-inspection",
            "qc_result": "fail",
            "value_base_units": "$steps.event_value.output",
            "write_to_event_store": "$inputs.write_to_event_store",
            "event_store_url": "http://event-store.local:8001",
            "fire_and_forget": False,
            "cooldown_seconds": 0,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "error_status",
            "selector": "$steps.vision_events.error_status",
        },
        {
            "type": "JsonField",
            "name": "message",
            "selector": "$steps.vision_events.message",
        },
    ],
}


# The value arrives as a workflow input; inputs are validated against the
# manifest, so they must already be whole numbers.
WORKFLOW_WITH_EVENT_VALUE_INPUT = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "event_value"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_vision_events@v1",
            "name": "vision_events",
            "input_image": "$inputs.image",
            "event_type": "quality_check",
            "solution": "surface-inspection",
            "qc_result": "fail",
            "value_base_units": "$inputs.event_value",
            "fire_and_forget": False,
            "cooldown_seconds": 0,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "error_status",
            "selector": "$steps.vision_events.error_status",
        },
        {
            "type": "JsonField",
            "name": "message",
            "selector": "$steps.vision_events.message",
        },
    ],
}


def _mock_response(status_code: int, body: dict) -> mock.MagicMock:
    response = mock.MagicMock()
    response.status_code = status_code
    response.json.return_value = body
    response.raise_for_status.return_value = None
    return response


def _route_requests(url: str, **_kwargs) -> mock.MagicMock:
    if url.endswith("/vision-events/upload"):
        return _mock_response(200, {"sourceId": "source-1"})
    return _mock_response(201, {"id": "evt-local-1"})


@pytest.mark.parametrize(
    "write_to_event_store, event_url_suffix, value_key",
    [
        (False, "/vision-events", "valueBaseUnits"),
        (True, "/v2/events", "value_base_units"),
    ],
)
def test_workflow_sends_computed_event_value(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    write_to_event_store: bool,
    event_url_suffix: str,
    value_key: str,
) -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_COMPUTED_EVENT_VALUE,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": "test-key",
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with mock.patch.object(requests, "post", side_effect=_route_requests) as post:
        result = execution_engine.run(
            runtime_parameters={
                "image": dogs_image,
                "write_to_event_store": write_to_event_store,
            }
        )

    # then
    assert result[0]["error_status"] is False, result[0]["message"]
    event_calls = [
        call for call in post.call_args_list if call.args[0].endswith(event_url_suffix)
    ]
    assert len(event_calls) == 1
    payload = event_calls[0].kwargs["json"]
    # rounded to whole base units
    assert payload[value_key] == round(dogs_image.shape[1] * VALUE_PER_PIXEL)
    assert dogs_image.shape[1] * VALUE_PER_PIXEL != payload[value_key]
    assert "valueCurrency" not in payload and "value_currency" not in payload


def _init_engine(workflow: dict, model_manager: ModelManager) -> ExecutionEngine:
    return ExecutionEngine.init(
        workflow_definition=workflow,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": "test-key",
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )


# A string input such as "1999" passes input validation, so it must be sent too.
@pytest.mark.parametrize("event_value", [1999, 1999.0, "1999"])
def test_workflow_sends_whole_number_event_value_input_as_int(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    event_value,
) -> None:
    # given
    execution_engine = _init_engine(WORKFLOW_WITH_EVENT_VALUE_INPUT, model_manager)

    # when
    with mock.patch.object(requests, "post", side_effect=_route_requests) as post:
        result = execution_engine.run(
            runtime_parameters={"image": dogs_image, "event_value": event_value}
        )

    # then
    assert result[0]["error_status"] is False, result[0]["message"]
    event_calls = [
        call for call in post.call_args_list if call.args[0].endswith("/vision-events")
    ]
    assert len(event_calls) == 1
    sent_value = event_calls[0].kwargs["json"]["valueBaseUnits"]
    assert sent_value == 1999
    assert type(sent_value) is int


@pytest.mark.parametrize("event_value", [12.6, 19.99 * 100, "lots", True])
def test_workflow_rejects_event_value_input_that_is_not_a_whole_number(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    event_value,
) -> None:
    # given
    execution_engine = _init_engine(WORKFLOW_WITH_EVENT_VALUE_INPUT, model_manager)

    # when
    with mock.patch.object(requests, "post", side_effect=_route_requests) as post:
        with pytest.raises(RuntimeInputError):
            execution_engine.run(
                runtime_parameters={"image": dogs_image, "event_value": event_value}
            )

    # then
    post.assert_not_called()
