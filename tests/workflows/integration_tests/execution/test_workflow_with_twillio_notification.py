import os

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOWS_TWILIO_ACCOUNT_SID = os.getenv("WORKFLOWS_TWILIO_ACCOUNT_SID")
WORKFLOWS_TWILIO_AUTH_TOKEN = os.getenv("WORKFLOWS_TWILIO_AUTH_TOKEN")
WORKFLOWS_TWILIO_PHONE_NUMBER = os.getenv("WORKFLOWS_TWILIO_PHONE_NUMBER")
WORKFLOWS_RECEIVER_PHONE_NUMBER = os.getenv("WORKFLOWS_RECEIVER_PHONE_NUMBER")


WORKFLOW_SENDING_PREDICTION_SUMMARY = {
    "version": "1.4.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-640",
        },
        {"type": "WorkflowParameter", "name": "account_sid"},
        {"type": "WorkflowParameter", "name": "auth_token"},
        {"type": "WorkflowParameter", "name": "sender_number"},
        {"type": "WorkflowParameter", "name": "receiver_number"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
        },
        {
            "type": "roboflow_core/twilio_sms_notification@v1",
            "name": "notification",
            "twilio_account_sid": "$inputs.account_sid",
            "twilio_auth_token": "$inputs.auth_token",
            "message": "Detected {{ $parameters.predictions }} objects",
            "sender_number": "$inputs.sender_number",
            "receiver_number": "$inputs.receiver_number",
            "message_parameters": {
                "predictions": "$steps.detection.predictions",
            },
            "message_parameters_operations": {
                "predictions": [{"type": "SequenceLength"}],
            },
            "fire_and_forget": False,
            "cooldown_seconds": 0,
            "cooldown_session_key": "some-unique-key",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "status",
            "selector": "$steps.notification.error_status",
        },
    ],
}


@add_to_workflows_gallery(
    category="Integration with external apps",
    use_case_title="Workflow sending SMS notification with Twilio",
    use_case_description="""
This Workflow illustrates how to send SMS notification with Twilio.
    """,
    workflow_definition=WORKFLOW_SENDING_PREDICTION_SUMMARY,
    workflow_name_in_app="basic-twilio-sms-notification",
)
@pytest.mark.skipif(
    WORKFLOWS_TWILIO_ACCOUNT_SID is None,
    reason="`WORKFLOWS_TWILIO_ACCOUNT_SID` variable not exported",
)
@pytest.mark.skipif(
    WORKFLOWS_TWILIO_AUTH_TOKEN is None,
    reason="`WORKFLOWS_TWILIO_AUTH_TOKEN` variable not exported",
)
@pytest.mark.skipif(
    WORKFLOWS_TWILIO_PHONE_NUMBER is None,
    reason="`WORKFLOWS_TWILIO_PHONE_NUMBER` variable not exported",
)
@pytest.mark.skipif(
    WORKFLOWS_RECEIVER_PHONE_NUMBER is None,
    reason="`WORKFLOWS_RECEIVER_PHONE_NUMBER` variable not exported",
)
def test_workflow_with_message_based_on_other_step_output(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_SENDING_PREDICTION_SUMMARY,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image],
            "account_sid": WORKFLOWS_TWILIO_ACCOUNT_SID,
            "auth_token": WORKFLOWS_TWILIO_AUTH_TOKEN,
            "sender_number": WORKFLOWS_TWILIO_PHONE_NUMBER,
            "receiver_number": WORKFLOWS_RECEIVER_PHONE_NUMBER,
        },
    )

    # then
    assert result[0]["status"] is False
