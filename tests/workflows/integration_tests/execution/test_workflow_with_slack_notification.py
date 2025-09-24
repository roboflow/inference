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

SLACK_TOKEN = os.getenv("SLACK_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")

WORKFLOW_WITH_PURE_TEXT_NOTIFICATION = {
    "version": "1.4.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "channel_id"},
    ],
    "steps": [
        {
            "type": "roboflow_core/environment_secrets_store@v1",
            "name": "vault",
            "variables_storing_secrets": ["SLACK_TOKEN"],
        },
        {
            "type": "roboflow_core/slack_notification@v1",
            "name": "notification",
            "slack_token": "$steps.vault.slack_token",
            "message": "This is example message",
            "channel": "$inputs.channel_id",
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


@pytest.mark.skipif(SLACK_TOKEN is None, reason="`SLACK_TOKEN` variable not exported")
@pytest.mark.skipif(
    SLACK_CHANNEL_ID is None, reason="`SLACK_CHANNEL_ID` variable not exported"
)
def test_minimalist_workflow_with_slack_notifications() -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PURE_TEXT_NOTIFICATION,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "channel_id": SLACK_CHANNEL_ID,
        },
    )

    # then
    assert result[0]["status"] is False


WORKFLOW_SENDING_PREDICTION_SUMMARY = {
    "version": "1.4.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-640",
        },
        {"type": "WorkflowParameter", "name": "channel_id"},
        {"type": "WorkflowParameter", "name": "slack_token"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
        },
        {
            "type": "roboflow_core/slack_notification@v1",
            "name": "notification",
            "slack_token": "$inputs.slack_token",
            "message": "Detected {{ $parameters.predictions }} objects",
            "channel": "$inputs.channel_id",
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
    use_case_title="Workflow sending notification to Slack",
    use_case_description="""
This Workflow illustrates how to send notification to Slack.
    """,
    workflow_definition=WORKFLOW_SENDING_PREDICTION_SUMMARY,
    workflow_name_in_app="basic-slack-notification",
)
@pytest.mark.skipif(SLACK_TOKEN is None, reason="`SLACK_TOKEN` variable not exported")
@pytest.mark.skipif(
    SLACK_CHANNEL_ID is None, reason="`SLACK_CHANNEL_ID` variable not exported"
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
            "image": [crowd_image, crowd_image],
            "channel_id": SLACK_CHANNEL_ID,
            "slack_token": SLACK_TOKEN,
        },
    )

    # then
    assert result[0]["status"] is False
    assert result[1]["status"] is False


WORKFLOW_SENDING_PREDICTION_SUMMARY_AND_FILES = {
    "version": "1.4.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-640",
        },
        {"type": "WorkflowParameter", "name": "channel_id"},
        {"type": "WorkflowParameter", "name": "slack_token"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "image_serialization",
            "data": "$inputs.image",
            "operations": [{"type": "ConvertImageToJPEG"}],
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "predictions_serialization",
            "data": "$steps.detection.predictions",
            "operations": [
                {"type": "DetectionsToDictionary"},
                {"type": "ConvertDictionaryToJSON"},
            ],
        },
        {
            "type": "roboflow_core/slack_notification@v1",
            "name": "notification",
            "slack_token": "$inputs.slack_token",
            "message": "Detected {{ $parameters.predictions }} objects",
            "channel": "$inputs.channel_id",
            "message_parameters": {
                "predictions": "$steps.detection.predictions",
            },
            "message_parameters_operations": {
                "predictions": [{"type": "SequenceLength"}],
            },
            "attachments": {
                "image.jpg": "$steps.image_serialization.output",
                "prediction.json": "$steps.predictions_serialization.output",
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
    use_case_title="Workflow sending notification with attachments to Slack",
    use_case_description="""
This Workflow illustrates how to send notification with attachments to Slack.
    """,
    workflow_definition=WORKFLOW_SENDING_PREDICTION_SUMMARY_AND_FILES,
    workflow_name_in_app="advanced-slack-notification",
)
@pytest.mark.skipif(SLACK_TOKEN is None, reason="`SLACK_TOKEN` variable not exported")
@pytest.mark.skipif(
    SLACK_CHANNEL_ID is None, reason="`SLACK_CHANNEL_ID` variable not exported"
)
def test_workflow_sending_attachments_to_slack(
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
        workflow_definition=WORKFLOW_SENDING_PREDICTION_SUMMARY_AND_FILES,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image, crowd_image],
            "channel_id": SLACK_CHANNEL_ID,
            "slack_token": SLACK_TOKEN,
        },
    )

    # then
    assert result[0]["status"] is False
    assert result[1]["status"] is False
