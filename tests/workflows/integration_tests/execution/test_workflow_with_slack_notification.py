from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine

WORKFLOW_WITH_PURE_TEXT_NOTIFICATION = {
    "version": "1.4.0",
    "inputs": [],
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
            "channel": "xxxs",
            "fire_and_forget": False,
            "cooldown_seconds": 0,
            "cooldown_session_key": "some-unique-key",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "status",
            "selector": "$steps.notification.error_status",
        },
    ],
}


def test_minimalist_workflow_with_slack_notifications() -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PURE_TEXT_NOTIFICATION,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={},
    )

    # then
    print(result)
