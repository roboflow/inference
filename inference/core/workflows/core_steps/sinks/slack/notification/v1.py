import hashlib
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field, field_validator
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from inference.core.cache.base import BaseCache
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    BYTES_KIND,
    INTEGER_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

CACHE_EXPIRE_TIME = 15 * 60

LONG_DESCRIPTION = """
The **Slack Notification** block 📩 enables sending notifications via Slack, with customizable messages, attachments, 
and cooldown mechanisms.

The block requires Slack setup - 
[this article](https://www.datacamp.com/tutorial/how-to-send-slack-messages-with-python) may help you 
configuring everything properly.


#### ✨ Key Features

* 📢 **Send Messages:** Deliver notifications to specified Slack channels.

* 🔗 **Dynamic Content:** Craft notifications based on outputs from other Workflow steps.

* 📎 **Attach Files:** Share reports, predictions or visualizations.

* 🕒 **Cooldown Control:** Prevent duplicate notifications within a set time frame.

* ⚙️ **Flexible Execution:** Execute in the background or block Workflow execution for debugging.
"""

PARAMETER_REGEX = re.compile(r"({{\s*\$parameters\.(\w+)\s*}})")


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Slack Notification",
            "version": "v1",
            "short_description": "Send notification via Slack.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "notifications",
                "icon": "far fa-brands fa-slack",
            },
        }
    )
    type: Literal["roboflow_core/slack_notification@v1"]
    slack_token: Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])] = Field(
        description="View the [Roboflow Blog](https://blog.roboflow.com/slack-notification-workflows/) or "
        "[Slack Documentation](https://api.slack.com/tutorials/tracks/getting-a-token) "
        "to learn how to generate a Slack API token.",
        private=True,
        examples=["$inputs.slack_token"],
    )
    channel: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Identifier of Slack channel.",
        examples=["$inputs.slack_channel_id"],
    )
    message: str = Field(
        description="Content of the message to be sent.",
        examples=[
            "During last 5 minutes detected {{ $parameters.num_instances }} instances"
        ],
        json_schema_extra={
            "multiline": True,
        },
    )
    message_parameters: Dict[
        str,
        Union[Selector(), Selector(), str, int, float, bool],
    ] = Field(
        description="Data to be used in the message content.",
        examples=[
            {
                "predictions": "$steps.model.predictions",
                "reference": "$inputs.reference_class_names",
            }
        ],
        default_factory=dict,
        json_schema_extra={
            "always_visible": True,
        },
    )
    message_parameters_operations: Dict[str, List[AllOperationsType]] = Field(
        description="Preprocessing operations to be performed on message parameters.",
        examples=[
            {
                "predictions": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            }
        ],
        default_factory=dict,
    )
    attachments: Dict[str, Selector(kind=[STRING_KIND, BYTES_KIND])] = Field(
        description="Attachments to be sent in the message, such as a csv file or jpg image.",
        default_factory=dict,
        examples=[{"report.csv": "$steps.csv_formatter.csv_content"}],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to run the block asynchronously (True) for faster workflows or  "
        "synchronously (False) for debugging and error handling.",
        examples=["$inputs.fire_and_forget", False],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Boolean flag to disable block execution.",
        examples=[False, "$inputs.disable_slack_notifications"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Number of seconds until a follow-up notification can be sent. "
        f"Maximum value: {CACHE_EXPIRE_TIME} seconds (15 minutes)",
        examples=["$inputs.cooldown_seconds", 3],
        json_schema_extra={
            "always_visible": True,
        },
    )
    cooldown_session_key: str = Field(
        description="Unique key used internally to implement cooldown. Must be unique for each step in your Workflow.",
        examples=["session-1v73kdhfse"],
        json_schema_extra={"hidden": True},
    )

    @field_validator("cooldown_seconds")
    @classmethod
    def ensure_cooldown_seconds_within_bounds(cls, value: Any) -> dict:
        if isinstance(value, int) and (value < 0 or value > CACHE_EXPIRE_TIME):
            raise ValueError(
                f"`cooldown_seconds` must be in range [0, {CACHE_EXPIRE_TIME}]"
            )
        return value

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="throttling_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class SlackNotificationBlockV1(WorkflowBlock):

    def __init__(
        self,
        cache: BaseCache,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._cache = cache
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._clients: Dict[str, WebClient] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["cache", "background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        slack_token: str,
        message: str,
        channel: str,
        message_parameters: Dict[str, Any],
        message_parameters_operations: Dict[str, List[AllOperationsType]],
        attachments: Dict[str, str],
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: int,
        cooldown_session_key: str,
    ) -> BlockResult:
        if disable_sink:
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Sink was disabled by parameter `disable_sink`",
            }
        token_hash = hashlib.sha256(slack_token.encode("utf-8")).hexdigest()
        cache_key = _generate_cache_key_for_cooldown_session(
            session_key=cooldown_session_key,
            token_hash=token_hash,
        )
        last_notification_fired = self._cache.get(cache_key)
        seconds_since_last_notification = cooldown_seconds
        if last_notification_fired is not None:
            last_notification_fired = datetime.fromisoformat(last_notification_fired)
            seconds_since_last_notification = (
                datetime.now() - last_notification_fired
            ).total_seconds()
        if seconds_since_last_notification < cooldown_seconds:
            logging.info(f"Activated `roboflow_core/slack_notification@v1` cooldown.")
            return {
                "error_status": False,
                "throttling_status": True,
                "message": "Sink cooldown applies",
            }
        if token_hash not in self._clients:
            self._clients[token_hash] = WebClient(token=slack_token)
        client = self._clients[token_hash]
        message = format_message(
            message=message,
            message_parameters=message_parameters,
            message_parameters_operations=message_parameters_operations,
        )
        send_notification_handler = partial(
            send_slack_notification,
            client=client,
            channel=channel,
            message=message,
            attachments=attachments,
        )
        last_notification_fired = datetime.now().isoformat()
        self._cache.set(
            key=cache_key, value=last_notification_fired, expire=CACHE_EXPIRE_TIME
        )
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(send_notification_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Notification sent in the background task",
            }
        if fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(send_notification_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Notification sent in the background task",
            }
        error_status, message = send_notification_handler()
        return {
            "error_status": error_status,
            "throttling_status": False,
            "message": message,
        }


def _generate_cache_key_for_cooldown_session(session_key: str, token_hash: str) -> str:
    return (
        f"workflows:steps_cache:roboflow_core/slack_notification@v1:"
        f"{token_hash}:{session_key}:last_notification_time"
    )


def format_message(
    message: str,
    message_parameters: Dict[str, Any],
    message_parameters_operations: Dict[str, List[AllOperationsType]],
) -> str:
    matching_parameters = PARAMETER_REGEX.findall(message)
    parameters_to_get_values = {
        p[1] for p in matching_parameters if p[1] in message_parameters
    }
    parameters_values = {}
    for parameter_name in parameters_to_get_values:
        parameter_value = message_parameters[parameter_name]
        operations = message_parameters_operations.get(parameter_name)
        if not operations:
            parameters_values[parameter_name] = parameter_value
            continue
        operations_chain = build_operations_chain(operations=operations)
        parameters_values[parameter_name] = operations_chain(
            parameter_value, global_parameters={}
        )
    parameter_to_placeholders = defaultdict(list)
    for placeholder, parameter_name in matching_parameters:
        if parameter_name not in parameters_to_get_values:
            continue
        parameter_to_placeholders[parameter_name].append(placeholder)
    for parameter_name, placeholders in parameter_to_placeholders.items():
        for placeholder in placeholders:
            message = message.replace(
                placeholder, str(parameters_values[parameter_name])
            )
    return message


def send_slack_notification(
    client: WebClient,
    channel: str,
    message: str,
    attachments: Dict[str, Union[str, bytes]],
) -> Tuple[bool, str]:
    try:
        _send_slack_notification(
            client=client,
            channel=channel,
            message=message,
            attachments=attachments,
        )
        return False, "Notification sent successfully"
    except SlackApiError as error:
        error_details = error.response.get("error", "Not Available.")
        logging.warning(f"Could not send Slack notification. Error: {error_details}")
        return (
            True,
            f"Failed to send Slack notification. Internal error details: {error_details}",
        )
    except Exception as error:
        logging.warning(f"Could not send Slack notification. Error: {str(error)}")
        return (
            True,
            f"Failed to send Slack notification. Internal error details: {error}",
        )


def _send_slack_notification(
    client: WebClient,
    channel: str,
    message: str,
    attachments: Dict[str, Union[str, bytes]],
) -> None:
    if not attachments:
        _ = client.chat_postMessage(
            channel=channel,
            text=message,
        )
    file_uploads = [
        {
            "title": name,
            "content": value,
        }
        for name, value in attachments.items()
    ]
    _ = client.files_upload_v2(
        channel=channel,
        initial_comment=message,
        file_uploads=file_uploads,
    )
