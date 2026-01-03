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
Send notifications to Slack channels with customizable message content featuring dynamic workflow data parameters, file attachments, cooldown throttling using cache-based session tracking, and optional async background execution for team alerts, monitoring, and real-time communication workflows.

## How This Block Works

This block sends notifications to Slack channels using the Slack Web API, integrating workflow execution results into message content. The block:

1. Checks if the sink is disabled via `disable_sink` flag (if disabled, returns immediately without sending)
2. Generates a cache key for cooldown tracking using the Slack token hash and `cooldown_session_key` (unique per workflow step)
3. Validates cooldown period by checking cache for the last notification timestamp (if enabled, throttles notifications within `cooldown_seconds` of the last sent notification, returning throttling status)
4. Creates or retrieves a Slack WebClient instance for the provided token (caches clients by token hash for efficiency)
5. Formats the message by processing dynamic parameters (replaces placeholders like `{{ $parameters.parameter_name }}` with actual workflow data from `message_parameters`)
6. Applies optional UQL operations to transform parameter values before insertion (e.g., extract class names from detections, calculate metrics, filter data) using `message_parameters_operations`
7. Sends the notification to the specified Slack channel:
   - **Without attachments**: Uses `chat_postMessage` API to send text-only messages
   - **With attachments**: Uses `files_upload_v2` API to upload files with the message as an initial comment
8. Updates the cache with the current notification timestamp (expires after 15 minutes)
9. Executes synchronously or asynchronously based on `fire_and_forget` setting:
   - **Synchronous mode** (`fire_and_forget=False`): Waits for Slack API call completion, returns actual error status for debugging
   - **Asynchronous mode** (`fire_and_forget=True`): Sends notification in background task, workflow continues immediately, error status always False
10. Returns status outputs indicating success, throttling, or errors (includes Slack API error details when available)

The block supports dynamic message content through parameter placeholders that are replaced with workflow data at runtime. Message parameters can be raw workflow outputs or transformed using UQL operations (e.g., extract properties, calculate counts, filter values). Attachments are sourced from other workflow blocks that produce string or binary content (e.g., CSV Formatter for reports, image outputs for visualizations). Cooldown prevents notification spam by enforcing minimum time between sends using cache-based tracking with session keys, enabling per-step throttling in distributed or multi-instance environments.

## Requirements

**Slack API Token**: Requires a Slack API token (Bot Token or User Token) with appropriate permissions:
- Token must have `chat:write` scope to send messages to channels
- Token must have `files:write` scope if using attachments
- Token can be provided via workflow inputs (recommended for security) or stored in workflow definitions
- View [Slack API documentation](https://api.slack.com/tutorials/tracks/getting-a-token) or [Roboflow Blog guide](https://blog.roboflow.com/slack-notification-workflows/) for token generation instructions

**Channel Configuration**: Requires a valid Slack channel identifier (channel ID or channel name starting with `#`). The bot or user associated with the token must be a member of the channel.

**Cooldown Session Key**: The `cooldown_session_key` must be unique for each Slack Notification step in your workflow to enable proper per-step cooldown tracking. The cooldown mechanism uses cache-based storage with a 15-minute expiration time, and cooldown seconds must be between 0 and 900 (15 minutes).

## Common Use Cases

- **Team Alert Notifications**: Send Slack alerts to team channels when specific conditions are detected (e.g., alert security team when unauthorized objects detected, notify operations when anomaly detected, send alerts when detection counts exceed thresholds), enabling real-time team collaboration and incident response
- **Workflow Execution Updates**: Send Slack notifications about workflow execution status and results (e.g., notify team when batch processing completes, send daily summary reports, alert about workflow failures), enabling team visibility into automated processes
- **Detection Summaries**: Send Slack messages with detection results and aggregated statistics (e.g., share lists of detected objects, send counts and classifications, include detection confidence summaries), enabling stakeholders to stay informed about workflow outputs via team communication channels
- **Report Distribution**: Upload and share generated reports and data exports via Slack (e.g., attach CSV reports from CSV Formatter, share exported detection data, include formatted analytics summaries), enabling automated data distribution through team channels
- **Real-Time Monitoring**: Send continuous monitoring updates and status notifications (e.g., notify about system health issues, send periodic performance metrics, alert about processing milestones), enabling real-time visibility for operational monitoring
- **Multi-Channel Broadcasting**: Send notifications to different Slack channels based on workflow conditions or routing logic (e.g., send alerts to different channels per detection type, route notifications by severity level, distribute reports to department-specific channels), enabling targeted communication and notification routing

## Connecting to Other Blocks

This block receives data from workflow steps and sends Slack notifications:

- **After detection or analysis blocks** (e.g., Object Detection, Instance Segmentation, Classification) to send alerts or summaries when objects are detected, classifications are made, or thresholds are exceeded, enabling real-time team notifications and collaboration
- **After data processing blocks** (e.g., Expression, Property Definition, Detections Filter) to include computed metrics, transformed data, or filtered results in Slack notifications, enabling customized reporting with processed data in team channels
- **After formatter blocks** (e.g., CSV Formatter) to attach formatted reports and exports to Slack messages, enabling automated distribution of structured data and analytics through team communication channels
- **In conditional workflows** (e.g., Continue If) to send notifications only when specific conditions are met, enabling event-driven alerting and team communication
- **After aggregation blocks** (e.g., Data Aggregator) to send periodic analytics summaries and statistical reports to Slack, enabling scheduled team updates and trend analysis
- **In monitoring workflows** to send status updates, error notifications, or health check reports to team channels, enabling automated system monitoring and incident management through Slack
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
        description="Slack API token (Bot Token or User Token) for authenticating with Slack API. Token must have 'chat:write' scope to send messages and 'files:write' scope if using attachments. Token is marked as private for security. Recommended to provide via workflow inputs using SECRET_KIND selectors rather than storing in workflow definitions. Generate tokens via Slack API apps or workspace administration. See [Slack API documentation](https://api.slack.com/tutorials/tracks/getting-a-token) or [Roboflow Blog guide](https://blog.roboflow.com/slack-notification-workflows/) for setup instructions.",
        private=True,
        examples=["$inputs.slack_token"],
    )
    channel: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Slack channel identifier where the notification will be sent. Can be a channel ID (e.g., 'C1234567890') or channel name starting with '#' (e.g., '#alerts', '#general'). The bot or user associated with the Slack token must be a member of the channel. Channel names are automatically converted to channel IDs by the Slack API.",
        examples=["$inputs.slack_channel_id"],
    )
    message: str = Field(
        description="Message content to send to the Slack channel (plain text). Supports dynamic parameters using placeholder syntax: {{ $parameters.parameter_name }}. Placeholders are replaced with values from message_parameters at runtime. Message can be multi-line text. If attachments are provided, this message becomes the initial comment attached to the file upload. Example: 'Detected {{ $parameters.num_objects }} objects. Classes: {{ $parameters.classes }}.'",
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
        description="Dictionary mapping parameter names (used in message placeholders) to workflow data sources. Keys are parameter names referenced in message as {{ $parameters.key }}, values are selectors to workflow step outputs or direct values. These values are substituted into message placeholders at runtime. Can optionally use message_parameters_operations to transform parameter values before substitution.",
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
        description="Optional dictionary mapping parameter names (from message_parameters) to UQL operation chains that transform parameter values before inserting them into the message. Operations are applied in sequence (e.g., extract class names from detections, calculate counts, filter values). Keys must match parameter names in message_parameters. Leave empty or omit parameters that don't need transformation.",
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
        description="Optional dictionary mapping attachment filenames to workflow step outputs that provide file content. Keys are the attachment filenames (e.g., 'report.csv', 'image.jpg'), values are selectors to blocks that output string or binary content (e.g., CSV Formatter outputs, image data, generated reports). Files are uploaded to Slack using files_upload_v2 API, and the message becomes the initial comment. Leave empty if no attachments are needed. Requires 'files:write' scope on the Slack token.",
        default_factory=dict,
        examples=[{"report.csv": "$steps.csv_formatter.csv_content"}],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Execution mode: True for asynchronous background sending (workflow continues immediately, error_status always False, faster execution), False for synchronous sending (waits for Slack API call completion, returns actual error status for debugging). Set to False during development and debugging to catch Slack API errors. Set to True in production for faster workflow execution when notification delivery timing is not critical.",
        examples=["$inputs.fire_and_forget", False],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Flag to disable Slack notification sending at runtime. When True, the block skips sending notification and returns a disabled message. Useful for conditional notification control via workflow inputs (e.g., allow callers to disable notifications for testing, enable/disable based on configuration). Set via workflow inputs for runtime control.",
        examples=[False, "$inputs.disable_slack_notifications"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Minimum seconds between consecutive Slack notifications to prevent notification spam. Defaults to 5 seconds. Set to 0 to disable cooldown (no throttling). Must be between 0 and 900 (15 minutes). During cooldown period, the block returns throttling_status=True and skips sending. Cooldown is tracked per step using cooldown_session_key with cache-based storage (15-minute expiration). Each Slack Notification step in a workflow should have a unique cooldown_session_key for proper per-step tracking.",
        examples=["$inputs.cooldown_seconds", 3],
        json_schema_extra={
            "always_visible": True,
        },
    )
    cooldown_session_key: str = Field(
        description="Unique identifier for this Slack Notification step's cooldown tracking session. Must be unique for each Slack Notification step in your workflow to enable proper per-step cooldown isolation. Used with the Slack token hash to create a cache key for tracking the last notification timestamp. In distributed or multi-instance environments, this ensures cooldown works correctly per step. Typically auto-generated or provided as a workflow input.",
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
