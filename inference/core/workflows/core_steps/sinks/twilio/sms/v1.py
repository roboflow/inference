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
from twilio.rest import Client

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
TRUNCATION_MARKER = "[...]"

LONG_DESCRIPTION = """
Send SMS text message notifications via Twilio SMS service with customizable message content featuring dynamic workflow data parameters, automatic message truncation for SMS length limits, cooldown throttling using cache-based session tracking, and optional async background execution for mobile alerts, urgent notifications, and real-time communication workflows.

## How This Block Works

This block sends SMS text messages to phone numbers using the Twilio SMS API, integrating workflow execution results into message content. The block:

1. Checks if the sink is disabled via `disable_sink` flag (if disabled, returns immediately without sending)
2. Generates a cache key for cooldown tracking using a hash of Twilio credentials (Account SID and Auth Token) and `cooldown_session_key` (unique per workflow step)
3. Validates cooldown period by checking cache for the last notification timestamp (if enabled, throttles notifications within `cooldown_seconds` of the last sent SMS, returning throttling status)
4. Creates or retrieves a Twilio Client instance for the provided credentials (caches clients by credential hash for efficiency)
5. Formats the message by processing dynamic parameters (replaces placeholders like `{{ $parameters.parameter_name }}` with actual workflow data from `message_parameters`)
6. Applies optional UQL operations to transform parameter values before insertion (e.g., extract class names from detections, calculate metrics, filter data) using `message_parameters_operations`
7. Truncates the message if it exceeds `length_limit` characters (appends truncation marker `[...]` to indicate truncation)
8. Sends the SMS message to the receiver phone number from the sender number using Twilio's `messages.create` API
9. Updates the cache with the current notification timestamp (expires after 15 minutes)
10. Executes synchronously or asynchronously based on `fire_and_forget` setting:
    - **Synchronous mode** (`fire_and_forget=False`): Waits for Twilio API call completion, returns actual error status for debugging
    - **Asynchronous mode** (`fire_and_forget=True`): Sends SMS in background task, workflow continues immediately, error status always False
11. Returns status outputs indicating success, throttling, or errors (includes Twilio API error details when available)

The block supports dynamic message content through parameter placeholders that are replaced with workflow data at runtime. Message parameters can be raw workflow outputs or transformed using UQL operations (e.g., extract properties, calculate counts, filter values). SMS messages are automatically truncated if they exceed the character limit to comply with SMS standards and prevent message delivery issues. Cooldown prevents SMS spam by enforcing minimum time between sends using cache-based tracking with session keys, enabling per-step throttling in distributed or multi-instance environments.

## Requirements

**Twilio Account Configuration**: Requires a Twilio account with SMS capabilities:
- `twilio_account_sid`: Twilio Account SID (found in [Twilio Console](https://twilio.com/console))
- `twilio_auth_token`: Twilio Auth Token (found in Twilio Console, marked as private for security)
- Credentials can be provided via workflow inputs (recommended for security) using SECRET_KIND selectors rather than storing in workflow definitions
- View [Twilio SMS tutorial](https://www.twilio.com/docs/messaging/tutorials/how-to-send-sms-messages/python) for setup instructions

**Phone Number Configuration**: Requires valid phone numbers in E.164 format (e.g., `+1234567890`):
- `sender_number`: Twilio phone number to send from (must be a Twilio-purchased number or verified number in your account)
- `receiver_number`: Destination phone number to receive the SMS

**Cooldown Session Key**: The `cooldown_session_key` must be unique for each Twilio SMS Notification step in your workflow to enable proper per-step cooldown tracking. The cooldown mechanism uses cache-based storage with a 15-minute expiration time, and cooldown seconds must be between 0 and 900 (15 minutes).

**Message Length Limit**: SMS messages have a default length limit of 160 characters. Messages exceeding `length_limit` are automatically truncated with a truncation marker (`[...]`). Adjust `length_limit` based on your needs, but note that SMS standards typically support 160 characters for single-part messages or 153 characters per segment for multi-part messages.

## Common Use Cases

- **Urgent Alert Notifications**: Send SMS alerts to mobile devices when critical conditions are detected (e.g., alert security personnel when unauthorized objects detected, notify operators when system anomalies occur, send alerts when detection counts exceed critical thresholds), enabling immediate mobile notification for time-sensitive incidents
- **Mobile Workflow Updates**: Send SMS notifications about workflow execution status and critical results (e.g., notify operators when batch processing completes, send urgent failure alerts, alert about system health issues), enabling mobile visibility into automated processes
- **Detection Summaries**: Send SMS messages with detection results and key statistics (e.g., share counts of detected objects, send classification summaries, include critical detection alerts), enabling stakeholders to receive important workflow outputs on their mobile devices
- **Emergency Notifications**: Send urgent SMS alerts for emergency situations and critical events (e.g., alert emergency responders, notify maintenance teams about critical issues, send immediate system failure notifications), enabling rapid mobile communication for emergency response
- **Real-Time Mobile Monitoring**: Send continuous SMS updates for real-time monitoring and status notifications (e.g., notify about system health issues, send periodic performance alerts, alert about processing milestones), enabling mobile visibility for operational monitoring
- **On-Call Alerts**: Send SMS notifications to on-call personnel for after-hours or critical issues (e.g., alert on-call engineers about system problems, notify on-call security about detected threats, send after-hours incident notifications), enabling mobile communication for on-call staff

## Connecting to Other Blocks

This block receives data from workflow steps and sends SMS notifications:

- **After detection or analysis blocks** (e.g., Object Detection, Instance Segmentation, Classification) to send urgent mobile alerts or summaries when objects are detected, classifications are made, or thresholds are exceeded, enabling real-time mobile notifications for critical events
- **After data processing blocks** (e.g., Expression, Property Definition, Detections Filter) to include computed metrics, transformed data, or filtered results in SMS notifications, enabling customized mobile reporting with processed data
- **In conditional workflows** (e.g., Continue If) to send SMS notifications only when specific conditions are met, enabling event-driven mobile alerting and urgent notifications
- **After aggregation blocks** (e.g., Data Aggregator) to send periodic analytics summaries and statistical reports via SMS, enabling scheduled mobile updates and trend analysis
- **In monitoring workflows** to send status updates, error notifications, or health check reports to mobile devices, enabling automated system monitoring and incident management via SMS
- **For emergency workflows** where immediate mobile notification is critical (e.g., security alerts, system failures, critical detections), enabling rapid mobile communication for time-sensitive situations
"""

PARAMETER_REGEX = re.compile(r"({{\s*\$parameters\.(\w+)\s*}})")


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Twilio SMS Notification",
            "version": "v1",
            "short_description": "Send notification via Twilio SMS service.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "notifications",
                "icon": "far fa-comment-sms",
            },
        }
    )
    type: Literal["roboflow_core/twilio_sms_notification@v1"]
    twilio_account_sid: Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])] = Field(
        title="Twilio Account SID",
        description="Twilio Account SID for authenticating with Twilio API. Found in the [Twilio Console](https://twilio.com/console) under Account Info. This field is marked as private for security. Recommended to provide via workflow inputs using SECRET_KIND selectors rather than storing in workflow definitions. Used together with twilio_auth_token to authenticate Twilio API requests for sending SMS messages.",
        private=True,
        examples=["$inputs.twilio_account_sid"],
    )
    twilio_auth_token: Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])] = Field(
        title="Twilio Auth Token",
        description="Twilio Auth Token for authenticating with Twilio API. Found in the [Twilio Console](https://twilio.com/console) under Account Info (click 'show' to reveal). This field is marked as private for security. Recommended to provide via workflow inputs using SECRET_KIND selectors rather than storing in workflow definitions. Used together with twilio_account_sid to authenticate Twilio API requests for sending SMS messages.",
        private=True,
        examples=["$inputs.twilio_auth_token"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    sender_number: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Twilio phone number to send SMS messages from. Must be in E.164 format (e.g., '+1234567890') and must be a Twilio-purchased phone number or a verified number in your Twilio account. This number appears as the sender in SMS messages received by recipients. You can purchase numbers in the Twilio Console or use trial numbers for testing.",
        examples=["+1234567890", "$inputs.sender_number"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    receiver_number: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Destination phone number to receive the SMS message. Must be in E.164 format (e.g., '+1234567890'). For Twilio trial accounts, receiver numbers must be verified in your Twilio account. For paid accounts, you can send to any valid phone number. This is the mobile number that will receive the notification.",
        examples=["+1234567890", "$inputs.receiver_number"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    message: str = Field(
        description="SMS message content (plain text). Supports dynamic parameters using placeholder syntax: {{ $parameters.parameter_name }}. Placeholders are replaced with values from message_parameters at runtime. Message can be multi-line text but will be sent as a single SMS. Messages exceeding length_limit will be automatically truncated with a truncation marker. Example: 'Detected {{ $parameters.num_objects }} objects. Alert: {{ $parameters.classes }}.'",
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
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Execution mode: True for asynchronous background sending (workflow continues immediately, error_status always False, faster execution), False for synchronous sending (waits for Twilio API call completion, returns actual error status for debugging). Set to False during development and debugging to catch Twilio API errors. Set to True in production for faster workflow execution when SMS delivery timing is not critical.",
        examples=["$inputs.fire_and_forget", False],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Flag to disable SMS notification sending at runtime. When True, the block skips sending SMS and returns a disabled message. Useful for conditional notification control via workflow inputs (e.g., allow callers to disable notifications for testing, enable/disable based on configuration). Set via workflow inputs for runtime control.",
        examples=[False, "$inputs.disable_sms_notifications"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Minimum seconds between consecutive SMS notifications to prevent SMS spam. Defaults to 5 seconds. Set to 0 to disable cooldown (no throttling). Must be between 0 and 900 (15 minutes). During cooldown period, the block returns throttling_status=True and skips sending. Cooldown is tracked per step using cooldown_session_key with cache-based storage (15-minute expiration). Each Twilio SMS Notification step in a workflow should have a unique cooldown_session_key for proper per-step tracking.",
        examples=["$inputs.cooldown_seconds", 3],
        json_schema_extra={
            "always_visible": True,
        },
    )
    cooldown_session_key: str = Field(
        description="Unique identifier for this Twilio SMS Notification step's cooldown tracking session. Must be unique for each Twilio SMS Notification step in your workflow to enable proper per-step cooldown isolation. Used with the Twilio credentials hash to create a cache key for tracking the last notification timestamp. In distributed or multi-instance environments, this ensures cooldown works correctly per step. Typically auto-generated or provided as a workflow input.",
        examples=["session-1v73kdhfse"],
        json_schema_extra={"hidden": True},
    )
    length_limit: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=160,
        description="Maximum number of characters allowed in the SMS message before truncation. Defaults to 160 characters (standard single-part SMS length). Messages exceeding this limit are automatically truncated, with the last characters replaced by a truncation marker ('[...]'). Must be greater than 0. Note: SMS standards support 160 characters for single-part messages or 153 characters per segment for multi-part messages. Adjust based on your needs, but longer messages may incur additional costs with Twilio.",
        examples=["$inputs.sms_length_limit", 160],
        json_schema_extra={
            "always_visible": True,
        },
    )

    @field_validator("cooldown_seconds")
    @classmethod
    def ensure_cooldown_seconds_within_bounds(cls, value: Any) -> dict:
        if isinstance(value, int) and (value < 0 or value > CACHE_EXPIRE_TIME):
            raise ValueError(
                f"`cooldown_seconds` must be in range [0, {CACHE_EXPIRE_TIME}]"
            )
        return value

    @field_validator("length_limit")
    @classmethod
    def ensure_length_limit_within_bounds(cls, value: Any) -> dict:
        if isinstance(value, int) and value <= 0:
            raise ValueError(f"Length limit for SMS must be greater than 0")
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


class TwilioSMSNotificationBlockV1(WorkflowBlock):

    def __init__(
        self,
        cache: BaseCache,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._cache = cache
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._clients: Dict[str, Client] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["cache", "background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        twilio_account_sid: str,
        twilio_auth_token: str,
        message: str,
        sender_number: str,
        receiver_number: str,
        message_parameters: Dict[str, Any],
        message_parameters_operations: Dict[str, List[AllOperationsType]],
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: int,
        cooldown_session_key: str,
        length_limit: int,
    ) -> BlockResult:
        if disable_sink:
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Sink was disabled by parameter `disable_sink`",
            }
        credentials_hash = _hash_credentials(
            twilio_account_sid=twilio_account_sid,
            twilio_auth_token=twilio_auth_token,
        )
        cache_key = _generate_cache_key_for_cooldown_session(
            session_key=cooldown_session_key,
            token_hash=credentials_hash,
        )
        last_notification_fired = self._cache.get(cache_key)
        seconds_since_last_notification = cooldown_seconds
        if last_notification_fired is not None:
            last_notification_fired = datetime.fromisoformat(last_notification_fired)
            seconds_since_last_notification = (
                datetime.now() - last_notification_fired
            ).total_seconds()
        if seconds_since_last_notification < cooldown_seconds:
            logging.info(
                f"Activated `roboflow_core/twilio_sms_notification@v1` cooldown."
            )
            return {
                "error_status": False,
                "throttling_status": True,
                "message": "Sink cooldown applies",
            }
        if credentials_hash not in self._clients:
            self._clients[credentials_hash] = Client(
                twilio_account_sid,
                twilio_auth_token,
            )
        client = self._clients[credentials_hash]
        message = format_message(
            message=message,
            message_parameters=message_parameters,
            message_parameters_operations=message_parameters_operations,
            length_limit=length_limit,
        )
        send_notification_handler = partial(
            send_sms_notification,
            client=client,
            message=message,
            sender_number=sender_number,
            receiver_number=receiver_number,
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


def _hash_credentials(twilio_account_sid: str, twilio_auth_token: str) -> str:
    sid_hash = hashlib.sha256(twilio_account_sid.encode("utf-8")).hexdigest()
    auth_token_hash = hashlib.sha256(twilio_auth_token.encode("utf-8")).hexdigest()
    return f"{sid_hash}:{auth_token_hash}"


def _generate_cache_key_for_cooldown_session(session_key: str, token_hash: str) -> str:
    return (
        f"workflows:steps_cache:roboflow_core/twilio_sms_notification@v1:"
        f"{token_hash}:{session_key}:last_notification_time"
    )


def format_message(
    message: str,
    message_parameters: Dict[str, Any],
    message_parameters_operations: Dict[str, List[AllOperationsType]],
    length_limit: int,
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
    if len(message) > length_limit:
        truncated_message = message[: length_limit - 1 - len(TRUNCATION_MARKER)]
        message = f"{truncated_message} {TRUNCATION_MARKER}"
    return message


def send_sms_notification(
    client: Client,
    message: str,
    sender_number: str,
    receiver_number: str,
) -> Tuple[bool, str]:
    try:
        client.messages.create(
            body=message,
            from_=sender_number,
            to=receiver_number,
        )
        return False, "Notification sent successfully"
    except Exception as error:
        logging.warning(f"Could not send Twilio SMS notification. Error: {str(error)}")
        return (
            True,
            f"Failed to send Twilio SMS notification. Internal error details: {error}",
        )
