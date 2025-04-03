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
The 📲 **Twilio SMS Notification** ✉️ block enables sending text message notifications via the Twilio SMS service, 
with flexible features such as dynamic content, message truncation, and cooldown management.

The block requires Twilio setup - 
[this article](https://www.twilio.com/docs/messaging/tutorials/how-to-send-sms-messages/python) may help you 
configuring everything properly.

#### ✨ Key Features

* 📢 **Send SMS**: Deliver SMS messages to designated recipients.

* 🔗 **Dynamic Content**: Craft notifications based on outputs from other Workflow steps.

* ✂️ **Message Truncation**: Automatically truncate messages exceeding the character limit.

* 🕒 **Cooldown Control:** Prevent duplicate notifications within a set time frame.

* ⚙️ **Flexible Execution:** Execute in the background or block Workflow execution for debugging.
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
        description="Twilio Account SID. Visit the "
        "[Twilio Console](https://twilio.com/console) "
        "to configure the SMS service and retrieve the value.",
        private=True,
        examples=["$inputs.twilio_account_sid"],
    )
    twilio_auth_token: Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])] = Field(
        title="Twilio Auth Token",
        description="Twilio Auth Token. Visit the "
        "[Twilio Console](https://twilio.com/console) "
        "to configure the SMS service and retrieve the value.",
        private=True,
        examples=["$inputs.twilio_auth_token"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    sender_number: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Sender phone number",
        examples=["+1234567890", "$inputs.sender_number"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    receiver_number: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Receiver phone number",
        examples=["+1234567890", "$inputs.receiver_number"],
        json_schema_extra={
            "hide_description": True,
        },
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
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to run the block asynchronously (True) for faster workflows or synchronously (False) for debugging and error handling.",
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
    length_limit: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=160,
        description="Maximum number of characters in SMS notification (longer messages will be truncated).",
        examples=["$inputs.sms_length_limit", 3],
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
