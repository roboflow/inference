"""Slack notification block: post a message, optionally with files, to a channel.

Port of V1 ``roboflow_core/slack_notification@v1``::

    disabled?  ─▶ error_status=False, V1's disabled message
    cooling?   ─▶ throttling_status=True (one window per step)
    message_parameters ── *_operations (UQL) ──▶ rendered text
    no attachments ─▶ chat.postMessage        attachments ─▶ files_upload_v2
    fire_and_forget and sink_effects ─▶ background, else sent now

The text is rendered and the token revealed in the calling thread, so an
operation error or a replayed secret fails the step before anything is sent.
The ``slack_api_url`` resource is the Web API base URL that ``slack_sdk``
calls; tests point it at a loopback server.

Differences from V1: the cooldown is one window per step, as in the other
notification blocks, instead of a cache entry per token and
``cooldown_session_key`` (that hidden parameter is gone, and so is its
15-minute limit on ``cooldown_seconds``); uploaded files keep their names
(V1 named them all "Uploaded file"); operation keys must name a
``message_parameters`` key; the token in failure messages is replaced by
``<secret NAME>``.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Union
from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
)
from roboflow_workflows.v2.blocks.query import Operation
from roboflow_workflows.v2.blocks.sinks.notification_message import (
    ParameterOperations,
    Scalar,
    referenced_parameters,
    render_message,
    validate_parameter_operations,
)
from roboflow_workflows.v2.blocks.sinks.runtime import (
    ERROR_STATUS_OUTPUT,
    MESSAGE_OUTPUT,
    BackgroundEffects,
    Cooldown,
    EffectOutcome,
    deliver,
    disabled_sink_message,
)
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Revealed,
    Secret,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


BLOCK_TYPE = "roboflow_core/slack_notification@v1"

SLACK_API_URL = "https://slack.com/api/"

THROTTLED_MESSAGE = "Sink cooldown applies"

SENT_MESSAGE = "Notification sent successfully"

FAILED_MESSAGE = "Failed to send Slack notification. Internal error details: {}"

class SlackNotificationBlock(Block):
    """Post a message to a Slack channel; attachments are uploaded as files.

    Example: ``message="{{ $parameters.count }} people at the door"`` with
    ``message_parameters={"count": "$steps.count.output"}`` posts
    "2 people at the door". With ``attachments`` the message becomes the
    files' comment.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "Slack Notification",
        "port_of": "roboflow_core/slack_notification@v1",
        "section": "notifications",
    }
    outputs = {
        "error_status": ERROR_STATUS_OUTPUT,
        "throttling_status": Output(
            BOOLEAN_KIND, description="True when the cooldown skipped this message."
        ),
        "message": MESSAGE_OUTPUT,
    }

    class Params(BlockParams):
        slack_token: StrictStr | Ref(SECRET_KIND, STRING_KIND) = Field(
            description=(
                "Bot or user token with 'chat:write' ('files:write' for "
                "attachments). Prefer a secret from a secrets store or a "
                "workflow input of kind secret."
            ),
            examples=["$steps.secrets.slack_token"],
        )
        channel: StrictStr | Ref(STRING_KIND) = Field(
            description=(
                "Channel the token's app is in: an ID such as C1234567890, or a "
                "name such as #alerts for messages without attachments."
            ),
            examples=["C1234567890", "$inputs.slack_channel"],
        )
        message: StrictStr = Field(
            description=(
                "Plain text. {{ $parameters.NAME }} is replaced by that message "
                "parameter."
            ),
            examples=["{{ $parameters.count }} people at the door"],
        )
        message_parameters: Dict[str, Scalar | Ref(allow_none=True)] = (  # noqa: F821
            Field(
                default_factory=dict,
                description="Message placeholder values: literals or selectors.",
                examples=[{"count": "$steps.count.output"}],
            )
        )
        message_parameters_operations: Dict[str, List[Operation]] = Field(
            default_factory=dict,
            description="UQL operations per message parameter, applied first.",
            examples=[
                {
                    "predictions": [
                        {
                            "type": "DetectionsPropertyExtract",
                            "property_name": "class_name",
                        }
                    ]
                }
            ],
        )
        attachments: Dict[str, Ref(STRING_KIND)] = Field(  # noqa: F821
            default_factory=dict,
            description=(
                "Files by name, e.g. a CSV report; text or bytes. The message "
                "becomes their comment."
            ),
            examples=[{"report.csv": "$steps.csv_formatter.csv_content"}],
        )
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Do not wait for Slack when the host provides background "
                "effects; otherwise the message is sent now."
            ),
            examples=[True],
        )
        disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Skip sending; outputs say the sink was disabled.",
            examples=[False, "$inputs.disable_slack"],
        )
        cooldown_seconds: StrictInt | Ref(INTEGER_KIND) = Field(
            default=5,
            ge=0,
            description=(
                "Least seconds between two sent messages of this step, over all "
                "inputs. Calls inside the window are throttled."
            ),
            examples=[5],
        )

        # Not a model validator: the engine reruns those per invocation.
        @field_validator("message_parameters_operations")
        @classmethod
        def _operations_compile(
            cls, value: Dict[str, List[Operation]], info: ValidationInfo
        ) -> Dict[str, List[Operation]]:
            checked = validate_parameter_operations(value, info, block_type=BLOCK_TYPE)

            return checked

    def __init__(
        self,
        *,
        slack_api_url: str = SLACK_API_URL,
        sink_effects: Optional[BackgroundEffects] = None,
        disable_sinks: bool = False,
        cooldown_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._slack_api_url = slack_api_url
        self._sink_effects = sink_effects
        self._disable_sinks = disable_sinks
        self._cooldown = Cooldown(cooldown_clock)
        self._operations = ParameterOperations(block_type=BLOCK_TYPE)

    def run(
        self,
        *,
        slack_token: Union[str, Secret],
        channel: str,
        message: str,
        message_parameters: Dict[str, Any],
        message_parameters_operations: Dict[str, List[Any]],
        attachments: Dict[str, Union[str, bytes]],
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: int,
    ) -> Dict[str, Any]:
        """Post one message unless disabled or cooling down.

        Args:
            slack_token: Slack token.
            channel: Channel ID.
            message: Text template.
            message_parameters: Placeholder values.
            message_parameters_operations: Operations per parameter.
            attachments: Files by name.
            fire_and_forget: Hand sending to background effects if any.
            disable_sink: Skip this message.
            cooldown_seconds: Least seconds between sent messages.

        Returns:
            ``error_status``, ``throttling_status`` and ``message``.

        Raises:
            QueryRuntimeError: When an operation cannot process a value.
            SecretUnavailableError: When the token was read back from a
                recording; nothing is sent.
        """
        if self._disable_sinks or disable_sink:
            disabled = disabled_sink_message(by_policy=self._disable_sinks)
            return _result(False, disabled)

        if self._cooldown.cooling(cooldown_seconds):
            return _result(False, THROTTLED_MESSAGE, throttled=True)

        names = referenced_parameters(message, message_parameters)
        values = self._operations.apply(
            message_parameters, message_parameters_operations, names=names
        )
        text = render_message(message, {name: str(values[name]) for name in names})
        revealed = reveal_secrets(to_secret(slack_token, name="slack_token"))
        files = [
            {"title": name, "filename": name, "content": content}
            for name, content in attachments.items()
        ]
        api_url = self._slack_api_url

        def send() -> EffectOutcome:
            outcome = send_slack_message(
                revealed,
                api_url=api_url,
                channel=channel,
                text=text,
                files=files,
            )

            return outcome

        # Another call may have started a window while this one prepared.
        if not self._cooldown.try_start(cooldown_seconds):
            return _result(False, THROTTLED_MESSAGE, throttled=True)

        error_status, outcome = deliver(
            send,
            fire_and_forget=fire_and_forget,
            sink_effects=self._sink_effects,
            label=BLOCK_TYPE,
        )
        result = _result(error_status, outcome)

        return result

def send_slack_message(
    token: Revealed,
    *,
    api_url: str,
    channel: str,
    text: str,
    files: List[Dict[str, Any]],
) -> EffectOutcome:
    """Send with ``slack_sdk`` and return V1's outcome.

    Args:
        token: The revealed token, used to scrub failure messages.
        api_url: Slack Web API base URL.
        channel: Channel ID.
        text: Rendered message.
        files: ``files_upload_v2`` items; empty for a plain message.

    Returns:
        ``(False, SENT_MESSAGE)`` or ``(True, FAILED_MESSAGE)`` with Slack's
        error code or the exception text.
    """
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=token.value, base_url=api_url)
    try:
        if files:
            client.files_upload_v2(
                channel=channel, initial_comment=text, file_uploads=files
            )
        else:
            client.chat_postMessage(channel=channel, text=text)
    except SlackApiError as error:
        details = error.response.get("error", "Not Available.")
        outcome = (True, FAILED_MESSAGE.format(token.scrub(str(details))))
        return outcome
    except Exception as error:
        outcome = (True, FAILED_MESSAGE.format(token.scrub(str(error))))
        return outcome

    return False, SENT_MESSAGE

def _result(
    error_status: bool, message: str, *, throttled: bool = False
) -> Dict[str, Any]:
    result = {
        "error_status": error_status,
        "throttling_status": throttled,
        "message": message,
    }

    return result
