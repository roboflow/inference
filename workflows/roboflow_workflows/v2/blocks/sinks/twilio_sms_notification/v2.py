"""Twilio SMS/MMS notification block.

Port of V1 ``roboflow_core/twilio_sms_notification@v2``::

    disabled?  ─▶ error_status=False, V1's disabled message
    Twilio fields missing? ─▶ error_status=True (custom Twilio only)
    cooling?   ─▶ throttling_status=True (one window per step, as in V1)
    message_parameters ── *_operations (UQL) ──▶ text (> 1600 chars truncated)
      ├ "Roboflow Managed API Key": JSON ─▶ platform_client.post("apiproxy/twilio")
      └ "Custom Twilio": twilio Client(sid, token).messages.create(...)
    fire_and_forget and sink_effects ─▶ background, else sent now

Text, media and credentials are prepared in the calling thread, so an
operation error or a replayed secret fails the step before anything is sent.

Media: URLs pass through. Images go to the managed service as base64 JPEG.
Twilio itself only fetches media from public URLs, so for custom Twilio a
host-provided ``media_uploader`` publishes each image and returns its URL.

Differences from V1: custom Twilio never sends images to a third-party file
host on its own (V1 uploaded them, and a placeholder image for texts over 160
characters, to a public temporary host); without ``media_uploader`` an image
fails the call, and long texts go as concatenated SMS. An upload failure
fails the call instead of silently dropping the image. Operation keys must
name a ``message_parameters`` key; credentials in failure messages are
replaced by ``<secret NAME>``; managed-provider errors other than 403/429
use the generic message.
"""

import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
)
from roboflow_workflows.v2.blocks.platform import (
    OFFLINE_PLATFORM_CLIENT,
    PlatformClient,
)
from roboflow_workflows.v2.blocks.query import Operation
from roboflow_workflows.v2.blocks.sinks.notification_message import (
    ManagedProviderError,
    ParameterOperations,
    Scalar,
    image_to_base64,
    image_to_jpeg,
    managed_error_handlers,
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
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Revealed,
    Secret,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


if TYPE_CHECKING:
    from twilio.http import HttpClient

BLOCK_TYPE = "roboflow_core/twilio_sms_notification@v2"

MANAGED_PROVIDER = "Roboflow Managed API Key"

TWILIO_PROVIDER = "Custom Twilio"

SMS_PROXY_ENDPOINT = "apiproxy/twilio"

SMS_CHAR_LIMIT = 160

MMS_CHAR_LIMIT = 1600

TRUNCATION_MARKER = "[...]"

THROTTLED_MESSAGE = "Sink cooldown applies"

SENT_MESSAGE = "Notification sent successfully"

FAILED_MESSAGE = "Failed to send Twilio SMS notification. Internal error details: {}"

MISSING_MESSAGE = (
    "Custom Twilio requires twilio_account_sid, twilio_auth_token, and sender_number"
)

NO_UPLOADER_MESSAGE = (
    "Custom Twilio needs public media URLs; this host provides no media_uploader "
    "to publish workflow images. Pass image URLs, or use the Roboflow Managed "
    "API Key provider."
)

PROXY_SENT_MESSAGE = "Notification sent successfully via Roboflow proxy"

PROXY_FAILED_MESSAGE = "Failed to send SMS via proxy. Internal error details: {}"

RATE_LIMIT_MESSAGE = (
    "Failed to send SMS: rate limit exceeded. The workspace has exceeded its SMS "
    "sending limits. Please wait before sending more messages."
)

CREDITS_MESSAGE = (
    "Failed to send SMS: workspace credits exceeded. Please add more credits to "
    "your workspace to continue sending messages."
)

MediaUploader = Callable[[bytes, str], str]

class TwilioSmsNotificationBlock(Block):
    """Send an SMS, or an MMS with media, through Roboflow or your Twilio account.

    Example: ``message="Gate: {{ $parameters.count }} trucks"`` with
    ``message_parameters={"count": "$steps.count.output"}`` texts
    "Gate: 4 trucks" to ``receiver_number``.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "Twilio SMS/MMS Notification",
        "port_of": "roboflow_core/twilio_sms_notification@v2",
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
        sms_provider: Literal["Roboflow Managed API Key", "Custom Twilio"] = Field(
            default=MANAGED_PROVIDER,
            description=(
                "Roboflow's managed service (uses the host's API key) or your "
                "own Twilio account."
            ),
            examples=[MANAGED_PROVIDER, TWILIO_PROVIDER],
        )
        receiver_number: StrictStr | Ref(STRING_KIND) = Field(
            description="Recipient phone number in E.164 format.",
            examples=["+15551234567", "$inputs.receiver_number"],
        )
        message: StrictStr = Field(
            description=(
                "Text. {{ $parameters.NAME }} is replaced by that message "
                "parameter. Texts over 1600 characters are truncated."
            ),
            examples=["Gate: {{ $parameters.count }} trucks"],
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
        media_url: Optional[StrictStr | List[StrictStr]] | Ref(
            STRING_KIND, LIST_OF_VALUES_KIND, IMAGE_KIND
        ) = Field(
            default=None,
            description=(
                "MMS media: public image URL(s) or workflow images. Custom "
                "Twilio needs the host's media_uploader for images."
            ),
            examples=["$steps.visualization.image", "https://example.com/a.jpg"],
        )
        twilio_account_sid: Optional[StrictStr] | Ref(SECRET_KIND, STRING_KIND) = Field(
            default=None,
            description="Twilio Account SID; required for Custom Twilio.",
            examples=["$steps.secrets.twilio_account_sid"],
        )
        twilio_auth_token: Optional[StrictStr] | Ref(SECRET_KIND, STRING_KIND) = Field(
            default=None,
            description=(
                "Twilio Auth Token; required for Custom Twilio. Prefer a "
                "secret from a secrets store or a workflow input of kind "
                "secret."
            ),
            examples=["$steps.secrets.twilio_auth_token"],
        )
        sender_number: Optional[StrictStr] | Ref(STRING_KIND) = Field(
            default=None,
            description="Your Twilio number (E.164); required for Custom Twilio.",
            examples=["+15557654321"],
        )
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Do not wait for sending when the host provides background "
                "effects; otherwise the message is sent now."
            ),
            examples=[True],
        )
        disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Skip sending; outputs say the sink was disabled.",
            examples=[False, "$inputs.disable_sms"],
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
        platform_client: PlatformClient = OFFLINE_PLATFORM_CLIENT,
        api_key: Optional[Union[str, Secret]] = None,
        twilio_http_client: Optional["HttpClient"] = None,
        media_uploader: Optional[MediaUploader] = None,
        sink_effects: Optional[BackgroundEffects] = None,
        disable_sinks: bool = False,
        cooldown_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._platform_client = platform_client
        self._api_key = to_secret(api_key, name="api_key")
        self._twilio_http_client = twilio_http_client
        self._media_uploader = media_uploader
        self._sink_effects = sink_effects
        self._disable_sinks = disable_sinks
        self._cooldown = Cooldown(cooldown_clock)
        self._operations = ParameterOperations(block_type=BLOCK_TYPE)

    def run(
        self,
        *,
        sms_provider: str,
        receiver_number: str,
        message: str,
        message_parameters: Dict[str, Any],
        message_parameters_operations: Dict[str, List[Any]],
        media_url: Union[None, str, ImageData, List[Union[str, ImageData]]],
        twilio_account_sid: Union[None, str, Secret],
        twilio_auth_token: Union[None, str, Secret],
        sender_number: Optional[str],
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: int,
    ) -> Dict[str, Any]:
        """Send one message unless disabled, misconfigured or cooling down.

        Args:
            sms_provider: Managed service or custom Twilio.
            receiver_number: Recipient number.
            message: Text template.
            message_parameters: Placeholder values.
            message_parameters_operations: Operations per parameter.
            media_url: MMS media URLs or images.
            twilio_account_sid: Twilio account SID.
            twilio_auth_token: Twilio auth token.
            sender_number: Twilio sender number.
            fire_and_forget: Hand sending to background effects if any.
            disable_sink: Skip this message.
            cooldown_seconds: Least seconds between sent messages.

        Returns:
            ``error_status``, ``throttling_status`` and ``message``.

        Raises:
            QueryRuntimeError: When an operation cannot process a value.
            SecretUnavailableError: When a credential was read back from a
                recording; nothing is sent.
        """
        if self._disable_sinks or disable_sink:
            disabled = disabled_sink_message(by_policy=self._disable_sinks)
            return _result(False, disabled)

        use_twilio = sms_provider == TWILIO_PROVIDER
        if use_twilio and not (
            twilio_account_sid and twilio_auth_token and sender_number
        ):
            return _result(True, MISSING_MESSAGE)

        media = _media_items(media_url)
        has_images = any(isinstance(item, ImageData) for item in media)
        if use_twilio and has_images and self._media_uploader is None:
            return _result(True, NO_UPLOADER_MESSAGE)

        if self._cooldown.cooling(cooldown_seconds):
            return _result(False, THROTTLED_MESSAGE, throttled=True)

        names = referenced_parameters(message, message_parameters)
        values = self._operations.apply(
            message_parameters, message_parameters_operations, names=names
        )
        text, needs_mms = sms_text(
            render_message(message, {name: str(values[name]) for name in names})
        )
        if use_twilio:
            effect = self._twilio_effect(
                text=text,
                sender=sender_number,
                receiver=receiver_number,
                media=media,
                credentials=reveal_secrets(
                    (
                        to_secret(twilio_account_sid, name="twilio_account_sid"),
                        to_secret(twilio_auth_token, name="twilio_auth_token"),
                    )
                ),
            )
        else:
            payload = managed_sms_payload(
                receiver=receiver_number, text=text, needs_mms=needs_mms, media=media
            )
            effect = self._managed_effect(payload)

        # Another call may have started a window while this one prepared.
        if not self._cooldown.try_start(cooldown_seconds):
            return _result(False, THROTTLED_MESSAGE, throttled=True)

        error_status, outcome = deliver(
            effect,
            fire_and_forget=fire_and_forget,
            sink_effects=self._sink_effects,
            label=BLOCK_TYPE,
        )
        result = _result(error_status, outcome)

        return result

    def _twilio_effect(
        self,
        *,
        text: str,
        sender: str,
        receiver: str,
        media: List[Union[str, ImageData]],
        credentials: Revealed,
    ) -> Callable[[], EffectOutcome]:
        # Encode here (host boundary); upload and send in the effect.
        prepared = [
            item if isinstance(item, str) else image_to_jpeg(item) for item in media
        ]
        uploader = self._media_uploader
        http_client = self._twilio_http_client

        def send() -> EffectOutcome:
            from twilio.rest import Client

            account_sid, auth_token = credentials.value
            try:
                media_urls = [
                    item if isinstance(item, str) else uploader(item, "image.jpg")
                    for item in prepared
                ]
                parameters: Dict[str, Any] = {
                    "body": text,
                    "from_": sender,
                    "to": receiver,
                }
                if media_urls:
                    parameters["media_url"] = media_urls
                client = Client(account_sid, auth_token, http_client=http_client)
                client.messages.create(**parameters)
            except Exception as error:
                outcome = (True, FAILED_MESSAGE.format(credentials.scrub(str(error))))
                return outcome

            return False, SENT_MESSAGE

        return send

    def _managed_effect(self, payload: Dict[str, Any]) -> Callable[[], EffectOutcome]:
        revealed = reveal_secrets(self._api_key)
        client = self._platform_client

        def send() -> EffectOutcome:
            try:
                client.post(
                    SMS_PROXY_ENDPOINT,
                    api_key=revealed.value,
                    payload=payload,
                    http_errors_handlers=managed_error_handlers((403, 429)),
                )
            except ManagedProviderError as error:
                outcome = (True, revealed.scrub(managed_sms_failure(error)))
                return outcome
            except Exception as error:
                outcome = (
                    True,
                    PROXY_FAILED_MESSAGE.format(revealed.scrub(str(error))),
                )
                return outcome

            return False, PROXY_SENT_MESSAGE

        return send

def sms_text(text: str) -> Tuple[str, bool]:
    """Apply V1's length rules to a rendered message.

    Args:
        text: Rendered message.

    Returns:
        The text, truncated with ``" [...]"`` to 1600 characters, and whether
        it is longer than one 160-character SMS.
    """
    needs_mms = len(text) > SMS_CHAR_LIMIT
    if len(text) > MMS_CHAR_LIMIT:
        kept = text[: MMS_CHAR_LIMIT - 1 - len(TRUNCATION_MARKER)]
        text = f"{kept} {TRUNCATION_MARKER}"

    return text, needs_mms

def managed_sms_payload(
    *,
    receiver: str,
    text: str,
    needs_mms: bool,
    media: List[Union[str, ImageData]],
) -> Dict[str, Any]:
    """Build the ``apiproxy/twilio`` request body, as V1 does.

    Args:
        receiver: Recipient number.
        text: Rendered, length-limited text.
        needs_mms: Whether the text exceeds one SMS.
        media: URLs and images.

    Returns:
        The body: URLs in ``media_urls``, images as base64 JPEG in
        ``media_base64``, and ``force_mms`` for a long text without media.
    """
    payload: Dict[str, Any] = {"receiver_number": receiver, "message": text}
    urls = [item for item in media if isinstance(item, str)]
    images = [
        {"base64": image_to_base64(item), "mimeType": "image/jpeg"}
        for item in media
        if isinstance(item, ImageData)
    ]
    if urls:
        payload["media_urls"] = urls
    if images:
        payload["media_base64"] = images
    if needs_mms and not media:
        payload["force_mms"] = True

    return payload

def managed_sms_failure(error: ManagedProviderError) -> str:
    """Return V1's message for a rejected managed SMS.

    Args:
        error: The API rejection.

    Returns:
        The outcome message.
    """
    detail = str(error)
    lowered = detail.lower()
    if error.status == 403:
        return f"Failed to send SMS: access forbidden. {detail}"
    if "rate limit" in lowered:
        return RATE_LIMIT_MESSAGE
    if "credits exceeded" in lowered:
        return CREDITS_MESSAGE

    message = f"Failed to send SMS via proxy. {detail}"

    return message

def _media_items(media_url: Any) -> List[Union[str, ImageData]]:
    if media_url is None:
        return []
    if isinstance(media_url, (str, ImageData)):
        return [media_url]

    items = [item for item in media_url if isinstance(item, (str, ImageData))]

    return items

def _result(
    error_status: bool, message: str, *, throttled: bool = False
) -> Dict[str, Any]:
    result = {
        "error_status": error_status,
        "throttling_status": throttled,
        "message": message,
    }

    return result
