"""E-mail notification block.

Port of V1 ``roboflow_core/email_notification@v2``::

    disabled?  ─▶ error_status=False, V1's disabled message
    SMTP fields missing? ─▶ error_status=True (custom SMTP only)
    cooling?   ─▶ throttling_status=True (one window per step, as in V1)
    message_parameters ── *_operations (UQL) ──▶ values (an error starts no window)
      ├ "Roboflow Managed API Key": JSON ─▶ platform_client.post("apiproxy/email")
      └ "Custom SMTP": HTML body, inline images, attachments ─▶ smtp_connect(...)
    fire_and_forget and sink_effects ─▶ background, else sent now

Everything is prepared in the calling thread: images become JPEG bytes there,
operation errors fail the step and a replayed secret raises
``SecretUnavailableError`` before anything is sent. The background effect
holds only detached bytes and strings.

Resources: ``platform_client`` and ``api_key`` for the managed provider
(offline by default, so nothing leaves without the host choosing a client),
and ``smtp_connect`` for custom SMTP (implicit TLS with system trust by
default, as in V1).

Differences from V1: CC and BCC addresses are delivered (V1 passed only the
To list to ``sendmail``) and no ``Bcc`` header is written (V1 showed it to
every recipient); attachment file names are quoted; the SMTP connection has
a timeout; operation keys must name a ``message_parameters`` key; a
placeholder value is never itself searched for placeholders; managed-provider
errors other than 403/413/429 use the generic message because V2 does not
import the host's exception classes; credentials in failure messages are
replaced by ``<secret NAME>``.
"""

import base64
import html
import json
import smtplib
import ssl
import time
from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

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
    as_list,
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
    Secret,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

BLOCK_TYPE = "v2/email_notification@v2"
MANAGED_PROVIDER = "Roboflow Managed API Key"
SMTP_PROVIDER = "Custom SMTP"
EMAIL_PROXY_ENDPOINT = "apiproxy/email"
SMTP_TIMEOUT_SECONDS = 30.0

THROTTLED_MESSAGE = "Sink cooldown applies"
SMTP_SENT_MESSAGE = "Notification sent successfully"
SMTP_FAILED_MESSAGE = "Failed to send e-mail. Internal error details: {}"
SMTP_MISSING_MESSAGE = (
    "Custom SMTP requires sender_email, smtp_server, and sender_email_password"
)
PROXY_SENT_MESSAGE = "Notification sent successfully via Roboflow proxy"
PROXY_FAILED_MESSAGE = "Failed to send e-mail via proxy. Internal error details: {}"
NON_MEMBER_MESSAGE = (
    "To prevent spam, you can only send emails to members of your Roboflow "
    "Workspace via the Roboflow Managed API Key. Add this email to your Workspace "
    "or switch to sending via your own SMTP server."
)
TOO_LARGE_MESSAGE = (
    "Failed to send email: attachment size exceeds the 5MB limit. For image "
    "attachments, use the Image Preprocessing block to resize images before "
    "sending. For other attachments (like CSV files), reduce the file size or "
    "send smaller data."
)
RATE_LIMIT_MESSAGE = (
    "Failed to send email: rate limit exceeded. The workspace has exceeded its "
    "email sending limits. Please wait before sending more emails or contact "
    "support to increase your limits."
)
CREDITS_MESSAGE = (
    "Failed to send email: workspace credits exceeded. Please add more credits "
    "to your workspace to continue sending emails."
)

Recipients = Union[StrictStr, List[StrictStr]]
SmtpConnect = Callable[[str, int], smtplib.SMTP]
"""Open an authenticated-ready SMTP connection to ``(host, port)``."""


# --- The e-mail block --------------------------------------------------------


class SmtpSslConnector:
    """Default ``smtp_connect``: implicit TLS (SMTPS), as V1 enforces.

    Args:
        context: TLS context; ``None`` uses the system trust store. Pass one
            that trusts a private CA only for a private or test server.
        timeout: Seconds for each socket operation.
    """

    def __init__(
        self,
        *,
        context: Optional[ssl.SSLContext] = None,
        timeout: float = SMTP_TIMEOUT_SECONDS,
    ) -> None:
        self._context = context
        self._timeout = timeout

    def __call__(self, host: str, port: int) -> smtplib.SMTP:
        """Connect to ``host:port`` over TLS.

        Args:
            host: SMTP server name.
            port: SMTP server port.

        Returns:
            The connected client; the caller closes it.
        """
        context = self._context or ssl.create_default_context()
        connection = smtplib.SMTP_SSL(
            host, port, context=context, timeout=self._timeout
        )

        return connection

    def __repr__(self) -> str:
        return f"SmtpSslConnector(timeout={self._timeout})"


DEFAULT_SMTP_CONNECTOR = SmtpSslConnector()
"""Shared stateless default of every ``smtp_connect`` resource."""


class EmailNotificationBlock(Block):
    """Send an e-mail through Roboflow's managed service or your SMTP server.

    Example: ``message="Detected {{ $parameters.count }} cars"`` with
    ``message_parameters={"count": "$steps.count.output"}`` sends
    "Detected 3 cars". ``message_parameters_operations`` can first turn
    predictions into plain values, e.g. class names.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "Email Notification",
        "port_of": "roboflow_core/email_notification@v2",
        "section": "notifications",
    }
    outputs = {
        "error_status": ERROR_STATUS_OUTPUT,
        "throttling_status": Output(
            BOOLEAN_KIND, description="True when the cooldown skipped this e-mail."
        ),
        "message": MESSAGE_OUTPUT,
    }

    class Params(BlockParams):
        email_provider: Literal["Roboflow Managed API Key", "Custom SMTP"] = Field(
            default=MANAGED_PROVIDER,
            description=(
                "Roboflow's managed service (uses the host's API key) or your "
                "own SMTP server."
            ),
            examples=[MANAGED_PROVIDER, SMTP_PROVIDER],
        )
        subject: StrictStr = Field(
            description="Subject line.", examples=["Workflow alert"]
        )
        receiver_email: Recipients | Ref(STRING_KIND, LIST_OF_VALUES_KIND) = Field(
            description="Recipient address or list of addresses.",
            examples=["receiver@example.com", "$inputs.receivers"],
        )
        message: StrictStr = Field(
            description=(
                "Body text. {{ $parameters.NAME }} is replaced by that message "
                "parameter."
            ),
            examples=["Detected {{ $parameters.count }} cars"],
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
        sender_email: Optional[StrictStr] | Ref(STRING_KIND) = Field(
            default=None,
            description="Sender address; required for Custom SMTP.",
            examples=["sender@example.com"],
        )
        smtp_server: Optional[StrictStr] | Ref(STRING_KIND) = Field(
            default=None,
            description="SMTP server host; required for Custom SMTP.",
            examples=["smtp.example.com", "$inputs.smtp_server"],
        )
        sender_email_password: Optional[StrictStr] | Ref(SECRET_KIND, STRING_KIND) = (
            Field(
                default=None,
                description=(
                    "SMTP password; required for Custom SMTP. Prefer a secret from "
                    "a secrets store or a workflow input of kind secret."
                ),
                examples=["$steps.secrets.smtp_password"],
            )
        )
        cc_receiver_email: Optional[Recipients] | Ref(
            STRING_KIND, LIST_OF_VALUES_KIND
        ) = Field(
            default=None,
            description="CC address or list of addresses.",
            examples=["cc@example.com"],
        )
        bcc_receiver_email: Optional[Recipients] | Ref(
            STRING_KIND, LIST_OF_VALUES_KIND
        ) = Field(
            default=None,
            description="BCC address or list of addresses; hidden from recipients.",
            examples=["bcc@example.com"],
        )
        smtp_port: StrictInt = Field(
            default=465,
            ge=1,
            le=65535,
            description="SMTP server port; the connection uses implicit TLS.",
            examples=[465],
        )
        attachments: Dict[str, Ref(STRING_KIND, IMAGE_KIND)] = Field(  # noqa: F821
            default_factory=dict,
            description=(
                "Files by name: text (e.g. CSV), bytes or images. Images are "
                "sent as JPEG; '.jpg' is appended to names without it."
            ),
            examples=[{"report.csv": "$steps.csv_formatter.csv_content"}],
        )
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Do not wait for sending when the host provides background "
                "effects; otherwise the e-mail is sent now."
            ),
            examples=[True],
        )
        disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Skip sending; outputs say the sink was disabled.",
            examples=[False, "$inputs.disable_email"],
        )
        cooldown_seconds: StrictInt | Ref(INTEGER_KIND) = Field(
            default=5,
            ge=0,
            description=(
                "Least seconds between two sent e-mails of this step, over all "
                "inputs. Calls inside the window are throttled."
            ),
            examples=[5],
        )

        @field_validator("receiver_email")
        @classmethod
        def _receivers_not_empty(cls, value: Any) -> Any:
            if isinstance(value, list) and not value:
                raise ValueError(
                    "E-mail notification must have at least one receiver defined."
                )

            return value

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
        smtp_connect: SmtpConnect = DEFAULT_SMTP_CONNECTOR,
        sink_effects: Optional[BackgroundEffects] = None,
        disable_sinks: bool = False,
        cooldown_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._platform_client = platform_client
        self._api_key = to_secret(api_key, name="api_key")
        self._smtp_connect = smtp_connect
        self._sink_effects = sink_effects
        self._disable_sinks = disable_sinks
        self._cooldown = Cooldown(cooldown_clock)
        self._operations = ParameterOperations(block_type=BLOCK_TYPE)

    def run(
        self,
        *,
        email_provider: str,
        subject: str,
        receiver_email: Union[str, List[str]],
        message: str,
        message_parameters: Dict[str, Any],
        message_parameters_operations: Dict[str, List[Any]],
        sender_email: Optional[str],
        smtp_server: Optional[str],
        sender_email_password: Union[None, str, Secret],
        cc_receiver_email: Union[None, str, List[str]],
        bcc_receiver_email: Union[None, str, List[str]],
        smtp_port: int,
        attachments: Dict[str, Any],
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: int,
    ) -> Dict[str, Any]:
        """Send one e-mail unless disabled, misconfigured or cooling down.

        Args:
            email_provider: Managed service or custom SMTP.
            subject: Subject line.
            receiver_email: To address(es).
            message: Body template.
            message_parameters: Placeholder values.
            message_parameters_operations: Operations per parameter.
            sender_email: SMTP sender.
            smtp_server: SMTP host.
            sender_email_password: SMTP password.
            cc_receiver_email: CC address(es).
            bcc_receiver_email: BCC address(es).
            smtp_port: SMTP port.
            attachments: Files by name.
            fire_and_forget: Hand sending to background effects if any.
            disable_sink: Skip this e-mail.
            cooldown_seconds: Least seconds between sent e-mails.

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

        use_smtp = email_provider == SMTP_PROVIDER
        if use_smtp and not (sender_email and smtp_server and sender_email_password):
            return _result(True, SMTP_MISSING_MESSAGE)

        receivers = as_list(receiver_email)
        if not receivers:
            return _result(True, "receiver_email has no address")

        if self._cooldown.cooling(cooldown_seconds):
            return _result(False, THROTTLED_MESSAGE, throttled=True)

        cc = as_list(cc_receiver_email)
        bcc = as_list(bcc_receiver_email)
        if use_smtp:
            password = to_secret(sender_email_password, name="sender_email_password")
            effect = self._prepare_smtp(
                sender=sender_email,
                receivers=receivers,
                cc=cc,
                bcc=bcc,
                subject=subject,
                message=message,
                parameters=message_parameters,
                operations=message_parameters_operations,
                attachments=attachments,
                host=smtp_server,
                port=smtp_port,
                password=password,
            )
        else:
            values = self._operations.apply(
                message_parameters,
                message_parameters_operations,
                names=list(message_parameters),
            )
            try:
                payload = managed_email_payload(
                    receivers=receivers,
                    cc=cc,
                    bcc=bcc,
                    subject=subject,
                    message=message,
                    values=values,
                    attachments=attachments,
                )
            except (TypeError, ValueError) as error:
                return _result(True, PROXY_FAILED_MESSAGE.format(error))
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

    def _prepare_smtp(
        self,
        *,
        sender: str,
        receivers: List[str],
        cc: Optional[List[str]],
        bcc: Optional[List[str]],
        subject: str,
        message: str,
        parameters: Dict[str, Any],
        operations: Dict[str, List[Any]],
        attachments: Dict[str, Any],
        host: str,
        port: int,
        password: Secret,
    ) -> Callable[[], EffectOutcome]:
        names = referenced_parameters(message, parameters)
        values = self._operations.apply(parameters, operations, names=names)
        body, inline_images = render_html(message, values)
        files = attachment_bytes(attachments)
        revealed = reveal_secrets(password)
        mime = build_mime_message(
            sender=sender,
            receivers=receivers,
            cc=cc,
            subject=subject,
            html_body=body,
            inline_images=inline_images,
            attachments=files,
        )
        envelope = receivers + (cc or []) + (bcc or [])
        connect = self._smtp_connect

        def send() -> EffectOutcome:
            try:
                with connect(host, port) as server:
                    server.login(sender, revealed.value)
                    server.sendmail(sender, envelope, mime)
            except Exception as error:
                outcome = (True, SMTP_FAILED_MESSAGE.format(revealed.scrub(str(error))))
                return outcome

            return False, SMTP_SENT_MESSAGE

        return send

    def _managed_effect(self, payload: Dict[str, Any]) -> Callable[[], EffectOutcome]:
        revealed = reveal_secrets(self._api_key)
        client = self._platform_client

        def send() -> EffectOutcome:
            try:
                client.post(
                    EMAIL_PROXY_ENDPOINT,
                    api_key=revealed.value,
                    payload=payload,
                    http_errors_handlers=managed_error_handlers((403, 413, 429)),
                )
            except ManagedProviderError as error:
                outcome = (True, revealed.scrub(managed_email_failure(error)))
                return outcome
            except Exception as error:
                outcome = (
                    True,
                    PROXY_FAILED_MESSAGE.format(revealed.scrub(str(error))),
                )
                return outcome

            return False, PROXY_SENT_MESSAGE

        return send


def managed_email_payload(
    *,
    receivers: List[str],
    cc: Optional[List[str]],
    bcc: Optional[List[str]],
    subject: str,
    message: str,
    values: Dict[str, Any],
    attachments: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the ``apiproxy/email`` request body; the service renders the message.

    Args:
        receivers: To addresses.
        cc: CC addresses.
        bcc: BCC addresses.
        subject: Subject line.
        message: Body template, sent unrendered as in V1.
        values: Every message parameter after operations.
        attachments: Files by name.

    Returns:
        A detached JSON body; images are base64 JPEG text.

    Raises:
        TypeError: When a parameter value is not JSON data.
    """
    payload: Dict[str, Any] = {
        "receiver_email": receivers,
        "subject": subject,
        "message": message,
        "message_parameters": images_as_base64(values),
    }
    if cc:
        payload["cc_receiver_email"] = cc
    if bcc:
        payload["bcc_receiver_email"] = bcc
    if attachments:
        payload["attachments"] = {
            name: base64.b64encode(content).decode("ascii")
            for name, content in attachment_bytes(attachments).items()
        }
    detached = detached_json(payload)

    return detached


def render_html(message: str, values: Dict[str, Any]) -> Tuple[str, Dict[str, bytes]]:
    """Render the SMTP body as HTML, as V1 does for both providers' look.

    Images become ``<img src="cid:image_NAME">`` with the JPEG returned for
    inline attachment; other values are HTML-escaped text. Newlines become
    ``<br>``. The template text itself is not escaped, so it may hold HTML.

    Args:
        message: Body template.
        values: Parameter values by name, after operations.

    Returns:
        The HTML body and inline JPEG images by content ID.
    """
    texts = {}
    inline_images = {}
    for name, value in values.items():
        if isinstance(value, ImageData):
            content_id = f"image_{name}"
            inline_images[content_id] = image_to_jpeg(value)
            texts[name] = (
                f'<img src="cid:{content_id}" alt="{name}" '
                'style="max-width: 600px; height: auto;">'
            )
        else:
            texts[name] = html.escape(str(value))
    body = render_message(message, texts).replace("\n", "<br>\n")

    return body, inline_images


def attachment_bytes(attachments: Dict[str, Any]) -> Dict[str, bytes]:
    """Turn attachment values into file bytes, as V1 does.

    Args:
        attachments: Values by file name: ``ImageData``, ``bytes``, ``str``
            or anything else (sent as its ``str``).

    Returns:
        Bytes by file name; image names get ``.jpg`` unless they end with
        ``.jpg`` or ``.jpeg``.
    """
    files = {}
    for name, value in attachments.items():
        if isinstance(value, ImageData):
            if not name.lower().endswith((".jpg", ".jpeg")):
                name = f"{name}.jpg"
            files[name] = image_to_jpeg(value)
        elif isinstance(value, bytes):
            files[name] = value
        else:
            files[name] = str(value).encode("utf-8")

    return files


def images_as_base64(value: Any) -> Any:
    """Replace images, also inside dicts and lists, by base64 JPEG text.

    Args:
        value: A parameter value.

    Returns:
        The value with every ``ImageData`` encoded; containers are rebuilt.
    """
    if isinstance(value, ImageData):
        return image_to_base64(value)
    if isinstance(value, dict):
        return {key: images_as_base64(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [images_as_base64(item) for item in value]

    return value


def detached_json(value: Any) -> Any:
    """Return a deep JSON copy of ``value``, sharing nothing with the workflow.

    Args:
        value: JSON data.

    Returns:
        The copy.

    Raises:
        TypeError: When a value is not JSON data, naming its type.
    """
    copy = json.loads(json.dumps(value))

    return copy


def build_mime_message(
    *,
    sender: str,
    receivers: List[str],
    cc: Optional[List[str]],
    subject: str,
    html_body: str,
    inline_images: Dict[str, bytes],
    attachments: Dict[str, bytes],
) -> str:
    """Build V1's ``multipart/related`` HTML message, without a ``Bcc`` header.

    Args:
        sender: From address.
        receivers: To addresses.
        cc: CC addresses.
        subject: Subject line.
        html_body: HTML body.
        inline_images: JPEG bytes by content ID.
        attachments: File bytes by name.

    Returns:
        The message as text for ``sendmail``.
    """
    mime = MIMEMultipart("related")
    mime["From"] = sender
    mime["To"] = ", ".join(receivers)
    if cc:
        mime["Cc"] = ", ".join(cc)
    mime["Subject"] = subject
    mime.attach(MIMEText(html_body, "html"))
    for content_id, image in inline_images.items():
        part = MIMEImage(image)
        part.add_header("Content-ID", f"<{content_id}>")
        part.add_header("Content-Disposition", "inline")
        mime.attach(part)
    for name, content in attachments.items():
        part = MIMEBase("application", "octet-stream")
        part.set_payload(content)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=name)
        mime.attach(part)
    text = mime.as_string()

    return text


def managed_email_failure(error: ManagedProviderError) -> str:
    """Return V1's message for a rejected managed e-mail.

    Args:
        error: The API rejection.

    Returns:
        The outcome message.
    """
    detail = str(error)
    lowered = detail.lower()
    if error.status == 403:
        if "non-workspace members" in lowered:
            return NON_MEMBER_MESSAGE
        return f"Failed to send email: access forbidden. {detail}"
    if error.status == 413 or "too large" in lowered:
        return TOO_LARGE_MESSAGE
    if "rate limit" in lowered:
        return RATE_LIMIT_MESSAGE
    if "credits exceeded" in lowered:
        return CREDITS_MESSAGE

    message = f"Failed to send email via proxy. {detail}"

    return message


def _result(
    error_status: bool, message: str, *, throttled: bool = False
) -> Dict[str, Any]:
    result = {
        "error_status": error_status,
        "throttling_status": throttled,
        "message": message,
    }

    return result
