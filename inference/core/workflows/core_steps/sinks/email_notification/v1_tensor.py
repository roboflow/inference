import logging
import re
import smtplib
import ssl
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import partial
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Type, Union

from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field, field_validator

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
    LIST_OF_VALUES_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

ROBOFLOW_EMAIL_ENDPOINT = "/notifications/email"

LONG_DESCRIPTION = """
Send email notifications via SMTP server with customizable subject, message content with dynamic workflow data parameters, recipient lists (To, CC, BCC), file attachments, cooldown throttling, and optional async background execution for alerting, reporting, and communication workflows.

## How This Block Works

This block sends email notifications through an SMTP server, integrating workflow execution results into email content. The block:

1. Checks if the sink is disabled via `disable_sink` flag (if disabled, returns immediately without sending)
2. Validates cooldown period (if enabled, throttles notifications within `cooldown_seconds` of the last sent email, returning throttling status)
3. Formats the email message by processing dynamic parameters (replaces placeholders like `{{ $parameters.parameter_name }}` with actual workflow data from `message_parameters`)
4. Applies optional UQL operations to transform parameter values before insertion (e.g., extract class names from detections, calculate metrics, filter data) using `message_parameters_operations`
5. Constructs email recipients lists from `receiver_email` (required), `cc_receiver_email`, and `bcc_receiver_email` (supports single email addresses or lists)
6. Processes attachments by retrieving content from referenced workflow step outputs and encoding them as email attachments
7. Establishes SSL-secured SMTP connection to the configured server (authentication using sender email and password)
8. Sends the email synchronously or asynchronously based on `fire_and_forget` setting:
   - **Synchronous mode** (`fire_and_forget=False`): Waits for email send completion, returns actual error status for debugging
   - **Asynchronous mode** (`fire_and_forget=True`): Sends email in background task, workflow continues immediately, error status always False
9. Returns status outputs indicating success, throttling, or errors

The block supports dynamic message content through parameter placeholders that are replaced with workflow data at runtime. Message parameters can be raw workflow outputs or transformed using UQL operations (e.g., extract properties, calculate counts, filter values). Attachments are sourced from other workflow blocks that produce string or binary content (e.g., CSV Formatter for reports, image outputs for screenshots). Cooldown prevents notification spam by enforcing minimum time between sends, though this only applies to video processing workflows (not HTTP request contexts).

## Requirements

**SMTP Server Configuration**: Requires access to an SMTP server with the following configuration:
- `smtp_server`: Hostname of the SMTP server (e.g., `smtp.gmail.com` for Google)
- `sender_email`: Email address to use as the sender
- `sender_email_password`: Password for the sender email account (or application-specific password for Gmail with 2FA)
- `smtp_port`: SMTP port (defaults to `465` for SSL)

**Gmail Users with 2FA**: If using Gmail with 2-step verification enabled, you must use an [application-specific password](https://support.google.com/accounts/answer/185833) instead of your regular Gmail password. Application passwords should be kept secure and provided via workflow inputs rather than stored in workflow definitions.

**Cooldown Limitations**: The cooldown mechanism (`cooldown_seconds`) only applies to video processing workflows. For HTTP request contexts (Roboflow Hosted API, Dedicated Deployment, or self-hosted servers), cooldown has no effect since each request is independent.

## Common Use Cases

- **Alert Notifications**: Send email alerts when specific conditions are detected (e.g., alert security team when unauthorized objects detected, notify operators when anomaly detected, send alerts when detection counts exceed thresholds), enabling real-time monitoring and incident response
- **Workflow Execution Reports**: Generate and email periodic or event-driven reports with workflow results (e.g., daily summary reports with detection statistics, batch processing completion notifications, performance metrics summaries), enabling automated reporting and documentation
- **Detection Summaries**: Send email summaries of detection results with aggregated statistics (e.g., email lists of detected objects, send counts and classifications, include detection confidence summaries), enabling stakeholders to stay informed about workflow outputs
- **Error and Status Notifications**: Notify administrators about workflow execution status and errors (e.g., send alerts when workflows fail, notify about processing completion, report system health issues), enabling monitoring and debugging for production deployments
- **Data Export Notifications**: Email generated data exports and reports (e.g., attach CSV reports from CSV Formatter, send exported detection data, include formatted analytics summaries), enabling automated data distribution and archival
- **Multi-Recipient Updates**: Send notifications to multiple stakeholders simultaneously using CC/BCC (e.g., notify team members about detections, send updates to multiple departments, distribute reports with CC for visibility), enabling efficient multi-party communication

## Connecting to Other Blocks

This block receives data from workflow steps and sends email notifications:

- **After detection or analysis blocks** (e.g., Object Detection, Instance Segmentation, Classification) to send alerts or summaries when objects are detected, classifications are made, or thresholds are exceeded, enabling real-time notification workflows
- **After data processing blocks** (e.g., Expression, Property Definition, Detections Filter) to include computed metrics, transformed data, or filtered results in email notifications, enabling customized reporting with processed data
- **After formatter blocks** (e.g., CSV Formatter) to attach formatted reports and exports to emails, enabling automated distribution of structured data and analytics
- **In conditional workflows** (e.g., Continue If) to send notifications only when specific conditions are met, enabling event-driven alerting and reporting
- **After aggregation blocks** (e.g., Data Aggregator) to email periodic analytics summaries and statistical reports, enabling scheduled reporting and trend analysis
- **In monitoring workflows** to send status updates, error notifications, or health check reports, enabling automated system monitoring and incident management
"""

PARAMETER_REGEX = re.compile(r"({{\s*\$parameters\.(\w+)\s*}})")


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Email Notification",
            "version": "v1",
            "short_description": "Send notification via e-mail.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "notifications",
                "icon": "far fa-envelope",
                "blockPriority": 0,
                "popular": True,
            },
        }
    )
    type: Literal["roboflow_core/email_notification@v1"]
    subject: str = Field(
        description="Email subject line for the notification. This is the text that appears in the email header and recipient's inbox subject field. Can include static text describing the notification purpose (e.g., 'Workflow Alert', 'Detection Summary', 'Daily Report').",
        examples=["Workflow alert"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    sender_email: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Email address to use as the sender of the notification. This email account must have access to the configured SMTP server and the password provided in sender_email_password. For Gmail with 2FA enabled, this should be the Gmail address that has an application-specific password configured.",
        examples=["sender@gmail.com"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    receiver_email: Union[
        str,
        List[str],
        Selector(kind=[STRING_KIND, LIST_OF_VALUES_KIND]),
    ] = Field(
        description="Primary recipient email address(es) for the notification. Required field - at least one recipient must be specified. Can be a single email address string or a list of email addresses for multiple recipients. Recipients will see their email addresses in the 'To' field of the received email.",
        examples=["receiver@gmail.com"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    message: str = Field(
        description="Email body content (plain text). Supports dynamic parameters using placeholder syntax: {{ $parameters.parameter_name }}. Placeholders are replaced with values from message_parameters at runtime. Message can be multi-line text. Example: 'Detected {{ $parameters.num_objects }} objects. Classes: {{ $parameters.classes }}.'",
        examples=[
            "During last 5 minutes detected {{ $parameters.num_instances }} instances"
        ],
        json_schema_extra={
            "hide_description": True,
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
    cc_receiver_email: Optional[
        Union[
            str,
            List[str],
            Selector(kind=[STRING_KIND, LIST_OF_VALUES_KIND]),
        ]
    ] = Field(
        default=None,
        description="Optional CC (Carbon Copy) recipient email address(es). Can be a single email address string or a list of email addresses. CC recipients receive a copy of the email and can see each other's addresses. Use for recipients who should be informed but don't need to take action.",
        examples=["cc-receiver@gmail.com"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    bcc_receiver_email: Optional[
        Union[
            str,
            List[str],
            Selector(kind=[STRING_KIND, LIST_OF_VALUES_KIND]),
        ]
    ] = Field(
        default=None,
        description="Optional BCC (Blind Carbon Copy) recipient email address(es). Can be a single email address string or a list of email addresses. BCC recipients receive a copy of the email but their addresses are hidden from other recipients. Use for recipients who should receive the notification privately.",
        examples=["bcc-receiver@gmail.com"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    attachments: Dict[str, Selector(kind=[STRING_KIND, BYTES_KIND])] = Field(
        description="Optional dictionary mapping attachment filenames to workflow step outputs that provide file content. Keys are the attachment filenames (e.g., 'report.csv', 'summary.pdf'), values are selectors to blocks that output string or binary content (e.g., CSV Formatter outputs, image data, generated reports). Attachments are encoded and attached to the email. Leave empty if no attachments are needed.",
        default_factory=dict,
        examples=[{"report.cvs": "$steps.csv_formatter.csv_content"}],
        json_schema_extra={
            "hide_description": True,
        },
    )
    smtp_server: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="SMTP server hostname to use for sending emails. Common examples: 'smtp.gmail.com' for Gmail, 'smtp.outlook.com' for Outlook, or your organization's SMTP server hostname. The block enforces SSL/TLS encryption for SMTP connections. Ensure the server supports SSL on the specified port.",
        examples=["$inputs.smtp_server", "smtp.google.com"],
    )
    sender_email_password: Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])] = (
        Field(
            description="Password for the sender email account to authenticate with the SMTP server. For Gmail with 2-step verification enabled, use an application-specific password instead of the regular Gmail password. This field is marked as private for security. Recommended to provide via workflow inputs rather than storing in workflow definitions. For Roboflow-hosted services, can use SECRET_KIND selectors for secure credential management.",
            private=True,
            examples=["$inputs.email_password"],
        )
    )
    smtp_port: int = Field(
        default=465,
        description="SMTP server port number. Defaults to 465 (standard SSL port for SMTP). Common alternatives: 587 for TLS (not supported - this block enforces SSL), 25 for unencrypted (not recommended). Ensure the port supports SSL encryption as required by this block.",
        examples=[465],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Execution mode: True for asynchronous background sending (workflow continues immediately, error_status always False, faster execution), False for synchronous sending (waits for email completion, returns actual error status for debugging). Set to False during development and debugging to catch email sending errors. Set to True in production for faster workflow execution when email delivery timing is not critical.",
        examples=["$inputs.fire_and_forget", False],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Flag to disable email sending at runtime. When True, the block skips sending email and returns a disabled message. Useful for conditional notification control via workflow inputs (e.g., allow callers to disable notifications for testing, enable/disable based on configuration). Set via workflow inputs for runtime control.",
        examples=[False, "$inputs.disable_email_notifications"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Minimum seconds between consecutive email notifications to prevent notification spam. Defaults to 5 seconds. Set to 0 to disable cooldown (no throttling). During cooldown period, the block returns throttling_status=True and skips sending. Note: Cooldown only applies to video processing workflows, not HTTP request contexts (Roboflow Hosted API, Dedicated Deployment, or self-hosted servers where each request is independent).",
        examples=["$inputs.cooldown_seconds", 3],
        json_schema_extra={
            "always_visible": True,
        },
    )

    @field_validator("receiver_email")
    @classmethod
    def ensure_receiver_email_is_not_an_empty_list(cls, value: Any) -> dict:
        if isinstance(value, list) and len(value) == 0:
            raise ValueError(
                "E-mail notification must have at least one receiver defined."
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


class EmailNotificationBlockV1(WorkflowBlock):

    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._last_notification_fired: Optional[datetime] = None

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        subject: str,
        message: str,
        sender_email: str,
        receiver_email: Union[str, List[str]],
        cc_receiver_email: Optional[Union[str, List[str]]],
        bcc_receiver_email: Optional[Union[str, List[str]]],
        message_parameters: Dict[str, Any],
        message_parameters_operations: Dict[str, List[AllOperationsType]],
        attachments: Dict[str, str],
        smtp_server: str,
        sender_email_password: str,
        smtp_port: int,
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: int,
    ) -> BlockResult:
        if disable_sink:
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Sink was disabled by parameter `disable_sink`",
            }
        seconds_since_last_notification = cooldown_seconds
        if self._last_notification_fired is not None:
            seconds_since_last_notification = (
                datetime.now() - self._last_notification_fired
            ).total_seconds()
        if seconds_since_last_notification < cooldown_seconds:
            logging.info(f"Activated `roboflow_core/email_notification@v1` cooldown.")
            return {
                "error_status": False,
                "throttling_status": True,
                "message": "Sink cooldown applies",
            }
        message = format_email_message(
            message=message,
            message_parameters=message_parameters,
            message_parameters_operations=message_parameters_operations,
        )
        receiver_email = (
            receiver_email if isinstance(receiver_email, list) else [receiver_email]
        )
        if cc_receiver_email is not None:
            cc_receiver_email = (
                cc_receiver_email
                if isinstance(cc_receiver_email, list)
                else [cc_receiver_email]
            )
        if bcc_receiver_email is not None:
            bcc_receiver_email = (
                bcc_receiver_email
                if isinstance(bcc_receiver_email, list)
                else [bcc_receiver_email]
            )
        send_email_handler = partial(
            send_email_using_smtp_server,
            sender_email=sender_email,
            receiver_email=receiver_email,
            cc_receiver_email=cc_receiver_email,
            bcc_receiver_email=bcc_receiver_email,
            subject=subject,
            message=message,
            attachments=attachments,
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            sender_email_password=sender_email_password,
        )
        self._last_notification_fired = datetime.now()
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(send_email_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Notification sent in the background task",
            }
        if fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(send_email_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Notification sent in the background task",
            }
        error_status, message = send_email_handler()
        return {
            "error_status": error_status,
            "throttling_status": False,
            "message": message,
        }


def format_email_message(
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


def send_email_using_smtp_server(
    sender_email: str,
    receiver_email: List[str],
    cc_receiver_email: Optional[List[str]],
    bcc_receiver_email: Optional[List[str]],
    subject: str,
    message: str,
    attachments: Dict[str, str],
    smtp_server: str,
    smtp_port: int,
    sender_email_password: str,
) -> Tuple[bool, str]:
    try:
        _send_email_using_smtp_server(
            sender_email=sender_email,
            receiver_email=receiver_email,
            cc_receiver_email=cc_receiver_email,
            bcc_receiver_email=bcc_receiver_email,
            subject=subject,
            message=message,
            attachments=attachments,
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            sender_email_password=sender_email_password,
        )
        return False, "Notification sent successfully"
    except Exception as error:
        logging.warning(
            f"Could not send e-mail using custom SMTP server. Error: {str(error)}"
        )
        return True, f"Failed to send e-mail. Internal error details: {error}"


def _send_email_using_smtp_server(
    sender_email: str,
    receiver_email: List[str],
    cc_receiver_email: Optional[List[str]],
    bcc_receiver_email: Optional[List[str]],
    subject: str,
    message: str,
    attachments: Dict[str, str],
    smtp_server: str,
    smtp_port: int,
    sender_email_password: str,
) -> None:
    e_mail_message = MIMEMultipart()
    e_mail_message["From"] = sender_email
    e_mail_message["To"] = ",".join(receiver_email)
    if cc_receiver_email:
        e_mail_message["Cc"] = ",".join(cc_receiver_email)
    if bcc_receiver_email:
        e_mail_message["Bcc"] = ",".join(bcc_receiver_email)
    e_mail_message["Subject"] = subject
    e_mail_message.attach(MIMEText(message, "plain"))
    for attachment_name, attachment_content in attachments.items():
        part = MIMEBase("application", "octet-stream")
        binary_payload = attachment_content
        if not isinstance(binary_payload, bytes):
            binary_payload = binary_payload.encode("utf-8")
        part.set_payload(binary_payload)
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {attachment_name}",
        )
        e_mail_message.attach(part)
    to_sent = e_mail_message.as_string()
    with establish_smtp_connection(
        smtp_server=smtp_server, smtp_port=smtp_port
    ) as server:
        server.login(sender_email, sender_email_password)
        server.sendmail(sender_email, receiver_email, to_sent)


@contextmanager
def establish_smtp_connection(
    smtp_server: str, smtp_port: int
) -> Generator[smtplib.SMTP_SSL, None, None]:
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
        yield server
