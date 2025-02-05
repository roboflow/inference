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
The **Email Notification** block allows users to send email notifications as part of a workflow. 
It **requires** SMTP server setup to send the notification 

### Customizable Email Content

* **Subject:** Set the subject field to define the subject line of the email.

* **Message:** Use the message field to write the body content of the email. **Message can be parametrised
with data generated during workflow run. See *Dynamic Parameters* section.**

* **Recipients (To, CC, BCC)**: Define who will receive the email using `receiver_email`, 
`cc_receiver_email`, and `bcc_receiver_email` properties. You can input a single email or a list.

### Dynamic Parameters

Content of the message can be parametrised with Workflow execution outcomes. Take a look at the example
message using dynamic parameters:

```
message = "This is example notification. Predicted classes: {{ '{{' }} $parameters.predicted_classes {{ '}}' }}"
```

Message parameters are delivered by Workflows Execution Engine by setting proper data selectors in
`message_parameters` field, for example:

```
message_parameters = {
    "predicted_classes": "$steps.model.predictions"
}
```

Selecting data is not the only option - data may be processed in the block. In the example below we wish to
extract names of predicted classes. We can apply transformation **for each parameter** by setting
`message_parameters_operations`:

```
message_parameters_operations = {
    "predictions": [
        {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
    ]
}
```

As a result, in the e-mail that will be sent, you can expect:

```
This is example notification. Predicted classes: ["class_a", "class_b"].
```

### Configuring SMTP server

Those are the parameters configuring SMTP server: 

* `smtp_server` - hostname of the SMTP server to use

* `sender_email` - e-mail account to be used as sender

* `sender_email_password` - password for sender e-mail account

* `smtp_port` - port of SMTP service - defaults to `465`

Block **enforces** SSL over SMTP.

Typical scenario for using custom SMTP server involves sending e-mail through Google SMTP server.
Take a look at [Google tutorial](https://support.google.com/a/answer/176600?hl=en) to configure the 
block properly. 

!!! note "GMAIL password will not work if 2-step verification is turned on"
    
    GMAIL users choosing custom SMTP server as e-mail service provider must configure 
    [application password](https://support.google.com/accounts/answer/185833) to avoid
    problems with 2-step verification protected account. Beware that **application
    password must be kept protected** - we recommend sending the password in Workflow 
    input and providing it each time by the caller, avoiding storing it in Workflow 
    definition.
    
### Cooldown

The block accepts `cooldown_seconds` (which **defaults to `5` seconds**) to prevent unintended bursts of 
notifications. Please adjust it according to your needs, setting `0` indicate no cooldown. 

During cooldown period, consecutive runs of the step will cause `throttling_status` output to be set `True`
and no notification will be sent.

!!! warning "Cooldown limitations"

    Current implementation of cooldown is limited to video processing - using this block in context of a 
    Workflow that is run behind HTTP service (Roboflow Hosted API, Dedicated Deployment or self-hosted 
    `inference` server) will have no effect for processing HTTP requests.  


### Attachments

You may specify attachment files to be send with your e-mail. Attachments can only be generated 
in runtime by dedicated blocks (for instance [CSV Formatter](https://inference.roboflow.com/workflows/csv_formatter/))

To include attachments, simply provide the attachment name and refer to other block outputs:

```
attachments = {
    "report.pdf": "$steps.report_generator.output"
}
```

### Async execution

Configure the `fire_and_forget` property. Set it to True if you want the email to be sent in the background, allowing the 
Workflow to proceed without waiting on e-mail to be sent. In this case you will not be able to rely on 
`error_status` output which will always be set to `False`, so we **recommend setting the `fire_and_forget=False` for
debugging purposes**.

### Disabling notifications based on runtime parameter

Sometimes it would be convenient to manually disable the e-mail notifier block. This is possible 
setting `disable_sink` flag to hold reference to Workflow input. with such setup, caller would be
able to disable the sink when needed sending agreed input parameter.
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
        description="Subject of the message.",
        examples=["Workflow alert"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    sender_email: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="E-mail to be used to send the message.",
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
        description="Destination e-mail address.",
        examples=["receiver@gmail.com"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    message: str = Field(
        description="Content of the message to be send.",
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
        description="Data to be used inside the message.",
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
    cc_receiver_email: Optional[
        Union[
            str,
            List[str],
            Selector(kind=[STRING_KIND, LIST_OF_VALUES_KIND]),
        ]
    ] = Field(
        default=None,
        description="Destination e-mail address.",
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
        description="Destination e-mail address.",
        examples=["bcc-receiver@gmail.com"],
        json_schema_extra={
            "hide_description": True,
        },
    )
    attachments: Dict[str, Selector(kind=[STRING_KIND, BYTES_KIND])] = Field(
        description="Attachments",
        default_factory=dict,
        examples=[{"report.cvs": "$steps.csv_formatter.csv_content"}],
        json_schema_extra={
            "hide_description": True,
        },
    )
    smtp_server: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Custom SMTP server to be used.",
        examples=["$inputs.smtp_server", "smtp.google.com"],
    )
    sender_email_password: Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])] = (
        Field(
            description="Sender e-mail password be used when authenticating to SMTP server.",
            private=True,
            examples=["$inputs.email_password"],
        )
    )
    smtp_port: int = Field(
        default=465,
        description="SMTP server port.",
        examples=[465],
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
        examples=[False, "$inputs.disable_email_notifications"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Number of seconds until a follow-up notification can be sent. ",
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
