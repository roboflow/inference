import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field, field_validator

from inference.core.roboflow_api import post_to_roboflow_api
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.core_steps.sinks.email_notification.v1 import (
    send_email_using_smtp_server,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition, WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    BYTES_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MANAGED_KEY,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
The **Email Notification** block allows users to send email notifications as part of a workflow. 

### Email Provider Options

This block supports two email delivery methods via a dropdown selector:

1. **Roboflow Managed API Key (Default)** - No SMTP configuration needed. Emails are sent through Roboflow's proxy service:
   * **Simplified setup** - just provide subject, message, and recipient
   * **Secure** - your workflow API key is used for authentication
   * **No SMTP server required**

2. **Custom SMTP** - Use your own SMTP server:
   * Full control over email delivery
   * Requires SMTP server configuration (host, port, credentials)
   * Supports CC and BCC recipients

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

### Using Custom SMTP Server

To use your own SMTP server, select "Custom SMTP" from the `email_provider` dropdown and configure 
the following parameters:

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

You may specify attachment files to be sent with your e-mail. Attachments can be generated 
in runtime by dedicated blocks or from image outputs.

**Supported attachment types:**
- **CSV/Text files**: From blocks like [CSV Formatter](https://inference.roboflow.com/workflows/csv_formatter/)
- **Images**: Any image output from visualization blocks (automatically converted to JPEG)
- **Binary data**: Any bytes output from compatible blocks

To include attachments, provide the attachment filename as the key and reference the block output:

```
attachments = {
    "report.csv": "$steps.csv_formatter.csv_content",
    "detection.jpg": "$steps.bounding_box_visualization.image"
}
```

**Note:** Image attachments are automatically converted to JPEG format. If the filename doesn't 
include a `.jpg` or `.jpeg` extension, it will be added automatically.

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
            "version": "v2",
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
    type: Literal["roboflow_core/email_notification@v2"]
    
    email_provider: Literal["Roboflow Managed API Key", "Custom SMTP"] = Field(
        default="Roboflow Managed API Key",
        description="Choose email delivery method: use Roboflow's managed service or configure your own SMTP server.",
        examples=["Roboflow Managed API Key", "Custom SMTP"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    
    subject: str = Field(
        description="Subject of the message.",
        examples=["Workflow alert"],
        json_schema_extra={
            "hide_description": True,
            "always_visible": True,
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
            "always_visible": True,
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
            "always_visible": True,
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
    
    # SMTP fields - hidden when using Roboflow Managed API Key
    sender_email: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="E-mail to be used to send the message.",
        examples=["sender@gmail.com"],
        json_schema_extra={
            "hide_description": True,
            "relevant_for": {
                "email_provider": {"values": ["Custom SMTP"], "required": True},
            },
        },
    )
    
    smtp_server: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Custom SMTP server to be used.",
        examples=["$inputs.smtp_server", "smtp.google.com"],
        json_schema_extra={
            "relevant_for": {
                "email_provider": {"values": ["Custom SMTP"], "required": True},
            },
        },
    )
    
    sender_email_password: Optional[Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])]] = Field(
        default=None,
        description="Sender e-mail password be used when authenticating to SMTP server.",
        private=True,
        examples=["$inputs.email_password"],
        json_schema_extra={
            "relevant_for": {
                "email_provider": {"values": ["Custom SMTP"], "required": True},
            },
        },
    )
    
    cc_receiver_email: Optional[
        Union[
            str,
            List[str],
            Selector(kind=[STRING_KIND, LIST_OF_VALUES_KIND]),
        ]
    ] = Field(
        default=None,
        description="CC e-mail address.",
        examples=["cc-receiver@gmail.com"],
        json_schema_extra={
            "hide_description": True,
            "relevant_for": {
                "email_provider": {"values": ["Custom SMTP"], "required": True},
            },
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
        description="BCC e-mail address.",
        examples=["bcc-receiver@gmail.com"],
        json_schema_extra={
            "hide_description": True,
            "relevant_for": {
                "email_provider": {"values": ["Custom SMTP"], "required": True},
            },
        },
    )
    
    smtp_port: int = Field(
        default=465,
        description="SMTP server port.",
        examples=[465],
        json_schema_extra={
            "relevant_for": {
                "email_provider": {"values": ["Custom SMTP"], "required": True},
            },
        },
    )
    
    attachments: Dict[str, Selector(kind=[STRING_KIND, BYTES_KIND, IMAGE_KIND])] = Field(
        description="Attachments",
        default_factory=dict,
        examples=[{"report.cvs": "$steps.csv_formatter.csv_content"}],
        json_schema_extra={
            "hide_description": True,
        },
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


class EmailNotificationBlockV2(WorkflowBlock):

    def __init__(
        self,
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
        api_key: Optional[str],
    ):
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor
        self._api_key = api_key
        self._last_notification_fired: Optional[datetime] = None

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor", "api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        subject: str,
        message: str,
        receiver_email: Union[str, List[str]],
        email_provider: str,
        sender_email: Optional[str],
        cc_receiver_email: Optional[Union[str, List[str]]],
        bcc_receiver_email: Optional[Union[str, List[str]]],
        message_parameters: Dict[str, Any],
        message_parameters_operations: Dict[str, List[AllOperationsType]],
        attachments: Dict[str, str],
        smtp_server: Optional[str],
        sender_email_password: Optional[str],
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
            logging.info(f"Activated `roboflow_core/email_notification@v2` cooldown.")
            return {
                "error_status": False,
                "throttling_status": True,
                "message": "Sink cooldown applies",
            }
        
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
        
        # Check if using Roboflow Managed API Key
        use_managed_service = email_provider == "Roboflow Managed API Key"
        
        if use_managed_service:
            send_email_handler = partial(
                send_email_via_roboflow_proxy,
                roboflow_api_key=self._api_key,
                receiver_email=receiver_email,
                cc_receiver_email=cc_receiver_email,
                bcc_receiver_email=bcc_receiver_email,
                subject=subject,
                message=message,
                message_parameters=message_parameters,
                message_parameters_operations=message_parameters_operations,
                attachments=attachments,
            )
        else:
            # Validate required SMTP fields
            if not sender_email or not smtp_server or not sender_email_password:
                return {
                    "error_status": True,
                    "throttling_status": False,
                    "message": "Custom SMTP requires sender_email, smtp_server, and sender_email_password",
                }
            
            # Format message with parameters before sending via SMTP
            formatted_message = format_email_message(
                message=message,
                message_parameters=message_parameters,
                message_parameters_operations=message_parameters_operations,
            )
            
            # Process attachments: convert images to bytes for SMTP
            processed_attachments = {}
            for filename, value in attachments.items():
                if isinstance(value, WorkflowImageData):
                    # Convert image to JPEG bytes
                    numpy_image = value.numpy_image
                    jpeg_bytes = encode_image_to_jpeg_bytes(numpy_image)
                    # Ensure filename has .jpg extension
                    if not filename.lower().endswith(('.jpg', '.jpeg')):
                        filename = f"{filename}.jpg"
                    processed_attachments[filename] = jpeg_bytes
                elif isinstance(value, bytes):
                    processed_attachments[filename] = value
                elif isinstance(value, str):
                    # String content (e.g., CSV)
                    processed_attachments[filename] = value.encode('utf-8')
                else:
                    # Fallback: convert to string then bytes
                    processed_attachments[filename] = str(value).encode('utf-8')
            
            send_email_handler = partial(
                send_email_using_smtp_server,
                sender_email=sender_email,
                receiver_email=receiver_email,
                cc_receiver_email=cc_receiver_email,
                bcc_receiver_email=bcc_receiver_email,
                subject=subject,
                message=formatted_message,
                attachments=processed_attachments,
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
    """Format email message by replacing parameter placeholders with actual values."""
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


def serialize_image_data(value: Any) -> Any:
    """
    Serialize WorkflowImageData objects to base64 strings for JSON transmission.
    Returns the value unchanged if it's not a WorkflowImageData object.
    """
    if isinstance(value, WorkflowImageData):
        # Get the base64 representation of the image
        base64_image = value.base64_image
        if base64_image:
            return base64_image
        # If no base64 available, try to convert numpy array
        numpy_image = value.numpy_image
        if numpy_image is not None:
            import cv2
            _, buffer = cv2.imencode('.jpg', numpy_image)
            import base64
            return base64.b64encode(buffer).decode('utf-8')
    elif isinstance(value, dict):
        return {k: serialize_image_data(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [serialize_image_data(item) for item in value]
    return value


def serialize_message_parameters(message_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert any WorkflowImageData objects in message_parameters to base64 strings
    so they can be serialized to JSON for the API call.
    """
    return {k: serialize_image_data(v) for k, v in message_parameters.items()}


def process_attachments(attachments: Dict[str, Any]) -> Dict[str, bytes]:
    """
    Process attachments dict to convert WorkflowImageData to JPEG bytes.
    Returns a dict with filename -> bytes mapping.
    """
    processed = {}
    for filename, value in attachments.items():
        if isinstance(value, WorkflowImageData):
            # Convert image to JPEG bytes
            numpy_image = value.numpy_image
            jpeg_bytes = encode_image_to_jpeg_bytes(numpy_image)
            processed[filename] = jpeg_bytes
        elif isinstance(value, bytes):
            # Already bytes, use as-is
            processed[filename] = value
        elif isinstance(value, str):
            # String data (e.g., CSV content)
            processed[filename] = value.encode('utf-8')
        else:
            # Fallback: convert to string then bytes
            processed[filename] = str(value).encode('utf-8')
    return processed


def send_email_via_roboflow_proxy(
    roboflow_api_key: str,
    receiver_email: List[str],
    cc_receiver_email: Optional[List[str]],
    bcc_receiver_email: Optional[List[str]],
    subject: str,
    message: str,
    message_parameters: Dict[str, Any],
    message_parameters_operations: Dict[str, List[AllOperationsType]],
    attachments: Dict[str, Any],
) -> Tuple[bool, str]:
    """Send email through Roboflow's proxy service."""
    try:
        # Serialize any WorkflowImageData objects to base64 strings
        serialized_parameters = serialize_message_parameters(message_parameters)
        
        payload = {
            "receiver_email": receiver_email,
            "subject": subject,
            "message": message,
            "message_parameters": serialized_parameters,
            "message_parameters_operations": message_parameters_operations,
        }
        
        if cc_receiver_email:
            payload["cc_receiver_email"] = cc_receiver_email
        if bcc_receiver_email:
            payload["bcc_receiver_email"] = bcc_receiver_email
        if attachments:
            # Process attachments: convert images to JPEG bytes, then base64 encode
            import base64
            processed_attachments = {}
            for filename, value in attachments.items():
                if isinstance(value, WorkflowImageData):
                    # Convert image to JPEG bytes
                    numpy_image = value.numpy_image
                    jpeg_bytes = encode_image_to_jpeg_bytes(numpy_image)
                    # Ensure filename has .jpg extension
                    if not filename.lower().endswith(('.jpg', '.jpeg')):
                        filename = f"{filename}.jpg"
                    # Base64 encode for JSON transmission
                    processed_attachments[filename] = base64.b64encode(jpeg_bytes).decode('utf-8')
                elif isinstance(value, bytes):
                    # Already bytes, base64 encode
                    processed_attachments[filename] = base64.b64encode(value).decode('utf-8')
                elif isinstance(value, str):
                    # String data (e.g., CSV content), base64 encode
                    processed_attachments[filename] = base64.b64encode(value.encode('utf-8')).decode('utf-8')
                else:
                    # Fallback: convert to string then bytes then base64
                    processed_attachments[filename] = base64.b64encode(str(value).encode('utf-8')).decode('utf-8')
            payload["attachments"] = processed_attachments
        
        endpoint = "apiproxy/email"
        
        response_data = post_to_roboflow_api(
            endpoint=endpoint,
            api_key=roboflow_api_key,
            payload=payload,
        )
        
        return False, "Notification sent successfully via Roboflow proxy"
    except Exception as error:
        logging.warning(
            f"Could not send e-mail via Roboflow proxy. Error: {str(error)}"
        )
        return True, f"Failed to send e-mail via proxy. Internal error details: {error}"
