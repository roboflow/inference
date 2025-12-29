import base64
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field
from twilio.rest import Client

from inference.core.roboflow_api import post_to_roboflow_api
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
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

LONG_DESCRIPTION = """
The **Twilio SMS/MMS Notification** block allows users to send text and multimedia messages as part of a workflow.

### SMS Provider Options

This block supports two SMS delivery methods via a dropdown selector:

1. **Roboflow Managed API Key (Default)** - No Twilio configuration needed. Messages are sent through Roboflow's proxy service:
   * **Simplified setup** - just provide message, recipient, and optional media
   * **Secure** - your workflow API key is used for authentication
   * **No Twilio account required**
   * **Pricing:** US/Canada: 30 messages per credit. International: 10 messages per credit. (SMS and MMS priced the same)

2. **Custom Twilio** - Use your own Twilio account:
   * Full control over message delivery
   * Requires Twilio credentials (Account SID, Auth Token, Phone Number)
   * You pay Twilio directly for messaging

### Message Content

* **Receiver Number:** Phone number to receive the message (must be in E.164 format, e.g., +15551234567)

* **Message:** The body content of the SMS/MMS. **Message can be parametrised with data generated during workflow run. See *Dynamic Parameters* section.**

* **Media URL (Optional):** For MMS messages, provide image URL(s) or image outputs from visualization blocks

### Dynamic Parameters

Message content can be parametrised with Workflow execution outcomes. Example:

```
message = "Alert! Detected {{ '{{' }} $parameters.num_detections {{ '}}' }} objects"
```

Message parameters are set via `message_parameters`:

```
message_parameters = {
    "num_detections": "$steps.model.predictions"
}
```

Transform data using `message_parameters_operations`:

```
message_parameters_operations = {
    "predictions": [
        {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
    ]
}
```

### MMS Support

Send images with your message by providing `media_url`:

* **Image URLs**: Provide publicly accessible image URLs as a string or list
* **Workflow Images**: Reference image outputs from visualization blocks  
* **Base64 Images**: Images are automatically converted for transmission

Example:

```
media_url = "$steps.bounding_box_visualization.image"
```

Or multiple images:

```
media_url = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
```

**Note:** MMS is primarily supported in US/Canada. International MMS availability varies by carrier.

### Using Custom Twilio

To use your own Twilio account, select "Custom Twilio" and configure:

* `twilio_account_sid` - Your Twilio Account SID from the [Twilio Console](https://twilio.com/console)
* `twilio_auth_token` - Your Twilio Auth Token  
* `sender_number` - Your Twilio phone number (must be in E.164 format)

### Cooldown

The block accepts `cooldown_seconds` (defaults to `5` seconds) to prevent notification bursts. Set `0` for no cooldown.

During cooldown, the `throttling_status` output is set to `True` and no message is sent.

!!! warning "Cooldown limitations"
    Cooldown is limited to video processing. Using this block in HTTP service workflows 
    (Roboflow Hosted API, Dedicated Deployment) has no cooldown effect for HTTP requests.

### Async Execution

Set `fire_and_forget=True` to send messages in the background, allowing the Workflow to proceed.  
With async mode, `error_status` is always `False`. **Set `fire_and_forget=False` for debugging.**

### Disabling Notifications

Set `disable_sink` flag to manually disable the notifier block via Workflow input.
"""

PARAMETER_REGEX = re.compile(r"({{\s*\$parameters\.(\w+)\s*}})")


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Twilio SMS/MMS Notification",
            "version": "v2",
            "short_description": "Send SMS/MMS notifications via Twilio.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "notifications",
                "icon": "far fa-comment-sms",
                "blockPriority": 0,
                "popular": True,
            },
        }
    )
    type: Literal["roboflow_core/twilio_sms_notification@v2"]

    sms_provider: Literal["Roboflow Managed API Key", "Custom Twilio"] = Field(
        default="Roboflow Managed API Key",
        description="Choose SMS delivery method: use Roboflow's managed service or configure your own Twilio account.",
        examples=["Roboflow Managed API Key", "Custom Twilio"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    receiver_number: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Phone number to receive the message (E.164 format, e.g., +15551234567)",
        examples=["+15551234567", "$inputs.receiver_number"],
        json_schema_extra={
            "hide_description": True,
            "always_visible": True,
        },
    )

    message: str = Field(
        description="Content of the message to be sent.",
        examples=["Alert! Detected {{ $parameters.num_detections }} objects"],
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
                "num_detections": "$steps.model.predictions",
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

    media_url: Optional[
        Union[
            str,
            List[str],
            Selector(kind=[STRING_KIND, LIST_OF_VALUES_KIND, IMAGE_KIND]),
        ]
    ] = Field(
        default=None,
        description="Optional media URL(s) for MMS. Provide publicly accessible image URLs or image outputs from workflow blocks.",
        examples=["$steps.visualization.image", "https://example.com/image.jpg"],
        json_schema_extra={
            "hide_description": True,
        },
    )

    # Twilio credentials - hidden when using Roboflow Managed API Key
    twilio_account_sid: Optional[
        Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])]
    ] = Field(
        default=None,
        title="Twilio Account SID",
        description="Twilio Account SID from the Twilio Console.",
        private=True,
        examples=["$inputs.twilio_account_sid"],
        json_schema_extra={
            "relevant_for": {
                "sms_provider": {"values": ["Custom Twilio"], "required": True},
            },
        },
    )

    twilio_auth_token: Optional[
        Union[str, Selector(kind=[STRING_KIND, SECRET_KIND])]
    ] = Field(
        default=None,
        title="Twilio Auth Token",
        description="Twilio Auth Token from the Twilio Console.",
        private=True,
        examples=["$inputs.twilio_auth_token"],
        json_schema_extra={
            "hide_description": True,
            "relevant_for": {
                "sms_provider": {"values": ["Custom Twilio"], "required": True},
            },
        },
    )

    sender_number: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Sender phone number (E.164 format, e.g., +15551234567)",
        examples=["+15551234567", "$inputs.sender_number"],
        json_schema_extra={
            "hide_description": True,
            "relevant_for": {
                "sms_provider": {"values": ["Custom Twilio"], "required": True},
            },
        },
    )

    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to run the block asynchronously (True) for faster workflows or synchronously (False) for debugging and error handling.",
        examples=["$inputs.fire_and_forget", False],
    )

    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Boolean flag to disable block execution.",
        examples=[False, "$inputs.disable_sms_notifications"],
    )

    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Number of seconds until a follow-up notification can be sent.",
        examples=["$inputs.cooldown_seconds", 3],
        json_schema_extra={
            "always_visible": True,
        },
    )

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


class TwilioSMSNotificationBlockV2(WorkflowBlock):

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
        self._clients: Dict[str, Client] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["background_tasks", "thread_pool_executor", "api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        sms_provider: str,
        receiver_number: str,
        message: str,
        message_parameters: Dict[str, Any],
        message_parameters_operations: Dict[str, List[AllOperationsType]],
        media_url: Optional[Union[str, List[str], WorkflowImageData]],
        twilio_account_sid: Optional[str],
        twilio_auth_token: Optional[str],
        sender_number: Optional[str],
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

        # Check cooldown
        seconds_since_last_notification = cooldown_seconds
        if self._last_notification_fired is not None:
            seconds_since_last_notification = (
                datetime.now() - self._last_notification_fired
            ).total_seconds()
        if seconds_since_last_notification < cooldown_seconds:
            logging.info(
                f"Activated `roboflow_core/twilio_sms_notification@v2` cooldown."
            )
            return {
                "error_status": False,
                "throttling_status": True,
                "message": "Sink cooldown applies",
            }

        # Check if using Roboflow Managed API Key
        use_managed_service = sms_provider == "Roboflow Managed API Key"

        if use_managed_service:
            send_sms_handler = partial(
                send_sms_via_roboflow_proxy,
                roboflow_api_key=self._api_key,
                receiver_number=receiver_number,
                message=message,
                message_parameters=message_parameters,
                message_parameters_operations=message_parameters_operations,
                media_url=media_url,
            )
        else:
            # Validate required Twilio fields
            if not twilio_account_sid or not twilio_auth_token or not sender_number:
                return {
                    "error_status": True,
                    "throttling_status": False,
                    "message": "Custom Twilio requires twilio_account_sid, twilio_auth_token, and sender_number",
                }

            # Format message
            formatted_message = format_message(
                message=message,
                message_parameters=message_parameters,
                message_parameters_operations=message_parameters_operations,
            )

            # Process media URLs for Custom Twilio
            processed_media_urls = None
            if media_url is not None:
                processed_media_urls = process_media_urls_for_twilio(media_url)

            # Get or create Twilio client
            credentials_key = f"{twilio_account_sid}:{twilio_auth_token}"
            if credentials_key not in self._clients:
                self._clients[credentials_key] = Client(
                    twilio_account_sid,
                    twilio_auth_token,
                )
            client = self._clients[credentials_key]

            send_sms_handler = partial(
                send_sms_using_twilio_client,
                client=client,
                message=formatted_message,
                sender_number=sender_number,
                receiver_number=receiver_number,
                media_urls=processed_media_urls,
            )

        self._last_notification_fired = datetime.now()
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(send_sms_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Notification sent in the background task",
            }
        if fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(send_sms_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Notification sent in the background task",
            }
        error_status, message = send_sms_handler()
        return {
            "error_status": error_status,
            "throttling_status": False,
            "message": message,
        }


def format_message(
    message: str,
    message_parameters: Dict[str, Any],
    message_parameters_operations: Dict[str, List[AllOperationsType]],
) -> str:
    """Format SMS/MMS message by replacing parameter placeholders with actual values."""
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


def process_media_urls_for_twilio(
    media_url: Union[str, List[str], WorkflowImageData]
) -> Optional[List[str]]:
    """
    Process media URLs for Twilio MMS.
    Converts WorkflowImageData to temporary public URLs.
    """
    if isinstance(media_url, WorkflowImageData):
        url = _upload_image_to_ephemeral_host(media_url)
        if url:
            return [url]
        logging.warning("Failed to upload WorkflowImageData to temporary storage")
        return None
    elif isinstance(media_url, str):
        return [media_url]
    elif isinstance(media_url, list):
        result = []
        for item in media_url:
            if isinstance(item, WorkflowImageData):
                url = _upload_image_to_ephemeral_host(item)
                if url:
                    result.append(url)
                else:
                    logging.warning(
                        "Failed to upload WorkflowImageData to temporary storage"
                    )
            else:
                result.append(item)
        return result if result else None
    return None


_EPHEMERAL_HOST_PARTS = ["tmp", "files", ".org"]


def _upload_image_to_ephemeral_host(image_data: WorkflowImageData) -> Optional[str]:
    """Upload WorkflowImageData to an ephemeral file hosting service."""
    import requests

    try:
        jpeg_bytes = encode_image_to_jpeg_bytes(image_data.numpy_image)
        host = "".join(_EPHEMERAL_HOST_PARTS)
        endpoint = f"https://{host}/api/v1/upload"

        files = {"file": ("image.jpg", jpeg_bytes, "image/jpeg")}
        response = requests.post(endpoint, files=files, timeout=10)
        response.raise_for_status()

        data = response.json()
        if data.get("status") == "success" and data.get("data", {}).get("url"):
            url = data["data"]["url"]
            if "/dl/" not in url and host in url:
                url = url.replace(f"{host}/", f"{host}/dl/")
            return url

        logging.warning(f"Unexpected ephemeral host response: {data}")
        return None

    except Exception as error:
        logging.warning(f"Failed to upload image to ephemeral host: {error}")
        return None


def serialize_media_for_api(
    media_url: Union[str, List[str], WorkflowImageData, None]
) -> Tuple[Optional[List[str]], Optional[List[Dict[str, str]]]]:
    """
    Serialize media for API transmission.
    Separates URL-based media from base64 image data.

    Returns:
        Tuple of (media_urls, media_base64) where:
        - media_urls: List of string URLs
        - media_base64: List of {"base64": str, "mimeType": str} objects
    """
    if media_url is None:
        return None, None

    media_urls: List[str] = []
    media_base64: List[Dict[str, str]] = []

    items = [media_url] if not isinstance(media_url, list) else media_url

    for item in items:
        if isinstance(item, WorkflowImageData):
            # Convert to base64 JPEG
            jpeg_bytes = encode_image_to_jpeg_bytes(item.numpy_image)
            media_base64.append(
                {
                    "base64": base64.b64encode(jpeg_bytes).decode("utf-8"),
                    "mimeType": "image/jpeg",
                }
            )
        elif isinstance(item, str):
            media_urls.append(item)

    return (media_urls if media_urls else None, media_base64 if media_base64 else None)


def send_sms_via_roboflow_proxy(
    roboflow_api_key: str,
    receiver_number: str,
    message: str,
    message_parameters: Dict[str, Any],
    message_parameters_operations: Dict[str, List[AllOperationsType]],
    media_url: Optional[Union[str, List[str], WorkflowImageData]],
) -> Tuple[bool, str]:
    """Send SMS/MMS through Roboflow's proxy service."""
    from inference.core.exceptions import (
        RoboflowAPIForbiddenError,
        RoboflowAPIUnsuccessfulRequestError,
    )

    # Custom error handler that preserves the API's error message
    def handle_sms_proxy_error(status_code: int, http_error: Exception) -> None:
        """Extract and preserve the actual error message from the API response."""
        try:
            response = http_error.response
            error_data = response.json()
            api_error_message = (
                error_data.get("details") or error_data.get("error") or str(http_error)
            )
        except Exception:
            api_error_message = str(http_error)

        if status_code == 403:
            raise RoboflowAPIForbiddenError(api_error_message) from http_error
        elif status_code == 429:
            raise RoboflowAPIUnsuccessfulRequestError(api_error_message) from http_error
        else:
            raise RoboflowAPIUnsuccessfulRequestError(api_error_message) from http_error

    custom_error_handlers = {
        403: lambda e: handle_sms_proxy_error(403, e),
        429: lambda e: handle_sms_proxy_error(429, e),
    }

    try:
        # Format message client-side before sending to proxy
        formatted_message = format_message(
            message=message,
            message_parameters=message_parameters,
            message_parameters_operations=message_parameters_operations,
        )

        payload = {
            "receiver_number": receiver_number,
            "message": formatted_message,
        }

        # Serialize media - separates URLs from base64 data
        if media_url is not None:
            media_urls, media_base64 = serialize_media_for_api(media_url)
            if media_urls:
                payload["media_urls"] = media_urls
            if media_base64:
                payload["media_base64"] = media_base64

        endpoint = "apiproxy/twilio"

        response_data = post_to_roboflow_api(
            endpoint=endpoint,
            api_key=roboflow_api_key,
            payload=payload,
            http_errors_handlers=custom_error_handlers,
        )

        return False, "Notification sent successfully via Roboflow proxy"
    except RoboflowAPIForbiddenError as error:
        error_message = str(error)
        logging.warning(
            f"SMS rejected by proxy due to access restrictions: {error_message}"
        )
        return True, f"Failed to send SMS: access forbidden. {error_message}"
    except RoboflowAPIUnsuccessfulRequestError as error:
        error_message = str(error)
        logging.warning(f"SMS proxy API error: {error_message}")

        if "rate limit" in error_message.lower():
            return True, (
                "Failed to send SMS: rate limit exceeded. "
                "The workspace has exceeded its SMS sending limits. "
                "Please wait before sending more messages."
            )
        elif "credits exceeded" in error_message.lower():
            return True, (
                "Failed to send SMS: workspace credits exceeded. "
                "Please add more credits to your workspace to continue sending messages."
            )
        else:
            return True, f"Failed to send SMS via proxy. {error_message}"
    except Exception as error:
        logging.warning(f"Could not send SMS via Roboflow proxy. Error: {str(error)}")
        return True, f"Failed to send SMS via proxy. Internal error details: {error}"


def send_sms_using_twilio_client(
    client: Client,
    message: str,
    sender_number: str,
    receiver_number: str,
    media_urls: Optional[List[str]],
) -> Tuple[bool, str]:
    """Send SMS/MMS using Twilio client directly."""
    try:
        message_params = {
            "body": message,
            "from_": sender_number,
            "to": receiver_number,
        }
        if media_urls:
            message_params["media_url"] = media_urls

        client.messages.create(**message_params)
        return False, "Notification sent successfully"
    except Exception as error:
        logging.warning(f"Could not send Twilio SMS notification. Error: {str(error)}")
        return (
            True,
            f"Failed to send Twilio SMS notification. Internal error details: {error}",
        )
