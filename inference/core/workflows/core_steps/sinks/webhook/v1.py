import logging
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import requests
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_API_KEY_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    TOP_CLASS_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
The **Webhook Sink** block enables sending a data from Workflow into external APIs 
by sending HTTP requests containing workflow results. It supports multiple HTTP methods 
(GET, POST, PUT) and can be configured to send:

* JSON payloads

* query parameters

* multipart-encoded files 

This block is designed to provide flexibility for integrating workflows with remote systems 
for data exchange, notifications, or other integrations.

### Setting Query Parameters
You can easily set query parameters for your request:

```
query_parameters = {
    "api_key": "$inputs.api_key",
}
```

will send request into the following URL: `https://your-host/some/resource?api_key=<API_KEY_VALUE>`

### Setting headers
Setting headers is as easy as setting query parameters:

```
headers = {
    "api_key": "$inputs.api_key",
}
```

### Sending JSON payloads

You can set the body of your message to be JSON document that you construct specifying `json_payload` 
and `json_payload_operations` properties. `json_payload` works similarly to `query_parameters` and 
`headers`, but you can optionally apply UQL operations on top of JSON body elements.

Let's assume that you want to send number of bounding boxes predicted by object detection model
in body field named `detections_number`, then you need to specify configuration similar to the 
following:

```
json_payload = {
    "detections_number": "$steps.model.predictions",
}
json_payload_operations = {
    "detections_number": [{"type": "SequenceLength"}]
}
```

### Multipart-Encoded Files in POST requests

Your endpoint may also accept multipart requests. You can form them in a way which is similar to 
JSON payloads - setting `multi_part_encoded_files` and `multi_part_encoded_files_operations`.

Let's assume you want to send the image in the request, then your configuration may be the following:

```
multi_part_encoded_files = {
    "image": "$inputs.image",
}
multi_part_encoded_files_operations = {
    "image": [{"type": "ConvertImageToJPEG"}]
}
```

### Cooldown

The block accepts `cooldown_seconds` (which **defaults to `5` seconds**) to prevent unintended bursts of 
notifications. Please adjust it according to your needs, setting `0` indicate no cooldown. 

During cooldown period, consecutive runs of the step will cause `throttling_status` output to be set `True`
and no notification will be sent.

!!! warning "Cooldown limitations"

    Current implementation of cooldown is limited to video processing - using this block in context of a 
    Workflow that is run behind HTTP service (Roboflow Hosted API, Dedicated Deployment or self-hosted 
    `inference` server) will have no effect for processing HTTP requests.  
    

### Async execution

Configure the `fire_and_forget` property. Set it to True if you want the request to be sent in the background, 
allowing the Workflow to proceed without waiting on data to be sent. In this case you will not be able to rely on 
`error_status` output which will always be set to `False`, so we **recommend setting the `fire_and_forget=False` for
debugging purposes**.

### Disabling notifications based on runtime parameter

Sometimes it would be convenient to manually disable the **Webhook sink** block. This is possible 
setting `disable_sink` flag to hold reference to Workflow input. with such setup, caller would be
able to disable the sink when needed sending agreed input parameter.
"""

QUERY_PARAMS_KIND = [
    STRING_KIND,
    INTEGER_KIND,
    FLOAT_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    ROBOFLOW_API_KEY_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    LIST_OF_VALUES_KIND,
    BOOLEAN_KIND,
    TOP_CLASS_KIND,
]
HEADER_KIND = [
    STRING_KIND,
    INTEGER_KIND,
    FLOAT_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    ROBOFLOW_API_KEY_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    BOOLEAN_KIND,
    TOP_CLASS_KIND,
]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Webhook Sink",
            "version": "v1",
            "short_description": "Send a request to a remote API with Workflow results.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-webhook",
                "blockPriority": 1,
                "popular": True,
            },
        }
    )
    type: Literal["roboflow_core/webhook_sink@v1"]
    url: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="URL of the resource to make request",
    )
    method: Literal["GET", "POST", "PUT"] = Field(
        description="HTTP method to be used",
    )
    query_parameters: Dict[
        str,
        Union[
            Selector(kind=QUERY_PARAMS_KIND),
            Selector(kind=QUERY_PARAMS_KIND),
            str,
            float,
            bool,
            int,
            List[Union[str, float, int, bool]],
        ],
    ] = Field(
        description="Request query parameters",
        default_factory=dict,
        examples=[{"api_key": "$inputs.api_key"}],
    )
    headers: Dict[
        str,
        Union[
            Selector(kind=HEADER_KIND),
            Selector(kind=HEADER_KIND),
            str,
            float,
            bool,
            int,
        ],
    ] = Field(
        description="Request headers",
        default_factory=dict,
        examples=[{"api_key": "$inputs.api_key"}],
    )
    json_payload: Dict[
        str,
        Union[
            Selector(),
            Selector(),
            str,
            float,
            bool,
            int,
            dict,
            list,
        ],
    ] = Field(
        description="Fields to put into JSON payload",
        default_factory=dict,
        examples=[{"field": "$steps.model.predictions"}],
    )
    json_payload_operations: Dict[str, List[AllOperationsType]] = Field(
        description="UQL definitions of operations to be performed on defined data w.r.t. each value of "
        "`json_payload` parameter",
        examples=[
            {
                "predictions": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            }
        ],
        default_factory=dict,
    )
    multi_part_encoded_files: Dict[
        str,
        Union[
            Selector(),
            Selector(),
            str,
            float,
            bool,
            int,
            dict,
            list,
        ],
    ] = Field(
        description="Data to POST as Multipart-Encoded File",
        default_factory=dict,
        examples=[{"image": "$steps.visualization.image"}],
    )
    multi_part_encoded_files_operations: Dict[str, List[AllOperationsType]] = Field(
        description="UQL definitions of operations to be performed on defined data w.r.t. each value of "
        "`multi_part_encoded_files` parameter",
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
        default_factory=dict,
    )
    form_data: Dict[
        str,
        Union[
            Selector(),
            Selector(),
            str,
            float,
            bool,
            int,
            dict,
            list,
        ],
    ] = Field(
        description="Fields to put into form-data",
        default_factory=dict,
        examples=[{"field": "$inputs.field_value"}],
    )
    form_data_operations: Dict[str, List[AllOperationsType]] = Field(
        description="UQL definitions of operations to be performed on defined data w.r.t. each value of "
        "`form_data` parameter",
        examples=[
            {
                "predictions": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            }
        ],
        default_factory=dict,
    )
    request_timeout: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=2,
        description="Number of seconds to wait for remote API response",
        examples=["$inputs.request_timeout", 10],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to run the block asynchronously (True) for faster workflows or  "
        "synchronously (False) for debugging and error handling.",
        examples=["$inputs.fire_and_forget", True],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Boolean flag to disable block execution.",
        examples=[False, "$inputs.disable_email_notifications"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Number of seconds to wait until follow-up notification can be sent.",
        json_schema_extra={
            "always_visible": True,
        },
        examples=["$inputs.cooldown_seconds", 10],
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
        return ">=1.3.0,<2.0.0"


class WebhookSinkBlockV1(WorkflowBlock):

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
        url: str,
        method: Literal["GET", "POST", "PUT"],
        query_parameters: Dict[str, Any],
        headers: Dict[str, Any],
        json_payload: Dict[str, Any],
        json_payload_operations: Dict[str, List[AllOperationsType]],
        multi_part_encoded_files: Dict[str, Any],
        multi_part_encoded_files_operations: Dict[str, List[AllOperationsType]],
        form_data: Dict[str, Any],
        form_data_operations: Dict[str, List[AllOperationsType]],
        request_timeout: int,
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
            logging.info(f"Activated `roboflow_core/webhook_notification@v1` cooldown.")
            return {
                "error_status": False,
                "throttling_status": True,
                "message": "Sink cooldown applies",
            }
        json_payload = execute_operations_on_parameters(
            parameters=json_payload,
            operations=json_payload_operations,
        )
        multi_part_encoded_files = execute_operations_on_parameters(
            parameters=multi_part_encoded_files,
            operations=multi_part_encoded_files_operations,
        )
        form_data = execute_operations_on_parameters(
            parameters=form_data,
            operations=form_data_operations,
        )
        request_handler = partial(
            execute_request,
            url=url,
            method=method,
            query_parameters=query_parameters,
            headers=headers,
            json_payload=json_payload,
            multi_part_encoded_files=multi_part_encoded_files,
            form_data=form_data,
            timeout=request_timeout,
        )
        self._last_notification_fired = datetime.now()
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(request_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Notification sent in the background task",
            }
        if fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(request_handler)
            return {
                "error_status": False,
                "throttling_status": False,
                "message": "Notification sent in the background task",
            }
        error_status, message = request_handler()
        return {
            "error_status": error_status,
            "throttling_status": False,
            "message": message,
        }


def execute_operations_on_parameters(
    parameters: Dict[str, Any],
    operations: Dict[str, List[AllOperationsType]],
) -> Dict[str, Any]:
    parameters = copy(parameters)
    for parameter_name, operations in operations.items():
        if not operations or parameter_name not in parameters:
            continue
        operations_chain = build_operations_chain(operations=operations)
        parameters[parameter_name] = operations_chain(
            parameters[parameter_name], global_parameters={}
        )
    return parameters


def execute_request(
    url: str,
    method: Literal["GET", "POST", "PUT"],
    query_parameters: Dict[str, Any],
    headers: Dict[str, Any],
    json_payload: Dict[str, Any],
    form_data: Dict[str, Any],
    multi_part_encoded_files: Dict[str, Any],
    timeout: int,
) -> Tuple[bool, str]:
    try:
        _execute_request(
            url=url,
            method=method,
            query_parameters=query_parameters,
            headers=headers,
            json_payload=json_payload,
            multi_part_encoded_files=multi_part_encoded_files,
            form_data=form_data,
            timeout=timeout,
        )
        return False, "Notification sent successfully"
    except Exception as error:
        logging.warning(f"Could not send webhook notification. Error: {str(error)}")
        return (
            True,
            f"Failed to send webhook notification. Internal error details: {error}",
        )


METHOD_TO_HANDLER = {"GET": requests.get, "POST": requests.post, "PUT": requests.put}


def _execute_request(
    url: str,
    method: Literal["GET", "POST", "PUT"],
    query_parameters: Dict[str, Any],
    headers: Dict[str, Any],
    json_payload: Dict[str, Any],
    form_data: Dict[str, Any],
    multi_part_encoded_files: Dict[str, Any],
    timeout: int,
) -> None:
    handler = METHOD_TO_HANDLER.get(method)
    if handler is None:
        raise ValueError(f"Handler for HTTP method `{method}` not registered")
    response = handler(
        url,
        params=query_parameters,
        headers=headers,
        json=json_payload,
        files=multi_part_encoded_files,
        data=form_data,
        timeout=timeout,
    )
    response.raise_for_status()
