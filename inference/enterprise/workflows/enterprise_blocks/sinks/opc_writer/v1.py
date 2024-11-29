import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import List, Literal, Optional, Tuple, Type, Union

from asyncua.client import Client as AsyncClient
from asyncua.sync import Client, sync_async_client_method
from asyncua.ua.uaerrors import BadNoMatch, BadTypeMismatch, BadUserAccessDenied
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

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

BLOCK_TYPE = "roboflow_enterprise/opc_writer_sink@v1"
LONG_DESCRIPTION = """
The **OPC Writer** block enables sending a data from Workflow into OPC server
by setting value of OPC object under OPC namespace.

This block is making use of [asyncua](https://github.com/FreeOpcUa/opcua-asyncio) in order to
perform communication with OPC servers.

Block will attempt to send:

* numbers (integers, floats)

* booleans

* strings

Type of sent data must match type of OPC object.

### Cooldown

The block accepts `cooldown_seconds` (which **defaults to `5` seconds**) to prevent unintended bursts of 
traffic sent to OPC server. Please adjust it according to your needs, setting `0` indicate no cooldown. 

During cooldown period, consecutive runs of the step will cause `throttling_status` output to be set `True`
and no data will be sent.

### Async execution

Configure the `fire_and_forget` property. Set it to True if you want the data to be sent in the background, 
allowing the Workflow to proceed without waiting on data to be sent. In this case you will not be able to rely on 
`error_status` output which will always be set to `False`, so we **recommend setting the `fire_and_forget=False` for
debugging purposes**.

### Disabling notifications based on runtime parameter

Sometimes it would be convenient to manually disable the **OPC Writer** block. This can be achieved by
setting `disable_sink` flag to hold reference to Workflow input. With such setup, caller cat disable the sink
by sending agreed input parameter.

!!! warning "Cooldown limitations"
    Current implementation of cooldown is limited to video processing - using this block in context of a 
    Workflow that is run behind HTTP service (Roboflow Hosted API, Dedicated Deployment or self-hosted 
    `inference` server) will have no effect with regards to cooldown timer.
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
            "name": "OPC Writer Sink",
            "version": "v1",
            "short_description": "Pushes data to OPC server, this block is making use of [asyncua](https://github.com/FreeOpcUa/opcua-asyncio)",
            "long_description": LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "sink",
        }
    )
    type: Literal[BLOCK_TYPE]
    url: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="URL of OPC server where data should be pushed to",
        examples=["$inputs.opc_url", "opc.tcp://localhost:4840/freeopcua/server/"],
    )
    namespace: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="OPC namespace",
        examples=["$inputs.opc_namespace", "http://examples.freeopcua.github.io"],
    )
    user_name: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Optional user name to be used for authentication when connecting to OPC server",
        examples=["$inputs.opc_user_name", "John"],
    )
    password: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Optional password to be used for authentication when connecting to OPC server",
        examples=["$inputs.opc_password", "secret"],
    )
    object_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Name of object to be searched in namespace",
        examples=["$inputs.opc_object_name", "Line1"],
    )
    variable_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Name of variable to be set under found object",
        examples=[
            "$inputs.opc_variable_name",
            "InspectionSuccess",
        ],
    )
    value: Union[
        Selector(kind=[BOOLEAN_KIND, FLOAT_KIND, INTEGER_KIND, STRING_KIND]),
        Union[bool, float, int, str],
    ] = Field(
        description="value to be written into variable",
        examples=["$other_block.result", "running"],
    )
    timeout: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=2,
        description="Number of seconds to wait for OPC server to respond",
        examples=["$inputs.timeout", 10],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag dictating if sink is supposed to be executed in the background, "
        "not waiting on status of registration before end of workflow run. Use `True` if best-effort "
        "registration is needed, use `False` while debugging and if error handling is needed",
        examples=["$inputs.fire_and_forget", True],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="boolean flag that can be also reference to input - to arbitrarily disable "
        "data collection for specific request",
        examples=[False, "$inputs.disable_opc_writers"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Number of seconds to wait until next value update can be sent",
        json_schema_extra={
            "always_visible": True,
        },
        examples=["$inputs.cooldown_seconds", 10],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="disabled", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="throttling_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class OPCWriterSinkBlockV1(WorkflowBlock):

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
        namespace: str,
        user_name: Optional[str],
        password: Optional[str],
        object_name: str,
        variable_name: str,
        value: Union[bool, float, int, str],
        timeout: int,
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: int,
    ) -> BlockResult:
        if disable_sink:
            return {
                "disabled": True,
                "throttling_status": False,
                "error_status": False,
                "message": "Sink was disabled by parameter `disable_sink`",
            }
        seconds_since_last_notification = cooldown_seconds
        if self._last_notification_fired is not None:
            seconds_since_last_notification = (
                datetime.now() - self._last_notification_fired
            ).total_seconds()
        if seconds_since_last_notification < cooldown_seconds:
            logging.info(f"Activated `{BLOCK_TYPE}` cooldown.")
            return {
                "disabled": False,
                "throttling_status": True,
                "error_status": False,
                "message": "Sink cooldown applies",
            }
        opc_writer_handler = partial(
            opc_connect_and_write_value,
            url=url,
            namespace=namespace,
            user_name=user_name,
            password=password,
            object_name=object_name,
            variable_name=variable_name,
            value=value,
            timeout=timeout,
        )
        self._last_notification_fired = datetime.now()
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(opc_writer_handler)
            return {
                "disabled": False,
                "error_status": False,
                "throttling_status": False,
                "message": "Writing to OPC in the background task",
            }
        if fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(opc_writer_handler)
            return {
                "disabled": False,
                "error_status": False,
                "throttling_status": False,
                "message": "Writing to OPC in the background task",
            }
        error_status, message = opc_writer_handler()
        return {
            "disabled": False,
            "error_status": error_status,
            "throttling_status": False,
            "message": message,
        }


def opc_connect_and_write_value(
    url: str,
    namespace: str,
    user_name: Optional[str],
    password: Optional[str],
    object_name: str,
    variable_name: str,
    value: Union[bool, float, int, str],
    timeout: int,
) -> Tuple[bool, str]:
    try:
        _opc_connect_and_write_value(
            url=url,
            namespace=namespace,
            user_name=user_name,
            password=password,
            object_name=object_name,
            variable_name=variable_name,
            value=value,
            timeout=timeout,
        )
        return False, "Value set successfully"
    except Exception as exc:
        return (
            True,
            f"Failed to write {value} to {object_name}:{variable_name} in {url}. Internal error details: {exc}.",
        )


def _opc_connect_and_write_value(
    url: str,
    namespace: str,
    user_name: Optional[str],
    password: Optional[str],
    object_name: str,
    variable_name: str,
    value: Union[bool, float, int, str],
    timeout: int,
):
    client = Client(url=url, sync_wrapper_timeout=timeout)
    if user_name and password:
        client.set_user(user_name)
        client.set_password(password)
    try:
        client.connect()
    except BadUserAccessDenied as exc:
        client.disconnect()
        raise Exception(f"AUTH ERROR: {exc}")
    except OSError as exc:
        client.disconnect()
        raise Exception(f"NETWORK ERROR: {exc}")
    except Exception as exc:
        client.disconnect()
        raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}")
    get_namespace_index = sync_async_client_method(AsyncClient.get_namespace_index)(
        client
    )

    try:
        nsidx = get_namespace_index(namespace)
    except ValueError as exc:
        client.disconnect()
        raise Exception(f"WRONG NAMESPACE ERROR: {exc}")
    except Exception as exc:
        client.disconnect()
        raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}")

    try:
        var = client.nodes.root.get_child(
            f"0:Objects/{nsidx}:{object_name}/{nsidx}:{variable_name}"
        )
    except BadNoMatch as exc:
        client.disconnect()
        raise Exception(f"WRONG OBJECT OR PROPERTY ERROR: {exc}")
    except Exception as exc:
        client.disconnect()
        raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}")

    try:
        var.write_value(value)
    except BadTypeMismatch as exc:
        client.disconnect()
        raise Exception(f"WRONG TYPE ERROR: {exc}")
    except Exception as exc:
        client.disconnect()
        raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}")

    client.disconnect()
