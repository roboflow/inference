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
The **OPC UA Writer** block enables you to write data to a variable on an OPC UA server, leveraging the 
[asyncua](https://github.com/FreeOpcUa/opcua-asyncio) library for seamless communication.

### Supported Data Types
This block supports writing the following data types to OPC UA server variables:
- Numbers (integers, floats)
- Booleans
- Strings

**Note:** The data type you send must match the expected type of the target OPC UA variable.

### Cooldown
To prevent excessive traffic to the OPC UA server, the block includes a `cooldown_seconds` parameter, 
which defaults to **5 seconds**. During the cooldown period:
- Consecutive executions of the block will set the `throttling_status` output to `True`.
- No data will be sent to the server.

You can customize the `cooldown_seconds` parameter based on your needs. Setting it to `0` disables 
the cooldown entirely.

### Asynchronous Execution
The block provides a `fire_and_forget` property for asynchronous execution:
- **When `fire_and_forget=True`**: The block sends data in the background, allowing the Workflow to 
  proceed immediately. However, the `error_status` output will always be set to `False`, so we do not 
  recommend this mode for debugging.
- **When `fire_and_forget=False`**: The block waits for confirmation before proceeding, ensuring errors 
  are captured in the `error_status` output.

### Disabling the Block Dynamically
You can disable the **OPC UA Writer** block during execution by linking the `disable_sink` parameter 
to a Workflow input. By providing a specific input value, you can dynamically prevent the block from 
executing.

### Cooldown Limitations
!!! warning "Cooldown Limitations"
    The cooldown feature is optimized for workflows involving video processing.  
    - In other contexts, such as Workflows triggered by HTTP services (e.g., Roboflow Hosted API, 
      Dedicated Deployment, or self-hosted `Inference` server), the cooldown timer will not be applied effectively.
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
            "name": "OPC UA Writer Sink",
            "version": "v1",
            "short_description": "Writes data to an OPC UA server using the [asyncua](https://github.com/FreeOpcUa/opcua-asyncio) library for communication.",
            "long_description": LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "sink",
        }
    )
    type: Literal[BLOCK_TYPE]
    url: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="URL of the OPC UA server to which data will be written.",
        examples=["opc.tcp://localhost:4840/freeopcua/server/", "$inputs.opc_url"],
    )
    namespace: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The OPC UA namespace URI or index used to locate objects and variables.",
        examples=["http://examples.freeopcua.github.io", "2", "$inputs.opc_namespace"],
    )
    user_name: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Optional username for authentication when connecting to the OPC UA server.",
        examples=["John", "$inputs.opc_user_name"],
    )
    password: Optional[Union[str, Selector(kind=[STRING_KIND])]] = Field(
        default=None,
        description="Optional password for authentication when connecting to the OPC UA server.",
        examples=["secret", "$inputs.opc_password"],
    )
    object_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The name of the target object in the namespace to search for.",
        examples=["Line1", "$inputs.opc_object_name"],
    )
    variable_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The name of the variable within the target object to be updated.",
        examples=[
            "InspectionSuccess",
            "$inputs.opc_variable_name",
        ],
    )
    value: Union[
        Selector(kind=[BOOLEAN_KIND, FLOAT_KIND, INTEGER_KIND, STRING_KIND]),
        str,
        bool,
        float,
        int,
    ] = Field(
        description="The value to be written to the target variable on the OPC UA server.",
        examples=["running", "$other_block.result"],
    )
    value_type: Union[
        Selector(kind=[STRING_KIND]),
        Literal["Boolean", "Float", "Integer", "String"],
    ] = Field(
        default="String",
        description="The type of the value to be written to the target variable on the OPC UA server.",
        examples=["Boolean", "Float", "Integer", "String"],
        json_schema_extra={
            "always_visible": True,
        },
    )
    timeout: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=2,
        description="The number of seconds to wait for a response from the OPC UA server before timing out.",
        examples=[10, "$inputs.timeout"],
    )
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to run the block asynchronously (True) for faster workflows or  "
        "synchronously (False) for debugging and error handling.",
        examples=[True, "$inputs.fire_and_forget"],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Boolean flag to disable block execution.",
        examples=[False, "$inputs.disable_opc_writers"],
    )
    cooldown_seconds: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="The minimum number of seconds to wait between consecutive updates to the OPC UA server.",
        json_schema_extra={
            "always_visible": True,
        },
        examples=[10, "$inputs.cooldown_seconds"],
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
        value: Union[str, bool, float, int],
        value_type: Literal["Boolean", "Float", "Integer", "String"] = "String",
        timeout: int = 2,
        fire_and_forget: bool = True,
        disable_sink: bool = False,
        cooldown_seconds: int = 5,
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

        value_str = str(value)
        try:
            if value_type in [BOOLEAN_KIND, "Boolean"]:
                decoded_value = value_str.strip().lower() in ("true", "1")
            elif value_type in [FLOAT_KIND, "Float"]:
                decoded_value = float(value_str)
            elif value_type in [INTEGER_KIND, "Integer"]:
                decoded_value = int(value_str)
            elif value_type in [STRING_KIND, "String"]:
                decoded_value = value_str
            else:
                raise ValueError(f"Unsupported value type: {value_type}")
        except ValueError as exc:
            return {
                "disabled": False,
                "error_status": True,
                "throttling_status": False,
                "message": f"Failed to convert value: {exc}",
            }

        opc_writer_handler = partial(
            opc_connect_and_write_value,
            url=url,
            namespace=namespace,
            user_name=user_name,
            password=password,
            object_name=object_name,
            variable_name=variable_name,
            value=decoded_value,
            timeout=timeout,
        )
        self._last_notification_fired = datetime.now()
        if fire_and_forget and self._background_tasks:
            self._background_tasks.add_task(opc_writer_handler)
            return {
                "disabled": False,
                "error_status": False,
                "throttling_status": False,
                "message": "Writing to the OPC UA server in the background task",
            }
        if fire_and_forget and self._thread_pool_executor:
            self._thread_pool_executor.submit(opc_writer_handler)
            return {
                "disabled": False,
                "error_status": False,
                "throttling_status": False,
                "message": "Writing to the OPC UA server in the background task",
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
        if namespace.isdigit():
            nsidx = int(namespace)
        else:
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
