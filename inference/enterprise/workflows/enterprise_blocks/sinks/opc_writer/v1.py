import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import List, Literal, Optional, Tuple, Type, Union

from asyncua import ua
from asyncua.client import Client as AsyncClient
from asyncua.sync import Client, sync_async_client_method
from asyncua.ua import VariantType
from asyncua.ua.uaerrors import BadNoMatch, BadTypeMismatch, BadUserAccessDenied
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field
from inference.core.logger import logger


class UnsupportedTypeError(Exception):
    """Raised when an unsupported value type is specified"""

    pass


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

### Node Lookup Mode
The block supports two methods for locating OPC UA nodes via the `node_lookup_mode` parameter:

- **`hierarchical` (default)**: Uses standard OPC UA hierarchical path navigation. The block navigates
  through the address space using `get_child()`. Each component in the `object_name` path is
  automatically prefixed with the namespace index.
  - **Example**: `object_name="Roboflow/Crane_11"` → path `0:Objects/2:Roboflow/2:Crane_11/2:Variable`
  - **Best for**: Traditional OPC UA servers with hierarchical address spaces

- **`direct`**: Uses direct NodeId string access. The block constructs a NodeId as
  `ns={namespace};s={object_name}/{variable_name}` and accesses it directly via `get_node()`.
  - **Example**: `object_name="[Sample_Tags]/Ramp"` → NodeId `ns=2;s=[Sample_Tags]/Ramp/South_Person_Count`
  - **Best for**: Ignition SCADA systems and other servers using string-based NodeId identifiers

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
            "ui_manifest": {
                "section": "industrial",
                "icon": "fal fa-industry",
                "blockPriority": 11,
                "enterprise_only": True,
                "local_only": True,
            },
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
        Literal[
            "Boolean",
            "Double",
            "Float",
            "Int16",
            "Int32",
            "Int64",
            "Integer",
            "SByte",
            "String",
            "UInt16",
            "UInt32",
            "UInt64",
        ],
    ] = Field(
        default="String",
        description="The type of the value to be written to the target variable on the OPC UA server. "
        "Supported types: Boolean, Double, Float, Int16, Int32, Int64, Integer (Int64 alias), SByte, String, UInt16, UInt32, UInt64.",
        examples=["Boolean", "Double", "Float", "Int32", "Int64", "String"],
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
    node_lookup_mode: Union[
        Selector(kind=[STRING_KIND]),
        Literal["hierarchical", "direct"],
    ] = Field(
        default="hierarchical",
        description="Method to locate the OPC UA node: 'hierarchical' uses path navigation, "
        "'direct' uses NodeId strings (for Ignition-style string-based tags).",
        examples=["hierarchical", "direct"],
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
        value_type: Literal[
            "Boolean",
            "Double",
            "Float",
            "Int16",
            "Int32",
            "Int64",
            "Integer",
            "SByte",
            "String",
            "UInt16",
            "UInt32",
            "UInt64",
        ] = "String",
        timeout: int = 2,
        fire_and_forget: bool = True,
        disable_sink: bool = False,
        cooldown_seconds: int = 5,
        node_lookup_mode: Literal["hierarchical", "direct"] = "hierarchical",
    ) -> BlockResult:
        if disable_sink:
            logger.debug("OPC Writer disabled by disable_sink parameter")
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
            logger.info(f"Activated `{BLOCK_TYPE}` cooldown.")
            return {
                "disabled": False,
                "throttling_status": True,
                "error_status": False,
                "message": "Sink cooldown applies",
            }

        if value_type in [BOOLEAN_KIND, "Boolean"] and isinstance(value, str):
            # handle boolean conversion explicitly if value is a string
            decoded_value = value.strip().lower() in ("true", "1")
        else:
            # Use value directly - OPC UA library will convert based on type specification
            decoded_value = value

        logger.debug(
            f"OPC Writer prepared value '{decoded_value}' for type {value_type}"
        )

        opc_writer_handler = partial(
            opc_connect_and_write_value,
            url=url,
            namespace=namespace,
            user_name=user_name,
            password=password,
            object_name=object_name,
            variable_name=variable_name,
            value=decoded_value,
            value_type=value_type,
            timeout=timeout,
            node_lookup_mode=node_lookup_mode,
        )
        self._last_notification_fired = datetime.now()
        if fire_and_forget and self._background_tasks:
            logger.debug("OPC Writer submitting write task to background tasks")
            self._background_tasks.add_task(opc_writer_handler)
            return {
                "disabled": False,
                "error_status": False,
                "throttling_status": False,
                "message": "Writing to the OPC UA server in the background task",
            }
        if fire_and_forget and self._thread_pool_executor:
            logger.debug("OPC Writer submitting write task to thread pool executor")
            self._thread_pool_executor.submit(opc_writer_handler)
            return {
                "disabled": False,
                "error_status": False,
                "throttling_status": False,
                "message": "Writing to the OPC UA server in the background task",
            }
        logger.debug("OPC Writer executing synchronous write")
        error_status, message = opc_writer_handler()
        logger.debug(
            f"OPC Writer write completed: error_status={error_status}, message={message}"
        )
        return {
            "disabled": False,
            "error_status": error_status,
            "throttling_status": False,
            "message": message,
        }


def get_available_namespaces(client: Client) -> List[str]:
    """
    Get list of available namespaces from OPC UA server.
    Returns empty list if unable to fetch namespaces.
    """
    try:
        get_namespace_array = sync_async_client_method(AsyncClient.get_namespace_array)(
            client
        )
        return get_namespace_array()
    except Exception as exc:
        logger.info(f"Failed to get namespace array (non-fatal): {exc}")
        return ["<unable to fetch namespaces>"]


def safe_disconnect(client: Client) -> None:
    """Safely disconnect from OPC UA server, swallowing any errors"""
    try:
        logger.debug("OPC Writer disconnecting from server")
        client.disconnect()
    except Exception as exc:
        logger.debug(f"OPC Writer disconnect error (non-fatal): {exc}")


def get_node_data_type(var) -> str:
    """
    Get the data type of an OPC UA node.
    Returns a string representation of the type, or "Unknown" if unable to read.
    """
    try:
        return str(var.read_data_type_as_variant_type())
    except Exception as exc:
        logger.info(f"Unable to read node data type: {exc}")
        return "Unknown"


def opc_connect_and_write_value(
    url: str,
    namespace: str,
    user_name: Optional[str],
    password: Optional[str],
    object_name: str,
    variable_name: str,
    value: Union[bool, float, int, str],
    timeout: int,
    node_lookup_mode: Literal["hierarchical", "direct"] = "hierarchical",
    value_type: str = "String",
) -> Tuple[bool, str]:
    logger.debug(
        f"OPC Writer attempting to connect and write value={value} to {url}/{object_name}/{variable_name}"
    )
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
            node_lookup_mode=node_lookup_mode,
            value_type=value_type,
        )
        logger.debug(
            f"OPC Writer successfully wrote value to {url}/{object_name}/{variable_name}"
        )
        return False, "Value set successfully"
    except Exception as exc:
        logger.error(f"OPC Writer failed to write value: {exc}")
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
    node_lookup_mode: Literal["hierarchical", "direct"] = "hierarchical",
    value_type: str = "String",
):
    logger.debug(f"OPC Writer creating client for {url} with timeout={timeout}")
    client = Client(url=url, sync_wrapper_timeout=timeout)
    if user_name and password:
        client.set_user(user_name)
        client.set_password(password)
    try:
        logger.debug(f"OPC Writer connecting to {url}")
        client.connect()
        logger.debug("OPC Writer successfully connected to server")
    except BadUserAccessDenied as exc:
        logger.error(f"OPC Writer authentication failed: {exc}")
        safe_disconnect(client)
        raise Exception(f"AUTH ERROR: {exc}")
    except OSError as exc:
        logger.error(f"OPC Writer network error during connection: {exc}")
        safe_disconnect(client)
        raise Exception(f"NETWORK ERROR: {exc}")
    except Exception as exc:
        logger.error(f"OPC Writer unhandled connection error: {type(exc)} {exc}")
        safe_disconnect(client)
        raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}")
    get_namespace_index = sync_async_client_method(AsyncClient.get_namespace_index)(
        client
    )

    try:
        if namespace.isdigit():
            nsidx = int(namespace)
            logger.debug(f"OPC Writer using numeric namespace index: {nsidx}")
        else:
            nsidx = get_namespace_index(namespace)
    except ValueError as exc:
        namespaces = get_available_namespaces(client)
        logger.error(f"OPC Writer invalid namespace: {exc}")
        logger.error(f"Available namespaces: {namespaces}")
        safe_disconnect(client)
        raise Exception(
            f"WRONG NAMESPACE ERROR: {exc}. Available namespaces: {namespaces}"
        )
    except Exception as exc:
        namespaces = get_available_namespaces(client)
        logger.error(f"OPC Writer unhandled namespace error: {type(exc)} {exc}")
        logger.error(f"Available namespaces: {namespaces}")
        safe_disconnect(client)
        raise Exception(
            f"UNHANDLED ERROR: {type(exc)} {exc}. Available namespaces: {namespaces}"
        )

    if node_lookup_mode == "direct":
        # Direct NodeId access for Ignition-style string identifiers
        try:
            node_id = f"ns={nsidx};s={object_name}/{variable_name}"
            logger.debug(f"OPC Writer using direct NodeId access: {node_id}")
            var = client.get_node(node_id)
            logger.debug(
                f"OPC Writer successfully found variable node with type {vt} using direct NodeId"
            )
        except Exception as exc:
            logger.error(f"OPC Writer direct NodeId access failed: {exc}")
            safe_disconnect(client)
            raise Exception(
                f"WRONG OBJECT OR PROPERTY ERROR: Could not find node with direct NodeId '{node_id}'. Error: {exc}"
            )
    else:
        # Hierarchical path navigation (standard OPC UA)
        try:
            # Split object_name on "/" and prepend namespace index to each component
            object_components = object_name.split("/")
            object_path = "/".join([f"{nsidx}:{comp}" for comp in object_components])
            node_path = f"0:Objects/{object_path}/{nsidx}:{variable_name}"
            logger.debug(f"OPC Writer using hierarchical path: {node_path}")
            var = client.nodes.root.get_child(node_path)
            logger.debug(
                f"OPC Writer successfully found variable node using hierarchical path"
            )
        except BadNoMatch as exc:
            logger.error(f"OPC Writer hierarchical path not found: {exc}")
            safe_disconnect(client)
            raise Exception(
                f"WRONG OBJECT OR PROPERTY ERROR: Could not find node at hierarchical path '{node_path}'. Error: {exc}"
            )
        except Exception as exc:
            logger.error(f"OPC Writer unhandled node lookup error: {type(exc)} {exc}")
            safe_disconnect(client)
            raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}")

    try:
        logger.debug(
            f"OPC Writer writing value '{value}' to variable with type '{value_type}'"
        )
        # Convert to primitive types before setting value
        if value_type in [BOOLEAN_KIND, "Boolean"]:
            var.set_value(bool(value), VariantType.Boolean)
        elif value_type == "Double":
            var.set_value(float(value), VariantType.Double)
        elif value_type in [FLOAT_KIND, "Float"]:
            var.set_value(float(value), VariantType.Float)
        elif value_type == "Int16":
            var.set_value(int(value), VariantType.Int16)
        elif value_type == "Int32":
            var.set_value(int(value), VariantType.Int32)
        elif value_type in ["Int64", INTEGER_KIND, "Integer"]:
            var.set_value(int(value), VariantType.Int64)
        elif value_type == "SByte":
            var.set_value(int(value), VariantType.SByte)
        elif value_type in [STRING_KIND, "String"]:
            var.set_value(str(value), VariantType.String)
        elif value_type == "UInt16":
            var.set_value(int(value), VariantType.UInt16)
        elif value_type == "UInt32":
            var.set_value(int(value), VariantType.UInt32)
        elif value_type == "UInt64":
            var.set_value(int(value), VariantType.UInt64)
        else:
            logger.error(f"OPC Writer unsupported value type: {value_type}")
            safe_disconnect(client)
            raise UnsupportedTypeError(f"Value type '{value_type}' is not supported.")
        logger.info(
            f"OPC Writer successfully wrote  '{value}'  to variable at {object_name}/{variable_name}"
        )
    except UnsupportedTypeError:
        raise
    except BadTypeMismatch as exc:
        node_type = get_node_data_type(var)
        logger.error(
            f"OPC Writer type mismatch: tried to write value '{value}' (type: {type(value).__name__}) to node with data type {node_type}. Error: {exc}"
        )
        safe_disconnect(client)
        raise Exception(
            f"WRONG TYPE ERROR: Tried to write value '{value}' (type: {type(value).__name__}) but node expects type {node_type}. {exc}"
        )
    except Exception as exc:
        logger.error(f"OPC Writer unhandled write error: {type(exc)} {exc}")
        safe_disconnect(client)
        raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}")

    safe_disconnect(client)
