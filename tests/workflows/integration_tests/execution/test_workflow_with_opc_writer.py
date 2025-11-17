import asyncio
import threading
import time
from typing import Optional, Union

import pytest
from asyncua import Server
from asyncua.client import Client as AsyncClient

try:
    from asyncua.server.users import User, UserRole
except ImportError:
    from asyncua.crypto.permission_rules import User, UserRole
from asyncua.sync import Client, sync_async_client_method
from asyncua.ua.uaerrors import BadNoMatch, BadTypeMismatch, BadUserAccessDenied
from asyncua.ua import NodeId

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_OPC_WRITER = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceParameter", "name": "opc_url"},
        {"type": "InferenceParameter", "name": "opc_namespace"},
        {"type": "InferenceParameter", "name": "opc_user_name"},
        {"type": "InferenceParameter", "name": "opc_password"},
        {"type": "InferenceParameter", "name": "opc_object_name"},
        {"type": "InferenceParameter", "name": "opc_variable_name"},
        {"type": "InferenceParameter", "name": "opc_value"},
        {"type": "InferenceParameter", "name": "opc_value_type"},
    ],
    "steps": [
        {
            "type": "roboflow_enterprise/opc_writer_sink@v1",
            "name": "opc_writer",
            "url": "$inputs.opc_url",
            "namespace": "$inputs.opc_namespace",
            "user_name": "$inputs.opc_user_name",
            "password": "$inputs.opc_password",
            "object_name": "$inputs.opc_object_name",
            "variable_name": "$inputs.opc_variable_name",
            "value": "$inputs.opc_value",
            "value_type": "$inputs.opc_value_type",
            "fire_and_forget": False,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "opc_writer_results",
            "selector": "$steps.opc_writer.*",
        }
    ],
}

WORKFLOW_OPC_WRITER_DIRECT_MODE = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceParameter", "name": "opc_url"},
        {"type": "InferenceParameter", "name": "opc_namespace"},
        {"type": "InferenceParameter", "name": "opc_user_name"},
        {"type": "InferenceParameter", "name": "opc_password"},
        {"type": "InferenceParameter", "name": "opc_object_name"},
        {"type": "InferenceParameter", "name": "opc_variable_name"},
        {"type": "InferenceParameter", "name": "opc_value"},
        {"type": "InferenceParameter", "name": "opc_value_type"},
    ],
    "steps": [
        {
            "type": "roboflow_enterprise/opc_writer_sink@v1",
            "name": "opc_writer",
            "url": "$inputs.opc_url",
            "namespace": "$inputs.opc_namespace",
            "user_name": "$inputs.opc_user_name",
            "password": "$inputs.opc_password",
            "object_name": "$inputs.opc_object_name",
            "variable_name": "$inputs.opc_variable_name",
            "value": "$inputs.opc_value",
            "value_type": "$inputs.opc_value_type",
            "fire_and_forget": False,
            "node_lookup_mode": "direct",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "opc_writer_results",
            "selector": "$steps.opc_writer.*",
        }
    ],
}


OPC_SERVER_STARTED = False
STOP_OPC_SERVER = False
SERVER_TASK = None


def start_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


users_db = {"user1": "password1"}


class UserManager:
    def get_user(self, iserver, username=None, password=None, certificate=None):
        if username in users_db and password == users_db[username]:
            return User(role=UserRole.User)
        return None


async def start_test_opc_server(
    url: str,
    namespace: str,
    object_name: str,
    variable_name: str,
    initial_value: float,
):
    global OPC_SERVER_STARTED
    global STOP_OPC_SERVER
    global SERVER_TASK

    server = Server(user_manager=UserManager())
    await server.init()
    server.set_endpoint(url)

    uri = namespace
    idx = await server.register_namespace(uri)

    # Support multi-level object hierarchy by splitting on "/"
    object_components = object_name.split("/")
    current_obj = server.nodes.objects

    # Create nested objects for each component in the path
    for component in object_components:
        current_obj = await current_obj.add_object(idx, component)

    # Add the main variable to the deepest object
    myvar = await current_obj.add_variable(idx, variable_name, initial_value)
    await myvar.set_writable()

    # Pre-create variables of all types for parametrized tests
    from asyncua.ua import VariantType

    # Boolean variables
    for val in [True, False]:
        var = await current_obj.add_variable(idx, f"BoolVar_{val}", val, varianttype=VariantType.Boolean)
        await var.set_writable()

    # Numeric type variables
    type_configs = [
        ("Double", 0.0, VariantType.Double),
        ("Float", 0.0, VariantType.Float),
        ("Int16", 0, VariantType.Int16),
        ("Int32", 0, VariantType.Int32),
        ("Int64", 0, VariantType.Int64),
        ("SByte", 0, VariantType.SByte),
        ("UInt16", 0, VariantType.UInt16),
        ("UInt32", 0, VariantType.UInt32),
        ("UInt64", 0, VariantType.UInt64),
    ]

    for type_name, init_val, variant_type in type_configs:
        var = await current_obj.add_variable(idx, f"{type_name}Var", init_val, varianttype=variant_type)
        await var.set_writable()

    # String variable
    str_var = await current_obj.add_variable(idx, "StringVar", "", varianttype=VariantType.String)
    await str_var.set_writable()

    # Also create a variable with string-based NodeId for direct mode testing
    # This allows both hierarchical and direct access to work
    direct_node_id = NodeId(f"{object_name}/{variable_name}", idx)
    direct_var = await server.nodes.objects.add_variable(direct_node_id, f"Direct_{variable_name}", initial_value)
    await direct_var.set_writable()

    OPC_SERVER_STARTED = True

    async def run_server():
        async with server:
            while not STOP_OPC_SERVER:
                await asyncio.sleep(0.1)

    loop = asyncio.get_event_loop()
    SERVER_TASK = loop.create_task(run_server())
    try:
        await SERVER_TASK
    except asyncio.CancelledError:
        pass


async def start_test_opc_server_with_string_nodeid(
    url: str,
    namespace: str,
    object_name: str,
    variable_name: str,
    initial_value: float,
):
    """Create an OPC UA server with string-based NodeIds (Ignition-style)"""
    global OPC_SERVER_STARTED
    global STOP_OPC_SERVER
    global SERVER_TASK

    server = Server(user_manager=UserManager())
    await server.init()
    server.set_endpoint(url)

    uri = namespace
    idx = await server.register_namespace(uri)

    # Create a variable with a string-based NodeId directly (like Ignition does)
    node_id = NodeId(f"{object_name}/{variable_name}", idx)
    myvar = await server.nodes.objects.add_variable(node_id, variable_name, initial_value)
    await myvar.set_writable()
    OPC_SERVER_STARTED = True

    async def run_server():
        async with server:
            while not STOP_OPC_SERVER:
                await asyncio.sleep(0.1)

    loop = asyncio.get_event_loop()
    SERVER_TASK = loop.create_task(run_server())
    try:
        await SERVER_TASK
    except asyncio.CancelledError:
        pass


def _opc_connect_and_read_value(
    url: str,
    namespace: str,
    user_name: Optional[str],
    password: Optional[str],
    object_name: str,
    variable_name: str,
    timeout: int,
    direct_mode: bool = False,
) -> Union[bool, float, int, str]:
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
        if direct_mode:
            # Direct NodeId access for Ignition-style string identifiers
            node_id = f"ns={nsidx};s={object_name}/{variable_name}"
            var = client.get_node(node_id)
        else:
            # Hierarchical path navigation - split object_name and prepend namespace to each component
            object_components = object_name.split("/")
            object_path = "/".join([f"{nsidx}:{comp}" for comp in object_components])
            var = client.nodes.root.get_child(
                f"0:Objects/{object_path}/{nsidx}:{variable_name}"
            )
    except BadNoMatch as exc:
        client.disconnect()
        raise Exception(f"WRONG OBJECT OR PROPERTY ERROR: {exc}")
    except Exception as exc:
        client.disconnect()
        raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}")

    try:
        value = var.read_value()
    except BadTypeMismatch as exc:
        client.disconnect()
        raise Exception(f"WRONG TYPE ERROR: {exc}")
    except Exception as exc:
        client.disconnect()
        raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}")

    client.disconnect()
    return value


@pytest.fixture(scope="module")
def test_opc_server():
    """Create one OPC server that will be reused for all tests"""
    global SERVER_TASK, OPC_SERVER_STARTED, STOP_OPC_SERVER

    # Reset global state
    OPC_SERVER_STARTED = False
    STOP_OPC_SERVER = False
    SERVER_TASK = None

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=start_loop, args=(loop,), daemon=True)
    t.start()

    opc_url = "opc.tcp://localhost:4840/freeopcua/server/"
    opc_namespace = "http://examples.freeopcua.github.io"
    opc_user_name = "user1"
    opc_password = users_db[opc_user_name]
    opc_object_name = "MyObject1"
    opc_variable_name = "MyVariable1"
    opc_initial_value = 1

    asyncio.run_coroutine_threadsafe(
        start_test_opc_server(
            url=opc_url,
            namespace=opc_namespace,
            object_name=opc_object_name,
            variable_name=opc_variable_name,
            initial_value=opc_initial_value,
        ),
        loop,
    )

    while not OPC_SERVER_STARTED:
        time.sleep(0.1)

    yield {
        "url": opc_url,
        "namespace": opc_namespace,
        "user_name": opc_user_name,
        "password": opc_password,
        "object_name": opc_object_name,
        "variable_name": opc_variable_name,
        "loop": loop,
        "thread": t,
    }

    # Cleanup
    STOP_OPC_SERVER = True
    if SERVER_TASK:
        SERVER_TASK.cancel()
        try:
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0.1), loop).result()
        except:
            pass
    loop.stop()
    t.join()


@add_to_workflows_gallery(
    category="Basic Workflows",
    use_case_title="Workflow writing data to OPC server",
    use_case_description="""
In this example data is written to OPC server.

In order to write to OPC this block is making use of [asyncua](https://github.com/FreeOpcUa/opcua-asyncio) package.

Writing to OPC enables workflows to expose insights extracted from camera to PLC controllers
allowing factory automation engineers to take advantage of machine vision when building PLC logic.
    """,
    workflow_definition=WORKFLOW_OPC_WRITER,
    workflow_name_in_app="opc_writer",
)
@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "value_type,variable_name,test_value,expected_value",
    [
        ("Boolean", "BoolVar_True", True, True),
        ("Boolean", "BoolVar_True", "true", True),
        ("Boolean", "BoolVar_True", "1", True),
        ("Boolean", "BoolVar_False", False, False),
        ("Boolean", "BoolVar_False", "false", False),
        ("Double", "DoubleVar", 3.14159265359, 3.14159265359),
        ("Float", "FloatVar", 3.14, 3.14),
        ("Int16", "Int16Var", 100, 100),
        ("Int16", "Int16Var", "100", 100),  # string to int conversion
        ("Int32", "Int32Var", 1000, 1000),
        ("Int32", "Int32Var", "1000", 1000),  # string to int conversion
        ("Int64", "Int64Var", 100000, 100000),
        ("Int64", "Int64Var", "100000", 100000),  # string to int conversion
        ("Integer", "Int64Var", 41, 41),  # backwards compatibility
        ("SByte", "SByteVar", -50, -50),
        ("SByte", "SByteVar", "-50", -50),  # string to int conversion
        ("String", "StringVar", "test", "test"),
        ("String", "StringVar", 123, "123"),  # int to string conversion
        ("String", "StringVar", 3.14, "3.14"),  # float to string conversion
        ("UInt16", "UInt16Var", 200, 200),
        ("UInt16", "UInt16Var", "200", 200),  # string to int conversion
        ("UInt32", "UInt32Var", 2000, 2000),
        ("UInt32", "UInt32Var", "2000", 2000),  # string to int conversion
        ("UInt64", "UInt64Var", 200000, 200000),
        ("UInt64", "UInt64Var", "200000", 200000),  # string to int conversion
    ],
)
def test_workflow_with_opc_writer_sink(test_opc_server, value_type, variable_name, test_value, expected_value) -> None:
    # given - use pre-created variables from the server
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_OPC_WRITER,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "opc_url": test_opc_server["url"],
            "opc_namespace": test_opc_server["namespace"],
            "opc_user_name": test_opc_server["user_name"],
            "opc_password": test_opc_server["password"],
            "opc_object_name": test_opc_server["object_name"],
            "opc_variable_name": variable_name,
            "opc_value": test_value,
            "opc_value_type": value_type,
        }
    )

    result_value = _opc_connect_and_read_value(
        url=test_opc_server["url"],
        namespace=test_opc_server["namespace"],
        user_name=test_opc_server["user_name"],
        password=test_opc_server["password"],
        object_name=test_opc_server["object_name"],
        variable_name=variable_name,
        timeout=1,
    )

    assert set(result[0].keys()) == {
        "opc_writer_results",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["opc_writer_results"]["error_status"] == False
    assert result[0]["opc_writer_results"]["disabled"] == False
    assert result[0]["opc_writer_results"]["throttling_status"] == False
    assert result[0]["opc_writer_results"]["message"] == "Value set successfully"

    # For floats, use approximate comparison
    if value_type in ["Float", "Double"]:
        assert abs(result_value - expected_value) < 0.01
    else:
        assert result_value == expected_value


@pytest.mark.timeout(5)
def test_workflow_with_opc_writer_sink_direct_mode(test_opc_server) -> None:
    """Test OPC writer with direct NodeId lookup mode using the shared server"""
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_OPC_WRITER_DIRECT_MODE,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when - use direct mode with the hierarchical server's object
    result = execution_engine.run(
        runtime_parameters={
            "opc_url": test_opc_server["url"],
            "opc_namespace": test_opc_server["namespace"],
            "opc_user_name": test_opc_server["user_name"],
            "opc_password": test_opc_server["password"],
            "opc_object_name": test_opc_server["object_name"],
            "opc_variable_name": test_opc_server["variable_name"],
            "opc_value": 42,
            "opc_value_type": "Integer",
        }
    )

    result_value = _opc_connect_and_read_value(
        url=test_opc_server["url"],
        namespace=test_opc_server["namespace"],
        user_name=test_opc_server["user_name"],
        password=test_opc_server["password"],
        object_name=test_opc_server["object_name"],
        variable_name=test_opc_server["variable_name"],
        timeout=1,
        direct_mode=True,
    )

    # then
    assert set(result[0].keys()) == {
        "opc_writer_results",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["opc_writer_results"]["error_status"] == False
    assert result[0]["opc_writer_results"]["disabled"] == False
    assert result[0]["opc_writer_results"]["throttling_status"] == False
    assert result[0]["opc_writer_results"]["message"] == "Value set successfully"
    assert result_value == 42
