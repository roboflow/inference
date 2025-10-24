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

    myobj = await server.nodes.objects.add_object(idx, object_name)
    myvar = await myobj.add_variable(idx, variable_name, initial_value)
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
        value = var.read_value()
    except BadTypeMismatch as exc:
        client.disconnect()
        raise Exception(f"WRONG TYPE ERROR: {exc}")
    except Exception as exc:
        client.disconnect()
        raise Exception(f"UNHANDLED ERROR: {type(exc)} {exc}")

    client.disconnect()
    return value


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
@pytest.mark.timeout(5)
def test_workflow_with_opc_writer_sink() -> None:
    # given
    global SERVER_TASK
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

    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_OPC_WRITER,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    while not OPC_SERVER_STARTED:
        time.sleep(0.1)

    # when
    result = execution_engine.run(
        runtime_parameters={
            "opc_url": opc_url,
            "opc_namespace": opc_namespace,
            "opc_user_name": opc_user_name,
            "opc_password": opc_password,
            "opc_object_name": opc_object_name,
            "opc_variable_name": opc_variable_name,
            "opc_value": 41,
            "opc_value_type": "Integer",
        }
    )

    result_value = _opc_connect_and_read_value(
        url=opc_url,
        namespace=opc_namespace,
        user_name=opc_user_name,
        password=opc_password,
        object_name=opc_object_name,
        variable_name=opc_variable_name,
        timeout=1,
    )

    STOP_OPC_SERVER = True
    if SERVER_TASK:
        SERVER_TASK.cancel()
        try:
            # Give the server a chance to clean up
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0.1), loop).result()
        except:
            pass
    loop.stop()
    t.join()

    assert set(result[0].keys()) == {
        "opc_writer_results",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["opc_writer_results"]["error_status"] == False
    assert result[0]["opc_writer_results"]["disabled"] == False
    assert result[0]["opc_writer_results"]["throttling_status"] == False
    assert result[0]["opc_writer_results"]["message"] == "Value set successfully"
    assert result_value == 41
