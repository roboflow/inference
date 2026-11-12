import asyncio
import os
import socket
import threading
from typing import Any, Optional

import pytest
from asyncua import Server, ua
from asyncua.common.callback import CallbackType
from roboflow_workflows.enterprise_blocks.sinks.opc_writer.v1 import (
    get_connection_manager,
)

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.conftest import bool_env

NAMESPACE = "http://roboflow.test/opc-writer-timeout"
# The server answers each Write this late: after the 1 s `timeout` of the workflow below,
# but within the 4 s that asyncua gives each request by default, which is what the block
# waited, whatever its `timeout` said, until it honored the field.
WRITE_ANSWER_DELAY_SECONDS = 3.5

WORKFLOW_OPC_WRITER_WITH_TIMEOUT = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceParameter", "name": "opc_url"},
    ],
    "steps": [
        {
            "type": "roboflow_enterprise/opc_writer_sink@v1",
            "name": "opc_writer",
            "url": "$inputs.opc_url",
            "namespace": NAMESPACE,
            "object_name": "Line1",
            "variable_name": "Count",
            "value": 7,
            "value_type": "Int32",
            "timeout": 1,
            "fire_and_forget": False,
            "max_retries": 1,
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


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class SlowWriteServer:
    """An asyncua server, on its own event loop thread, that answers each Write from a
    client `WRITE_ANSWER_DELAY_SECONDS` late. Only the answer waits: the server loop
    keeps running, so the client's own timeout is all that ends the wait."""

    def __init__(self) -> None:
        self.url = f"opc.tcp://127.0.0.1:{free_port()}/opc-writer-timeout/"
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._server: Optional[Server] = None
        self._released: Optional[asyncio.Event] = None

    def _call(self, coroutine) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(60)

    def start(self) -> None:
        self._call(self._start())

    async def _start(self) -> None:
        self._released = asyncio.Event()
        server = Server()
        await server.init()
        server.set_endpoint(self.url)
        idx = await server.register_namespace(NAMESPACE)
        line = await server.nodes.objects.add_object(idx, "Line1")
        count = await line.add_variable(
            idx, "Count", 0, varianttype=ua.VariantType.Int32
        )
        await count.set_writable()
        server.subscribe_server_callback(CallbackType.PreWrite, self._answer_late)
        await server.start()
        self._server = server

    async def _answer_late(self, event, _dispatcher) -> None:
        # The server's own writes (during setup) are not held
        if not event.is_external:
            return

        try:
            await asyncio.wait_for(
                self._released.wait(), timeout=WRITE_ANSWER_DELAY_SECONDS
            )
        except asyncio.TimeoutError:
            pass

    def shutdown(self) -> None:
        # Answer a held Write now rather than make the shutdown wait for it
        self._loop.call_soon_threadsafe(self._released.set)
        if self._server is not None:
            self._call(self._server.stop())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


@pytest.fixture(scope="module")
def slow_write_server():
    server = SlowWriteServer()
    server.start()
    yield server
    # Close the pooled session before its server goes away
    get_connection_manager().invalidate_connection(server.url, None, None)
    server.shutdown()


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_OPC_WRITER_TEST", False)),
    reason="Skipping OPC writer test due to flag `SKIP_OPC_WRITER_TEST`",
)
@pytest.mark.timeout(30)
def test_workflow_with_opc_writer_stops_waiting_for_the_server_after_its_timeout(
    slow_write_server,
) -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_OPC_WRITER_WITH_TIMEOUT,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"opc_url": slow_write_server.url})

    # then
    assert result[0]["opc_writer_results"]["error_status"] is True
    assert result[0]["opc_writer_results"]["message"].startswith("Failed to write 7")
