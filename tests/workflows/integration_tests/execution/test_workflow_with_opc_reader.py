import asyncio
import socket
import threading
import time
from typing import Any, Callable, Optional

import pytest
from asyncua import Server, ua

try:
    from asyncua.server.users import User, UserRole
except ImportError:
    from asyncua.crypto.permission_rules import User, UserRole

from roboflow_workflows.enterprise_blocks.sinks.opc_reader.v1 import OPCReaderBlockV1

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine

NAMESPACE = "http://roboflow.test/opc-reader"
USERS = {"operator": "secret"}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class UserManager:
    def get_user(self, iserver, username=None, password=None, certificate=None):
        if username is None:
            return User(role=UserRole.User)
        if USERS.get(username) == password:
            return User(role=UserRole.User)
        return None


class OPCTestServer:
    """An asyncua server on its own event loop thread that can be stopped and restarted
    on the same port, to simulate an outage."""

    def __init__(self):
        self.port = free_port()
        self.url = f"opc.tcp://127.0.0.1:{self.port}/opc-reader-test/"
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._server: Optional[Server] = None
        self.nodes = {}

    def _call(self, coroutine, timeout: float = 60.0) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout)

    def start(self) -> None:
        self._call(self._start())

    async def _start(self) -> None:
        server = Server(user_manager=UserManager())
        await server.init()
        server.set_endpoint(self.url)
        idx = await server.register_namespace(NAMESPACE)
        line = await server.nodes.objects.add_object(idx, "Line1")
        self.nodes["running"] = await line.add_variable(
            idx, "LineRunning", False, varianttype=ua.VariantType.Boolean
        )
        self.nodes["speed"] = await line.add_variable(
            idx, "Speed", 1.5, varianttype=ua.VariantType.Double
        )
        # the Roboflow edge OPC UA server's layout: string NodeIds `Folder.Tag`
        self.nodes["temperature"] = await server.nodes.objects.add_variable(
            ua.NodeId("Sensors.Temperature", idx), "Temperature", 20.0
        )
        await server.start()
        self._server = server

    def stop(self) -> None:
        self._call(self._server.stop())
        self._server = None

    def reset(self) -> None:
        """Restore the initial values (and the server, if a test left it stopped)."""
        if self._server is None:
            self.start()
        self.write("running", False)
        self.write("speed", 1.5)
        self.write("temperature", 20.0)

    def subscription_count(self) -> int:
        return len(self._server.iserver.subscription_service.subscriptions)

    def write(self, name: str, value: Any) -> None:
        self._call(self.nodes[name].write_value(value))

    def write_data_value(self, name: str, data_value: ua.DataValue) -> None:
        self._call(self.nodes[name].write_value(data_value))

    def shutdown(self) -> None:
        if self._server is not None:
            self.stop()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


@pytest.fixture(scope="module")
def opc_test_server():
    # one server for the module: starting one loads the whole OPC UA address space
    server = OPCTestServer()
    server.start()
    yield server
    server.shutdown()


@pytest.fixture
def opc_server(opc_test_server):
    opc_test_server.reset()
    return opc_test_server


def reader_kwargs(server: OPCTestServer, **overrides) -> dict:
    kwargs = {
        "url": server.url,
        "namespace": NAMESPACE,
        "object_name": "Line1",
        "variable_name": "LineRunning",
        "publishing_interval_ms": 50,
        "timeout": 5.0,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.timeout(30)
def test_first_run_returns_current_value_and_later_runs_repeat_it(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        # when
        first = block.run(**reader_kwargs(opc_server))
        second = block.run(**reader_kwargs(opc_server))

        # then
        assert first == {
            "value": False,
            "is_new": True,
            "error_status": False,
            "error_message": None,
        }
        assert second == {**first, "is_new": False}
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_server_side_change_is_pushed_without_polling(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        block.run(**reader_kwargs(opc_server))

        # when
        opc_server.write("running", True)

        # then
        results = []
        assert wait_until(
            lambda: results.append(block.run(**reader_kwargs(opc_server)))
            or results[-1]["value"] is True
        )
        assert results[-1] == {
            "value": True,
            "is_new": True,
            "error_status": False,
            "error_message": None,
        }
        assert block.run(**reader_kwargs(opc_server))["is_new"] is False
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_runs_after_the_first_do_not_wait_for_the_server(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        block.run(**reader_kwargs(opc_server))

        # when
        started = time.perf_counter()
        for _ in range(100):
            block.run(**reader_kwargs(opc_server))
        elapsed = time.perf_counter() - started

        # then: 100 runs read memory only, far below one round trip each
        assert elapsed < 0.05
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_direct_mode_reads_edge_server_style_string_node_ids(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        # when
        result = block.run(
            **reader_kwargs(
                opc_server,
                object_name="Sensors.Temperature",
                variable_name="",
                node_lookup_mode="direct",
            )
        )

        # then
        assert result == {
            "value": 20.0,
            "is_new": True,
            "error_status": False,
            "error_message": None,
        }
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_namespace_index_is_accepted(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        # when
        result = block.run(**reader_kwargs(opc_server, namespace="2"))

        # then
        assert result["error_status"] is False
        assert result["value"] is False
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_unknown_variable_is_reported(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        # when
        result = block.run(**reader_kwargs(opc_server, variable_name="Missing"))

        # then
        assert result["value"] is None
        assert result["error_status"] is True
        assert "WRONG OBJECT OR PROPERTY ERROR" in result["error_message"]
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_unknown_direct_node_id_is_reported(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        # when
        result = block.run(
            **reader_kwargs(
                opc_server,
                object_name="Sensors.Missing",
                variable_name="",
                node_lookup_mode="direct",
            )
        )

        # then
        assert result["error_status"] is True
        assert "BadNodeIdUnknown" in result["error_message"]
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_unknown_namespace_lists_the_available_ones(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        # when
        result = block.run(**reader_kwargs(opc_server, namespace="http://wrong"))

        # then
        assert result["error_status"] is True
        assert "WRONG NAMESPACE ERROR" in result["error_message"]
        assert NAMESPACE in result["error_message"]
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_valid_credentials_are_used(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        # when
        result = block.run(
            **reader_kwargs(opc_server, user_name="operator", password="secret")
        )

        # then
        assert result["error_status"] is False
        assert result["value"] is False
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_rejected_credentials_are_reported_and_not_retried(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        # when
        first = block.run(
            **reader_kwargs(opc_server, user_name="operator", password="wrong")
        )
        time.sleep(1.5)
        later = block.run(
            **reader_kwargs(opc_server, user_name="operator", password="wrong")
        )

        # then
        assert first["error_status"] is True
        assert first["error_message"].startswith("AUTH ERROR")
        assert later == first
        assert block._worker._finished.is_set()
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_value_with_bad_quality_is_flagged(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        block.run(**reader_kwargs(opc_server))

        # when
        opc_server.write_data_value(
            "running",
            ua.DataValue(
                ua.Variant(True, ua.VariantType.Boolean),
                StatusCode_=ua.StatusCode(ua.StatusCodes.BadCommunicationError),
            ),
        )

        # then
        results = []
        assert wait_until(
            lambda: results.append(block.run(**reader_kwargs(opc_server)))
            or results[-1]["error_status"]
        )
        # a Bad value carries no meaningful data, so the server may send it as null
        assert results[-1]["is_new"] is True
        assert results[-1]["error_message"] == (
            "Value quality is not Good: BadCommunicationError"
        )
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_value_that_is_bad_at_startup_is_reported_then_recovers(opc_server):
    # given the tag already carries a Bad quality, e.g. Ignition lost its PLC
    opc_server.write_data_value(
        "running",
        ua.DataValue(
            ua.Variant(False, ua.VariantType.Boolean),
            StatusCode_=ua.StatusCode(ua.StatusCodes.BadCommunicationError),
        ),
    )
    block = OPCReaderBlockV1()

    try:
        # when
        first = block.run(**reader_kwargs(opc_server))
        opc_server.write("running", True)

        # then the bad value is reported, not retried as a connection failure
        assert first["error_status"] is True
        assert first["error_message"] == (
            "Value quality is not Good: BadCommunicationError"
        )
        assert block._worker._subscribed_once is True
        results = []
        assert wait_until(
            lambda: results.append(block.run(**reader_kwargs(opc_server)))
            or (results[-1]["value"] is True and not results[-1]["error_status"])
        )
    finally:
        block.close()


@pytest.mark.timeout(60)
def test_outage_repeats_last_value_without_waiting_then_resubscribes(opc_server):
    # given
    block = OPCReaderBlockV1()

    try:
        assert block.run(**reader_kwargs(opc_server))["value"] is False

        # when the server goes away
        opc_server.stop()
        results = []
        assert wait_until(
            lambda: results.append(block.run(**reader_kwargs(opc_server)))
            or results[-1]["error_status"],
            timeout=10,
        )
        started = time.perf_counter()
        during_outage = block.run(**reader_kwargs(opc_server))
        outage_run_seconds = time.perf_counter() - started

        # then the last value is repeated, flagged, and the run did not wait
        assert during_outage["value"] is False
        assert during_outage["is_new"] is False
        assert during_outage["error_status"] is True
        assert "reconnecting in the background" in during_outage["error_message"]
        assert outage_run_seconds < 0.01

        # when the server comes back with a different value
        opc_server.start()
        opc_server.write("running", True)

        # then the block reconnects, re-subscribes and reports it
        assert wait_until(
            lambda: results.append(block.run(**reader_kwargs(opc_server)))
            or (results[-1]["value"] is True and not results[-1]["error_status"]),
            timeout=20,
        )
        opc_server.write("running", False)
        assert wait_until(
            lambda: results.append(block.run(**reader_kwargs(opc_server)))
            or results[-1]["value"] is False
        )
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_unreachable_server_is_reported_and_later_runs_do_not_wait():
    # given
    block = OPCReaderBlockV1()
    url = f"opc.tcp://127.0.0.1:{free_port()}/nothing-here/"

    try:
        # when
        first = block.run(url=url, namespace="2", object_name="A", variable_name="B")
        started = time.perf_counter()
        second = block.run(url=url, namespace="2", object_name="A", variable_name="B")
        second_run_seconds = time.perf_counter() - started

        # then
        assert first["error_status"] is True
        assert first["value"] is None
        assert "CONNECTION ERROR" in first["error_message"]
        assert second["error_status"] is True
        assert second_run_seconds < 0.01
    finally:
        block.close()


@pytest.mark.timeout(30)
def test_close_ends_the_session_and_the_thread(opc_server):
    # given
    before = opc_server.subscription_count()
    block = OPCReaderBlockV1()
    block.run(**reader_kwargs(opc_server))
    worker = block._worker
    assert opc_server.subscription_count() == before + 1

    # when
    block.close()

    # then
    assert not worker._thread.is_alive()
    assert wait_until(lambda: opc_server.subscription_count() == before)


@pytest.mark.timeout(30)
def test_close_during_retry_backoff_is_prompt(opc_server):
    # given a worker that just failed and is closing its session before backing off
    block = OPCReaderBlockV1()
    block.run(**reader_kwargs(opc_server, variable_name="Missing"))
    worker = block._worker

    # when
    started = time.perf_counter()
    block.close()
    close_seconds = time.perf_counter() - started

    # then
    assert close_seconds < 2.0
    assert not worker._thread.is_alive()


@pytest.mark.timeout(30)
def test_garbage_collected_block_closes_its_session(opc_server):
    # given
    before = opc_server.subscription_count()
    block = OPCReaderBlockV1()
    block.run(**reader_kwargs(opc_server))
    worker = block._worker
    assert opc_server.subscription_count() == before + 1

    # when
    del block

    # then
    assert not worker._thread.is_alive()
    assert wait_until(lambda: opc_server.subscription_count() == before)


WORKFLOW_OPC_READER = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "opc_url"},
        {"type": "WorkflowParameter", "name": "opc_namespace"},
    ],
    "steps": [
        {
            "type": "roboflow_enterprise/opc_reader@v1",
            "name": "line_running",
            "url": "$inputs.opc_url",
            "namespace": "$inputs.opc_namespace",
            "object_name": "Line1",
            "variable_name": "LineRunning",
            "publishing_interval_ms": 50,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "line_running",
            "selector": "$steps.line_running.*",
        }
    ],
}


@pytest.mark.timeout(30)
def test_workflow_with_opc_reader_keeps_the_subscription_across_runs(opc_server):
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_OPC_READER,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    runtime_parameters = {"opc_url": opc_server.url, "opc_namespace": NAMESPACE}

    # when
    first = execution_engine.run(runtime_parameters=runtime_parameters)
    opc_server.write("running", True)
    results = []
    assert wait_until(
        lambda: results.append(
            execution_engine.run(runtime_parameters=runtime_parameters)
        )
        or results[-1][0]["line_running"]["value"] is True
    )

    # then
    assert first == [
        {
            "line_running": {
                "value": False,
                "is_new": True,
                "error_status": False,
                "error_message": None,
            }
        }
    ]
    assert results[-1][0]["line_running"] == {
        "value": True,
        "is_new": True,
        "error_status": False,
        "error_message": None,
    }
