import os
import threading
import time
from typing import Callable

import pytest
from roboflow_workflows.enterprise_blocks.sinks.mqtt_reader.v1 import MQTTReaderBlockV1

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.conftest import FakeMQTTBroker

RUNNING = b'{"state": "RUNNING"}'
PAUSED = b'{"state": "PAUSED"}'


def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def broker():
    broker = FakeMQTTBroker(keep_serving=True)
    thread = threading.Thread(target=broker.start, daemon=True)
    thread.start()
    yield broker
    broker.finish()
    thread.join(timeout=2)


def reader_kwargs(broker: FakeMQTTBroker, **overrides) -> dict:
    kwargs = {
        "host": broker.host,
        "port": broker.port,
        "topic": "plc/#",
        "timeout": 2.0,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.timeout(15)
def test_retained_message_returned_on_first_run(broker):
    # given
    broker.retained["plc/state"] = RUNNING
    block = MQTTReaderBlockV1()

    try:
        # when
        first = block.run(**reader_kwargs(broker))
        second = block.run(**reader_kwargs(broker))

        # then
        assert first == {
            "value": '{"state": "RUNNING"}',
            "payload": {"state": "RUNNING"},
            "topic": "plc/state",
            "is_new": True,
            "error_status": False,
            "error_message": None,
        }
        assert second == {**first, "is_new": False}
        assert broker.subscriptions == [("plc/#", 0)]
    finally:
        block.close()


@pytest.mark.timeout(15)
def test_empty_first_run_then_latest_message_delivered(broker):
    # given
    block = MQTTReaderBlockV1()

    try:
        # when
        empty = block.run(**reader_kwargs(broker))
        for payload in (b"A", b"B", PAUSED):
            broker.publish("plc/state", payload)
        # the one-slot buffer holds a message as soon as A lands; wait for the
        # newest one to have replaced it
        assert wait_until(
            lambda: bool(block._state.messages)
            and block._state.messages[-1][1] == PAUSED
        )
        result = block.run(**reader_kwargs(broker))

        # then
        assert empty["value"] is None
        assert empty["is_new"] is False
        assert empty["error_status"] is False
        assert result["value"] == '{"state": "PAUSED"}'
        assert result["payload"] == {"state": "PAUSED"}
        assert result["is_new"] is True
    finally:
        block.close()


@pytest.mark.timeout(15)
def test_sequential_mode_delivers_backlog_one_per_run(broker):
    # given
    block = MQTTReaderBlockV1()
    kwargs = reader_kwargs(broker, read_mode="sequential")

    try:
        # when
        block.run(**kwargs)
        broker.publish("plc/a", b"A")
        broker.publish("plc/b", b"B")
        broker.publish("plc/c", b"C")
        assert wait_until(lambda: len(block._state.messages) == 3)
        results = [block.run(**kwargs) for _ in range(4)]

        # then
        assert [r["value"] for r in results] == ["A", "B", "C", "C"]
        assert [r["topic"] for r in results] == ["plc/a", "plc/b", "plc/c", "plc/c"]
        assert [r["is_new"] for r in results] == [True, True, True, False]
    finally:
        block.close()


@pytest.mark.timeout(15)
def test_qos2_message_delivered_once(broker):
    # given
    block = MQTTReaderBlockV1()
    kwargs = reader_kwargs(broker, qos=2, read_mode="sequential")

    try:
        # when
        block.run(**kwargs)
        broker.publish("plc/state", RUNNING, qos=2)
        assert wait_until(lambda: len(block._state.messages) == 1)
        time.sleep(0.2)
        first = block.run(**kwargs)
        second = block.run(**kwargs)

        # then
        assert broker.subscriptions == [("plc/#", 2)]
        assert first["payload"] == {"state": "RUNNING"}
        assert first["is_new"] is True
        assert second["is_new"] is False
    finally:
        block.close()


@pytest.mark.timeout(15)
def test_refused_subscription_reported_and_connection_kept():
    # given
    broker = FakeMQTTBroker(keep_serving=True, suback_reason_code=0x80)
    thread = threading.Thread(target=broker.start, daemon=True)
    thread.start()
    block = MQTTReaderBlockV1()

    try:
        # when
        result = block.run(**reader_kwargs(broker))

        # then
        assert result["error_status"] is True
        assert "refused" in result["error_message"].lower()
        assert block._state.connected.is_set()
        assert block._client is not None
    finally:
        block.close()
        broker.finish()
        thread.join(timeout=2)


@pytest.mark.timeout(20)
def test_subscription_restored_after_reconnect(broker):
    # given - a retained state the broker re-sends on every subscription
    broker.retained["plc/state"] = RUNNING
    block = MQTTReaderBlockV1()

    try:
        first = block.run(**reader_kwargs(broker))
        assert first["payload"] == {"state": "RUNNING"}
        assert first["is_new"] is True

        # when - the broker drops the connection and accepts a new one
        broker.drop_connection()
        assert wait_until(lambda: not block._state.connected.is_set())
        second_thread = threading.Thread(target=broker.start, daemon=True)
        second_thread.start()
        assert wait_until(lambda: len(broker.subscriptions) == 2, timeout=10)
        assert wait_until(lambda: len(block._state.messages) == 1)
        redelivered = block.run(**reader_kwargs(broker))
        broker.publish("plc/state", PAUSED)
        assert wait_until(
            lambda: bool(block._state.messages)
            and block._state.messages[-1][1] == PAUSED
        )
        updated = block.run(**reader_kwargs(broker))

        # then - the subscription is live again, the retained copy is not new
        assert broker.connections_accepted == 2
        assert redelivered == {**first, "is_new": False}
        assert updated["payload"] == {"state": "PAUSED"}
        assert updated["is_new"] is True
    finally:
        block.close()


@pytest.mark.timeout(15)
def test_unreachable_broker_leaves_nothing_behind_and_later_run_recovers():
    # given - bound but not listening, so the TCP connect is refused
    broker = FakeMQTTBroker(listening=False, keep_serving=True)
    block = MQTTReaderBlockV1()
    thread = None

    try:
        # when
        first = block.run(**reader_kwargs(broker, timeout=0.3))
        assert block._client is None

        broker.listen()
        broker.retained["plc/state"] = RUNNING
        thread = threading.Thread(target=broker.start, daemon=True)
        thread.start()
        second = block.run(**reader_kwargs(broker))

        # then
        assert first["error_status"] is True
        assert second["error_status"] is False
        assert second["payload"] == {"state": "RUNNING"}
    finally:
        block.close()
        broker.finish()
        if thread is not None:
            thread.join(timeout=2)


@pytest.fixture
def tls_broker(mqtt_test_certificates):
    broker = FakeMQTTBroker(
        keep_serving=True, tls_context=mqtt_test_certificates.server_context
    )
    thread = threading.Thread(target=broker.start, daemon=True)
    thread.start()
    yield broker
    broker.finish()
    thread.join(timeout=2)


@pytest.mark.timeout(15)
def test_retained_message_over_tls_with_ca_certificate(
    tls_broker, mqtt_test_certificates
):
    # given
    tls_broker.retained["plc/state"] = RUNNING
    block = MQTTReaderBlockV1(allow_access_to_file_system=True)

    try:
        # when
        result = block.run(
            **reader_kwargs(
                tls_broker,
                encryption="tls",
                ca_certificate_path=mqtt_test_certificates.ca_path,
                # the TLS handshake, CONNACK, SUBACK and the retained message
                # each get this budget; CI runners are slower than a laptop
                timeout=5.0,
            )
        )

        # then
        assert result["error_status"] is False, result["error_message"]
        assert result["payload"] == {"state": "RUNNING"}
        assert result["is_new"] is True
        assert tls_broker.connections_accepted == 1
        assert tls_broker.handshake_failures == 0
    finally:
        block.close()


@pytest.mark.timeout(15)
def test_tls_without_ca_certificate_is_rejected_by_verification(tls_broker):
    # given - the test CA is not in the system trust store
    block = MQTTReaderBlockV1()

    try:
        # when
        result = block.run(**reader_kwargs(tls_broker, encryption="tls"))

        # then - the client refused the certificate; nothing is kept
        assert result["error_status"] is True
        assert "not connected" in result["error_message"].lower()
        assert "TLS is enabled" in result["error_message"]
        assert block._client is None
        assert wait_until(lambda: tls_broker.handshake_failures == 1)
    finally:
        block.close()


@pytest.mark.timeout(15)
def test_tls_with_unreadable_ca_bundle_is_reported_by_paho(tls_broker, tmp_path):
    # given - a real file that is not a PEM bundle: this exercises paho's
    # loader, not a mock
    bad_bundle = tmp_path / "not-a-bundle.pem"
    bad_bundle.write_text("this is not a certificate")
    block = MQTTReaderBlockV1(allow_access_to_file_system=True)

    try:
        # when
        result = block.run(
            **reader_kwargs(
                tls_broker, encryption="tls", ca_certificate_path=str(bad_bundle)
            )
        )

        # then
        assert result["error_status"] is True
        assert "could not load CA bundle" in result["error_message"]
        assert block._client is None
        assert tls_broker.connections_accepted == 0
    finally:
        block.close()


@pytest.mark.timeout(15)
def test_tls_against_plain_broker_reports_not_connected(broker):
    # given - the broker never speaks TLS; like a real broker it drops a
    # connection whose first bytes are not an MQTT CONNECT, so the client's
    # handshake fails at once
    block = MQTTReaderBlockV1()

    try:
        # when
        result = block.run(**reader_kwargs(broker, encryption="tls", timeout=1.0))

        # then
        assert result["error_status"] is True
        assert "not connected" in result["error_message"].lower()
        assert block._client is None
    finally:
        block.close()


# --- persistent sessions (client_id) ------------------------------------------


@pytest.fixture
def session_broker():
    # granted QoS caps delivery, so the broker must grant up to QoS 2 to queue
    # anything for an offline persistent session
    broker = FakeMQTTBroker(keep_serving=True, session_aware=True, suback_reason_code=2)
    thread = threading.Thread(target=broker.serve, daemon=True)
    thread.start()
    yield broker
    broker.finish()
    thread.join(timeout=2)


def persistent_kwargs(broker: FakeMQTTBroker, **overrides) -> dict:
    kwargs = reader_kwargs(broker, client_id="line1", qos=1, read_mode="sequential")
    kwargs.update(overrides)
    return kwargs


def session_is_away(broker: FakeMQTTBroker, client_id: str = "line1") -> bool:
    session = broker.sessions.get(client_id)
    return session is not None and session.connection is None


def subscribe_and_leave(broker: FakeMQTTBroker, **overrides) -> None:
    """A block instance with a persistent session subscribes, then goes away."""
    block = MQTTReaderBlockV1()
    try:
        result = block.run(**persistent_kwargs(broker, **overrides))
        assert result["error_status"] is False, result["error_message"]
    finally:
        block.close()
    assert wait_until(lambda: session_is_away(broker))


@pytest.mark.timeout(20)
def test_persistent_session_delivers_backlog_after_restart(session_broker, caplog):
    # given - a subscriber with a fixed id went away and three QoS 1 messages
    # were published meanwhile
    with caplog.at_level("INFO", logger="inference"):
        subscribe_and_leave(session_broker)
    assert session_broker.connects == [("line1", False)]
    assert "session present: False" in caplog.text
    caplog.clear()
    for payload in (b"A", b"B", b"C"):
        session_broker.publish("plc/state", payload, qos=1)
    block = MQTTReaderBlockV1()

    try:
        # when - a new instance (a restarted pipeline) resumes the session
        with caplog.at_level("INFO", logger="inference"):
            results = [block.run(**persistent_kwargs(session_broker)) for _ in range(3)]

        # then - the broker resumed the session and delivered the backlog in order
        assert session_broker.connects[-1] == ("line1", False)
        assert "session present: True" in caplog.text
        assert [r["value"] for r in results] == ["A", "B", "C"]
        assert [r["is_new"] for r in results] == [True, True, True]
        assert [r["error_status"] for r in results] == [False, False, False]
    finally:
        block.close()


@pytest.mark.timeout(20)
def test_persistent_session_in_latest_mode_keeps_newest_of_backlog(session_broker):
    # given
    subscribe_and_leave(session_broker, read_mode="latest")
    for payload in (b"A", b"B", b"C"):
        session_broker.publish("plc/state", payload, qos=1)
    block = MQTTReaderBlockV1()

    try:
        # when
        first = block.run(**persistent_kwargs(session_broker, read_mode="latest"))
        second = block.run(**persistent_kwargs(session_broker, read_mode="latest"))

        # then - only the newest message of the burst survives
        assert first["value"] == "C"
        assert first["is_new"] is True
        assert second["is_new"] is False
    finally:
        block.close()


@pytest.mark.timeout(20)
def test_persistent_session_does_not_queue_qos0_messages(session_broker):
    # given - the publisher used QoS 0, which a broker never queues
    subscribe_and_leave(session_broker)
    for payload in (b"A", b"B", b"C"):
        session_broker.publish("plc/state", payload, qos=0)
    block = MQTTReaderBlockV1()

    try:
        # when
        result = block.run(**persistent_kwargs(session_broker))

        # then - the session resumed, but there was nothing to deliver
        assert session_broker.connects[-1] == ("line1", False)
        assert result["error_status"] is False
        assert result["value"] is None
        assert result["is_new"] is False
    finally:
        block.close()


@pytest.mark.timeout(20)
def test_persistent_session_survives_reconnect_of_the_same_instance(session_broker):
    # given - a live persistent subscription
    block = MQTTReaderBlockV1()

    try:
        first = block.run(**persistent_kwargs(session_broker))
        assert first["error_status"] is False

        # when - the connection drops, a message is published while the block is
        # away, and the background loop reconnects
        session_broker.drop_connection()
        assert wait_until(lambda: not block._state.connected.is_set())
        assert wait_until(lambda: session_is_away(session_broker))
        session_broker.publish("plc/state", PAUSED, qos=1)
        assert wait_until(lambda: len(block._state.messages) == 1, timeout=10)
        result = block.run(**persistent_kwargs(session_broker))

        # then - the queued message reached the same instance after the reconnect
        assert session_broker.connections_accepted == 2
        assert session_broker.connects == [("line1", False), ("line1", False)]
        assert result["payload"] == {"state": "PAUSED"}
        assert result["is_new"] is True
    finally:
        block.close()


@pytest.mark.timeout(20)
def test_second_connection_with_the_same_client_id_evicts_the_first(session_broker):
    # given - two block instances sharing one client id on one broker
    first = MQTTReaderBlockV1()
    second = MQTTReaderBlockV1()

    try:
        assert first.run(**persistent_kwargs(session_broker))["error_status"] is False

        # when - the second connects with the same id
        result = second.run(**persistent_kwargs(session_broker))

        # then - the broker drops the first; its reconnect would evict the second
        # in turn, so the first is closed here (only the first eviction is tested)
        assert wait_until(lambda: not first._state.connected.is_set())
        first.close()
        assert result["error_status"] is False
        assert session_broker.connects[:2] == [("line1", False), ("line1", False)]
        assert wait_until(
            lambda: second._state.connected.is_set()
            and session_broker.sessions["line1"].connection is not None,
            timeout=10,
        )
        session_broker.publish("plc/state", PAUSED, qos=1)
        assert wait_until(lambda: len(second._state.messages) == 1, timeout=10)
        updated = second.run(**persistent_kwargs(session_broker))
        assert updated["payload"] == {"state": "PAUSED"}
        assert updated["is_new"] is True
    finally:
        first.close()
        second.close()


@pytest.mark.timeout(20)
def test_writer_without_client_id_delivers_to_an_away_persistent_subscriber(
    session_broker,
):
    # given - a persistent subscriber that went away, and the unchanged MQTT Writer
    from roboflow_workflows.enterprise_blocks.sinks.mqtt_writer.v1 import (
        MQTTWriterSinkBlockV1,
    )

    subscribe_and_leave(session_broker)
    writer = MQTTWriterSinkBlockV1()
    reader = MQTTReaderBlockV1()

    try:
        # when - the writer publishes at QoS 1 with a broker-generated id
        published = writer.run(
            host=session_broker.host,
            port=session_broker.port,
            topic="plc/state",
            message=PAUSED.decode("utf-8"),
            qos=1,
            timeout=2.0,
        )
        writer.close()
        result = reader.run(**persistent_kwargs(session_broker))

        # then - queued delivery depends only on the publish QoS, not on a writer id
        assert published["error_status"] is False, published["message"]
        assert published["message"] == "Message published successfully"
        assert session_broker.connects[1] == ("", True)
        assert result["payload"] == {"state": "PAUSED"}
        assert result["is_new"] is True
    finally:
        writer.close()
        reader.close()


GATED_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "host"},
        {"type": "WorkflowParameter", "name": "port"},
    ],
    "steps": [
        {
            "type": "roboflow_enterprise/mqtt_reader@v1",
            "name": "reader",
            "host": "$inputs.host",
            "port": "$inputs.port",
            "topic": "plc/state",
            "timeout": 2.0,
        },
        {
            "type": "roboflow_core/continue_if@v1",
            "name": "gate",
            "condition_statement": {
                "type": "StatementGroup",
                "statements": [
                    {
                        "type": "UnaryStatement",
                        "operand": {
                            "type": "DynamicOperand",
                            "operand_name": "raw",
                            "operations": [
                                {
                                    "type": "StringMatches",
                                    "regex": '"state"\\s*:\\s*"RUNNING"',
                                }
                            ],
                        },
                        "operator": {"type": "(Boolean) is True"},
                    }
                ],
            },
            "evaluation_parameters": {"raw": "$steps.reader.value"},
            "next_steps": ["$steps.shout"],
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "shout",
            "data": "$steps.reader.value",
            "operations": [{"type": "StringToUpperCase"}],
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "state", "selector": "$steps.reader.payload"},
        {"type": "JsonField", "name": "is_new", "selector": "$steps.reader.is_new"},
        {"type": "JsonField", "name": "shouted", "selector": "$steps.shout.output"},
    ],
}


@pytest.fixture
def enterprise_blocks_enabled(monkeypatch):
    from inference.core.env import ENTERPRISE_BLOCKS_PLUGIN
    from inference.core.workflows.execution_engine.introspection import blocks_loader
    from inference.core.workflows.execution_engine.v1.compiler.core import (
        COMPILATION_CACHE,
    )

    plugins = [p for p in os.getenv("WORKFLOWS_PLUGINS", "").split(",") if p]
    if ENTERPRISE_BLOCKS_PLUGIN not in plugins:  # env.py may have expanded it already;
        plugins.append(ENTERPRISE_BLOCKS_PLUGIN)  # a duplicate trips the clash check
    monkeypatch.setenv("WORKFLOWS_PLUGINS", ",".join(plugins))
    blocks_loader.clear_caches()
    yield
    blocks_loader.clear_caches()
    # drop graphs compiled with enterprise blocks from the process-wide cache
    with COMPILATION_CACHE._cache_lock:
        COMPILATION_CACHE._cache.clear()
        COMPILATION_CACHE._keys_buffer.clear()


@pytest.mark.timeout(20)
def test_workflow_gates_downstream_step_on_received_state(
    broker, enterprise_blocks_enabled
):
    # given - a retained RUNNING state, then a PAUSED update between frames
    broker.retained["plc/state"] = RUNNING
    execution_engine = ExecutionEngine.init(
        workflow_definition=GATED_WORKFLOW,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    runtime_parameters = {"host": broker.host, "port": broker.port}
    reader_block = execution_engine._engine._compiled_workflow.steps["reader"].step

    try:
        # when
        running_frame = execution_engine.run(runtime_parameters=runtime_parameters)
        broker.publish("plc/state", PAUSED)
        assert wait_until(lambda: len(reader_block._state.messages) == 1)
        paused_frame = execution_engine.run(runtime_parameters=runtime_parameters)
        repeated_frame = execution_engine.run(runtime_parameters=runtime_parameters)

        # then - one block instance served every frame
        assert running_frame[0]["state"] == {"state": "RUNNING"}
        assert running_frame[0]["is_new"] is True
        assert running_frame[0]["shouted"] == '{"STATE": "RUNNING"}'
        assert paused_frame[0]["state"] == {"state": "PAUSED"}
        assert paused_frame[0]["is_new"] is True
        assert paused_frame[0].get("shouted") is None
        assert repeated_frame[0]["state"] == {"state": "PAUSED"}
        assert repeated_frame[0]["is_new"] is False
        assert broker.connections_accepted == 1
    finally:
        reader_block.close()
