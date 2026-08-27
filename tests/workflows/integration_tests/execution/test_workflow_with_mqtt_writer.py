import copy
import socket
import threading
import time
from unittest.mock import patch

import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer import v1
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import (
    MQTTWriterSinkBlockV1,
)
from tests.workflows.integration_tests.execution.conftest import FakeMQTTBroker


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def closed_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.mark.timeout(10)
def test_successful_connection_and_publishing(fake_mqtt_broker):
    # given
    block = MQTTWriterSinkBlockV1()
    published_message = "Test message"
    expected_message = "Message published successfully"

    fake_mqtt_broker.messages_count_to_wait_for = 1
    broker_thread = threading.Thread(target=fake_mqtt_broker.start)
    broker_thread.start()

    try:
        # when
        result = block.run(
            host=fake_mqtt_broker.host,
            port=fake_mqtt_broker.port,
            topic="RoboflowTopic",
            message=published_message,
            timeout=2.0,
        )

        broker_thread.join(timeout=2)

        # then
        assert result["error_status"] is False, "No error expected"
        assert result["message"] == expected_message

        assert published_message.encode() in fake_mqtt_broker.messages[-1]
    finally:
        block.close()


@pytest.mark.timeout(10)
def test_rejected_connack_is_not_treated_as_connected():
    # given
    broker = FakeMQTTBroker(connack_reason_code=5)
    broker.messages_count_to_wait_for = 0
    broker_thread = threading.Thread(target=broker.start)
    broker_thread.start()
    block = MQTTWriterSinkBlockV1()

    try:
        # when
        result = block.run(
            host=broker.host,
            port=broker.port,
            topic="RoboflowTopic",
            message="should not be delivered",
            timeout=1.0,
        )
        broker_thread.join(timeout=2)

        # then
        assert result["error_status"] is True
        assert not block._connected.is_set()
        assert broker.messages == []
    finally:
        block.close()
        broker.finish()


@pytest.mark.timeout(15)
def test_block_recovers_when_broker_becomes_available_after_first_run():
    # given - the port is bound but refuses connections, so the first TCP
    # attempt fails; the failed run must leave no client or thread behind
    broker = FakeMQTTBroker(listening=False)
    broker.messages_count_to_wait_for = 1
    block = MQTTWriterSinkBlockV1()

    try:
        # when - broker is not accepting connections yet
        first_result = block.run(
            host=broker.host,
            port=broker.port,
            topic="RoboflowTopic",
            message="first message",
            timeout=0.3,
        )

        assert block.mqtt_client is None

        # broker becomes available, the next run connects from scratch
        broker.listen()
        broker_thread = threading.Thread(target=broker.start)
        broker_thread.start()

        second_result = block.run(
            host=broker.host,
            port=broker.port,
            topic="RoboflowTopic",
            message="second message",
            timeout=2.0,
        )
        broker_thread.join(timeout=2)

        # then
        assert first_result["error_status"] is True
        assert second_result["error_status"] is False
        assert b"second message" in broker.messages[-1]
    finally:
        block.close()
        broker.finish()


@pytest.mark.timeout(15)
def test_lost_connection_clears_readiness_and_run_reports_it(fake_mqtt_broker):
    # given
    block = MQTTWriterSinkBlockV1()
    fake_mqtt_broker.messages_count_to_wait_for = 1
    broker_thread = threading.Thread(target=fake_mqtt_broker.start)
    broker_thread.start()

    try:
        # when
        first_result = block.run(
            host=fake_mqtt_broker.host,
            port=fake_mqtt_broker.port,
            topic="RoboflowTopic",
            message="delivered",
            timeout=2.0,
        )
        broker_thread.join(timeout=2)
        fake_mqtt_broker.finish()

        assert wait_until(lambda: not block._connected.is_set())

        second_result = block.run(
            host=fake_mqtt_broker.host,
            port=fake_mqtt_broker.port,
            topic="RoboflowTopic",
            message="lost",
            timeout=2.0,
        )

        # then
        assert first_result["error_status"] is False
        assert second_result["error_status"] is True
        assert "not connected" in second_result["message"].lower()
    finally:
        block.close()


@pytest.mark.timeout(10)
def test_missing_puback_reported_as_delivery_unknown_not_final_failure(
    fake_mqtt_broker,
):
    # given - broker accepts the PUBLISH but never sends PUBACK
    block = MQTTWriterSinkBlockV1()
    published_message = "qos1 message"
    fake_mqtt_broker.messages_count_to_wait_for = 1
    broker_thread = threading.Thread(target=fake_mqtt_broker.start)
    broker_thread.start()

    try:
        # when
        result = block.run(
            host=fake_mqtt_broker.host,
            port=fake_mqtt_broker.port,
            topic="RoboflowTopic",
            message=published_message,
            qos=1,
            timeout=1.0,
        )
        broker_thread.join(timeout=2)

        # then - the block reports delivery-unknown, yet the broker received it
        assert result["error_status"] is True
        assert "delivery status unknown" in result["message"].lower()
        assert published_message.encode() in fake_mqtt_broker.messages[-1]
    finally:
        block.close()


@pytest.mark.timeout(10)
def test_second_broker_is_rejected_instead_of_publishing_to_first(fake_mqtt_broker):
    # given
    block = MQTTWriterSinkBlockV1()
    other_broker = FakeMQTTBroker()
    fake_mqtt_broker.messages_count_to_wait_for = 1
    broker_thread = threading.Thread(target=fake_mqtt_broker.start)
    broker_thread.start()

    try:
        # when
        first_result = block.run(
            host=fake_mqtt_broker.host,
            port=fake_mqtt_broker.port,
            topic="RoboflowTopic",
            message="for broker A",
            timeout=2.0,
        )
        broker_thread.join(timeout=2)

        second_result = block.run(
            host=other_broker.host,
            port=other_broker.port,
            topic="RoboflowTopic",
            message="for broker B",
            timeout=2.0,
        )

        # then
        assert first_result["error_status"] is False
        assert second_result["error_status"] is True
        assert "parameters" in second_result["message"].lower()
        assert other_broker.messages == []
        assert all(
            b"for broker B" not in message for message in fake_mqtt_broker.messages
        )
    finally:
        block.close()
        other_broker.finish()


MQTT_SINK_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "host"},
        {"type": "WorkflowParameter", "name": "port"},
        {"type": "WorkflowParameter", "name": "message"},
    ],
    "steps": [
        {
            "type": "roboflow_enterprise/mqtt_writer_sink@v1",
            "name": "mqtt_sink",
            "host": "$inputs.host",
            "port": "$inputs.port",
            "topic": "RoboflowTopic",
            "message": "$inputs.message",
            "timeout": 0.2,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "status",
            "selector": "$steps.mqtt_sink.error_status",
        },
    ],
}


@pytest.fixture
def enterprise_blocks_enabled():
    from inference.core.workflows.execution_engine.introspection import blocks_loader

    previous_value = blocks_loader.LOAD_ENTERPRISE_BLOCKS
    blocks_loader.LOAD_ENTERPRISE_BLOCKS = True
    blocks_loader.load_core_workflow_blocks.cache_clear()
    yield
    blocks_loader.LOAD_ENTERPRISE_BLOCKS = previous_value
    blocks_loader.load_core_workflow_blocks.cache_clear()


@pytest.mark.timeout(15)
def test_workflow_with_failing_mqtt_sink_logs_and_continues(
    enterprise_blocks_enabled,
):
    # given - nothing listens on the port, so every sink run fails
    execution_engine = ExecutionEngine.init(
        workflow_definition=MQTT_SINK_WORKFLOW,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    runtime_parameters = {
        "host": "localhost",
        "port": closed_port(),
        "message": "frame payload",
    }

    # when - two consecutive frames
    with patch.object(v1, "logger") as mock_logger:
        first_frame_result = execution_engine.run(runtime_parameters=runtime_parameters)
        second_frame_result = execution_engine.run(
            runtime_parameters=runtime_parameters
        )

    # then - failures are surfaced in logs and outputs, execution continues
    assert any(
        call.args[0].startswith("MQTT Writer failure")
        for call in mock_logger.error.call_args_list
    )
    assert first_frame_result[0]["status"] is True
    assert second_frame_result[0]["status"] is True


@pytest.mark.timeout(15)
def test_workflow_with_unwired_failing_mqtt_sink_still_logs_and_continues(
    enterprise_blocks_enabled,
):
    # given - no MQTT outputs are wired, so logs are the only failure signal
    workflow_definition = copy.deepcopy(MQTT_SINK_WORKFLOW)
    workflow_definition["outputs"] = [
        {"type": "JsonField", "name": "echo", "selector": "$inputs.message"},
    ]
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    runtime_parameters = {
        "host": "localhost",
        "port": closed_port(),
        "message": "frame payload",
    }

    # when
    with patch.object(v1, "logger") as mock_logger:
        first_frame_result = execution_engine.run(runtime_parameters=runtime_parameters)
        second_frame_result = execution_engine.run(
            runtime_parameters=runtime_parameters
        )

    # then
    assert any(
        call.args[0].startswith("MQTT Writer failure")
        for call in mock_logger.error.call_args_list
    )
    assert first_frame_result[0]["echo"] == "frame payload"
    assert second_frame_result[0]["echo"] == "frame payload"


@pytest.mark.timeout(15)
def test_workflow_with_failing_mqtt_sink_and_fail_fast_raises(
    enterprise_blocks_enabled,
):
    # given
    workflow_definition = copy.deepcopy(MQTT_SINK_WORKFLOW)
    workflow_definition["steps"][0]["fail_fast"] = True
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when / then
    with pytest.raises(Exception, match="not connected"):
        execution_engine.run(
            runtime_parameters={
                "host": "localhost",
                "port": closed_port(),
                "message": "frame payload",
            }
        )
