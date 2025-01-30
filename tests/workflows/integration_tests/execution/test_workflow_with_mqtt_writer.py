import threading

import pytest

from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import (
    MQTTWriterSinkBlockV1,
)


@pytest.mark.timeout(5)
def test_successful_connection_and_publishing(fake_mqtt_broker):
    # given
    block = MQTTWriterSinkBlockV1()
    published_message = "Test message"
    expected_message = "Message published successfully"

    fake_mqtt_broker.messages_count_to_wait_for = 1
    broker_thread = threading.Thread(target=fake_mqtt_broker.start)
    broker_thread.start()

    # when
    result = block.run(
        host=fake_mqtt_broker.host,
        port=fake_mqtt_broker.port,
        topic="RoboflowTopic",
        message=published_message,
    )

    broker_thread.join(timeout=2)

    # then
    assert result["error_status"] is False, "No error expected"
    assert result["message"] == expected_message

    assert published_message.encode() in fake_mqtt_broker.messages[-1]
