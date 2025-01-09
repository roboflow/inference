import pytest
from unittest.mock import patch, MagicMock
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import MQTTWriterSinkBlockV1


@pytest.mark.timeout(5)
def test_mqtt_writer_sink_block_v1():
    # given
    host = "localhost"
    port = 1883
    topic = "test/topic"
    message = "Hello, MQTT!"
    qos = 1
    retain = False

    mqtt_block = MQTTWriterSinkBlockV1()

    with patch('paho.mqtt.client.Client') as MockClient:
        mock_client_instance = MockClient.return_value
        mock_client_instance.connect.return_value = 0
        mock_client_instance.publish.return_value = (0, 0)

        # when
        result = mqtt_block.run(host=host, port=port, topic=topic, message=message, qos=qos, retain=retain)

        # then
        mock_client_instance.connect.assert_called_once_with(host, port)
        mock_client_instance.publish.assert_called_once_with(topic, message, qos=qos, retain=retain)
        mock_client_instance.disconnect.assert_called_once()

        assert result["status"] == "Message published successfully"
