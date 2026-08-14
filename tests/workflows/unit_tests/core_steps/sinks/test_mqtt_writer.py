import unittest
from typing import get_args
from unittest.mock import MagicMock, patch

from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import (
    BlockManifest,
    MQTTWriterSinkBlockV1,
)


class TestMQTTWriterSinkBlockV1(unittest.TestCase):
    @patch(
        "inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1.mqtt.Client"
    )
    def test_successful_connection_and_publishing(self, MockMQTTClient):
        # Arrange
        mock_client = MockMQTTClient.return_value
        mock_client.is_connected.return_value = False
        mock_client.publish.return_value.is_published.return_value = True

        block = MQTTWriterSinkBlockV1()
        block._connected = MagicMock()
        block._connected.wait.return_value = True

        # Act
        result = block.run(
            host="localhost",
            port=1883,
            topic="test/topic",
            message="Hello, MQTT!",
            username="lenny",
            password="roboflow",
        )

        # Assert
        self.assertFalse(result["error_status"])
        self.assertEqual(result["message"], "Message published successfully")

    @patch(
        "inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1.mqtt.Client"
    )
    def test_connection_failure(self, MockMQTTClient):
        # Arrange
        mock_client = MockMQTTClient.return_value
        mock_client.is_connected.return_value = False
        mock_client.connect.side_effect = Exception("Connection failed")

        block = MQTTWriterSinkBlockV1()

        # Act
        result = block.run(
            host="localhost",
            port=1883,
            topic="test/topic",
            message="Hello, MQTT!",
            username="lenny",
            password="roboflow",
        )

        # Assert
        self.assertTrue(result["error_status"])
        self.assertIn("Failed to connect to MQTT broker", result["message"])

    @patch(
        "inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1.mqtt.Client"
    )
    def test_publishing_failure(self, MockMQTTClient):
        # Arrange
        mock_client = MockMQTTClient.return_value
        mock_client.is_connected.return_value = True
        mock_client.publish.return_value.is_published.return_value = False

        block = MQTTWriterSinkBlockV1()
        block._connected = MagicMock()
        block._connected.wait.return_value = True

        # Act
        result = block.run(
            host="localhost",
            port=1883,
            topic="test/topic",
            message="Hello, MQTT!",
            username="lenny",
            password="roboflow",
        )

        # Assert
        self.assertTrue(result["error_status"])
        self.assertEqual(result["message"], "Failed to publish payload")


class TestMQTTWriterSinkBlockV1Manifest(unittest.TestCase):
    def test_primary_identifier_is_namespaced(self):
        identifiers = get_args(BlockManifest.model_fields["type"].annotation)

        self.assertEqual(identifiers[0], "roboflow_enterprise/mqtt_writer_sink@v1")

    def test_legacy_identifier_still_accepted(self):
        manifest = BlockManifest.model_validate(
            {
                "type": "mqtt_writer_sink@v1",
                "name": "mqtt",
                "host": "localhost",
                "port": 1883,
                "topic": "test/topic",
                "message": "Hello, MQTT!",
            }
        )

        self.assertEqual(manifest.type, "mqtt_writer_sink@v1")


if __name__ == "__main__":
    unittest.main()
