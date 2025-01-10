import unittest
import time
import paho.mqtt.client as mqtt
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import MQTTWriterSinkBlockV1
import pytest

@pytest.mark.timeout(5)
class MQTTWriterSinkBlockV1IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.broker_host = 'test.mosquitto.org'
        self.broker_port = 1883
        self.topic = 'test/topic'
        self.received_messages = []

        self.listener_client = mqtt.Client()
        self.listener_client.on_message = self.on_message
        self.listener_client.connect(self.broker_host, self.broker_port)
        self.listener_client.subscribe(self.topic)
        self.listener_client.loop_start()

    def on_message(self, client, userdata, msg):
        self.received_messages.append(msg.payload.decode())

    def tearDown(self):
        self.listener_client.loop_stop()
        self.listener_client.disconnect()

    def test_successful_connection_and_publishing(self):
        # given
        block = MQTTWriterSinkBlockV1()
        message = 'Test message'
        expected_error_status = False
        expected_message = 'Message published successfully'

        # when
        result = block.run(
            host=self.broker_host,
            port=self.broker_port,
            topic=self.topic,
            message=message
        )

        time.sleep(2)

        # then
        self.assertEqual(result['error_status'], expected_error_status)
        self.assertEqual(result['message'], expected_message)
        self.assertIn(message, self.received_messages)

if __name__ == '__main__':
    unittest.main() 