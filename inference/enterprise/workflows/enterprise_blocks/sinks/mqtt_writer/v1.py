import threading
from typing import List, Literal, Optional, Type, Union

import paho.mqtt.client as mqtt
from pydantic import ConfigDict, Field

from inference.core.logger import logger
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
MQTT Writer block for publishing messages to an MQTT broker.

This block is blocking on connect and publish operations.

Outputs:
    - error_status (bool): Indicates if an error occurred during the MQTT publishing process.
                          True if there was an error, False if successful.
    - message (str): Status message describing the result of the operation.
                    Contains error details if error_status is True,
                    or success confirmation if error_status is False.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "MQTT Writer",
            "version": "v1",
            "short_description": "Publishes messages to an MQTT broker.",
            "long_description": LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "sink",
        }
    )
    type: Literal["mqtt_writer_sink@v1"]
    host: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Host of the MQTT broker.",
        examples=["localhost", "$inputs.mqtt_host"],
    )
    port: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        description="Port of the MQTT broker.",
        examples=[1883, "$inputs.mqtt_port"],
    )
    topic: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="MQTT topic to publish the message to.",
        examples=["sensors/temperature", "$inputs.mqtt_topic"],
    )
    message: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Message to be published.",
        examples=["Hello, MQTT!", "$inputs.mqtt_message"],
    )
    qos: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=0,
        description="Quality of Service level for the message.",
        examples=[0, 1, 2],
    )
    retain: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Whether the message should be retained by the broker.",
        examples=[True, False],
    )
    timeout: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.5,
        description="Timeout for connecting to the MQTT broker and for sending MQTT messages.",
        examples=[0.5],
    )
    username: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=None,
        description="Username for MQTT broker authentication.",
        examples=["$inputs.mqtt_username"],
    )
    password: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=None,
        description="Password for MQTT broker authentication.",
        examples=["$inputs.mqtt_password"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class MQTTWriterSinkBlockV1(WorkflowBlock):
    def __init__(self):
        self.mqtt_client: Optional[mqtt.Client] = None
        self._connected = threading.Event()

    def __del__(self):
        try:
            if self.mqtt_client is not None:
                self.mqtt_client.disconnect()
                self.mqtt_client.loop_stop()
        except Exception as e:
            logger.error("Failed to disconnect MQTT client: %s", e)

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        host: str,
        port: int,
        topic: str,
        message: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        qos: int = 0,
        retain: bool = False,
        timeout: float = 0.5,
    ) -> BlockResult:
        if self.mqtt_client is None:
            self.mqtt_client = mqtt.Client()
            if username and password:
                self.mqtt_client.username_pw_set(username, password)
            self.mqtt_client.on_connect = self.mqtt_on_connect
            self.mqtt_client.on_connect_fail = self.mqtt_on_connect_fail
            self.mqtt_client.reconnect_delay_set(
                min_delay=timeout, max_delay=2 * timeout
            )
            try:
                # TODO: blocking, consider adding fire_and_forget like in OPC writer
                print("Connecting")
                self.mqtt_client.connect(host, port)
                self.mqtt_client.loop_start()

                if not self._connected.wait(timeout=timeout):
                    raise Exception("Connection timeout")
            except Exception as e:
                logger.error("Failed to connect to MQTT broker: %s", e)
                return {
                    "error_status": True,
                    "message": f"Failed to connect to MQTT broker: {e}",
                }

        if not self.mqtt_client.is_connected():
            try:
                # TODO: blocking
                print("Reconnecting")
                self.mqtt_client.reconnect()
                if not self._connected.wait(timeout=timeout):
                    raise Exception("Connection timeout")
            except Exception as e:
                logger.error("Failed to connect to MQTT broker: %s", e)
                return {
                    "error_status": True,
                    "message": f"Failed to connect to MQTT broker: {e}",
                }

        try:
            res: mqtt.MQTTMessageInfo = self.mqtt_client.publish(
                topic, message, qos=qos, retain=retain
            )
            # TODO: this is blocking
            res.wait_for_publish(timeout=timeout)
            if res.is_published():
                return {
                    "error_status": False,
                    "message": "Message published successfully",
                }
            else:
                return {"error_status": True, "message": "Failed to publish payload"}
        except Exception as e:
            logger.error("Failed to publish message: %s", e)
            return {"error_status": True, "message": f"Unhandled error - {e}"}

    def mqtt_on_connect(self, client, userdata, flags, reason_code, properties=None):
        logger.info("Connected with result code %s", reason_code)
        self._connected.set()

    def mqtt_on_connect_fail(
        self, client, userdata, flags, reason_code, properties=None
    ):
        logger.error(f"Failed to connect with result code %s", reason_code)
        self._connected.clear()
