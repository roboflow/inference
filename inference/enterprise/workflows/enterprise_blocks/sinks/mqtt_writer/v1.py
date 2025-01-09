import logging
from typing import List, Literal, Optional, Type, Union

import paho.mqtt.client as mqtt
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "MQTT Writer",
            "version": "v1",
            "short_description": "Publishes messages to an MQTT broker.",
            "long_description": "This block allows publishing messages to a specified MQTT broker and topic.",
            "license": "Apache-2.0",
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

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="status", kind=[STRING_KIND])]


class MQTTWriterSinkBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        host: str,
        port: int,
        topic: str,
        message: str,
        qos: int = 0,
        retain: bool = False,
    ) -> BlockResult:
        client = mqtt.Client()
        try:
            client.connect(host, port)
            client.publish(topic, message, qos=qos, retain=retain)
            client.disconnect()
            return {"status": "Message published successfully"}
        except Exception as e:
            logging.error(f"Failed to publish message: {e}")
            return {"status": f"Failed to publish message: {e}"}
