"""Selectively exported active blocks for the opt-in V2 catalogue."""

from roboflow_workflows.v2.blocks.industrial.kafka import (
    KafkaConsumerBlock,
    KafkaProducerSinkBlock,
)
from roboflow_workflows.v2.blocks.industrial.mqtt import (
    MQTTReaderBlock,
    MQTTWriterBlock,
)
from roboflow_workflows.v2.blocks.industrial.onvif import (
    OnvifTrackingBlock,
)
from roboflow_workflows.v2.blocks.industrial.opc import (
    OPCWriterSinkBlock,
)
from roboflow_workflows.v2.blocks.industrial.plc import (
    PlcReaderBlock,
    PlcWriterBlock,
)
from roboflow_workflows.v2.blocks.industrial.sql import (
    PostgreSQLSinkBlock,
    SQLServerSinkBlock,
)

BLOCKS = (
    PlcReaderBlock,
    PlcWriterBlock,
    PostgreSQLSinkBlock,
    SQLServerSinkBlock,
    MQTTReaderBlock,
    MQTTWriterBlock,
    KafkaConsumerBlock,
    KafkaProducerSinkBlock,
    OPCWriterSinkBlock,
    OnvifTrackingBlock,
)

__all__ = [
    "BLOCKS",
    "PlcReaderBlock",
    "PlcWriterBlock",
    "PostgreSQLSinkBlock",
    "SQLServerSinkBlock",
    "MQTTReaderBlock",
    "MQTTWriterBlock",
    "KafkaConsumerBlock",
    "KafkaProducerSinkBlock",
    "OPCWriterSinkBlock",
    "OnvifTrackingBlock",
]
