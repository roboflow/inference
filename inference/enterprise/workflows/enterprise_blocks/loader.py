from typing import List, Type

from inference.core.workflows.prototypes.block import WorkflowBlock
from inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1 import (
    EventWriterSinkBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.microsoft_sql_server.v1 import (
    MicrosoftSQLServerSinkBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.microsoft_sql_server.v2 import (
    MicrosoftSQLServerSinkBlockV2,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import (
    MQTTWriterSinkBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v2 import (
    MQTTWriterSinkBlockV2,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.opc_writer.v1 import (
    OPCWriterSinkBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.plc.v1 import (
    PLCReaderBlockV1,
    PLCWriterBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1 import (
    ModbusTCPBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v2 import (
    ModbusTCPBlockV2,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1 import (
    PLCBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v2 import (
    PLCBlockV2,
)


def load_enterprise_blocks() -> List[Type[WorkflowBlock]]:
    return [
        OPCWriterSinkBlockV1,
        MQTTWriterSinkBlockV1,
        MQTTWriterSinkBlockV2,
        PLCBlockV1,
        PLCBlockV2,
        PLCReaderBlockV1,
        PLCWriterBlockV1,
        ModbusTCPBlockV1,
        ModbusTCPBlockV2,
        MicrosoftSQLServerSinkBlockV1,
        MicrosoftSQLServerSinkBlockV2,
        EventWriterSinkBlockV1,
    ]
