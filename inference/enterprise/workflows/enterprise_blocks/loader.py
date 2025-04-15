from typing import List, Type

from inference.core.workflows.prototypes.block import WorkflowBlock
from inference.enterprise.workflows.enterprise_blocks.sinks.microsoft_sql_server.v1 import (
    MicrosoftSQLServerSinkBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import (
    MQTTWriterSinkBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.opc_writer.v1 import (
    OPCWriterSinkBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1 import (
    ModbusTCPBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1 import (
    PLCBlockV1,
)


def load_enterprise_blocks() -> List[Type[WorkflowBlock]]:
    return [
        OPCWriterSinkBlockV1,
        MQTTWriterSinkBlockV1,
        PLCBlockV1,
        ModbusTCPBlockV1,
        MicrosoftSQLServerSinkBlockV1,
    ]
