from typing import List, Type

from inference.core.workflows.prototypes.block import WorkflowBlock
from inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1 import (
    EventWriterSinkBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.microsoft_sql_server.v1 import (
    MicrosoftSQLServerSinkBlockV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import (
    MQTTWriterSinkBlockV1,
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
from inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1 import (
    PLCBlockV1,
)

# Plugin interface (see blocks_loader._load_blocks_from_plugin). Enterprise
# blocks keep the core block source so the `workflows_core.*` init parameters
# the server supplies - `disable_sinks` in particular - still resolve for them.
BLOCKS_SOURCE = "workflows_core"


def load_blocks() -> List[Type[WorkflowBlock]]:
    return load_enterprise_blocks()


def load_enterprise_blocks() -> List[Type[WorkflowBlock]]:
    return [
        OPCWriterSinkBlockV1,
        MQTTWriterSinkBlockV1,
        PLCBlockV1,
        PLCReaderBlockV1,
        PLCWriterBlockV1,
        ModbusTCPBlockV1,
        MicrosoftSQLServerSinkBlockV1,
        EventWriterSinkBlockV1,
    ]
