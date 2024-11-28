from typing import List, Type

from inference.enterprise.workflows.enterprise_steps.sinks.opc_writer.v1 import OPCWriterSinkBlockV1
from inference.core.workflows.prototypes.block import WorkflowBlock


def load_enterprise_blocks() -> List[Type[WorkflowBlock]]:
    return [
        OPCWriterSinkBlockV1,
    ]
