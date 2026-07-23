from typing import Dict, List, Literal, Optional, Type

from inference.core.workflows.core_steps.sinks.noop import (
    DisableSink,
    versioned_sink_manifest_config,
)
from inference.core.workflows.execution_engine.entities.base import (
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest
from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1 import (
    ModbusTCPBlockManifest as ModbusTCPBlockManifestV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1 import (
    ModbusTCPBlockV1,
)


class ModbusTCPBlockManifest(ModbusTCPBlockManifestV1):
    model_config = versioned_sink_manifest_config(
        ModbusTCPBlockManifestV1, version="v2"
    )
    type: Literal["roboflow_core/modbus_tcp@v2"]
    disable_sink: DisableSink = False


class ModbusTCPBlockV2(ModbusTCPBlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ModbusTCPBlockManifest

    def run(
        self,
        plc_ip: str,
        plc_port: int,
        mode: str,
        registers_to_read: List[int],
        registers_to_write: Dict[int, int],
        depends_on: object,
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
        disable_sink: bool = False,
    ) -> dict:
        if disable_sink:
            return {"modbus_results": []}
        return super().run(
            plc_ip=plc_ip,
            plc_port=plc_port,
            mode=mode,
            registers_to_read=registers_to_read,
            registers_to_write=registers_to_write,
            depends_on=depends_on,
            image=image,
            metadata=metadata,
        )
