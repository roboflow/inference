from typing import Dict, List, Literal, Optional, Type, Union

from inference.core.workflows.core_steps.sinks.noop import (
    DisableSink,
    versioned_sink_manifest_config,
)
from inference.core.workflows.execution_engine.entities.base import (
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest
from inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1 import (
    PLCBlockManifest as BlockManifestV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1 import (
    PLCBlockV1,
)


class BlockManifest(BlockManifestV1):
    model_config = versioned_sink_manifest_config(BlockManifestV1, version="v2")
    type: Literal["roboflow_core/sinks@v2"]
    disable_sink: DisableSink = False


class PLCBlockV2(PLCBlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        plc_ip: str,
        mode: str,
        tags_to_read: List[str],
        tags_to_write: Dict[str, Union[int, float, str]],
        depends_on: object,
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
        disable_sink: bool = False,
    ) -> dict:
        if disable_sink:
            return {"plc_results": []}
        return super().run(
            plc_ip=plc_ip,
            mode=mode,
            tags_to_read=tags_to_read,
            tags_to_write=tags_to_write,
            depends_on=depends_on,
            image=image,
            metadata=metadata,
        )
