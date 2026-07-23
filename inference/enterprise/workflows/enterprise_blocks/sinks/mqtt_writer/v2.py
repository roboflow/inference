from typing import Literal, Optional, Type

from inference.core.workflows.core_steps.sinks.noop import (
    DisableSink,
    disabled_sink_response,
    versioned_sink_manifest_config,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import (
    BlockManifest as BlockManifestV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import (
    MQTTWriterSinkBlockV1,
)


class BlockManifest(BlockManifestV1):
    model_config = versioned_sink_manifest_config(BlockManifestV1, version="v2")
    type: Literal["mqtt_writer_sink@v2"]
    disable_sink: DisableSink = False


class MQTTWriterSinkBlockV2(MQTTWriterSinkBlockV1):
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
        disable_sink: bool = False,
    ) -> BlockResult:
        if disable_sink:
            return disabled_sink_response()
        return super().run(
            host=host,
            port=port,
            topic=topic,
            message=message,
            username=username,
            password=password,
            qos=qos,
            retain=retain,
            timeout=timeout,
        )
