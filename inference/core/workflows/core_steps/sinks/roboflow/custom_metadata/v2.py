from typing import Literal, Type, Union

import supervision as sv

from inference.core.workflows.core_steps.sinks.noop import (
    DisableSink,
    disabled_sink_response,
    versioned_sink_manifest_config,
)
from inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1 import (
    BlockManifest as BlockManifestV1,
)
from inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1 import (
    RoboflowCustomMetadataBlockV1,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest


class BlockManifest(BlockManifestV1):
    model_config = versioned_sink_manifest_config(BlockManifestV1, version="v2")
    type: Literal["roboflow_core/roboflow_custom_metadata@v2"]
    disable_sink: DisableSink = False


class RoboflowCustomMetadataBlockV2(RoboflowCustomMetadataBlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        fire_and_forget: bool,
        field_name: str,
        field_value: str,
        predictions: Union[sv.Detections, dict],
        disable_sink: bool = False,
    ) -> BlockResult:
        if disable_sink:
            return disabled_sink_response()
        return super().run(
            fire_and_forget=fire_and_forget,
            field_name=field_name,
            field_value=field_value,
            predictions=predictions,
        )
