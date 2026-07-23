from typing import Literal, Type

from inference.core.workflows.core_steps.sinks.local_file.v1 import (
    BlockManifest as BlockManifestV1,
)
from inference.core.workflows.core_steps.sinks.local_file.v1 import LocalFileSinkBlockV1
from inference.core.workflows.core_steps.sinks.noop import (
    DisableSink,
    disabled_sink_response,
    versioned_sink_manifest_config,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest


class BlockManifest(BlockManifestV1):
    model_config = versioned_sink_manifest_config(BlockManifestV1, version="v2")
    type: Literal["roboflow_core/local_file_sink@v2"]
    disable_sink: DisableSink = False


class LocalFileSinkBlockV2(LocalFileSinkBlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        content: str,
        file_type: Literal["csv", "json", "txt"],
        output_mode: Literal["append_log", "separate_files"],
        target_directory: str,
        file_name_prefix: str,
        max_entries_per_file: int,
        disable_sink: bool = False,
    ) -> BlockResult:
        if disable_sink:
            return disabled_sink_response()
        return super().run(
            content=content,
            file_type=file_type,
            output_mode=output_mode,
            target_directory=target_directory,
            file_name_prefix=file_name_prefix,
            max_entries_per_file=max_entries_per_file,
        )
