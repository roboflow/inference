from typing import Literal, Optional, Type

from inference.core.workflows.core_steps.sinks.noop import (
    DisableSink,
    disabled_sink_response,
    versioned_sink_manifest_config,
)
from inference.core.workflows.core_steps.sinks.s3.v1 import (
    BlockManifest as BlockManifestV1,
)
from inference.core.workflows.core_steps.sinks.s3.v1 import S3SinkBlockV1
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest


class BlockManifest(BlockManifestV1):
    model_config = versioned_sink_manifest_config(BlockManifestV1, version="v2")
    type: Literal["roboflow_core/s3_sink@v2"]
    disable_sink: DisableSink = False


class S3SinkBlockV2(S3SinkBlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        content: str,
        file_type: Literal["csv", "json", "txt"],
        output_mode: Literal["append_log", "separate_files"],
        bucket_name: str,
        s3_prefix: str,
        file_name_prefix: str,
        max_entries_per_file: int,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        disable_sink: bool = False,
    ) -> BlockResult:
        if disable_sink:
            return disabled_sink_response()
        return super().run(
            content=content,
            file_type=file_type,
            output_mode=output_mode,
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            file_name_prefix=file_name_prefix,
            max_entries_per_file=max_entries_per_file,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
        )
