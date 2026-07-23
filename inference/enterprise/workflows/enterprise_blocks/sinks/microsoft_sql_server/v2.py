from typing import Any, Dict, List, Literal, Optional, Type, Union

from inference.core.workflows.core_steps.sinks.noop import (
    DisableSink,
    disabled_sink_response,
    versioned_sink_manifest_config,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest
from inference.enterprise.workflows.enterprise_blocks.sinks.microsoft_sql_server.v1 import (
    BlockManifest as BlockManifestV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.microsoft_sql_server.v1 import (
    MicrosoftSQLServerSinkBlockV1,
)


class BlockManifest(BlockManifestV1):
    model_config = versioned_sink_manifest_config(BlockManifestV1, version="v2")
    type: Literal["roboflow_core/microsoft_sql_server_sink@v2"]
    disable_sink: DisableSink = False


class MicrosoftSQLServerSinkBlockV2(MicrosoftSQLServerSinkBlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        host: str,
        port: int,
        database: str,
        table_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        username: Optional[str] = None,
        password: Optional[str] = None,
        fire_and_forget: bool = True,
        disable_sink: bool = False,
    ) -> BlockResult:
        if disable_sink:
            return disabled_sink_response()
        return super().run(
            host=host,
            port=port,
            database=database,
            table_name=table_name,
            data=data,
            username=username,
            password=password,
            fire_and_forget=fire_and_forget,
        )
