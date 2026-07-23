from copy import deepcopy
from typing import Union

from pydantic import ConfigDict, Field
from typing_extensions import Annotated

from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

DisableSink = Annotated[
    Union[bool, Selector(kind=[BOOLEAN_KIND])],
    Field(
        description="When true, prevents this sink from producing side effects.",
        examples=[False, "$inputs.disable_sink"],
    ),
]


def disabled_sink_response() -> BlockResult:
    return {
        "error_status": False,
        "message": "Sink was disabled by parameter `disable_sink`",
    }


def versioned_sink_manifest_config(
    manifest: type[WorkflowBlockManifest],
    version: str,
) -> ConfigDict:
    json_schema_extra = deepcopy(manifest.model_config.get("json_schema_extra", {}))
    json_schema_extra["version"] = version
    return ConfigDict(json_schema_extra=json_schema_extra)
