from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference.enterprise.workflows.stream_camera_parameters.service import (
    apply_camera_register_parameters,
)

LONG_DESCRIPTION = """
Apply runtime camera register or property changes to the active edge inference stream.

Pass a **parameters** object using the same keys as Device Manager stream camera settings
(for example `v4l2_camera_properties`, `lucid_camera_settings`, or top-level GenICam keys
such as `AcquisitionLineRate`). Map PLC values upstream with an **Expression** block.

When **stream_name** is omitted, the block targets the only active pipeline on the device.
Provide **stream_name** when multiple streams are running.
"""


class SetStreamCameraParametersBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Set Stream Camera Parameters",
            "version": "v1",
            "short_description": "Apply runtime camera register changes on an edge stream.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "industrial",
                "icon": "fal fa-camera",
                "blockPriority": 15,
                "enterprise_only": True,
                "local_only": True,
            },
        }
    )

    type: Literal["roboflow_enterprise/set_stream_camera_parameters@v1"]

    parameters: Union[
        Dict[str, Any], WorkflowParameterSelector(kind=[DICTIONARY_KIND])
    ] = Field(
        description="Camera settings delta to apply, matching Device Manager stream config keys.",
        examples=[
            {"v4l2_camera_properties": {"exposure_absolute": 200}},
            {"AcquisitionLineRate": 128000},
        ],
    )
    stream_name: Union[
        str, WorkflowParameterSelector(kind=[STRING_KIND])
    ] = Field(
        default="",
        description="Optional stream name. Required when multiple pipelines are active.",
        examples=["line-scan-1", "$inputs.stream_name"],
    )
    persist: bool = Field(
        default=False,
        description="Persist merged settings to the device config file after a successful apply.",
    )
    only_if_changed: bool = Field(
        default=True,
        description="Skip the hardware write when merged settings match the current stream config.",
    )
    depends_on: Selector() = Field(
        description="Reference to the step output this block depends on.",
        examples=["$steps.plc_read.output"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="success", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="applied", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="failed", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="skipped", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class SetStreamCameraParametersBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SetStreamCameraParametersBlockManifest

    def run(
        self,
        parameters: Dict[str, Any],
        stream_name: str,
        persist: bool,
        only_if_changed: bool,
        depends_on: any,
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> Dict[str, Any]:
        resolved_stream_name = stream_name.strip() if stream_name else None
        result = apply_camera_register_parameters(
            parameters,
            stream_name=resolved_stream_name,
            persist=persist,
            only_if_changed=only_if_changed,
        )
        return {
            "success": result.success,
            "applied": result.applied,
            "failed": result.failed,
            "skipped": result.skipped,
            "message": result.message or "",
        }
