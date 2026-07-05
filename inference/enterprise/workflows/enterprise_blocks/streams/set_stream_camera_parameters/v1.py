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
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference.enterprise.workflows.stream_camera_parameters.register_catalog import (
    REGISTER_LABELS,
    build_parameter_delta,
    registers_for_camera_family,
)
from inference.enterprise.workflows.stream_camera_parameters.service import (
    apply_camera_register_parameters,
)

LONG_DESCRIPTION = """
Apply a runtime camera register value on an edge inference stream.

Wire an upstream step or workflow input as **value** (already transformed/scaled as needed).
Pick the target **stream** and a friendly **register** for the camera family (focus, line rate,
brightness, etc.) instead of typing raw Device Manager JSON keys.

Use **manual** register mode to write a specific key when needed.
"""


class SetStreamCameraParametersBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "name": "Set Stream Camera Parameters",
            "version": "v1",
            "short_description": "Write a camera register on an edge stream from a workflow value.",
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

    value: Union[
        float,
        int,
        str,
        WorkflowParameterSelector(kind=[INTEGER_KIND, FLOAT_KIND, STRING_KIND]),
    ] = Field(
        description="Register value from an upstream step or workflow input.",
        examples=["$steps.plc_read.value", "$inputs.line_speed"],
    )
    register_key: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        default="manual",
        alias="register",
        description="Friendly register name for the selected camera family.",
        examples=["focus", "line_rate", "brightness", "manual"],
    )
    camera_family: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        default="",
        description="Camera family for register mapping (usb, ai1, basler, basler_line_scan, lucid, lucid_line_scan).",
        examples=["ai1", "basler_line_scan", "lucid_line_scan"],
    )
    stream_name: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        default="",
        description="Edge stream / pipeline id. Leave empty when only one stream is active.",
        examples=["aione", "line-scan-1"],
    )
    device_id: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        default="",
        description="Optional workspace device id (editor metadata; runtime uses stream_name).",
    )
    manual_register_key: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        default="",
        description="Hardware register key when register is manual.",
        examples=["AcquisitionLineRate", "lens_position"],
    )
    parameters: Union[Dict[str, Any], WorkflowParameterSelector(kind=[DICTIONARY_KIND])] = Field(
        default_factory=dict,
        description="Advanced: raw parameter object (overrides register mapping when non-empty).",
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
        description="Reference to the upstream step output.",
        examples=["$inputs.image", "$steps.plc_read.output"],
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
        value: Any,
        register_key: str,
        camera_family: str,
        stream_name: str,
        device_id: str,
        manual_register_key: str,
        parameters: Dict[str, Any],
        persist: bool,
        only_if_changed: bool,
        depends_on: any,
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> Dict[str, Any]:
        resolved_stream_name = stream_name.strip() if stream_name else None
        resolved_parameters = parameters if isinstance(parameters, dict) else {}

        if not resolved_parameters:
            try:
                resolved_parameters = build_parameter_delta(
                    str(register_key or "manual"),
                    value,
                    camera_family=str(camera_family or ""),
                    manual_register_key=str(manual_register_key or "") or None,
                )
            except ValueError as exc:
                return {
                    "success": False,
                    "applied": [],
                    "failed": [],
                    "skipped": False,
                    "message": str(exc),
                }

        result = apply_camera_register_parameters(
            resolved_parameters,
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


__all__ = [
    "SetStreamCameraParametersBlockManifest",
    "SetStreamCameraParametersBlockV1",
    "REGISTER_LABELS",
    "registers_for_camera_family",
]
