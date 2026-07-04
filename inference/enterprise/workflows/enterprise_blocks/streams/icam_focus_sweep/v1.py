from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Selector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference.enterprise.workflows.stream_camera_parameters.focus_sweep import (
    FocusSweepState,
    run_focus_sweep_tick,
)

LONG_DESCRIPTION = """
Incrementally sweep ICAM / AI1 lens focus on a fixed interval.

Each time the cooldown elapses, adds **step** to an internal counter (0..**max_focus**,
then wraps to 0) and writes **lens_position** via the edge `POST /cameras/configure` API.

Set **edge_base_url** to `http://127.0.0.1:8000` when the workflow runs on the ICAM device,
or `http://<icam-ip>:8000` when driving the camera from another edge host on the same LAN.
"""


class IcamFocusSweepBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "ICAM Focus Sweep",
            "version": "v1",
            "short_description": "Sweep lens focus on a timer for ICAM / AI1 devices.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "industrial",
                "icon": "fal fa-crosshairs",
                "blockPriority": 16,
                "enterprise_only": True,
                "local_only": True,
            },
        }
    )

    type: Literal["roboflow_enterprise/icam_focus_sweep@v1"]

    interval_seconds: Union[float, WorkflowParameterSelector(kind=[FLOAT_KIND])] = Field(
        default=10.0,
        description="Minimum seconds between focus updates.",
        ge=0.1,
    )
    step: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(
        default=5,
        description="Amount added to the focus counter on each tick.",
        ge=1,
    )
    max_focus: Union[int, WorkflowParameterSelector(kind=[INTEGER_KIND])] = Field(
        default=100,
        description="Inclusive upper bound before wrapping back to 0.",
        ge=1,
    )
    video_reference: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        default="0",
        description="USB video device index for the ICAM (typically 0).",
    )
    edge_base_url: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        default="http://127.0.0.1:8000",
        description="Base URL of the edge inference server that owns the camera.",
        examples=["http://127.0.0.1:8000", "http://192.168.0.15:8000"],
    )
    depends_on: Selector() = Field(
        description="Reference to the upstream step output.",
        examples=["$inputs.image"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="focus", kind=[INTEGER_KIND]),
            OutputDefinition(name="updated", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="success", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class IcamFocusSweepBlockV1(WorkflowBlock):

    def __init__(self) -> None:
        self._state = FocusSweepState()

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return IcamFocusSweepBlockManifest

    def run(
        self,
        interval_seconds: float,
        step: int,
        max_focus: int,
        video_reference: str,
        edge_base_url: str,
        depends_on: any,
        image: Optional[WorkflowImageData] = None,
        metadata: Optional[VideoMetadata] = None,
    ) -> Dict[str, Any]:
        state, updated, result = run_focus_sweep_tick(
            self._state,
            interval_seconds=float(interval_seconds),
            step=int(step),
            max_focus=int(max_focus),
            video_reference=str(video_reference),
            edge_base_url=str(edge_base_url).rstrip("/"),
        )
        self._state = state

        message = ""
        success = True
        if result is not None:
            success = result.success
            message = result.message or ""
            if result.failed:
                message = f"{message} failed={result.failed}".strip()

        return {
            "focus": self._state.focus_value,
            "updated": updated,
            "success": success,
            "message": message,
        }
