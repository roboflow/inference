from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.exceptions import FeatureDeprecatedError
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    IMAGE_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
**DEPRECATED.** L2CS Gaze detection has been removed from inference along
with the MediaPipe dependency. Invoking this block raises
`FeatureDeprecatedError` (HTTP 410 Gone).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Gaze Detection",
            "version": "v1",
            "short_description": "Detect faces and estimate gaze direction (deprecated).",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["gaze", "face"],
            "deprecated": True,
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-eyes",
                "blockPriority": 13.5,
            },
        },
        protected_namespaces=(),
    )

    type: Literal["roboflow_core/gaze@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    do_run_face_detection: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Whether to run face detection. Set to False if input images are pre-cropped face images.",
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="face_predictions",
                kind=[KEYPOINT_DETECTION_PREDICTION_KIND],
                description="Facial landmark predictions",
            ),
            OutputDefinition(
                name="yaw_degrees",
                kind=[FLOAT_KIND],
                description="Yaw angle in degrees (-180 to 180, negative is left)",
            ),
            OutputDefinition(
                name="pitch_degrees",
                kind=[FLOAT_KIND],
                description="Pitch angle in degrees (-90 to 90, negative is down)",
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class GazeBlockV1(WorkflowBlock):
    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        do_run_face_detection: bool,
    ) -> BlockResult:
        raise FeatureDeprecatedError(
            feature="roboflow_core/gaze@v1",
            reason="MediaPipe dependency removed from inference",
        )
