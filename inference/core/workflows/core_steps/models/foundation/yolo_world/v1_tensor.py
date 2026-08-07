"""Tensor-native sibling of `roboflow_core/yolo_world_model@v1`.

DEPRECATION STUB. YOLO-World has no inference_models tensor-native path, so rather
than port it, it is deprecated in the tensor-native Workflows pipeline (same pattern
as `gaze/v1.py`). The manifest is kept identical to the numpy block so existing
workflow JSON still loads/validates; invoking the block raises
`FeatureDeprecatedError` (HTTP 410 Gone) at runtime. Flag-off keeps the working
numpy `YoloWorldModelBlockV1`.
"""

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
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    FloatZeroToOne,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
**DEPRECATED.** YOLO-World is deprecated in the tensor-native Workflows pipeline.
Invoking this block raises `FeatureDeprecatedError` (HTTP 410 Gone).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "YOLO-World Model",
            "version": "v1",
            "short_description": "Run a zero-shot object detection model (deprecated).",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "deprecated": True,
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 8,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/yolo_world_model@v1", "YoloWorldModel", "YoloWorld"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    class_names: Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]] = Field(
        description="One or more classes that you want YOLO-World to detect. The model accepts any string as an input, though does best with short descriptions of common objects.",
        examples=[["person", "car", "license plate"], "$inputs.class_names"],
    )
    version: Union[
        Literal["v2-s", "v2-m", "v2-l", "v2-x", "s", "m", "l", "x"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="v2-s",
        description="Variant of YoloWorld model",
        examples=["v2-s", "$inputs.variant"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.005,
        description="Confidence threshold for detections",
        examples=[0.005, "$inputs.confidence"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_supported_model_variants(cls) -> Optional[List[str]]:
        """Return list of model_id variants that can satisfy this block."""
        return [
            "yolo_world/v2-s",
            "yolo_world/v2-m",
            "yolo_world/v2-l",
            "yolo_world/v2-x",
            "yolo_world/s",
            "yolo_world/m",
            "yolo_world/l",
            "yolo_world/x",
        ]


class YoloWorldModelBlockV1(WorkflowBlock):

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
        class_names: List[str],
        version: str,
        confidence: Optional[float],
    ) -> BlockResult:
        raise FeatureDeprecatedError(
            feature="roboflow_core/yolo_world_model@v1",
            reason="YOLO-World is deprecated in the tensor-native pipeline (no inference_models path)",
        )
