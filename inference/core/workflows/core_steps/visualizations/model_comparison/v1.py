from typing import Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.annotators.model_comparison import (
    ModelComparisonAnnotator,
)
from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    PredictionsVisualizationBlock,
    VisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    FloatZeroToOne,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/model_comparison_visualization@v1"
SHORT_DESCRIPTION = "Visualizes the difference between two models' detections."
LONG_DESCRIPTION = """
The `ModelComparisonVisualization` block draws all areas
predicted by neither model with a specified color,
lets overlapping areas of the predictions shine through,
and colors areas predicted by only one model with a distinct color.
"""


class ModelComparisonManifest(VisualizationManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Model Comparison Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-not-equal",
            },
        }
    )

    predictions_a: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions",
        examples=["$steps.object_detection_model.predictions"],
    )

    color_a: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the areas Model A predicted that Model B did not..",
        default="GREEN",
        examples=["GREEN", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.color_a"],
    )

    predictions_b: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions",
        examples=["$steps.object_detection_model.predictions"],
    )

    color_b: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the areas Model B predicted that Model A did not.",
        default="RED",
        examples=["RED", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.color_b"],
    )

    background_color: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the areas neither model predicted.",
        default="BLACK",
        examples=["BLACK", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.background_color"],
    )

    opacity: Union[FloatZeroToOne, WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Transparency of the overlay.",
        default=0.7,
        examples=[0.7, "$inputs.opacity"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class ModelComparisonVisualizationBlockV1(PredictionsVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ModelComparisonManifest

    def getAnnotator(
        self,
        color_a: str,
        color_b: str,
        background_color: str,
        opacity: float,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_a,
                    color_b,
                    background_color,
                    opacity,
                ],
            )
        )

        if key not in self.annotatorCache:
            color_a = str_to_color(color_a)
            color_b = str_to_color(color_b)
            background_color = str_to_color(background_color)
            self.annotatorCache[key] = ModelComparisonAnnotator(
                color_a=color_a,
                color_b=color_b,
                background_color=background_color,
                opacity=opacity,
            )

        return self.annotatorCache[key]

    def run(
        self,
        image: WorkflowImageData,
        predictions_a: sv.Detections,
        color_a: str,
        predictions_b: sv.Detections,
        color_b: str,
        background_color: str,
        opacity: Optional[float],
        copy_image: bool,
    ) -> BlockResult:
        annotator = self.getAnnotator(
            color_a=color_a,
            color_b=color_b,
            background_color=background_color,
            opacity=opacity,
        )

        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections_a=predictions_a,
            detections_b=predictions_b,
        )

        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=annotated_image,
        )

        return {OUTPUT_IMAGE_KEY: output}
