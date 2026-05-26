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
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    STRING_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/model_comparison_visualization@v1"
SHORT_DESCRIPTION = "Visualize the difference between two models' detections."
LONG_DESCRIPTION = """
Compare predictions from two different models by color-coding areas where only one model detected objects, highlighting model differences while leaving overlapping predictions unchanged to visualize model agreement and disagreement.

## How This Block Works

This block takes an image and predictions from two models (Model A and Model B) and creates a visual comparison overlay that highlights differences between the models. The block:

1. Takes an image and two sets of predictions (predictions_a and predictions_b) as input
2. Creates masks for areas predicted by each model (using bounding boxes or segmentation masks if available)
3. Identifies four distinct regions:
   - Areas predicted only by Model A (colored with color_a, default green)
   - Areas predicted only by Model B (colored with color_b, default red)
   - Areas predicted by both models (left unchanged, allowing the original image to show through)
   - Areas predicted by neither model (colored with background_color, default black)
4. Applies colored overlays to the identified regions using the specified opacity
5. Returns an annotated image where model differences are visually distinguished with color coding

The block creates a side-by-side comparison visualization that makes it easy to see where models agree (unchanged areas) and where they disagree (color-coded areas). Areas where both models made predictions are left unchanged, allowing the original image to "shine through" and clearly showing model consensus. This visualization helps identify model strengths, weaknesses, and differences in detection behavior. The block works with object detection predictions (using bounding boxes) or instance segmentation predictions (using masks), making it versatile for comparing different model types.

## Common Use Cases

- **Model Evaluation and Comparison**: Compare two models' detection performance side-by-side to identify where models agree, disagree, or have different detection behaviors for model evaluation, benchmarking, or selection workflows
- **Model Development and Debugging**: Visualize differences between model versions, architectures, or configurations to understand how changes affect detection behavior, identify improvement opportunities, or debug model performance issues
- **Ensemble Model Analysis**: Compare predictions from different models in ensemble workflows to understand model agreement patterns, identify complementary strengths, or analyze consensus areas for ensemble decision-making
- **Training Data Analysis**: Compare model predictions to ground truth annotations or between training runs to identify patterns in detection differences, validate training improvements, or analyze model behavior across datasets
- **A/B Testing and Model Selection**: Visually compare candidate models to evaluate relative performance, identify detection differences, or make informed model selection decisions for deployment
- **Quality Assurance and Validation**: Validate model consistency, compare model performance on edge cases, or identify systematic differences between models for quality assurance, validation, or compliance workflows

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Model blocks** (e.g., Object Detection Model, Instance Segmentation Model) to receive predictions_a and predictions_b from different models for comparison
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save comparison visualizations for documentation, reporting, or analysis
- **Webhook blocks** to send comparison visualizations to external systems, APIs, or web applications for display in dashboards, model monitoring tools, or evaluation interfaces
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send comparison visualizations as visual evidence in alerts or reports for model performance monitoring
- **Video output blocks** to create annotated video streams or recordings with model comparison visualizations for live model evaluation, performance monitoring, or post-processing analysis
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
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-not-equal",
                "blockPriority": 16,
                "supervision": True,
                "warnings": [
                    {
                        "property": "copy_image",
                        "value": False,
                        "message": "This setting will mutate its input image. If the input is used by other blocks, it may cause unexpected behavior.",
                    }
                ],
            },
        }
    )

    predictions_a: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions from Model A (the first model being compared). Can be object detection, instance segmentation, or keypoint detection predictions. Areas predicted only by Model A (and not by Model B) will be colored with color_a. Works with bounding boxes or masks depending on prediction type.",
        examples=["$steps.object_detection_model.predictions"],
    )

    color_a: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color used to highlight areas predicted only by Model A (that Model B did not predict). Can be specified as a color name (e.g., 'GREEN', 'BLUE'), hex color code (e.g., '#00FF00', '#FFFFFF'), or RGB format (e.g., 'rgb(0, 255, 0)'). Default is GREEN to indicate Model A's unique predictions.",
        default="GREEN",
        examples=["GREEN", "#FFFFFF", "rgb(255, 255, 255)", "$inputs.color_a"],
    )

    predictions_b: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Predictions from Model B (the second model being compared). Can be object detection, instance segmentation, or keypoint detection predictions. Areas predicted only by Model B (and not by Model A) will be colored with color_b. Works with bounding boxes or masks depending on prediction type.",
        examples=["$steps.object_detection_model.predictions"],
    )

    color_b: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color used to highlight areas predicted only by Model B (that Model A did not predict). Can be specified as a color name (e.g., 'RED', 'BLUE'), hex color code (e.g., '#FF0000', '#FFFFFF'), or RGB format (e.g., 'rgb(255, 0, 0)'). Default is RED to indicate Model B's unique predictions.",
        default="RED",
        examples=["RED", "#FFFFFF", "rgb(255, 255, 255)", "$inputs.color_b"],
    )

    background_color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color used for areas predicted by neither model. Can be specified as a color name (e.g., 'BLACK', 'GRAY'), hex color code (e.g., '#000000', '#808080'), or RGB format (e.g., 'rgb(0, 0, 0)'). Default is BLACK to indicate areas where both models missed detections.",
        default="BLACK",
        examples=["BLACK", "#FFFFFF", "rgb(255, 255, 255)", "$inputs.background_color"],
    )

    opacity: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Opacity of the comparison overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls how transparent the color-coded overlays appear over the original image. Lower values create more transparent overlays where original image details remain more visible, while higher values create more opaque overlays with stronger color emphasis. Typical values range from 0.5 to 0.8 for balanced visibility.",
        default=0.7,
        examples=[0.7, "$inputs.opacity"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


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
