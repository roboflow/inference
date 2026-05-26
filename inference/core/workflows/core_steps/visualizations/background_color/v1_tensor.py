from typing import Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.annotators.background_color import (
    BackgroundColorAnnotator,
)
from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    PredictionsVisualizationBlock,
    PredictionsVisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    STRING_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/background_color_visualization@v1"
SHORT_DESCRIPTION = (
    "Apply a mask to cover all areas outside the detected regions in an image."
)
LONG_DESCRIPTION = """
Apply a colored overlay to areas outside detected regions, effectively masking the background while preserving detected objects at their original appearance.

## How This Block Works

This block takes an image and detection predictions and applies a colored overlay to all areas outside of the detected objects, leaving the detected regions unchanged. The block:

1. Takes an image and predictions as input
2. Creates a colored mask layer with the specified background color
3. Identifies detected regions from bounding boxes or segmentation masks (preserves detected objects)
4. Applies the colored overlay to all areas outside the detected regions with the specified opacity
5. Blends the colored overlay with the original image based on the opacity setting
6. Returns an annotated image where detected objects appear unchanged, while the background is filled with the specified color

The block works with both object detection predictions (using bounding boxes) and instance segmentation predictions (using masks). When masks are available, it preserves the exact shape of detected objects; otherwise, it uses bounding box regions. The opacity parameter controls how transparent or opaque the background overlay is, allowing you to create effects ranging from subtle background dimming (low opacity) to complete background replacement (high opacity). This creates a visual focus effect that highlights the detected objects by de-emphasizing or completely hiding the background.

## Common Use Cases

- **Object Focus and Highlighting**: Highlight detected objects by dimming or replacing the background, making objects stand out for presentations, documentation, or user interfaces
- **Background Removal Effects**: Create images where backgrounds are replaced with solid colors or semi-transparent overlays for product photography, content creation, or design workflows
- **Privacy and Anonymization**: Mask backgrounds while preserving detected objects (e.g., people, vehicles) to anonymize images, protect privacy, or comply with data protection requirements
- **Visual Debugging and Validation**: Dim backgrounds to focus attention on detected regions when validating model performance, checking detection accuracy, or debugging detection results
- **Presentation and Documentation**: Create clean, professional visualizations for reports, presentations, or documentation where you want to emphasize detected objects without distracting backgrounds
- **Content Creation and Editing**: Prepare images for further processing, compositing, or editing by isolating detected objects with colored backgrounds for easier manipulation or integration into other workflows

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Bounding Box Visualization, Polygon Visualization) to add additional annotations on top of the background-colored image for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with background coloring for documentation, reporting, or archiving
- **Webhook blocks** to send visualized results with background coloring to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with background coloring as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with background coloring for live monitoring, tracking visualization, or post-processing analysis
"""


class BackgroundColorManifest(PredictionsVisualizationManifest):
    type: Literal[f"{TYPE}", "BackgroundColorVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Background Color Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-fill-drip",
                "blockPriority": 3,
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

    color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color to use for the background overlay. Areas outside detected regions will be filled with this color. Can be a color name (e.g., 'BLACK', 'WHITE') or color code in HEX format (e.g., '#000000') or RGB format (e.g., 'rgb(0, 0, 0)').",
        default="BLACK",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.background_color"],
    )

    opacity: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Opacity of the background overlay, ranging from 0.0 (fully transparent, original background visible) to 1.0 (fully opaque, complete background replacement). Values between 0.0 and 1.0 create a blend between the original image and the background color. Lower values create subtle background dimming, while higher values create stronger background replacement effects.",
        default=0.5,
        examples=[0.5, "$inputs.opacity"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BackgroundColorVisualizationBlockV1(PredictionsVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BackgroundColorManifest

    def getAnnotator(
        self,
        color: str,
        opacity: float,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color,
                    opacity,
                ],
            )
        )

        if key not in self.annotatorCache:
            background_color = str_to_color(color)
            self.annotatorCache[key] = BackgroundColorAnnotator(
                color=background_color,
                opacity=opacity,
            )

        return self.annotatorCache[key]

    def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        color: str,
        opacity: Optional[float],
    ) -> BlockResult:
        annotator = self.getAnnotator(color, opacity)
        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
