from typing import Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    PredictionsVisualizationBlock,
    PredictionsVisualizationManifest,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/blur_visualization@v1"
SHORT_DESCRIPTION = "Blur detected objects in an image."
LONG_DESCRIPTION = """
Apply blur effects to detected objects in an image, obscuring their details while preserving the background, useful for privacy protection, content filtering, or visual emphasis.

## How This Block Works

This block takes an image and detection predictions and applies a blur effect to the detected objects, leaving the background unchanged. The block:

1. Takes an image and predictions as input
2. Identifies detected regions from bounding boxes or segmentation masks
3. Applies a blur effect (using average pooling) to the detected object regions
4. Preserves the background and areas outside detected objects unchanged
5. Returns an annotated image where detected objects are blurred, while the rest of the image remains sharp

The block works with both object detection predictions (using bounding boxes) and instance segmentation predictions (using masks). When masks are available, it blurs the exact shape of detected objects; otherwise, it blurs rectangular bounding box regions. The blur intensity is controlled by the kernel size parameter, where larger kernel sizes create stronger blur effects. This creates a visual effect that obscures or anonymizes detected objects while maintaining context from the surrounding image, making it ideal for privacy protection, content filtering, or focusing attention on the background.

## Common Use Cases

- **Privacy Protection and Anonymization**: Blur faces, people, license plates, or other sensitive information in images or videos to protect privacy, comply with data protection regulations, or anonymize content before sharing or publishing
- **Content Filtering and Moderation**: Obscure inappropriate or sensitive content in images or videos for content moderation workflows, safe content previews, or user-generated content filtering
- **Visual Emphasis and Focus**: Blur detected objects to draw attention to other parts of the image, create visual contrast between blurred foreground objects and sharp backgrounds, or emphasize specific elements in composition
- **Product Photography and E-commerce**: Blur detected distracting elements or secondary products in images to keep the main subject sharp and prominent for product photography, catalog creation, or e-commerce image preparation
- **Security and Surveillance**: Anonymize people, vehicles, or other identifiable elements in security footage or surveillance images while preserving scene context for analysis, reporting, or public sharing
- **Documentation and Reporting**: Create anonymized or censored versions of images for reports, documentation, or case studies where sensitive information needs to be obscured but overall context should remain visible

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Bounding Box Visualization, Polygon Visualization) to add additional annotations on top of blurred objects for comprehensive visualization or to indicate what was blurred
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save blurred images for documentation, reporting, or archiving privacy-protected content
- **Webhook blocks** to send blurred images to external systems, APIs, or web applications for content moderation, privacy-compliant sharing, or anonymized analysis
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send blurred images as privacy-protected visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with blurred objects for live monitoring, privacy-compliant video processing, or post-processing analysis
"""


class BlurManifest(PredictionsVisualizationManifest):
    type: Literal[f"{TYPE}", "BlurVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Blur Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "fad fa-glasses",
                "blockPriority": 4,
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

    kernel_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Size of the blur kernel used for average pooling. Larger values create stronger blur effects, making objects more obscured. Smaller values create subtle blur effects. Typical values range from 5 (light blur) to 51 (strong blur). Must be an odd number for optimal blurring performance.",
        default=15,
        examples=[15, "$inputs.kernel_size"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BlurVisualizationBlockV1(PredictionsVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlurManifest

    def getAnnotator(
        self,
        kernel_size: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(map(str, [kernel_size]))

        if key not in self.annotatorCache:
            self.annotatorCache[key] = sv.BlurAnnotator(kernel_size=kernel_size)
        return self.annotatorCache[key]

    def run(
        self,
        image: WorkflowImageData,
        predictions: sv.Detections,
        copy_image: bool,
        kernel_size: Optional[int],
    ) -> BlockResult:
        annotator = self.getAnnotator(kernel_size)
        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
