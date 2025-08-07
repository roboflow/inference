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
The Background Color Visualization block lets you change the color of regions not detected by a detection model.

This is useful for if you want to highlight detected regions so they are easier to see in an image, or if you want to remove backgrounds from an image.

By default, this block makes regions not detected by a detection model opaque. You can also configure the block to change the colour of undetected regions.

This block works with:

- Object detection models
- Segmentation models

Here is an example of the block in use:

![](https://docs.roboflow.com/~gitbook/image?url=https%3A%2F%2F662926385-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252F-M6S9nPJhEX9FYH6clfW%252Fuploads%252FSenp0sD01hM934sNQ9cP%252FScreenshot%25202025-05-23%2520at%252018.40.50.png%3Falt%3Dmedia%26token%3Ddf7c9760-7327-4820-954c-9be5baba70ec&width=768&dpr=4&quality=100&sign=f638faf3&sv=2)

You can change the colour and opacity of the background from the block configuration options.
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
        description="Color of the background.",
        default="BLACK",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.background_color"],
    )

    opacity: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Transparency of the Mask overlay.",
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
