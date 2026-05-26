from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BackgroundSubtractionManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/background_subtraction@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Background Subtraction",
            "version": "v1",
            "short_description": "Subtract an image from its background history.",
            "long_description": (
                """
Create motion masks from video streams using OpenCV's background subtraction algorithm.

## How This Block Works

This block uses background subtraction (specifically the MOG2 algorithm) to identify pixels that differ from a learned background model and outputs a mask image highlighting motion areas. The block maintains state across frames to build and update the background model:

1. **Initializes background model** - on the first frame, creates a background subtractor using the specified history and threshold parameters
2. **Processes each frame** - applies background subtraction to identify pixels that differ from the learned background model
3. **Creates motion mask** - generates a foreground mask where white pixels represent motion areas and black pixels represent the background
4. **Converts to image format** - converts the single-channel mask to a 3-channel image format required by workflows
5. **Returns mask image** - outputs the motion mask as an image that can be visualized or processed further

The output mask image shows motion areas as white pixels against a black background, making it easy to visualize where motion occurred in the frame. This mask can be used for further analysis, visualization, or as input to other processing steps.

## Common Use Cases

- **Motion Visualization**: Create visual motion masks to see where movement occurs in video streams for monitoring, analysis, or debugging purposes
- **Preprocessing for Motion Models**: Generate motion masks as input data for training or inference with motion-based models that require mask data
- **Motion Area Extraction**: Extract regions of motion from video frames for further processing, analysis, or feature extraction
- **Video Analysis**: Analyze motion patterns by processing mask images to identify movement trends, activity levels, or motion characteristics
- **Background Removal**: Use motion masks to separate foreground (moving) objects from static background for segmentation or isolation tasks
- **Motion-based Filtering**: Use motion masks to filter or focus processing on areas where motion occurs, ignoring static background regions

## Connecting to Other Blocks

The motion mask image from this block can be connected to:

- **Visualization blocks** to display the motion mask overlayed on original images or as standalone visualizations
- **Object detection blocks** to run detection models only on motion regions identified by the mask
- **Image processing blocks** to apply additional transformations, filters, or analysis to motion mask images
- **Data storage blocks** (e.g., Local File Sink, Roboflow Dataset Upload) to save motion masks for training data, analysis, or documentation
- **Conditional logic blocks** to route workflow execution based on the presence or absence of motion in mask images
- **Model training blocks** to use motion masks as training data for motion-based models or segmentation tasks
"""
            ),
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-circle-minus",
                "blockPriority": 8,
                "opencv": True,
                "video": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image or video frame to process for background subtraction. The block processes frames sequentially to build a background model - each frame updates the background model and creates a motion mask showing areas that differ from the learned background. Can be connected from workflow inputs or previous steps.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    threshold: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="Threshold",
        description="Threshold value for the squared Mahalanobis distance used by the MOG2 background subtraction algorithm. Controls sensitivity to motion - smaller values increase sensitivity (detect smaller changes) but may produce more false positives, larger values decrease sensitivity (only detect significant changes) but may miss subtle motion. Recommended range is 8-32. Default is 16.",
        examples=[16, 8, 24, 32],
        validation_alias=AliasChoices("threshold"),
        default=16,
    )

    history: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="History",
        description="Number of previous frames used to build the background model. Controls how quickly the background adapts to changes - larger values (e.g., 50-100) create a more stable background model that's less sensitive to temporary changes but adapts slowly to permanent background changes. Smaller values (e.g., 10-20) allow faster adaptation but may treat moving objects as background if they stop moving. Default is 30 frames.",
        examples=[30, 50, 100],
        validation_alias=AliasChoices("history"),
        default=30,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[IMAGE_KIND],
                description="Motion mask image showing areas of detected motion. White pixels represent motion areas (foreground), black pixels represent the background. The mask is a 3-channel image suitable for visualization or further processing. Can be used as input to other blocks or saved as training data.",
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BackgroundSubtractionBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.back_sub = None

    @classmethod
    def get_manifest(cls) -> Type[BackgroundSubtractionManifest]:
        return BackgroundSubtractionManifest

    def run(
        self, image: WorkflowImageData, threshold: int, history: int, *args, **kwargs
    ) -> BlockResult:
        if not self.back_sub:
            self.back_sub = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=threshold, detectShadows=True
            )

        frame = image.numpy_image

        # apply background subtraction
        fg_mask = self.back_sub.apply(frame)

        # workflows require 3 channel images
        color_mask = np.stack((fg_mask,) * 3, axis=-1)

        output_image = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=color_mask,
        )

        return {
            OUTPUT_IMAGE_KEY: output_image,
        }
