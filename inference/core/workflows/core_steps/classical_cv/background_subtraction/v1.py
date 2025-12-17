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

SHORT_DESCRIPTION: str = "Subtract an image from it's background history."
LONG_DESCRIPTION: str = """
This block uses background subtraction to detect motion in an image in order to highlight areas of motion.
The output of the block can be used to train and infer on motion based models.
"""


class BackgroundSubtractionManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/background_subtraction@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Background Subtraction",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
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
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    threshold: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="Threshold",
        description="The threshold value for the squared Mahalanobis distance for background subtraction."
        " Smaller values increase sensitivity to motion.",
        examples=[16],
        validation_alias=AliasChoices("threshold"),
        default=16,
    )

    history: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        title="History",
        description="The number of previous frames to use for background subtraction.",
        examples=[30],
        validation_alias=AliasChoices("history"),
        default=30,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[IMAGE_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BackgroundSubtractionBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_motion = False
        self.backSub = None
        self.threshold = None
        self.history = None
        self.frame_count = 0

    @classmethod
    def get_manifest(cls) -> Type[BackgroundSubtractionManifest]:
        return BackgroundSubtractionManifest

    def run(
        self, image: WorkflowImageData, threshold: int, history: int, *args, **kwargs
    ) -> BlockResult:
        if not self.backSub or self.threshold != threshold or self.history != history:
            self.threshold = threshold
            self.history = history
            self.backSub = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=threshold, detectShadows=True
            )

        frame = image.numpy_image

        # apply background subtraction
        fg_mask = self.backSub.apply(frame)

        # workflows require 3 channel images
        color_mask = np.stack((fg_mask,) * 3, axis=-1)

        output_image = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=color_mask,
        )

        return {
            OUTPUT_IMAGE_KEY: output_image,
        }
