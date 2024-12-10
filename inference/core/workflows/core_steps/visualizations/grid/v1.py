from typing import Literal, Optional, Type, Union, List

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
    ImageParentMetadata,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

import cv2
import math
import uuid
import numpy as np

TYPE: str = "roboflow_core/grid_visualization@v1"
SHORT_DESCRIPTION = "Shows an array of images in a grid."
LONG_DESCRIPTION = """
The `GridVisualization` block displays an array of images in a grid.
It will automatically resize the images to in the specified width and
height. The first image will be in the top left corner, and the rest
will be added to the right of the previous image until the row is full.
"""


class GridVisualizationManifest(WorkflowBlockManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Grid Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-grid",
            },
        }
    )

    images: Selector(
        kind=[
            LIST_OF_VALUES_KIND
        ]
    ) = Field(  # type: ignore
        description="Images to visualize",
        examples=["$steps.buffer.output"],
    )

    width: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Width of the output image.",
        default=2560,
        examples=[2560, "$inputs.width"],
    )

    height: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Height of the output image.",
        default=1440,
        examples=[1440, "$inputs.height"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    IMAGE_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class GridVisualizationBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_input = None
        self.prev_output = None

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return GridVisualizationManifest

    def run(
        self,
        images: List[WorkflowImageData],
        width: int,
        height: int
    ) -> BlockResult:
        # use previous result if input hasn't changed
        if self.prev_output is not None:
            if len(self.prev_input) == len(images) and all(
                self.prev_input[i] == images[i] for i in range(len(images))
            ):
                return {OUTPUT_IMAGE_KEY: self.prev_output}

        output = getImageFor(images, width, height)
        
        self.prev_input = images
        self.prev_output = output

        return {OUTPUT_IMAGE_KEY: output}

def getImageFor(images: List[WorkflowImageData], width: int, height: int) -> WorkflowImageData:
    if images is None or len(images) == 0:
        return getEmptyImage(width, height)
    else:
        np_image = createGrid(images, width, height)
        return WorkflowImageData.copy_and_replace(
            origin_image_data=images[0], numpy_image=np_image
        )

def getEmptyImage(width: int, height: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=str(uuid.uuid4())),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8)
    )

def createGrid(images: List[WorkflowImageData], width: int, height: int) -> WorkflowImageData:
    grid_size = math.ceil(math.sqrt(len(images)))
    img = np.zeros((height, width, 3), dtype=np.uint8)

    cell_width = width // grid_size
    cell_height = height // grid_size

    for r in range(grid_size):
        for c in range(grid_size):
            index = r * grid_size + c

            if index >= len(images):
                break
            
            if images[index] is None:
                continue

            img_data = images[index].numpy_image
            # resize, preserving aspect ratio & fit within cell width and height
            img_data = resizeImage(img_data, cell_width, cell_height)
            img_data_height, img_data_width, _ = img_data.shape

            # place image in cell (centered)
            start_x = c * cell_width + (cell_width - img_data_width) // 2
            start_y = r * cell_height + (cell_height - img_data_height) // 2

            # Clamp to avoid negative indices
            start_x = max(start_x, 0)
            start_y = max(start_y, 0)

            end_x = start_x + img_data_width
            end_y = start_y + img_data_height

            # Ensure we do not exceed the canvas boundaries
            end_x = min(end_x, width)
            end_y = min(end_y, height)

            # If for some reason the image doesn't fit perfectly, we crop it
            target_height = end_y - start_y
            target_width = end_x - start_x

            img[start_y:end_y, start_x:end_x] = img_data[:target_height, :target_width]
    
    return img

def resizeImage(img: np.ndarray, width: int, height: int) -> np.ndarray:
    img_height, img_width, _ = img.shape
    scale_w = width / img_width
    scale_h = height / img_height
    scale = min(scale_w, scale_h)  # choose the scale that fits both dimensions
    
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
