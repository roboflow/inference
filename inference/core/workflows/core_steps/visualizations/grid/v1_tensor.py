import math
import uuid
from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
from pydantic import ConfigDict, Field

from inference.core.cache.lru_cache import LRUCache
from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
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

TYPE: str = "roboflow_core/grid_visualization@v1"
SHORT_DESCRIPTION = "Shows an array of images in a grid."
LONG_DESCRIPTION = """
Arrange multiple images in a grid layout, automatically organizing a list of images into a square grid pattern with automatic resizing and cell-based positioning for side-by-side comparison, thumbnail displays, or batch visualization.

## How This Block Works

This block takes a list of images and arranges them into a grid layout within a single output image. The block:

1. Takes a list of images and output dimensions (width and height) as input
2. Calculates the grid size based on the number of images (creates a square grid with dimensions equal to the square root of the image count, rounded up)
3. Divides the output canvas into equal-sized cells based on the grid dimensions
4. Resizes each input image to fit within its assigned cell while maintaining aspect ratio (images are scaled to fit the cell dimensions without distortion)
5. Places images in the grid starting from the top-left corner, filling left-to-right and top-to-bottom (row-major order)
6. Centers each resized image within its cell, creating evenly spaced grid layout
7. Returns a single output image containing all input images arranged in the grid

The block automatically organizes multiple images into a grid for easy comparison or batch viewing. Each image is resized to fit its grid cell while preserving aspect ratio, and images are centered within their cells. The grid dimensions are automatically calculated to create a roughly square grid (e.g., 4 images = 2x2, 9 images = 3x3, 10 images = 4x4). This creates a compact, organized layout ideal for comparing multiple images, displaying thumbnails, or creating batch visualization outputs. The block uses caching to optimize performance when the same images are reused.

## Common Use Cases

- **Batch Image Comparison**: Arrange multiple images side-by-side in a grid for easy comparison, allowing you to visualize results from different models, time periods, or processing steps simultaneously
- **Thumbnail Gallery Creation**: Create thumbnail grids from collections of images for gallery displays, image browsers, or preview interfaces where multiple images need to be shown in a compact layout
- **Multi-Image Workflow Results**: Display results from multi-image workflows (like batch processing, image slicer outputs, or buffer collections) in an organized grid format for overview visualization
- **Before/After Comparisons**: Arrange before and after images, original and processed versions, or multiple workflow outputs in a grid for comparison and validation workflows
- **Time-Series Visualization**: Display images from different time points, frames, or snapshots in a grid to visualize temporal changes, sequences, or progression over time
- **Quality Control and Review**: Create grid layouts for quality control workflows, batch review, or inspection processes where multiple images need to be viewed together for evaluation or validation

## Connecting to Other Blocks

The grid output image from this block can be connected to:

- **Image processing blocks** (e.g., Buffer, Image Slicer, Dynamic Crop) to receive lists of images that are arranged into grid layouts
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save grid images for documentation, reporting, or batch review purposes
- **Webhook blocks** to send grid visualizations to external systems, APIs, or web applications for display in dashboards, galleries, or batch viewing interfaces
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send grid images as visual evidence in alerts or reports containing multiple images
- **Video output blocks** to create video streams or recordings with grid layouts for live multi-image monitoring or batch visualization workflows
- **Other visualization blocks** that can accept single images, allowing grid outputs to be further processed or combined with additional annotations
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
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-grid",
            },
        }
    )

    images: Selector(kind=[LIST_OF_VALUES_KIND]) = Field(  # type: ignore
        description="List of images to arrange in a grid layout. Can be a list of image outputs from blocks like Buffer, Image Slicer, Dynamic Crop, or other blocks that output multiple images. Images will be automatically arranged in a square grid (calculated from the number of images) and resized to fit their grid cells while maintaining aspect ratio.",
        examples=["$steps.buffer.output"],
    )

    width: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Width of the output grid image in pixels. Controls the total width of the canvas where the image grid will be arranged. The width is divided into equal-sized cells based on the grid dimensions. Typical values range from 1280 to 3840 pixels depending on desired output size and number of images.",
        default=2560,
        examples=[2560, "$inputs.width"],
    )

    height: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Height of the output grid image in pixels. Controls the total height of the canvas where the image grid will be arranged. The height is divided into equal-sized cells based on the grid dimensions. Typical values range from 720 to 2160 pixels depending on desired output size and number of images.",
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

        self.thumbCache = LRUCache()

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return GridVisualizationManifest

    def run(
        self, images: List[WorkflowImageData], width: int, height: int
    ) -> BlockResult:
        # use previous result if input hasn't changed
        if self.prev_output is not None:
            if len(self.prev_input) == len(images) and all(
                self.prev_input[i] == images[i] for i in range(len(images))
            ):
                return {OUTPUT_IMAGE_KEY: self.prev_output}

        self.thumbCache.set_max_size(len(images) + 1)
        output = self.getImageFor(images, width, height)

        self.prev_input = images
        self.prev_output = output

        return {OUTPUT_IMAGE_KEY: output}

    def getImageFor(
        self, images: List[WorkflowImageData], width: int, height: int
    ) -> WorkflowImageData:
        if images is None or len(images) == 0:
            return self.getEmptyImage(width, height)
        else:
            np_image = self.createGrid(images, width, height)
            return WorkflowImageData.copy_and_replace(
                origin_image_data=images[0], numpy_image=np_image
            )

    def getEmptyImage(self, width: int, height: int) -> WorkflowImageData:
        return WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id=str(uuid.uuid4())),
            numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
        )

    def createGrid(
        self, images: List[WorkflowImageData], width: int, height: int
    ) -> WorkflowImageData:
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

                cacheKey = f"{id(images[index])}_{cell_width}_{cell_height}"
                if self.thumbCache.get(cacheKey) is None:
                    self.thumbCache.set(
                        cacheKey,
                        self.resizeImage(
                            images[index].numpy_image, cell_width, cell_height
                        ),
                    )
                img_data = self.thumbCache.get(cacheKey)

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

                img[start_y:end_y, start_x:end_x] = img_data[
                    :target_height, :target_width
                ]

        return img

    def resizeImage(self, img: np.ndarray, width: int, height: int) -> np.ndarray:
        img_height, img_width, _ = img.shape
        scale_w = width / img_width
        scale_h = height / img_height
        scale = min(scale_w, scale_h)  # choose the scale that fits both dimensions

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
