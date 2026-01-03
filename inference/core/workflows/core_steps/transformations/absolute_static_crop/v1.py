from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Extract a fixed rectangular region from input images using absolute pixel coordinates specified by center point and dimensions, creating consistent crops from the same image location across all inputs for region-of-interest extraction and fixed-area analysis workflows.

## How This Block Works

This block crops a fixed rectangular region from input images using absolute pixel coordinates, unlike dynamic cropping which uses detection bounding boxes. The block:

1. Receives input images and absolute coordinate specifications (x_center, y_center, width, height)
2. Calculates the crop boundaries from the center point and dimensions:
   - Computes x_min and y_min by subtracting half the width/height from the center coordinates
   - Computes x_max and y_max by adding the width/height to the minimum coordinates
   - Rounds coordinate values to integer pixel positions
3. Extracts the rectangular region from the image using array slicing (from y_min to y_max, x_min to x_max)
4. Validates that the cropped region has content (returns None if the crop would be empty, such as when coordinates are outside image bounds)
5. Creates a cropped image object with metadata tracking the crop's origin (original image, offset coordinates, unique crop identifier)
6. Preserves video metadata if the input is from video (maintains frame information and temporal context)
7. Returns the cropped image for each input image

The block uses fixed coordinates, so the same region is extracted from all images in a batch, making it suitable for extracting consistent regions across multiple images (e.g., always cropping the top-right corner, extracting a fixed area of interest, or focusing on a specific image section). The center-based coordinate system allows specifying crops by their center point rather than corner coordinates, which can be more intuitive for defining regions. The block handles edge cases gracefully by returning None for invalid crops (coordinates outside image bounds or resulting in empty regions).

## Common Use Cases

- **Fixed Region Extraction**: Extract the same image region from multiple images for consistent analysis (e.g., crop a specific area of interest like a logo zone, extract a fixed region for watermark detection, crop a consistent area for pattern matching), enabling standardized region analysis across image batches
- **Region-of-Interest Focus**: Isolate specific areas of images for detailed processing (e.g., crop a specific quadrant of surveillance frames, extract a fixed region for text recognition, focus on a known area of interest), enabling focused analysis of predetermined image regions
- **Multi-Stage Workflow Preparation**: Extract fixed regions for secondary processing steps (e.g., crop a specific area from full images, then run OCR or classification on the cropped region), enabling hierarchical workflows with fixed region focus
- **Standardized Crop Generation**: Create consistent crops from images for training or analysis (e.g., extract a fixed region from all images for dataset creation, crop a standard area for comparison, generate uniform crops for feature extraction), enabling standardized data preparation workflows
- **Fixed-Area Monitoring**: Monitor specific image regions across time or batches (e.g., crop the same area from video frames for change detection, extract a fixed region for tracking analysis, focus on a consistent monitoring zone), enabling temporal analysis of fixed regions
- **Pre-Processing for Specialized Blocks**: Extract fixed regions before processing with specialized models (e.g., crop a specific area before running OCR, extract a fixed region for fine-grained classification, isolate a region for specialized analysis), enabling optimized processing of known image regions

## Connecting to Other Blocks

This block receives images and produces cropped images from fixed regions:

- **After image loading blocks** to extract a fixed region of interest before processing, enabling focused analysis of predetermined image areas without processing entire images
- **Before classification or analysis blocks** that need region-focused inputs (e.g., OCR for text in a fixed area, fine-grained classification for cropped regions, specialized models for specific image areas), enabling optimized processing of consistent regions
- **In video processing workflows** to extract the same region from multiple frames (e.g., crop a fixed area from each video frame for temporal analysis, extract a consistent monitoring zone for tracking, focus on a specific region across frames), enabling temporal analysis of fixed regions
- **After detection blocks** where you know the approximate location and want to extract a fixed-size region around it (e.g., detect objects in a general area, then crop a fixed region around that area for detailed analysis), enabling region-focused multi-stage workflows
- **Before visualization blocks** that display specific regions (e.g., display only the cropped region, visualize a fixed area of interest, show isolated region annotations), enabling focused visualization of extracted regions
- **In batch processing workflows** where the same region needs to be extracted from all images for consistent analysis or comparison, enabling standardized region extraction across image sets
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Absolute Static Crop",
            "version": "v1",
            "short_description": "Crop an image using fixed pixel coordinates.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "far fa-crop-alt",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/absolute_static_crop@v1", "AbsoluteStaticCrop"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    x_center: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="X coordinate of the center point of the crop region in absolute pixel coordinates. Must be a positive integer. The crop region is centered at this X coordinate. The actual crop boundaries are calculated as x_min = x_center - width/2 and x_max = x_min + width. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty.",
        examples=[40, "$inputs.center_x"],
    )
    y_center: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Y coordinate of the center point of the crop region in absolute pixel coordinates. Must be a positive integer. The crop region is centered at this Y coordinate. The actual crop boundaries are calculated as y_min = y_center - height/2 and y_max = y_min + height. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty.",
        examples=[40, "$inputs.center_y"],
    )
    width: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Width of the crop region in pixels. Must be a positive integer. Defines the horizontal extent of the crop. The crop extends width/2 pixels to the left and right of the x_center coordinate. Total crop width equals this value. If the calculated crop extends beyond the image's width, it will be clipped to image boundaries.",
        examples=[40, "$inputs.width"],
    )
    height: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Height of the crop region in pixels. Must be a positive integer. Defines the vertical extent of the crop. The crop extends height/2 pixels above and below the y_center coordinate. Total crop height equals this value. If the calculated crop extends beyond the image's height, it will be clipped to image boundaries.",
        examples=[40, "$inputs.height"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class AbsoluteStaticCropBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        x_center: int,
        y_center: int,
        width: int,
        height: int,
    ) -> BlockResult:
        return [
            {
                "crops": take_static_crop(
                    image=image,
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                )
            }
            for image in images
        ]


def take_static_crop(
    image: WorkflowImageData,
    x_center: int,
    y_center: int,
    width: int,
    height: int,
) -> Optional[WorkflowImageData]:
    x_min = round(x_center - width / 2)
    y_min = round(y_center - height / 2)
    x_max = round(x_min + width)
    y_max = round(y_min + height)
    cropped_image = image.numpy_image[y_min:y_max, x_min:x_max]
    if not cropped_image.size:
        return None
    return WorkflowImageData.create_crop(
        origin_image_data=image,
        crop_identifier=f"absolute_static_crop.{uuid4()}",
        cropped_image=cropped_image,
        offset_x=x_min,
        offset_y=y_min,
        preserve_video_metadata=True,
    )
