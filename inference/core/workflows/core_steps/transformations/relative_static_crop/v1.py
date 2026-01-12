from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    FloatZeroToOne,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Extract a fixed rectangular region from input images using relative coordinates (normalized 0.0-1.0 values proportional to image dimensions) specified by center point and dimensions, creating consistent proportional crops from the same relative location across images of different sizes for region-of-interest extraction and size-agnostic fixed-area analysis workflows.

## How This Block Works

This block crops a fixed rectangular region from input images using relative coordinates (normalized 0.0-1.0 values), unlike absolute static cropping which uses pixel coordinates. The relative coordinates adapt to different image sizes, making it ideal for extracting the same proportional region from images of varying dimensions. The block:

1. Receives input images and relative coordinate specifications (x_center, y_center, width, height) as values between 0.0 and 1.0
2. Converts relative coordinates to absolute pixel coordinates by multiplying with image dimensions:
   - Converts x_center from relative (0.0-1.0) to absolute pixels: `x_center_pixels = image_width * x_center`
   - Converts y_center from relative (0.0-1.0) to absolute pixels: `y_center_pixels = image_height * y_center`
   - Converts width from relative (0.0-1.0) to absolute pixels: `width_pixels = image_width * width`
   - Converts height from relative (0.0-1.0) to absolute pixels: `height_pixels = image_height * height`
3. Calculates the crop boundaries from the converted center point and dimensions:
   - Computes x_min and y_min by subtracting half the width/height from the center coordinates
   - Computes x_max and y_max by adding the width/height to the minimum coordinates
   - Rounds coordinate values to integer pixel positions
4. Extracts the rectangular region from the image using array slicing (from y_min to y_max, x_min to x_max)
5. Validates that the cropped region has content (returns None if the crop would be empty, such as when coordinates are outside image bounds)
6. Creates a cropped image object with metadata tracking the crop's origin (original image, offset coordinates, unique crop identifier)
7. Preserves video metadata if the input is from video (maintains frame information and temporal context)
8. Returns the cropped image for each input image

The block uses relative coordinates (0.0-1.0), so the same proportional region is extracted from all images regardless of their size. For example, x_center=0.5, y_center=0.5, width=0.4, height=0.4 extracts a 40% by 40% region centered in the image, whether the image is 100x100 pixels or 2000x2000 pixels. This makes the block particularly useful for extracting consistent regions from images of varying sizes (e.g., always cropping the top-right 20% corner, extracting a fixed percentage of the image center, or focusing on a specific proportional area). The center-based coordinate system allows specifying crops by their center point rather than corner coordinates, which can be more intuitive for defining proportional regions.

## Common Use Cases

- **Size-Agnostic Region Extraction**: Extract the same proportional region from images of different sizes for consistent analysis (e.g., crop the top-right 20% corner from images regardless of resolution, extract a fixed percentage of the image center, crop a consistent proportional area for pattern matching), enabling standardized region analysis across images with varying dimensions
- **Multi-Resolution Image Processing**: Extract consistent proportional regions from images with different resolutions (e.g., crop the same relative area from high-resolution and low-resolution images, extract proportional regions from resized images, maintain consistent cropping across different image sizes), enabling size-independent region extraction
- **Proportional Region-of-Interest Focus**: Isolate specific proportional areas of images for detailed processing (e.g., crop a specific relative quadrant of images, extract a fixed percentage region for text recognition, focus on a known proportional area of interest), enabling focused analysis of predetermined proportional regions
- **Multi-Stage Workflow Preparation**: Extract fixed proportional regions for secondary processing steps (e.g., crop a specific relative area from full images, then run OCR or classification on the cropped region), enabling hierarchical workflows with proportional region focus
- **Standardized Crop Generation**: Create consistent proportional crops from images for training or analysis (e.g., extract a fixed relative region from all images for dataset creation, crop a standard proportional area for comparison, generate uniform proportional crops for feature extraction), enabling standardized data preparation workflows across varying image sizes
- **Video Frame Proportional Cropping**: Extract the same proportional region from video frames of different resolutions (e.g., crop a fixed percentage area from each video frame for temporal analysis, extract a consistent proportional monitoring zone for tracking, focus on a specific relative region across frames), enabling temporal analysis of proportional regions across varying frame sizes

## Connecting to Other Blocks

This block receives images and produces cropped images from fixed proportional regions:

- **After image loading blocks** to extract a fixed proportional region of interest before processing, enabling focused analysis of predetermined image areas without processing entire images, particularly useful when working with images of varying sizes
- **Before classification or analysis blocks** that need region-focused inputs (e.g., OCR for text in a fixed proportional area, fine-grained classification for cropped regions, specialized models for specific proportional image areas), enabling optimized processing of consistent proportional regions
- **In video processing workflows** to extract the same proportional region from multiple frames regardless of resolution changes (e.g., crop a fixed percentage area from each video frame for temporal analysis, extract a consistent proportional monitoring zone for tracking, focus on a specific relative region across frames), enabling temporal analysis of proportional regions
- **After detection blocks** where you know the approximate relative location and want to extract a fixed-size proportional region around it (e.g., detect objects in a general relative area, then crop a fixed proportional region around that area for detailed analysis), enabling region-focused multi-stage workflows with size-agnostic cropping
- **Before visualization blocks** that display specific regions (e.g., display only the cropped proportional region, visualize a fixed relative area of interest, show isolated region annotations), enabling focused visualization of extracted proportional regions
- **In batch processing workflows** where the same proportional region needs to be extracted from all images for consistent analysis or comparison, regardless of individual image sizes, enabling standardized proportional region extraction across image sets with varying dimensions
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Relative Static Crop",
            "version": "v1",
            "short_description": "Crop an image proportional (%) to its dimensions.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "far fa-crop-alt",
                "blockPriority": 2,
            },
        }
    )
    type: Literal["roboflow_core/relative_statoic_crop@v1", "RelativeStaticCrop"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    x_center: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        description="X coordinate of the center point of the crop region as a relative value (0.0 to 1.0). 0.0 represents the left edge of the image, 1.0 represents the right edge, and 0.5 represents the center horizontally. The crop region is centered at this X coordinate after converting to absolute pixels. The actual crop boundaries are calculated as x_min = x_center - width/2 and x_max = x_min + width, where all values are converted to pixels based on image width. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty. Relative coordinates adapt to different image sizes - the same relative value extracts the same proportional position from images of any size.",
        examples=[0.3, "$inputs.center_x"],
    )
    y_center: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        description="Y coordinate of the center point of the crop region as a relative value (0.0 to 1.0). 0.0 represents the top edge of the image, 1.0 represents the bottom edge, and 0.5 represents the center vertically. The crop region is centered at this Y coordinate after converting to absolute pixels. The actual crop boundaries are calculated as y_min = y_center - height/2 and y_max = y_min + height, where all values are converted to pixels based on image height. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty. Relative coordinates adapt to different image sizes - the same relative value extracts the same proportional position from images of any size.",
        examples=[0.3, "$inputs.center_y"],
    )
    width: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        description="Width of the crop region as a relative value (0.0 to 1.0). 1.0 represents 100% of the image width, 0.5 represents 50% of the image width, etc. Defines the horizontal extent of the crop as a proportion of the image width. The crop extends width/2 pixels to the left and right of the x_center coordinate after converting to absolute pixels. Total crop width equals this relative value multiplied by the image width. If the calculated crop extends beyond the image's width, it will be clipped to image boundaries. Relative width adapts to different image sizes - the same relative value extracts the same proportional width from images of any size.",
        examples=[0.3, "$inputs.width"],
    )
    height: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        description="Height of the crop region as a relative value (0.0 to 1.0). 1.0 represents 100% of the image height, 0.5 represents 50% of the image height, etc. Defines the vertical extent of the crop as a proportion of the image height. The crop extends height/2 pixels above and below the y_center coordinate after converting to absolute pixels. Total crop height equals this relative value multiplied by the image height. If the calculated crop extends beyond the image's height, it will be clipped to image boundaries. Relative height adapts to different image sizes - the same relative value extracts the same proportional height from images of any size.",
        examples=[0.3, "$inputs.height"],
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


class RelativeStaticCropBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[BaseModel]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        x_center: float,
        y_center: float,
        width: float,
        height: float,
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
    x_center: float,
    y_center: float,
    width: float,
    height: float,
) -> Optional[WorkflowImageData]:
    x_center = round(image.numpy_image.shape[1] * x_center)
    y_center = round(image.numpy_image.shape[0] * y_center)
    width = round(image.numpy_image.shape[1] * width)
    height = round(image.numpy_image.shape[0] * height)
    x_min = round(x_center - width / 2)
    y_min = round(y_center - height / 2)
    x_max = round(x_min + width)
    y_max = round(y_min + height)
    cropped_image = image.numpy_image[y_min:y_max, x_min:x_max]
    if not cropped_image.size:
        return None
    return WorkflowImageData.create_crop(
        origin_image_data=image,
        crop_identifier=f"relative_static_crop.{uuid4()}",
        cropped_image=cropped_image,
        offset_x=x_min,
        offset_y=y_min,
        preserve_video_metadata=True,
    )
