from dataclasses import replace
from typing import List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt
from supervision import crop_image
from typing_extensions import Annotated

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Split input images into overlapping tiles or slices using the Slicing Adaptive Inference (SAHI) technique to enable small object detection by processing smaller image regions where objects appear larger relative to the image size, improving detection accuracy for small objects in large images through tiled inference workflows with equal-sized slices and deduplication.

## How This Block Works

This block implements the first step of the SAHI (Slicing Adaptive Inference) technique by dividing large images into smaller overlapping tiles. This approach helps detect small objects that might be missed when processing the entire image at once. The block:

1. Receives an input image and slicing configuration:
   - Takes an input image to be sliced
   - Receives slice dimensions (width and height in pixels)
   - Receives overlap ratios for width and height (controls overlap between adjacent slices)
2. Calculates slice positions:
   - Generates a grid of slice coordinates across the image
   - Positions slices with specified overlap between consecutive slices
   - Overlap helps ensure objects at slice boundaries are not missed
   - Adjusts border slice positions to ensure all slices are equal size (pushes border slices toward image center)
3. Creates image slices:
   - Extracts each slice from the original image using calculated coordinates
   - Creates WorkflowImageData objects for each slice with crop metadata
   - Stores offset information (x, y coordinates) for each slice relative to original image
   - Maintains parent image reference for coordinate mapping
4. Deduplicates slices:
   - Removes any duplicate slice coordinates that may occur from overlap calculations
   - Ensures each unique slice position appears only once in the output
   - Prevents redundant processing of identical image regions
5. Handles edge cases:
   - Filters out empty slices (if any occur)
   - Ensures all slices fit within image boundaries
   - Creates crop identifiers for tracking each slice
6. Returns list of slices:
   - Outputs all unique slices as a list of images
   - All slices have equal dimensions (border slices adjusted to match)
   - Increases dimensionality by 1 (one image becomes multiple slices)
   - Each slice can be processed independently by downstream blocks

The SAHI technique works by making small objects appear larger relative to the slice size. When an object is only a few pixels in a large image, scaling the image down to model input size makes the object too small to detect. By slicing the image and processing each slice separately, the same object occupies more pixels in each slice, making detection more reliable. Overlapping slices ensure objects near slice boundaries are detected in at least one slice.

## Common Use Cases

- **Small Object Detection**: Detect small objects in large images using SAHI technique (e.g., detect small vehicles in aerial images, find license plates in wide-angle camera views, detect insects in high-resolution photos), enabling small object detection workflows
- **High-Resolution Image Processing**: Process high-resolution images by slicing them into manageable pieces (e.g., process satellite imagery, analyze medical imaging scans, process large document images), enabling high-resolution processing workflows
- **Aerial and Drone Imagery**: Detect objects in aerial photography where objects are small relative to image size (e.g., detect vehicles in drone footage, find people in aerial surveillance, detect structures in satellite images), enabling aerial detection workflows
- **Wide-Angle Camera Monitoring**: Improve detection in wide-angle camera views where objects appear small (e.g., monitor large parking lots, detect objects in panoramic views, analyze traffic in wide camera coverage), enabling wide-angle monitoring workflows
- **Medical Imaging Analysis**: Analyze medical images by processing regions separately (e.g., detect lesions in large scans, find anomalies in medical images, analyze radiology images), enabling medical imaging workflows
- **Document and Text Processing**: Process large documents by slicing into regions (e.g., OCR large documents, detect text regions in scanned documents, analyze document layouts), enabling document processing workflows

## Connecting to Other Blocks

This block receives images and produces image slices:

- **After image input or preprocessing blocks** to slice images for SAHI processing (e.g., slice input images, process preprocessed images, slice transformed images), enabling image-to-slice workflows
- **Before detection model blocks** (Object Detection Model, Instance Segmentation Model) to process slices for small object detection (e.g., detect objects in slices, run detection on each slice, process slices with models), enabling slice-to-detection workflows
- **Before Detections Stitch block** (required after detection models) to merge detections from slices back to original image coordinates (e.g., merge slice detections, combine detection results, reconstruct full-image predictions), enabling slice-detection-stitch workflows
- **In SAHI workflows** following the pattern: Image Slicer → Detection Model → Detections Stitch to implement complete SAHI technique for small object detection
- **Before filtering or analytics blocks** to process slice-level results before stitching (e.g., filter detections per slice, analyze slice results, process slice outputs), enabling slice-to-analysis workflows
- **As part of multi-stage detection pipelines** where slices are processed independently and results are combined (e.g., multi-scale detection, hierarchical detection, parallel slice processing), enabling multi-stage detection workflows

## Version Differences

This version (v2) includes the following enhancements over v1:

- **Equal-Sized Slices**: All slices generated by the slicer have equal dimensions. Border slices that would normally be smaller in v1 are adjusted by pushing them toward the image center, ensuring consistent slice sizes. This provides more predictable processing behavior and ensures all slices are processed with the same dimensions, which can be important for model inference consistency.
- **Deduplication**: Duplicate slice coordinates are automatically removed, ensuring each unique slice position appears only once in the output. This prevents redundant processing of identical image regions that could occur due to overlap calculations, improving efficiency and preventing duplicate detections.

## Requirements

This block requires an input image. The slice dimensions (width and height) should ideally match the model's input size for optimal performance. If slice size differs from model input size, slices will be resized during inference which may affect accuracy. Default slice size is 640x640 pixels, but this should be adjusted based on your model's input size (e.g., use 320x320 for models with 320 input size, 1280x1280 for models with 1280 input size). Overlap ratios (default 0.2 or 20%) help ensure objects at slice boundaries are detected, but higher overlap increases processing time. The block should be used with object detection or instance segmentation models, followed by Detections Stitch block to merge results. For more information on SAHI technique, see: https://ieeexplore.ieee.org/document/9897990. For a practical guide, visit: https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Image Slicer",
            "version": "v2",
            "short_description": "Tile the input image into a list of smaller images to perform small object detection.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-scissors",
                "blockPriority": 9,
                "opencv": True,
            },
        }
    )
    type: Literal["roboflow_core/image_slicer@v2"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Image to slice",
        description="Input image to be sliced into smaller tiles. The image will be divided into overlapping slices based on the slice dimensions and overlap ratios. Each slice maintains metadata about its position in the original image for coordinate mapping. All slices will have equal dimensions (border slices are adjusted to match). Used in SAHI (Slicing Adaptive Inference) workflows to enable small object detection by processing image regions separately.",
        examples=["$inputs.image", "$steps.preprocessing.output", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    slice_width: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=640,
        description="Width of each slice in pixels. Should ideally match your detection model's input width for optimal performance. If different, slices will be resized during model inference which may affect accuracy. Common values: 320 (for models with 320px input), 640 (default, for most YOLO models), 1280 (for high-resolution models). Larger slices process fewer total slices but may miss very small objects. Smaller slices detect smaller objects but increase processing time. All slices will have equal width (border slices adjusted to match).",
        examples=[320, 640, 1280, "$inputs.slice_width"],
    )
    slice_height: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=640,
        description="Height of each slice in pixels. Should ideally match your detection model's input height for optimal performance. If different, slices will be resized during model inference which may affect accuracy. Common values: 320 (for models with 320px input), 640 (default, for most YOLO models), 1280 (for high-resolution models). Larger slices process fewer total slices but may miss very small objects. Smaller slices detect smaller objects but increase processing time. All slices will have equal height (border slices adjusted to match).",
        examples=[320, 640, 1280, "$inputs.slice_height"],
    )
    overlap_ratio_width: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.2,
        description="Overlap ratio between consecutive slices in the width (horizontal) dimension. Range: 0.0 to <1.0. Specifies what fraction of the slice width overlaps with adjacent slices. Default 0.2 means 20% overlap. Higher overlap (e.g., 0.3-0.5) ensures objects at slice boundaries are more likely to be detected but increases processing time since more slices are created. Lower overlap (e.g., 0.1) is faster but may miss objects at boundaries. Typical values: 0.1-0.3 for most use cases, 0.3-0.5 for critical small object detection. Duplicate slices created by overlap are automatically removed.",
        examples=[0.1, 0.2, 0.3, 0.5, "$inputs.overlap_ratio_width"],
    )
    overlap_ratio_height: Union[
        Annotated[float, Field(ge=0.0, lt=1.0)],
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.2,
        description="Overlap ratio between consecutive slices in the height (vertical) dimension. Range: 0.0 to <1.0. Specifies what fraction of the slice height overlaps with adjacent slices. Default 0.2 means 20% overlap. Higher overlap (e.g., 0.3-0.5) ensures objects at slice boundaries are more likely to be detected but increases processing time since more slices are created. Lower overlap (e.g., 0.1) is faster but may miss objects at boundaries. Typical values: 0.1-0.3 for most use cases, 0.3-0.5 for critical small object detection. Duplicate slices created by overlap are automatically removed.",
        examples=[0.1, 0.2, 0.3, 0.5, "$inputs.overlap_ratio_height"],
    )

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="slices", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ImageSlicerBlockV2(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        slice_width: int,
        slice_height: int,
        overlap_ratio_width: float,
        overlap_ratio_height: float,
    ) -> BlockResult:
        image_numpy = image.numpy_image
        resolution_wh = (image_numpy.shape[1], image_numpy.shape[0])
        offsets = generate_offsets(
            resolution_wh=resolution_wh,
            slice_wh=(slice_width, slice_height),
            overlap_ratio_wh=(overlap_ratio_width, overlap_ratio_height),
        )
        slices = []
        for offset in offsets:
            x_min, y_min, _, _ = offset
            crop_numpy = crop_image(image=image_numpy, xyxy=offset)
            if crop_numpy.size:
                cropped_image = WorkflowImageData.create_crop(
                    origin_image_data=image,
                    crop_identifier=f"image_slicer.{uuid4()}",
                    cropped_image=crop_numpy,
                    offset_x=x_min,
                    offset_y=y_min,
                )
                slices.append({"slices": cropped_image})
            else:
                slices.append({"slices": None})
        return slices


def generate_offsets(
    resolution_wh: Tuple[int, int],
    slice_wh: Tuple[int, int],
    overlap_ratio_wh: Tuple[float, float],
) -> np.ndarray:
    """
    This is modification of the function from block v1, which
    makes sure that the "border" crops are pushed towards the center of
    the image, making sure:
        * all crops will be the same size
        * deduplication of crops coordinates is done
    """
    slice_width, slice_height = slice_wh
    image_width, image_height = resolution_wh
    slice_width = min(slice_width, image_width)
    slice_height = min(slice_height, image_height)
    overlap_width = int(overlap_ratio_wh[0] * slice_width)
    overlap_height = int(overlap_ratio_wh[1] * slice_height)
    width_stride = slice_width - overlap_width
    height_stride = slice_height - overlap_height
    ws = np.arange(0, image_width, width_stride)
    ws_left_over = np.clip(ws + slice_width - image_width, 0, slice_width)
    hs = np.arange(0, image_height, height_stride)
    hs_left_over = np.clip(hs + slice_height - image_height, 0, slice_height)
    anchors_ws = ws - ws_left_over
    anchors_hs = hs - hs_left_over
    xmin, ymin = np.meshgrid(anchors_ws, anchors_hs)
    xmax = np.clip(xmin + slice_width, 0, image_width)
    ymax = np.clip(ymin + slice_height, 0, image_height)
    results = np.stack([xmin, ymin, xmax, ymax], axis=-1).reshape(-1, 4)
    deduplicated_results = []
    already_seen = set()
    for xyxy in results:
        xyxy_tuple = tuple(xyxy)
        if xyxy_tuple in already_seen:
            continue
        deduplicated_results.append(xyxy)
        already_seen.add(xyxy_tuple)
    return np.array(deduplicated_results)
