from typing import List, Literal, Optional, Type

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
    IMAGE_KEYPOINTS_KIND,
    IMAGE_KIND,
    NUMPY_ARRAY_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

####

SHORT_DESCRIPTION: str = "Apply SIFT to an image."
LONG_DESCRIPTION = """
Detect and describe distinctive visual features in images using SIFT (Scale-Invariant Feature Transform), extracting keypoints (interest points) and computing 128-dimensional feature descriptors that are invariant to scale, rotation, and lighting conditions, enabling feature-based image matching, object recognition, and image similarity detection workflows.

## How This Block Works

This block detects distinctive visual features in an image using SIFT and computes feature descriptors for each detected keypoint. The block:

1. Receives an input image to analyze for feature detection
2. Converts the image to grayscale (SIFT operates on grayscale images for efficiency and robustness)
3. Creates a SIFT detector using OpenCV's SIFT implementation
4. Detects keypoints and computes descriptors simultaneously using detectAndCompute:
   - **Keypoint Detection**: Identifies distinctive interest points (keypoints) in the image that are stable across different viewing conditions
   - Keypoints are detected at multiple scales (pyramid of scale-space images) to handle scale variations
   - Keypoints are detected with orientation assignment to handle rotation variations
   - Each keypoint has properties: position (x, y coordinates), size (scale at which it was detected), angle (orientation), response (strength), octave (scale level), and class_id
   - **Descriptor Computation**: Computes 128-dimensional feature descriptors for each keypoint that describe the local image region around the keypoint
   - Descriptors encode gradient information in the local region, making them distinctive and robust to lighting changes
   - Descriptors are normalized to be partially invariant to illumination changes
5. Draws keypoints on the original image for visualization:
   - Uses OpenCV's drawKeypoints to overlay keypoint markers on the image
   - Visualizes keypoint locations, orientations, and scales
   - Creates a visual representation showing where features were detected
6. Converts keypoints to dictionary format:
   - Extracts keypoint properties (position, size, angle, response, octave, class_id) into dictionaries
   - Makes keypoint data accessible for downstream processing and analysis
7. Returns the image with keypoints drawn, the keypoints data (as dictionaries), and the descriptors (as numpy array)

SIFT features are scale-invariant (work at different zoom levels), rotation-invariant (handle rotated images), and partially lighting-invariant (robust to illumination changes). This makes them highly effective for matching the same object or scene across different images taken from different viewpoints, distances, angles, or lighting conditions. The 128-dimensional descriptors provide rich information about local image regions, enabling robust feature matching and comparison.

## Common Use Cases

- **Feature-Based Image Matching**: Detect features for matching objects or scenes across different images (e.g., match objects in multiple images, find corresponding features across viewpoints, identify matching regions in image pairs), enabling feature-based matching workflows
- **Object Recognition**: Use SIFT features for object recognition and identification (e.g., recognize objects using feature matching, identify objects by their distinctive features, match object features for classification), enabling feature-based object recognition workflows
- **Image Similarity Detection**: Detect similar images by comparing SIFT features (e.g., find similar images in databases, detect duplicate images, identify matching scenes), enabling image similarity workflows
- **Feature Extraction for Analysis**: Extract distinctive features from images for further analysis (e.g., extract features for processing, analyze image characteristics, identify interesting regions), enabling feature extraction workflows
- **Visual Localization**: Use SIFT features for visual localization and mapping (e.g., localize objects in scenes, track features across frames, map feature correspondences), enabling visual localization workflows
- **Image Registration**: Align images using SIFT feature correspondences (e.g., register images for stitching, align images from different viewpoints, match images for alignment), enabling image registration workflows

## Connecting to Other Blocks

This block receives an image and produces SIFT keypoints and descriptors:

- **After image input blocks** to extract SIFT features from input images (e.g., detect features in camera feeds, extract features from image inputs, analyze features in images), enabling SIFT feature extraction workflows
- **After preprocessing blocks** to extract features from preprocessed images (e.g., detect features after filtering, extract features from enhanced images, analyze features after preprocessing), enabling preprocessed feature extraction workflows
- **Before SIFT Comparison blocks** to provide SIFT descriptors for image comparison (e.g., provide descriptors for matching, prepare features for comparison, supply descriptors for similarity detection), enabling SIFT-based image comparison workflows
- **Before filtering or logic blocks** that use feature counts or properties for decision-making (e.g., filter based on feature count, make decisions based on detected features, apply logic based on feature properties), enabling feature-based conditional workflows
- **Before data storage blocks** to store feature data (e.g., store keypoints and descriptors, save feature information, record feature data for analysis), enabling feature data storage workflows
- **Before visualization blocks** to display detected features (e.g., visualize keypoints, display feature locations, show feature analysis results), enabling feature visualization workflows
"""


class SIFTDetectionManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/sift@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SIFT",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-grid-2-plus",
                "blockPriority": 4,
                "opencv": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Input image to analyze for SIFT feature detection. The image will be converted to grayscale internally for SIFT processing. SIFT works best on images with good texture and detail - images with rich visual content (edges, corners, patterns) produce more keypoints than uniform or smooth images. Each detected keypoint will have a 128-dimensional descriptor computed. The output includes an image with keypoints drawn for visualization, keypoint data (position, size, angle, response, octave), and descriptor arrays for matching and comparison. SIFT features are scale and rotation invariant, making them effective for matching across different viewpoints and conditions.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[IMAGE_KIND],
            ),
            OutputDefinition(
                name="keypoints",
                kind=[IMAGE_KEYPOINTS_KIND],
            ),
            OutputDefinition(
                name="descriptors",
                kind=[NUMPY_ARRAY_KIND],
            ),
        ]


class SIFTBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[SIFTDetectionManifest]:
        return SIFTDetectionManifest

    def run(self, image: WorkflowImageData, *args, **kwargs) -> BlockResult:
        img_with_kp, keypoints, descriptors = apply_sift(image.numpy_image)
        output_image = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=img_with_kp,
        )
        return {
            OUTPUT_IMAGE_KEY: output_image,
            "keypoints": keypoints,
            "descriptors": descriptors,
        }


def apply_sift(image: np.ndarray) -> (np.ndarray, list, np.ndarray):
    """
    Applies SIFT to the image.
    Args:
        image: Input image.
    Returns:
        np.ndarray: Image with keypoints drawn.
        list: Keypoints detected.
        np.ndarray: Descriptors of the keypoints.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    # Draw onto a copy so we do not mutate the caller's cached numpy_image (e.g. WorkflowImageData)
    img_with_kp = cv2.drawKeypoints(gray, kp, image.copy())
    # Convert keypoints to the desired format
    keypoints = [
        {
            "pt": (point.pt[0], point.pt[1]),
            "size": point.size,
            "angle": point.angle,
            "response": point.response,
            "octave": point.octave,
            "class_id": point.class_id,
        }
        for point in kp
    ]
    return img_with_kp, keypoints, des
