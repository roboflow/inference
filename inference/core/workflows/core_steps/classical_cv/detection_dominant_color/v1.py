from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Get the dominant color as a hex string for each detection region in an image."
LONG_DESCRIPTION = """
Extract the most prevalent dominant color from each detected object region in an image using K-means clustering on pixel colors, processing each detection's bounding box (or instance segmentation mask) independently, analyzing color distribution within each region, identifying color clusters, and returning the hex color string (#rrggbb) of the most frequently occurring color cluster for per-detection color analysis, object color identification, color-based filtering, and detection-level visual analysis workflows.

## How This Block Works

This block analyzes the color distribution within each detected object region using K-means clustering to identify and extract the dominant (most prevalent) color per detection. The block:

1. Receives an input image and a set of detection predictions (from an object detection, instance segmentation, or keypoint detection model) to analyze for dominant colors on a per-detection basis
2. For each detection, extracts the bounding box region from the source image using the detection's coordinates (x1, y1, x2, y2), cropping the image to isolate the detected object
3. For instance segmentation detections that include a mask, uses the segmentation mask to extract only the pixels that fall inside the detected object's boundary, ignoring background pixels within the bounding box for more accurate color analysis
4. Downsamples the pixel data if the detection region contains too many pixels to optimize processing speed:
   - Calculates the maximum pixel count based on the target_size parameter (target_size × target_size)
   - If the region exceeds this count, randomly subsamples pixels down to the maximum
   - Smaller target_size values speed up processing but may slightly reduce precision
5. Initializes K-means clustering:
   - Selects random pixel colors as initial cluster centroids (number of clusters = color_clusters, capped at the number of available pixels)
   - Creates initial color clusters to group similar pixel colors
6. Performs iterative K-means clustering (up to max_iterations), using the same algorithm as the Dominant Color block:
   - Assigns each pixel to the nearest color cluster (based on Euclidean distance in color space)
   - Updates cluster centroids by computing the mean color values of pixels in each cluster
   - Handles empty clusters by reinitializing them to random pixel colors
   - Checks for convergence (centroids stop changing significantly) and exits early if converged
7. Counts pixels in each color cluster to determine which cluster contains the most pixels
8. Selects the cluster with the highest pixel count as the dominant color cluster
9. Extracts the color values from the dominant cluster's centroid and converts from BGR to RGB color order
10. Clips RGB values to the valid 0-255 integer range and formats the result as a hex color string (#rrggbb)
11. Returns one hex color string per detection, producing a list of results that matches the detection order

The block uses K-means clustering to group similar pixel colors together within each detection region, then identifies the largest color group as the dominant color. Processing time depends on the number of detections, region sizes, color complexity, and parameter settings (color_clusters, max_iterations, target_size). The downsampling step balances speed and accuracy - reducing pixel count speeds up clustering while still capturing the overall color distribution within each detected object. If a detection region contains no valid pixels (e.g., a zero-area bounding box), the block returns None for that detection.

## Common Use Cases

- **Product Color Extraction**: Extract dominant colors from detected products in retail or e-commerce images for catalog enrichment, inventory management, and color-based search (e.g., detect products on a shelf and extract each product's primary color for database indexing), enabling per-product color analysis workflows
- **Vehicle Color Identification**: Identify the color of detected vehicles from traffic cameras, parking lot surveillance, or dashcam footage (e.g., detect cars in a traffic scene and determine each vehicle's dominant color for identification or filtering), enabling vehicle color classification workflows
- **Quality Control and Inspection**: Verify expected colors in detected components on assembly lines or manufacturing processes (e.g., detect parts on a conveyor belt and check that each part's color matches the specification), enabling color-based quality control workflows
- **Wildlife Monitoring and Species Identification**: Identify species or track individuals by analyzing color patterns from detected animals in camera trap or drone footage (e.g., detect birds and extract plumage colors for species classification), enabling wildlife color analysis workflows
- **Fashion and Retail Analysis**: Extract clothing or accessory colors from detected people or garments in retail settings, fashion photography, or social media images (e.g., detect clothing items on a person and determine the dominant color of each garment), enabling fashion color extraction workflows
- **Sorting and Classification**: Automatically sort or categorize detected objects by their dominant color (e.g., detect items on a conveyor and route them based on color, or classify detected fruits by ripeness color), enabling color-based sorting workflows

## Connecting to Other Blocks

This block receives an image and detection predictions, and produces a hex color string per detection:

- **After object detection or instance segmentation model blocks** to analyze the dominant color of each detected object (e.g., run a product detection model then extract the color of each detected product, run an instance segmentation model for more precise color extraction using masks), enabling detection-to-color analysis workflows
- **After tracking blocks** to track color changes of detected objects over time (e.g., detect and track objects across frames, then monitor if their dominant color changes, useful for detecting state changes like ripening fruit or temperature-based color shifts), enabling color tracking workflows
- **Before filtering or logic blocks** that use color information for decision-making (e.g., filter detections to keep only objects of a specific color, route detections based on color matching criteria, trigger alerts when unexpected colors are detected), enabling color-based conditional workflows
- **Before sink blocks** to store color metadata alongside detection data (e.g., save dominant color hex values with detection coordinates to a database, log color information to a webhook, write color analysis results to a local file), enabling color metadata storage workflows
- **In multi-stage workflows** combining detection, color analysis, and downstream processing (e.g., detect objects → extract colors → filter by color → visualize results, or detect products → extract colors → compare to expected colors → flag anomalies), enabling comprehensive detection-and-color analysis pipelines
- **After crop blocks or image preprocessing** to analyze colors in specific image regions before or after detection (e.g., crop to a region of interest, detect objects within the crop, then extract per-detection colors for focused analysis), enabling region-specific color analysis workflows
"""


class DetectionDominantColorManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/detection_dominant_color@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detection Dominant Color",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-palette",
                "blockPriority": 1,
            },
        }
    )
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The source image from which detection regions will be cropped for color analysis.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Detections",
        description="Detection predictions whose bounding boxes (and optionally masks) define the regions to analyze.",
        examples=["$steps.my_object_detection_model.predictions"],
        validation_alias=AliasChoices("predictions", "detections"),
    )
    color_clusters: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Color Clusters",
        description="Number of color clusters (K) for K-means clustering. Higher values may improve precision for complex regions but increase processing time.",
        default=4,
        examples=[4, "$inputs.color_clusters"],
        gt=0,
        le=10,
    )
    max_iterations: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Max Iterations",
        description="Maximum number of K-means iterations. The algorithm stops early if convergence is reached.",
        default=100,
        examples=[100, "$inputs.max_iterations"],
        gt=0,
        le=500,
    )
    target_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Target Size",
        description="If the number of pixels in a detection region exceeds target_size*target_size, the pixels are randomly subsampled to that count for faster clustering.",
        default=100,
        examples=[100, "$inputs.target_size"],
        gt=0,
        le=250,
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["image", "predictions"]

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="dominant_color_hex", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionDominantColorBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[DetectionDominantColorManifest]:
        return DetectionDominantColorManifest

    def run(
        self,
        image: Batch[WorkflowImageData],
        predictions: Batch[sv.Detections],
        color_clusters: int,
        max_iterations: int,
        target_size: int,
    ) -> BlockResult:
        return [
            _process_single_image(
                image=img,
                detections=dets,
                color_clusters=color_clusters,
                max_iterations=max_iterations,
                target_size=target_size,
            )
            for img, dets in zip(image, predictions)
        ]


def _process_single_image(
    image: WorkflowImageData,
    detections: sv.Detections,
    color_clusters: int,
    max_iterations: int,
    target_size: int,
) -> List[dict]:
    if len(detections) == 0:
        return []

    image_np = image.numpy_image
    results = []

    for idx in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[idx].round().astype(int)
        region = image_np[y1:y2, x1:x2]

        if detections.mask is not None:
            mask_crop = detections.mask[idx][y1:y2, x1:x2]
            pixels = region[mask_crop]  # shape (M, 3)
        else:
            pixels = region.reshape(-1, 3)

        if pixels.size == 0:
            results.append({"dominant_color_hex": None})
            continue

        pixels = pixels.astype(np.float32)

        # Downsample if needed
        max_pixels = target_size * target_size
        if pixels.shape[0] > max_pixels:
            indices = np.random.choice(pixels.shape[0], max_pixels, replace=False)
            pixels = pixels[indices]

        actual_clusters = min(color_clusters, pixels.shape[0])

        # K-means clustering
        centroids = pixels[
            np.random.choice(pixels.shape[0], actual_clusters, replace=False)
        ]
        for _ in range(max_iterations):
            distances = np.sqrt(
                ((pixels[:, np.newaxis] - centroids) ** 2).sum(axis=2)
            )
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for i in range(actual_clusters):
                cluster_points = pixels[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = pixels[np.random.choice(pixels.shape[0])]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        _, counts = np.unique(labels, return_counts=True)
        dominant_color_bgr = centroids[np.argmax(counts)]

        b, g, r = [int(np.clip(round(x), 0, 255)) for x in dominant_color_bgr]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        results.append({"dominant_color_hex": hex_color})

    return results
