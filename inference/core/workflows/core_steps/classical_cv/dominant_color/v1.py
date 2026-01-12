from typing import List, Literal, Optional, Type, Union

import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    RGB_COLOR_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Get the dominant color of an image in RGB format."
LONG_DESCRIPTION = """
Extract the most prevalent dominant color from an image using K-means clustering on pixel colors, analyzing color distribution, identifying color clusters, and returning the RGB value of the most frequently occurring color cluster for color analysis, theme extraction, color-based filtering, and visual analysis workflows.

## How This Block Works

This block analyzes the color distribution in an image using K-means clustering to identify and extract the dominant (most prevalent) color. The block:

1. Receives an input image (assumes RGB format) to analyze for dominant color
2. Downsamples the image to optimize processing speed:
   - Calculates a scale factor based on the smallest dimension and target_size parameter
   - Reduces image resolution while preserving color characteristics
   - Smaller target_size values speed up processing but may slightly reduce precision
3. Reshapes the image pixels into a 2D array where each row represents one pixel's RGB values
4. Initializes K-means clustering:
   - Selects random pixel colors as initial cluster centroids (number of clusters = color_clusters)
   - Creates initial color clusters to group similar pixel colors
5. Performs iterative K-means clustering (up to max_iterations):
   - Assigns each pixel to the nearest color cluster (based on Euclidean distance in RGB color space)
   - Updates cluster centroids by computing the mean RGB values of pixels in each cluster
   - Handles empty clusters by reinitializing them to random pixel colors
   - Checks for convergence (centroids stop changing significantly) and exits early if converged
6. Counts pixels in each color cluster to determine which cluster contains the most pixels
7. Selects the cluster with the highest pixel count as the dominant color cluster
8. Extracts the RGB values from the dominant cluster's centroid
9. Converts and clips RGB values to valid 0-255 integer range
10. Returns the dominant color as an RGB tuple (R, G, B values)

The block uses K-means clustering to group similar pixel colors together, then identifies the largest color group as the dominant color. Processing time depends on image size, color complexity, and parameter settings (color_clusters, max_iterations, target_size). Most images complete in under half a second with default settings. The downsampling step balances speed and accuracy - reducing resolution speeds up clustering while still capturing the overall color distribution.

## Common Use Cases

- **Color Theme Extraction**: Extract dominant colors from images to identify color themes or palettes (e.g., extract dominant colors from product images, identify color themes in photographs, analyze color palettes in images), enabling color theme analysis workflows
- **Image Color Analysis**: Analyze images to determine their primary color characteristics (e.g., identify dominant colors in images, analyze color distribution, extract color signatures from images), enabling color-based image analysis
- **Color-Based Filtering**: Use dominant colors for filtering or categorizing images (e.g., filter images by dominant color, categorize images by color themes, group images by color characteristics), enabling color-based classification workflows
- **Visual Analysis and Reporting**: Extract color information for visual analysis or reporting (e.g., generate color reports for images, analyze color trends in image collections, extract color metadata for image databases), enabling color reporting workflows
- **Design and Branding Analysis**: Analyze images for design or branding purposes (e.g., extract brand colors from images, analyze design color schemes, identify color usage in branded content), enabling design analysis workflows
- **Quality Control**: Use dominant color analysis for quality control or inspection (e.g., verify expected colors in products, detect color anomalies, validate color characteristics), enabling color-based quality control workflows

## Connecting to Other Blocks

This block receives an image and produces a dominant RGB color:

- **After image input blocks** to extract dominant colors from input images (e.g., analyze colors in camera feeds, extract colors from image inputs, analyze color characteristics of images), enabling color analysis workflows
- **After crop blocks** to analyze dominant colors in specific image regions (e.g., extract dominant colors from cropped regions, analyze colors in specific areas, identify colors in selected regions), enabling region-based color analysis workflows
- **Before filtering or logic blocks** that use color information for decision-making (e.g., filter based on dominant colors, make decisions based on color characteristics, apply logic based on color analysis), enabling color-based conditional workflows
- **Before visualization blocks** to use dominant colors for visualization (e.g., use extracted colors for annotations, apply dominant colors to visualizations, customize visualizations with extracted colors), enabling color-enhanced visualization workflows
- **In color analysis pipelines** where dominant color extraction is part of a larger analysis workflow (e.g., analyze colors in multi-stage workflows, extract colors for comprehensive analysis, process color information in pipelines), enabling color analysis pipeline workflows
- **Before data storage blocks** to store color information along with images (e.g., store dominant colors with image metadata, save color analysis results, record color characteristics), enabling color metadata storage workflows
"""


class DominantColorManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/dominant_color@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Dominant Color",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-palette",
                "blockPriority": 1,
                "opencv": True,
            },
        }
    )
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Input image to analyze for dominant color extraction. The image is assumed to be in RGB format. The block analyzes all pixels in the image to determine the most prevalent color. Processing time depends on image size and complexity. The image is automatically downsampled during processing to optimize speed while preserving color characteristics.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    color_clusters: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Color Clusters",
        description="Number of color clusters (K) to use in K-means clustering. Must be between 1 and 10. The algorithm groups pixel colors into this many clusters, then selects the largest cluster as the dominant color. Higher values (e.g., 6-8) can improve precision for images with complex color distributions but increase processing time. Lower values (e.g., 2-3) are faster but may be less precise for multi-color images. Default is 4, which provides a good balance. Use fewer clusters for images with simple color schemes, more clusters for images with varied colors.",
        default=4,
        examples=[4, "$inputs.color_clusters"],
        gt=0,
        le=10,
    )
    max_iterations: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Max Iterations",
        description="Maximum number of K-means clustering iterations to perform. Must be between 1 and 500. The algorithm iteratively refines color clusters and stops early if convergence is reached (centroids stop changing). Higher values allow more refinement and can improve precision but increase processing time. Lower values are faster but may result in less refined color clusters. Default is 100, which is typically sufficient for convergence. Most images converge well before reaching the maximum. Increase if you need more precise clustering, decrease if speed is critical.",
        default=100,
        examples=[100, "$inputs.max_iterations"],
        gt=0,
        le=500,
    )
    target_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Target Size",
        description="Target size in pixels for the smallest dimension of the downsampled image used for clustering. Must be between 1 and 250. The image is downsampled before clustering to speed up processing - the smallest dimension is resized to approximately this size while maintaining aspect ratio. Lower values (e.g., 50-75) speed up processing significantly but may slightly reduce precision for images with fine color details. Higher values (e.g., 150-200) preserve more detail but are slower. Default is 100 pixels, which provides a good balance. Use lower values for speed-critical applications, higher values for maximum precision.",
        default=100,
        examples=[100, "$inputs.target_size"],
        gt=0,
        le=250,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="rgb_color", kind=[RGB_COLOR_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DominantColorBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[DominantColorManifest]:
        return DominantColorManifest

    def run(
        self,
        image: WorkflowImageData,
        color_clusters: Optional[int],
        max_iterations: Optional[int],
        target_size: Optional[int],
        *args,
        **kwargs
    ) -> BlockResult:
        np_image = image.numpy_image

        # Downsample the image to speed up processing
        height, width = np_image.shape[:2]
        scale_factor = max(1, min(width, height) // target_size)
        np_image = np_image[::scale_factor, ::scale_factor]

        pixels = np_image.reshape(-1, 3).astype(np.float32)

        centroids = pixels[
            np.random.choice(pixels.shape[0], color_clusters, replace=False)
        ]

        for _ in range(max_iterations):
            # Assign pixels to nearest centroid
            distances = np.sqrt(((pixels[:, np.newaxis] - centroids) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(color_clusters):
                cluster_points = pixels[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    # If cluster is empty, reinitialize to a random point
                    new_centroids[i] = pixels[np.random.choice(pixels.shape[0])]

            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        # Get the colors and their counts
        colors = centroids
        _, counts = np.unique(labels, return_counts=True)

        # Find the most dominant color
        dominant_color = colors[np.argmax(counts)]
        rgb_color = tuple(
            int(np.clip(round(x), 0, 255)) for x in reversed(dominant_color)
        )

        return {"rgb_color": rgb_color}
