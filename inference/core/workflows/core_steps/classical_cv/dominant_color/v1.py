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
Extract the dominant color from an input image using K-means clustering.

This block identifies the most prevalent color in an image.
Processing time is dependant on color complexity and image size.
Most images should complete in under half a second.

The output is a list of RGB values representing the dominant color, making it easy 
to use in further processing or visualization tasks.

Note: The block operates on the assumption that the input image is in RGB format. 
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
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    color_clusters: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Color Clusters",
        description="Number of dominant colors to identify. Higher values increase precision but may slow processing.",
        default=4,
        examples=[4, "$inputs.color_clusters"],
        gt=0,
        le=10,
    )
    max_iterations: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Max Iterations",
        description="Max number of iterations to perform. Higher values increase precision but may slow processing.",
        default=100,
        examples=[100, "$inputs.max_iterations"],
        gt=0,
        le=500,
    )
    target_size: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        title="Target Size",
        description="Sets target for the smallest dimension of the downsampled image in pixels. Lower values increase speed but may reduce precision.",
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
