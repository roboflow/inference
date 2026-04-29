from typing import List, Literal, Optional, Type

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = (
    "Smooth the image while preserving edges using bilateral filtering."
)
LONG_DESCRIPTION = """
Apply bilateral filtering to denoise an image while preserving sharp edges. Unlike Gaussian blur which blurs everything uniformly, bilateral filtering considers both spatial distance and intensity difference, making it ideal for preprocessing when you need to remove noise without softening important boundaries.

## How This Block Works

Bilateral filtering is a non-linear filter that averages pixel values based on:
1. **Spatial distance**: Pixels spatially close to the center pixel have more weight
2. **Intensity distance**: Pixels with similar intensity to the center pixel have more weight

This means edges (where intensity changes sharply) naturally receive less smoothing than flat regions, resulting in "edge-preserving" denoising.

## Common Use Cases

- **Medical imaging**: Denoise CT/MRI while preserving tissue boundaries
- **Industrial quality control**: Remove surface noise from reflective or textured materials without blurring edges
- **Document processing**: Clean up scanner noise from document photos while keeping text sharp
- **Autonomous vehicles**: Denoise road scene while preserving lane/road boundaries
- **Microscopy**: Smooth cellular background noise while preserving nuclear/membrane boundaries
- **Preprocessing for edge detection**: Bilateral filter is often better than Gaussian for edge detection tasks

## Input Parameters

**image** : Input image to filter (color or grayscale)
- Can be single-channel, 3-channel (BGR), or 4-channel (BGRA)
- Applied independently to each channel

**diameter** : Diameter of the pixel neighborhood in pixels (default: 9)
- Range: 3-51 (must be odd)
- Small (3-5): minimal smoothing, fast
- Medium (7-15): good balance for most applications
- Large (21-51): strong smoothing, significantly slower

**sigma_color** : Intensity standard deviation for the range Gaussian (default: 75)
- Range: 1-255
- Controls how much difference in intensity prevents blurring across an edge
- Small (5-20): only very similar intensities get blurred together (sharp edges)
- Medium (50-100): more permissive blurring across gradual intensity transitions
- Large (150-255): almost everything gets blurred (only geometry matters)

**sigma_space** : Spatial standard deviation for the spatial Gaussian (default: 75)
- Range: 1-255
- Controls decay of spatial influence with distance from center
- Typically set equal to sigma_color for balanced results
- Increasing beyond sigma_color can make the filter "brighter" (more distant pixels contribute)

## Outputs

**image** : Filtered image with same shape and type as input

## Notes

- **Computational cost**: Bilateral filtering is approximately O(n * d^2) where n is the number of pixels and d is diameter. It's significantly slower than Gaussian blur — typical 640x480 image with diameter=9 takes ~100-200ms.
- **Disk I/O bottleneck**: With large images (>4K), memory bandwidth can be limiting
- **Alpha channel handling**: For BGRA input, the alpha channel is premultiplied before filtering and un-premultiplied after. This ensures semi-transparent pixels don't overstate their weight in the bilateral filter, and the transparency map is smoothed consistently with the color signals.
- **Parameter tuning**: Start with diameter=9, sigma_color=75, sigma_space=75, then adjust:
  - Increase sigma_color if edges are still too noisy
  - Increase diameter if you need more smoothing in flat regions
  - Decrease both sigmas to preserve more detail
"""


class BilateralFilterManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/bilateral_filter@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Bilateral Filter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-water",
                "blockPriority": 11,
                "opencv": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Image to filter (color or grayscale). Bilateral filtering is applied to each channel independently.",
        examples=["$inputs.image", "$steps.preprocessing.image"],
        validation_alias=AliasChoices("image", "images"),
    )

    diameter: int = Field(
        default=9,
        ge=3,
        le=51,
        description="Diameter of the pixel neighborhood in pixels. Must be odd. 3-5: minimal smoothing (fast). 7-15: typical. 21-51: strong smoothing (slow).",
    )

    sigma_color: float = Field(
        default=75.0,
        ge=1.0,
        le=255.0,
        description="Intensity standard deviation for the range Gaussian. Controls how much intensity difference prevents blurring across edges. Small (5-20): preserve sharp edges. Large (150-255): almost uniform blurring.",
    )

    sigma_space: float = Field(
        default=75.0,
        ge=1.0,
        le=255.0,
        description="Spatial standard deviation for the spatial Gaussian. Controls how far spatial influence extends. Typically set equal to sigma_color.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="image",
                kind=[IMAGE_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BilateralFilterBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[BilateralFilterManifest]:
        return BilateralFilterManifest

    def run(
        self,
        image: WorkflowImageData,
        diameter: int,
        sigma_color: float,
        sigma_space: float,
        *args,
        **kwargs,
    ) -> BlockResult:
        filtered_image = apply_bilateral_filter(
            image=image,
            diameter=diameter,
            sigma_color=sigma_color,
            sigma_space=sigma_space,
        )

        return {
            "image": filtered_image,
        }


def apply_bilateral_filter(
    image: WorkflowImageData,
    diameter: int,
    sigma_color: float,
    sigma_space: float,
) -> WorkflowImageData:
    """Apply bilateral filtering to denoise while preserving edges."""

    np_img = image.numpy_image.copy()

    # Ensure diameter is odd
    d = int(diameter)
    if d % 2 == 0:
        d += 1
    d = max(3, min(d, 51))

    sigma_c = float(sigma_color)
    sigma_s = float(sigma_space)

    # Handle different image formats
    if len(np_img.shape) == 2:
        # Grayscale
        filtered = cv2.bilateralFilter(np_img, d, sigma_c, sigma_s)
        return WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=filtered,
        )
    elif np_img.shape[2] == 1:
        # Single channel
        filtered = cv2.bilateralFilter(np_img[:, :, 0], d, sigma_c, sigma_s)
        return WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=filtered,
        )
    elif np_img.shape[2] == 4:
        # BGRA: check if alpha is uniform (all fully opaque)
        alpha_channel = np_img[:, :, 3]
        if np.all(alpha_channel == 255):
            # Fully opaque: filter BGR directly as 3-channel image (faster)
            bgr = np_img[:, :, :3]
            filtered_bgr = cv2.bilateralFilter(bgr, d, sigma_c, sigma_s)
            filtered_img = np.concatenate(
                [filtered_bgr, alpha_channel[:, :, np.newaxis]], axis=2
            )
            return WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=filtered_img,
            )
        else:
            # Variable alpha: premultiply, filter each channel separately, then un-premultiply
            # This ensures semi-transparent pixels don't overstate their weight in the filter
            bgr = np_img[:, :, :3].astype(np.float32)
            alpha = alpha_channel.astype(np.float32) / 255.0  # 2D array

            # Premultiply: scale color by alpha so transparent pixels have less influence
            bgr_premultiplied = bgr * alpha[:, :, np.newaxis]

            # Filter each channel separately (OpenCV only supports 1 or 3 channels)
            filtered_b = cv2.bilateralFilter(
                bgr_premultiplied[:, :, 0].astype(np.uint8), d, sigma_c, sigma_s
            ).astype(np.float32)
            filtered_g = cv2.bilateralFilter(
                bgr_premultiplied[:, :, 1].astype(np.uint8), d, sigma_c, sigma_s
            ).astype(np.float32)
            filtered_r = cv2.bilateralFilter(
                bgr_premultiplied[:, :, 2].astype(np.uint8), d, sigma_c, sigma_s
            ).astype(np.float32)
            filtered_alpha = cv2.bilateralFilter(
                (alpha * 255.0).astype(np.uint8), d, sigma_c, sigma_s
            ).astype(np.float32) / 255.0

            # Stack filtered channels
            filtered_bgr_premult = np.stack([filtered_b, filtered_g, filtered_r], axis=2)

            # Un-premultiply: divide by alpha to recover original color space
            # Avoid division by zero by adding small epsilon
            epsilon = 1e-6
            filtered_bgr = filtered_bgr_premult / (filtered_alpha[:, :, np.newaxis] + epsilon)

            # Clamp to valid range and convert back to uint8
            filtered_bgr = np.clip(filtered_bgr, 0, 255).astype(np.uint8)
            filtered_alpha = (np.clip(filtered_alpha, 0, 1) * 255).astype(np.uint8)

            # Expand alpha back to 4th channel dimension
            filtered_img = np.concatenate(
                [filtered_bgr, filtered_alpha[:, :, np.newaxis]], axis=2
            )
            return WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=filtered_img,
            )
    else:
        # BGR
        filtered = cv2.bilateralFilter(np_img, d, sigma_c, sigma_s)
        return WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=filtered,
        )
