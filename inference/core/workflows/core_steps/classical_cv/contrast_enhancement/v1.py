from typing import List, Literal, Optional, Type

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
    "Enhance image contrast by normalizing the histogram to use the full dynamic range."
)
LONG_DESCRIPTION = """
Enhance image contrast using histogram normalization (the algorithm from GIMP's Auto Levels). This block stretches the image histogram to use the full available range [0-255], improving visibility of features with low contrast.

## How This Block Works

1. **Channel Analysis**: For grayscale images, find min/max directly. For color images, analyze each channel independently.
2. **Histogram Normalization**: For each channel, stretch values from [min, max] to [0, 255] using linear scaling: `output = (input - min) / (max - min) * 255`
3. **Clipping**: Values outside [0, 255] are clipped to the valid range

## Common Use Cases

- **Low-contrast medical imaging**: Normalize tissue visibility across varying acquisition parameters
- **Industrial inspection**: Enhance subtle surface defects on dull materials
- **Surveillance footage**: Improve nighttime or backlit scene visibility
- **Document scanning**: Brighten poorly lit document photos
- **Microscopy**: Boost signal from weak fluorescence or phase-contrast images

## Input Parameters

**image** : Input image to enhance (color or grayscale)
- Can be single-channel, 3-channel (BGR), or 4-channel (BGRA)
- Each channel is normalized independently for color images

**clip_limit** : Percentage of histogram range to skip at extremes (default: 0)
- Range: 0-50
- 0: No clipping, entire histogram from min to max is used
- 1-3: Skip dark and bright outliers (robust to noise)
- 5-10: Very aggressive outlier removal
- 20-50: Extreme outlier removal, may lose subtle details

**contrast_multiplier** : Multiplier for contrast scaling after normalization (default: 1.0)
- Range: 0.1-5.0
- 1.0: No additional scaling, just normalization
- 0.5-0.9: Reduce contrast for smoother images
- 1.1-2.0: Increase contrast for more dramatic enhancement

**normalize_brightness** : Apply brightness normalization using midtone equalization (default: False)
- False: Only histogram normalization and contrast scaling
- True: After histogram normalization, apply midtone adjustment for balanced brightness

## Outputs

**image** : Enhanced image with normalized contrast, same shape and type as input

## Notes

- **Sensitive to outliers**: Extreme min/max values (single dark/bright pixels) stretch most of the histogram into a narrow range. Use morphological opening (Morphological Transformation v2) as preprocessing to remove spurious dark/bright specks.
- **Color shift**: For color images, each channel stretches independently, which can shift hue if channels have very different dynamic ranges
- **Efficiency**: Very fast — linear scan for min/max, then linear transformation per pixel
- **Brightness normalization**: When enabled, applies midtone stretch (gamma ≈ 1.3) for more balanced perceived brightness
"""


class ContrastEnhancementManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/contrast_enhancement@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Contrast Enhancement",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-adjust",
                "blockPriority": 10,
                "opencv": True,
            },
        }
    )

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Image to enhance (color or grayscale). Each channel is normalized independently.",
        examples=["$inputs.image", "$steps.preprocessing.image"],
        validation_alias=AliasChoices("image", "images"),
    )

    clip_limit: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Percentage of histogram range to skip at dark and bright extremes. 0: use full range from min to max. 1-3: skip outliers (robust). 5-10: very aggressive outlier removal.",
    )

    contrast_multiplier: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Multiplier for contrast scaling after normalization. 1.0: no additional scaling (just histogram normalization). <1.0: reduce contrast. >1.0: increase contrast for more dramatic enhancement.",
    )

    normalize_brightness: bool = Field(
        default=False,
        description="Apply brightness normalization using midtone equalization. When False, only histogram normalization and contrast scaling are applied. When True, applies midtone adjustment for more balanced perceived brightness.",
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


class ContrastEnhancementBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[ContrastEnhancementManifest]:
        return ContrastEnhancementManifest

    def run(
        self,
        image: WorkflowImageData,
        clip_limit: int,
        contrast_multiplier: float,
        normalize_brightness: bool,
        *args,
        **kwargs,
    ) -> BlockResult:
        enhanced_image = enhance_contrast(
            image=image,
            clip_limit=clip_limit,
            contrast_multiplier=contrast_multiplier,
            normalize_brightness=normalize_brightness,
        )

        return {
            "image": enhanced_image,
        }


def enhance_contrast(
    image: WorkflowImageData,
    clip_limit: int,
    contrast_multiplier: float = 1.0,
    normalize_brightness: bool = False,
) -> WorkflowImageData:
    """Enhance image contrast using histogram normalization with optional brightness normalization."""

    np_img = image.numpy_image.copy()
    clip_pct = float(clip_limit) / 100.0
    contrast_mult = float(contrast_multiplier)
    gamma_val = (
        1.0 / 1.3 if normalize_brightness else 1.0
    )  # Midtone brightening when enabled

    # Handle different image formats
    if len(np_img.shape) == 2:
        # Grayscale
        return _enhance_channel_contrast(
            image, np_img, clip_pct, contrast_mult, gamma_val
        )
    elif np_img.shape[2] == 1:
        # Single channel
        np_img = np_img[:, :, 0]
        return _enhance_channel_contrast(
            image, np_img, clip_pct, contrast_mult, gamma_val
        )
    elif np_img.shape[2] == 4:
        # BGRA: enhance BGR separately, keep alpha
        bgr = np_img[:, :, :3]
        alpha = np_img[:, :, 3:4]
        enhanced_bgr = _enhance_multichannel_contrast(
            bgr, clip_pct, contrast_mult, gamma_val
        )
        enhanced_img = np.concatenate([enhanced_bgr, alpha], axis=2)
        return WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=enhanced_img.astype(np.uint8),
        )
    else:
        # BGR
        enhanced_img = _enhance_multichannel_contrast(
            np_img, clip_pct, contrast_mult, gamma_val
        )
        return WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=enhanced_img.astype(np.uint8),
        )


def _enhance_channel_contrast(
    image: WorkflowImageData,
    channel: np.ndarray,
    clip_pct: float,
    contrast_mult: float,
    gamma_val: float,
) -> WorkflowImageData:
    """Enhance contrast of a single grayscale channel."""
    channel_float = channel.astype(np.float32)

    # Compute histogram percentiles for clipping
    flat = channel_float.flatten()
    if clip_pct > 0:
        lower_pct = clip_pct * 100
        upper_pct = 100 - clip_pct * 100
        v_min = np.percentile(flat, lower_pct)
        v_max = np.percentile(flat, upper_pct)
    else:
        v_min = flat.min()
        v_max = flat.max()

    # Avoid division by zero
    if v_max <= v_min:
        return image

    # Normalize: (x - min) / (max - min) * 255
    normalized = (channel_float - v_min) / (v_max - v_min) * 255.0

    # Apply contrast multiplier
    if contrast_mult != 1.0:
        center = 128.0
        normalized = center + (normalized - center) * contrast_mult

    # Apply gamma correction (brightness normalization when gamma_val != 1.0)
    if gamma_val != 1.0:
        normalized = np.clip(normalized, 0, 255)
        normalized = np.power(normalized / 255.0, gamma_val) * 255.0

    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        numpy_image=normalized,
    )


def _enhance_multichannel_contrast(
    image: np.ndarray,
    clip_pct: float,
    contrast_mult: float,
    gamma_val: float,
) -> np.ndarray:
    """Enhance contrast of each color channel independently."""
    enhanced = np.zeros_like(image, dtype=np.float32)

    for c in range(image.shape[2]):
        channel = image[:, :, c].astype(np.float32)

        # Compute histogram percentiles for clipping
        if clip_pct > 0:
            lower_pct = clip_pct * 100
            upper_pct = 100 - clip_pct * 100
            v_min, v_max = np.quantile(channel, [lower_pct / 100, upper_pct / 100])
        else:
            v_min = channel.min()
            v_max = channel.max()

        # Avoid division by zero
        if v_max > v_min:
            normalized = (channel - v_min) / (v_max - v_min) * 255.0

            # Apply contrast multiplier
            if contrast_mult != 1.0:
                center = 128.0
                normalized = center + (normalized - center) * contrast_mult

            # Apply gamma correction (brightness normalization when gamma_val != 1.0)
            if gamma_val != 1.0:
                normalized = np.clip(normalized, 0, 255)
                normalized = np.power(normalized / 255.0, gamma_val) * 255.0

            enhanced[:, :, c] = np.clip(normalized, 0, 255)
        else:
            enhanced[:, :, c] = channel

    return enhanced.astype(np.uint8)
