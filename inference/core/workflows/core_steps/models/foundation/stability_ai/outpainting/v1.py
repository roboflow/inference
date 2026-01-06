"""
Credits to: https://github.com/Fafruch for origin idea
"""

from enum import Enum
from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
import requests
import supervision as sv
from pydantic import ConfigDict, Field
from supervision import Color

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Extend images beyond their borders using Stability AI's outpainting API to generate new content that seamlessly continues the image in specified directions.

## How This Block Works

This block uses Stability AI's outpainting API to extend images beyond their original boundaries by generating new content that matches and continues the existing image. The block:

1. Takes an image and direction parameters (left, right, up, down) specifying how many pixels to extend in each direction
2. Sends the image to Stability AI's API with the specified extension directions
3. The API analyzes the image content and generates new content that seamlessly continues the scene beyond the borders
4. Returns the extended image with outpainted regions added in the specified directions

Outpainting extends images by generating contextually appropriate content that matches the style, lighting, and subject matter of the original image. You can extend the image in any combination of the four directions (left, right, top, bottom) by specifying pixel values. The creativity parameter controls how much the generated content diverges from the original style - higher values allow more creative variations. Optional prompts and presets can guide the style and content of the generated extensions.

## Common Use Cases

- **Aspect Ratio Conversion**: Extend images to different aspect ratios (e.g., convert portrait to landscape, square to wide) while maintaining visual coherence
- **Panoramic Expansion**: Expand panoramic or landscape photos to show more of the scene beyond the original frame boundaries
- **Composition Enhancement**: Extend images to improve composition by adding space around subjects or adjusting framing
- **Content Creation**: Create larger images from smaller originals for social media, print, or digital displays while maintaining visual quality
- **Background Extension**: Expand backgrounds in product photos, portraits, or architectural images to create more space or adjust framing
- **Creative Image Expansion**: Generate artistic extensions of images for creative projects, allowing AI to imaginatively continue scenes

## Connecting to Other Blocks

The outpainted images from this block can be connected to:

- **Object detection blocks** combined with crop blocks to first detect and isolate regions, then extend the image around those regions
- **Image transformation blocks** to further process or modify the extended images
- **Data storage blocks** (e.g., Local File Sink, Roboflow Dataset Upload) to save the generated images
- **Visualization blocks** to display the outpainted results
- **Additional image processing blocks** for further editing, cropping, or analysis of the extended content
- **Conditional logic blocks** (e.g., Continue If) to route workflows based on the success or quality of outpainting results

## Requirements

This block requires a Stability AI API key. The outpainting operation is performed remotely via Stability AI's API, so an active internet connection is required. At least one direction parameter (left, right, up, or down) must be specified with a value greater than 0, otherwise the original image is returned unchanged.
"""

SHORT_DESCRIPTION = "Use object detection bounding box to crop the image and to outpaint within given directions."

API_HOST = "https://api.stability.ai"
ENDPOINT = "/v2beta/stable-image/edit/outpaint"


class StabilityAIPresets(Enum):
    THREE_D_MODEL = "3d-model"
    ANALOG_FILM = "analog-film"
    ANIME = "anime"
    CINEMATIC = "cinematic"
    COMIC_BOOK = "comic-book"
    DIGITAL_ART = "digital-art"
    ENHANCE = "enhance"
    FANTASY_ART = "fantasy-art"
    ISOMETRIC = "isometric"
    LINE_ART = "line-art"
    LOW_POLY = "low-poly"
    MODELING_COMPOUND = "modeling-compound"
    NEON_PUNK = "neon-punk"
    ORIGAMI = "origami"
    PHOTOGRAPHIC = "photographic"
    PIXEL_ART = "pixel-art"
    TILE_TEXTURE = "tile-texture"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Stability AI Outpainting",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Stability AI",
                "stability.ai",
                "outpainting",
                "image generation",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-palette",
                "blockPriority": 14,
            },
        }
    )
    type: Literal["roboflow_core/stability_ai_outpainting@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="The input image to extend beyond its borders. This image will be used as the base, and new content will be generated to extend it in the specified directions (left, right, up, down).",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    creativity: Union[
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        float,
    ] = Field(
        default=0.5,
        description="Creativity parameter controlling how much the generated content can diverge from the original image style. Range from 0.0 to 1.0. Lower values (closer to 0) produce more conservative, style-consistent extensions. Higher values (closer to 1) allow more creative and varied content generation. Default is 0.5 for a balanced approach.",
        examples=[0.3, 0.5, 0.8],
    )
    left: Optional[
        Union[
            Selector(kind=[INTEGER_KIND]),
            int,
        ]
    ] = Field(
        default=None,
        description="Number of pixels to extend the image on the left side. Maximum value is 2000 pixels. At least one direction (left, right, up, or down) must be specified. Set to None or 0 to skip extending in this direction.",
        examples=[200, 500, 1000],
    )
    right: Optional[
        Union[
            Selector(kind=[INTEGER_KIND]),
            int,
        ]
    ] = Field(
        default=None,
        description="Number of pixels to extend the image on the right side. Maximum value is 2000 pixels. At least one direction (left, right, up, or down) must be specified. Set to None or 0 to skip extending in this direction.",
        examples=[200, 500, 1000],
    )
    up: Optional[
        Union[
            Selector(kind=[INTEGER_KIND]),
            int,
        ]
    ] = Field(
        default=None,
        description="Number of pixels to extend the image on the top (up) side. Maximum value is 2000 pixels. At least one direction (left, right, up, or down) must be specified. Set to None or 0 to skip extending in this direction.",
        examples=[200, 500, 1000],
    )
    down: Optional[
        Union[
            Selector(kind=[INTEGER_KIND]),
            int,
        ]
    ] = Field(
        default=None,
        description="Number of pixels to extend the image on the bottom (down) side. Maximum value is 2000 pixels. At least one direction (left, right, up, or down) must be specified. Set to None or 0 to skip extending in this direction.",
        examples=[200, 500, 1000],
    )
    prompt: Optional[
        Union[
            Selector(kind=[STRING_KIND]),
            str,
        ]
    ] = Field(
        default=None,
        description="Optional text prompt to guide the content and style of the generated extensions. Use this to describe what you want to see in the extended regions (e.g., 'ocean waves', 'mountain landscape', 'urban cityscape'). If not provided, the AI will automatically continue the scene based on the existing image content.",
        examples=["ocean waves and sky", "mountain landscape", "$inputs.prompt"],
        json_schema_extra={
            "multiline": True,
        },
    )
    preset: Optional[StabilityAIPresets] = Field(
        default=None,
        description=f"Optional artistic preset to control the style of the generated extensions. Presets influence the visual style of the outpainted content. Available presets: {', '.join(m.value for m in StabilityAIPresets)}. For example, use 'photographic' for realistic results, 'anime' for anime-style, 'cinematic' for film-like quality, or 'digital-art' for artistic styles. If not provided, the default outpainting style matches the original image.",
        examples=[StabilityAIPresets.PHOTOGRAPHIC, StabilityAIPresets.CINEMATIC],
    )
    seed: Optional[
        Union[
            Selector(kind=[INTEGER_KIND]),
            int,
        ]
    ] = Field(
        default=None,
        description="Optional seed value to control the randomness of content generation. Using the same seed with the same image and parameters will produce reproducible results. Must be a number between 0 and 4294967294. If not provided, a random seed is used, resulting in different outputs each time.",
        examples=[200, 12345, "$inputs.seed"],
    )
    api_key: Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str] = Field(
        description="Your Stability AI API key required to access the outpainting API. You can obtain an API key from https://platform.stability.ai. This field is kept private for security.",
        examples=[
            "sk-xxx-xxx",
            "$inputs.stability_ai_api_key",
            "$secrets.stability_api_key",
        ],
        private=True,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class StabilityAIOutpaintingBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        api_key: str,
        creativity: float,
        left: Optional[int] = None,
        right: Optional[int] = None,
        up: Optional[int] = None,
        down: Optional[int] = None,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
        preset: Optional[StabilityAIPresets] = None,
    ) -> BlockResult:
        if not any([left, right, up, down]):
            return {
                "image": WorkflowImageData.copy_and_replace(
                    origin_image_data=image,
                    numpy_image=image.numpy_image.copy(),
                ),
            }
        left = min(2000, left) if left else 0
        right = min(2000, right) if right else 0
        up = min(2000, up) if up else 0
        down = min(2000, down) if down else 0
        creativity = max(0, min(1, creativity))
        seed = max(0, min(4294967294, seed)) if seed else None
        preset = (
            preset.value if preset in set(e.value for e in StabilityAIPresets) else None
        )

        request_data = {
            "output_format": "jpeg",
            "creativity": creativity,
        }
        if left:
            request_data["left"] = left
        if right:
            request_data["right"] = right
        if up:
            request_data["up"] = up
        if down:
            request_data["down"] = down
        if preset:
            request_data["preset"] = preset
        if prompt:
            request_data["prompt"] = prompt
        if seed:
            request_data["seed"] = seed

        response = requests.post(
            f"{API_HOST}{ENDPOINT}",
            headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
            files={
                "image": numpy_array_to_jpeg_bytes(image=image.numpy_image),
            },
            data=request_data,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Request to StabilityAI API failed: {str(response.json())}"
            )
        result_image = bytes_to_opencv_image(payload=response.content)
        return {
            "image": WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=result_image,
            ),
        }


def numpy_array_to_jpeg_bytes(
    image: np.ndarray,
) -> bytes:
    _, img_encoded = cv2.imencode(".jpg", image)
    return np.array(img_encoded).tobytes()


def bytes_to_opencv_image(
    payload: bytes, array_type: np.number = np.uint8
) -> np.ndarray:
    bytes_array = np.frombuffer(payload, dtype=array_type)
    decoding_result = cv2.imdecode(bytes_array, cv2.IMREAD_UNCHANGED)
    if decoding_result is None:
        raise ValueError("Could not encode bytes to OpenCV image.")
    return decoding_result
