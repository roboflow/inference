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
    BOOLEAN_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
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
Use Stability AI's inpainting API to intelligently fill or replace masked regions in images with AI-generated content based on text prompts.

## How This Block Works

This block uses Stability AI's inpainting API to replace specific regions of an image with AI-generated content. The block:

1. Takes an image and a segmentation mask (from instance segmentation predictions) that defines which areas to modify
2. Processes the mask to create a clear boundary for the inpainting region
3. Sends the image, mask, and text prompt to Stability AI's API
4. The API generates new content that matches the prompt and seamlessly blends with the surrounding image
5. Returns the inpainted image with the masked regions filled or replaced with the generated content

The segmentation mask determines which parts of the image will be modified - areas covered by the mask are replaced with AI-generated content based on your prompt. You can invert the mask to inpaint the background instead of the foreground objects. The block supports various artistic presets (photographic, cinematic, anime, digital art, etc.) and optional negative prompts to guide what should not appear in the generated content.

## Common Use Cases

- **Object Replacement**: Remove unwanted objects from images and replace them with AI-generated alternatives (e.g., replace a person with a different person, swap products, change backgrounds)
- **Content Removal and Filling**: Remove objects or people from photos and intelligently fill the space with contextually appropriate content (e.g., remove tourists from scenic photos, clean up product photos)
- **Creative Image Editing**: Transform specific regions of images with creative prompts (e.g., turn a car into a futuristic vehicle, change clothing styles, add artistic elements)
- **Product Photography**: Replace backgrounds, remove reflections or shadows, or modify product appearances for marketing materials
- **Content Moderation**: Replace inappropriate or unwanted content with AI-generated alternatives while maintaining visual coherence
- **Scene Enhancement**: Add or modify elements in scenes based on text descriptions (e.g., add clouds to sky, change weather, add decorative elements)

## Connecting to Other Blocks

The inpainted images from this block can be connected to:

- **Instance segmentation blocks** (e.g., Instance Segmentation Model, Segment Anything 3) to generate masks that define which regions to inpaint
- **Object detection blocks** combined with transformation blocks to first detect objects, then create masks for inpainting
- **Data storage blocks** (e.g., Local File Sink, Roboflow Dataset Upload) to save the generated images
- **Visualization blocks** to display the inpainted results
- **Additional image processing blocks** for further editing or analysis of the generated content
- **Conditional logic blocks** (e.g., Continue If) to route workflows based on the success or quality of inpainting results

## Requirements

This block requires a Stability AI API key. The inpainting operation is performed remotely via Stability AI's API, so an active internet connection is required.
"""

SHORT_DESCRIPTION = "Use segmentation masks to inpaint objects within an image."

API_HOST = "https://api.stability.ai"
ENDPOINT = "/v2beta/stable-image/edit/inpaint"


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
            "name": "Stability AI Inpainting",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Stability AI",
                "stability.ai",
                "inpainting",
                "image generation",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-palette",
                "blockPriority": 14,
            },
        }
    )
    type: Literal["roboflow_core/stability_ai_inpainting@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="The input image to inpaint. This is the image that will be modified by replacing masked regions with AI-generated content based on the prompt.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    segmentation_mask: Selector(kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]) = Field(
        name="Segmentation Mask",
        description="Instance segmentation predictions that define which regions of the image to inpaint. The mask determines which areas will be replaced with AI-generated content. Typically comes from instance segmentation model outputs (e.g., Instance Segmentation Model, Segment Anything 3). You can invert the mask using the invert_segmentation_mask parameter to inpaint the background instead of the foreground.",
        examples=["$steps.segmentation.predictions", "$steps.model.predictions"],
    )
    prompt: Union[
        Selector(kind=[STRING_KIND]),
        str,
    ] = Field(
        description="Text prompt describing what you want to see in the inpainted regions. The AI will generate content based on this prompt to fill the masked areas. Be descriptive and specific for best results (e.g., 'a red sports car', 'a sunny beach with palm trees', 'modern office furniture').",
        examples=[
            "a red sports car",
            "a sunny beach with palm trees",
            "$inputs.prompt",
        ],
        json_schema_extra={
            "multiline": True,
        },
    )
    negative_prompt: Optional[
        Union[
            Selector(kind=[STRING_KIND]),
            str,
        ]
    ] = Field(
        default=None,
        description="Optional negative prompt describing what you do not want to see in the generated content. Use this to guide the AI away from unwanted elements (e.g., 'blurry, distorted, low quality', 'people, faces', 'text, watermarks').",
        examples=[
            "blurry, distorted, low quality",
            "people, faces",
            "$inputs.negative_prompt",
        ],
    )
    api_key: Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str] = Field(
        description="Your Stability AI API key required to access the inpainting API. You can obtain an API key from https://platform.stability.ai. This field is kept private for security.",
        examples=[
            "sk-xxx-xxx",
            "$inputs.stability_ai_api_key",
            "$secrets.stability_api_key",
        ],
        private=True,
    )
    invert_segmentation_mask: Union[
        Selector(kind=[BOOLEAN_KIND]),
        bool,
    ] = Field(
        default=False,
        description="Invert the segmentation mask to inpaint the background instead of the foreground objects. When True, the areas outside the segmentation mask (background) will be replaced with generated content instead of the masked regions. Useful for changing backgrounds while keeping foreground objects intact.",
    )
    preset: Optional[StabilityAIPresets] = Field(
        default=None,
        description=f"Optional artistic preset to apply when inpainting the image. Presets control the style and appearance of the generated content. Available presets: {', '.join(m.value for m in StabilityAIPresets)}. For example, use 'photographic' for realistic results, 'anime' for anime-style, 'cinematic' for film-like quality, or 'digital-art' for artistic styles. If not provided, the default inpainting style is used.",
        examples=[StabilityAIPresets.PHOTOGRAPHIC, StabilityAIPresets.CINEMATIC],
    )
    seed: Optional[
        Union[
            Selector(kind=[INTEGER_KIND]),
            int,
        ]
    ] = Field(
        default=None,
        description="Optional seed value to control the randomness of content generation. Using the same seed with the same prompt and image will produce reproducible results. Must be a number between 0 and 4294967294. If not provided, a random seed is used, resulting in different outputs each time.",
        examples=[200, 12345, "$inputs.seed"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class StabilityAIInpaintingBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        segmentation_mask: sv.Detections,
        prompt: str,
        negative_prompt: str,
        api_key: str,
        invert_segmentation_mask: bool,
        preset: Optional[StabilityAIPresets] = None,
        seed: Optional[int] = None,
    ) -> BlockResult:
        black_image = np.zeros_like(image.numpy_image)
        mask_annotator = sv.MaskAnnotator(color=Color.WHITE, opacity=1.0)
        mask = mask_annotator.annotate(black_image, segmentation_mask)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        encoded_image = numpy_array_to_jpeg_bytes(image=image.numpy_image)
        if invert_segmentation_mask:
            mask = cv2.bitwise_not(mask)
        encoded_mask = numpy_array_to_jpeg_bytes(image=mask)
        request_data = {
            "prompt": prompt,
            "output_format": "jpeg",
        }
        preset = (
            preset.value if preset in set(e.value for e in StabilityAIPresets) else None
        )
        if preset:
            request_data["preset"] = preset
        seed = max(0, min(4294967294, seed)) if seed else None
        if seed:
            request_data["seed"] = seed
        response = requests.post(
            f"{API_HOST}{ENDPOINT}",
            headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
            files={
                "image": encoded_image,
                "mask": encoded_mask,
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
