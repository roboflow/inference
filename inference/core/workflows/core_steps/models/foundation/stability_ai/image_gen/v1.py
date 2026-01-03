import base64
import uuid
from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
import requests
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
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
Generate new images from text prompts or create variations of existing images using Stability AI's image generation API.

## How This Block Works

This block uses Stability AI's image generation API to create images based on text prompts. The block:

1. Takes a text prompt describing the image you want to generate
2. Optionally accepts an existing image to use as a starting point for image-to-image generation
3. Sends the prompt (and optional image) to Stability AI's API
4. The API generates a new image based on your prompt, optionally influenced by the input image
5. Returns the generated image

The block supports three model options: **core** (balanced quality and speed), **ultra** (highest quality), and **sd3** (Stable Diffusion 3). When an image is provided, the `strength` parameter controls how much the input image influences the output - lower values keep more of the original image structure, while higher values allow more creative variation. You can also use negative prompts to specify what you don't want in the generated image.

## Common Use Cases

- **Text-to-Image Generation**: Create images from scratch based on text descriptions for creative projects, concept visualization, or content creation
- **Image Variation**: Generate variations of existing images by providing an input image and a prompt, useful for exploring different styles or compositions
- **Creative Content Creation**: Produce artwork, illustrations, or visual concepts for marketing, social media, or design projects
- **Prototype Visualization**: Quickly generate visual mockups or prototypes from text descriptions before creating final assets
- **Style Transfer**: Transform existing images into different styles by using the image as input with a style-describing prompt
- **Product Visualization**: Generate product images or variations for e-commerce, catalog creation, or marketing materials

## Connecting to Other Blocks

The generated images from this block can be connected to:

- **Image processing blocks** (e.g., crop, resize, transform) to further modify or refine the generated images
- **Inpainting or outpainting blocks** (e.g., Stability AI Inpainting, Stability AI Outpainting) to edit or extend specific regions of generated images
- **Data storage blocks** (e.g., Local File Sink, Roboflow Dataset Upload) to save the generated images
- **Visualization blocks** to display the generated results
- **Object detection or classification blocks** to analyze or validate the content of generated images
- **Conditional logic blocks** (e.g., Continue If) to route workflows based on generation success or to generate multiple variations

## Requirements

This block requires a Stability AI API key. Image generation is performed remotely via Stability AI's API, so an active internet connection is required.
"""

SHORT_DESCRIPTION = (
    "generate new images from text, or create variations of existing images."
)

API_HOST = "https://api.stability.ai"
ENDPOINT = {
    "ultra": "/v2beta/stable-image/generate/ultra",
    "core": "/v2beta/stable-image/generate/core",
    "sd3": "/v2beta/stable-image/generate/sd3",
}


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Stability AI Image Generation",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "Stability AI",
                "stability.ai",
                "image variation",
                "image generation",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-palette",
            },
        }
    )
    type: Literal["roboflow_core/stability_ai_image_gen@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Optional input image to use as a starting point for image-to-image generation. When provided, the generated image will be influenced by this image based on the strength parameter. Leave empty (None) to generate images purely from text prompts. Use this for creating variations of existing images or style transfer.",
        examples=["$inputs.image", "$steps.previous_step.image"],
        default=None,
    )
    strength: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        description="Controls how much the input image influences the generated output when an image is provided. Range from 0.0 to 1.0. Lower values (closer to 0) keep more of the original image structure and appearance. Higher values (closer to 1) allow more creative variation, generating images that deviate more from the input. A value of 0 would keep the image identical to the input. A value of 1 would generate as if no image was provided. Default is 0.3 for moderate influence.",
        default=0.3,
        examples=[0.2, 0.3, 0.7, "$inputs.strength"],
    )
    prompt: Union[
        Selector(kind=[STRING_KIND]),
        Selector(kind=[STRING_KIND]),
        str,
    ] = Field(
        description="Text prompt describing the image you want to generate. Be descriptive and specific for best results (e.g., 'a red sports car on a mountain road at sunset', 'a futuristic cityscape with flying cars', 'a peaceful lake with mountains in the background'). The AI will generate an image based on this description.",
        examples=["a red sports car on a mountain road", "a futuristic cityscape", "$inputs.prompt"],
        json_schema_extra={
            "multiline": True,
        },
    )
    negative_prompt: Optional[
        Union[
            Selector(kind=[STRING_KIND]),
            Selector(kind=[STRING_KIND]),
            str,
        ]
    ] = Field(
        default=None,
        description="Optional negative prompt describing what you do not want to see in the generated image. Use this to guide the AI away from unwanted elements or styles (e.g., 'blurry, distorted, low quality', 'people, faces', 'text, watermarks', 'cartoon style'). Helps refine the output by excluding undesired features.",
        examples=["blurry, distorted, low quality", "people, faces", "$inputs.negative_prompt"],
    )
    model: Optional[
        Union[
            Selector(kind=[STRING_KIND]),
            Selector(kind=[STRING_KIND]),
            str,
        ]
    ] = Field(
        default="core",
        description="Stability AI model to use for image generation. Choose from: 'core' (balanced quality and speed, default), 'ultra' (highest quality, slower), or 'sd3' (Stable Diffusion 3, latest model). The 'core' model provides a good balance for most use cases. Use 'ultra' for maximum quality when speed is not critical. Use 'sd3' for access to the latest Stable Diffusion capabilities.",
        examples=["core", "ultra", "sd3", "$inputs.model"],
    )
    api_key: Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str] = Field(
        description="Your Stability AI API key required to access the image generation API. You can obtain an API key from https://platform.stability.ai. This field is kept private for security.",
        examples=["sk-xxx-xxx", "$inputs.stability_ai_api_key", "$secrets.stability_api_key"],
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


class StabilityAIImageGenBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        prompt: str,
        negative_prompt: str,
        model: str,
        api_key: str,
        image: WorkflowImageData,
        strength: float = 0.3,
    ) -> BlockResult:
        request_data = {
            "prompt": prompt,
            "output_format": "jpeg",
        }
        files_to_send = {"none": ""}
        if image is not None:
            encoded_image = numpy_array_to_jpeg_bytes(image=image.numpy_image)
            files_to_send = {
                "image": encoded_image,
            }
            request_data["strength"] = strength

        if negative_prompt is not None:
            request_data["negative_prompt"] = negative_prompt
        if model not in ENDPOINT.keys():
            model = "core"
        response = requests.post(
            f"{API_HOST}{ENDPOINT[model]}",
            headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
            files=files_to_send,
            data=request_data,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Request to StabilityAI API failed: {str(response.json())}"
            )
        new_image_base64 = base64.b64encode(response.content).decode("utf-8")
        parent_metadata = ImageParentMetadata(parent_id=str(uuid.uuid1()))
        return {
            "image": WorkflowImageData(parent_metadata, base64_image=new_image_base64),
        }


def numpy_array_to_jpeg_bytes(
    image: np.ndarray,
) -> bytes:
    _, img_encoded = cv2.imencode(".jpg", image)
    return np.array(img_encoded).tobytes()
