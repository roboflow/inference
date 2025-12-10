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
The block wraps 
[Stability AI outpainting API](https://platform.stability.ai/docs/api-reference#tag/Edit/paths/~1v2beta~1stable-image~1edit~1outpaint/post) and 
let users use object detection results to change the content of images in a creative way.

The block sends crop of the image to the API together with directions where to outpaint.
As a result, the API returns the image with outpainted regions.
At least one of `left`, `right`, `up`, `down` must be provided, otherwise original image is returned.
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
            "access_third_party": True,
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-palette",
                "blockPriority": 14,
            },
        }
    )
    type: Literal["roboflow_core/stability_ai_outpainting@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="The image to outpaint.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    creativity: Union[
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
        float,
    ] = Field(
        default=0.5,
        description="Creativity parameter for outpainting. Higher values result in more creative outpainting.",
        examples=[0.5],
    )
    left: Optional[
        Union[
            Selector(kind=[INTEGER_KIND]),
            int,
        ]
    ] = Field(
        default=None,
        description="Number of pixels to outpaint on the left side of the image. Max value is 2000.",
        examples=[200],
    )
    right: Optional[
        Union[
            Selector(kind=[INTEGER_KIND]),
            int,
        ]
    ] = Field(
        default=None,
        description="Number of pixels to outpaint on the right side of the image. Max value is 2000.",
        examples=[200],
    )
    up: Optional[
        Union[
            Selector(kind=[INTEGER_KIND]),
            int,
        ]
    ] = Field(
        default=None,
        description="Number of pixels to outpaint on the top side of the image. Max value is 2000.",
        examples=[200],
    )
    down: Optional[
        Union[
            Selector(kind=[INTEGER_KIND]),
            int,
        ]
    ] = Field(
        default=None,
        description="Number of pixels to outpaint on the bottom side of the image. Max value is 2000.",
        examples=[200],
    )
    prompt: Optional[
        Union[
            Selector(kind=[STRING_KIND]),
            str,
        ]
    ] = Field(
        default=None,
        description="Optional prompt to apply when outpainting the image (what you wish to see)."
        " If not provided, the image will be outpainted without any prompt.",
        examples=["my prompt", "$inputs.prompt"],
        json_schema_extra={
            "multiline": True,
        },
    )
    preset: Optional[StabilityAIPresets] = Field(
        default=None,
        description="Optional preset to apply when outpainting the image (what you wish to see)."
        " If not provided, the image will be outpainted without any preset."
        f" Avaliable presets: {', '.join(m.value for m in StabilityAIPresets)}",
        examples=[StabilityAIPresets.THREE_D_MODEL],
    )
    seed: Optional[
        Union[
            Selector(kind=[INTEGER_KIND]),
            int,
        ]
    ] = Field(
        default=None,
        description="A specific value that is used to guide the 'randomness' of the generation."
        " If not provided, a random seed is used."
        " Must be a number between 0 and 4294967294",
        examples=[200],
    )
    api_key: Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str] = Field(
        description="Your Stability AI API key.",
        examples=["xxx-xxx", "$inputs.stability_ai_api_key"],
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
