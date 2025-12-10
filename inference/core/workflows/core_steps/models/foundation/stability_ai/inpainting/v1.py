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
The block wraps 
[Stability AI inpainting API](https://platform.stability.ai/docs/legacy/grpc-api/features/inpainting#Python) and 
let users use instance segmentation results to change the content of images in a creative way.
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
            "access_third_party": False,
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-palette",
                "blockPriority": 14,
            },
        }
    )
    type: Literal["roboflow_core/stability_ai_inpainting@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="The image to inpaint.",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    segmentation_mask: Selector(kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]) = Field(
        name="Segmentation Mask",
        description="Model predictions from segmentation model.",
        examples=["$steps.model.predictions"],
    )
    prompt: Union[
        Selector(kind=[STRING_KIND]),
        str,
    ] = Field(
        description="Prompt to inpainting model (what you wish to see).",
        examples=["my prompt", "$inputs.prompt"],
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
        description="Negative prompt to inpainting model (what you do not wish to see).",
        examples=["my prompt", "$inputs.prompt"],
    )
    api_key: Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str] = Field(
        description="Your Stability AI API key.",
        examples=["xxx-xxx", "$inputs.stability_ai_api_key"],
        private=True,
    )
    invert_segmentation_mask: Union[
        Selector(kind=[BOOLEAN_KIND]),
        bool,
    ] = Field(
        default=False,
        description="Invert segmentation mask to inpaint background instead of foreground.",
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
