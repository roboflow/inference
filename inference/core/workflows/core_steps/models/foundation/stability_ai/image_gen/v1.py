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
The block wraps [Stability AI image generation API](https://platform.stability.ai/docs/api-reference#tag/Generate) and let users generate new images from text, or create variations of existing images.
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
        description="The image to use as the starting point for the generation.",
        examples=["$inputs.image"],
        default=None,
    )
    strength: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        description="controls how much influence the image parameter has on the generated image. A value of 0 would yield an image that is identical to the input. A value of 1 would be as if you passed in no image at all.",
        default=0.3,
        examples=[0.3, "$inputs.strength"],
    )
    prompt: Union[
        Selector(kind=[STRING_KIND]),
        Selector(kind=[STRING_KIND]),
        str,
    ] = Field(
        description="Prompt to generate new images from text (what you wish to see)",
        examples=["my prompt", "$inputs.prompt"],
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
        description="Negative prompt to image generation model (what you do not wish to see)",
        examples=["my prompt", "$inputs.prompt"],
    )
    model: Optional[
        Union[
            Selector(kind=[STRING_KIND]),
            Selector(kind=[STRING_KIND]),
            str,
        ]
    ] = Field(
        default="core",
        description="choose one of {'core', 'ultra', 'sd3'}. Default 'core' ",
        examples=["my prompt", "$inputs.prompt"],
    )
    api_key: Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str] = Field(
        description="Your Stability AI API key",
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
