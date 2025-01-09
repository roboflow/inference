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
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
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
[Stability AI image generation API](https://platform.stability.ai/docs/api-reference#tag/Generate) and 
let users generate new images from text, or create variations of existing images.
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
        description="The image which was the base to generate VLM prediction",
        examples=["$inputs.image"],
        default=None,
    )
    prompt: Union[
        Selector(kind=[STRING_KIND]),
        Selector(kind=[STRING_KIND]),
        str,
    ] = Field(
        description="Prompt to generate new images from text (what you wish to see)",
        examples=["my prompt", "$inputs.prompt"],
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
        image: WorkflowImageData,
        prompt: str,
        negative_prompt: str,
        model: str,
        api_key: str,
    ) -> BlockResult:
        files_to_send = {"none": ""}
        if image is not None:
            encoded_image = numpy_array_to_jpeg_bytes(image=image.numpy_image)
            files_to_send = {
                "image": encoded_image,
            }
        request_data = {
            "prompt": prompt,
            "output_format": "jpeg",
        }
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
