"""
Credits to: https://github.com/Fafruch for origin idea
"""

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
    STRING_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
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

SHORT_DESCRIPTION = "Uses segmentation masks to inpaint objects into image"

API_HOST = "https://api.stability.ai"
ENDPOINT = "/v2beta/stable-image/edit/inpaint"


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
            },
        }
    )
    type: Literal["roboflow_core/stability_ai_inpainting@v1"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="The image which was the base to generate VLM prediction",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    segmentation_mask: StepOutputSelector(
        kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]
    ) = Field(
        name="Segmentation Mask",
        description="Segmentation masks",
        examples=["$steps.model.predictions"],
    )
    prompt: Union[
        WorkflowParameterSelector(kind=[STRING_KIND]),
        StepOutputSelector(kind=[STRING_KIND]),
        str,
    ] = Field(
        description="Prompt to inpainting model (what you wish to see)",
        examples=["my prompt", "$inputs.prompt"],
    )
    negative_prompt: Optional[
        Union[
            WorkflowParameterSelector(kind=[STRING_KIND]),
            StepOutputSelector(kind=[STRING_KIND]),
            str,
        ]
    ] = Field(
        default=None,
        description="Negative prompt to inpainting model (what you do not wish to see)",
        examples=["my prompt", "$inputs.prompt"],
    )
    api_key: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
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
        return ">=1.0.0,<2.0.0"


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
    ) -> BlockResult:
        black_image = np.zeros_like(image.numpy_image)
        mask_annotator = sv.MaskAnnotator(color=Color.WHITE, opacity=1.0)
        mask = mask_annotator.annotate(black_image, segmentation_mask)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        encoded_image = numpy_array_to_jpeg_bytes(image=image.numpy_image)
        encoded_mask = numpy_array_to_jpeg_bytes(image=mask)
        request_data = {
            "prompt": prompt,
            "output_format": "jpeg",
        }
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
            "image": WorkflowImageData(
                parent_metadata=image.parent_metadata,
                workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
                numpy_image=result_image,
            )
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
