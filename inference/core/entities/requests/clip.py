from typing import Dict, List, Optional, Union

from pydantic import Field, ValidationInfo, field_validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import CLIP_VERSION_ID


class ClipInferenceRequest(BaseRequest):
    """Request for CLIP inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        clip_version_id (Optional[str]): The version ID of CLIP to be used for this request.
    """

    clip_version_id: Optional[str] = Field(
        default=CLIP_VERSION_ID,
        examples=["ViT-B-16"],
        description="The version ID of CLIP to be used for this request. Must be one of RN101, RN50, RN50x16, RN50x4, RN50x64, ViT-B-16, ViT-B-32, ViT-L-14-336px, and ViT-L-14.",
    )
    model_id: Optional[str] = Field(None, validate_default=True)

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value, info: ValidationInfo):
        if value is not None:
            return value

        clip_version_id = info.data.get("clip_version_id")
        if clip_version_id is None:
            return None
        return f"clip/{clip_version_id}"


class ClipImageEmbeddingRequest(ClipInferenceRequest):
    """Request for CLIP image embedding.

    Attributes:
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) to be embedded.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]


class ClipTextEmbeddingRequest(ClipInferenceRequest):
    """Request for CLIP text embedding.

    Attributes:
        text (Union[List[str], str]): A string or list of strings.
    """

    text: Union[List[str], str] = Field(
        examples=["The quick brown fox jumps over the lazy dog"],
        description="A string or list of strings",
    )


class ClipCompareRequest(ClipInferenceRequest):
    """Request for CLIP comparison.

    Attributes:
        subject (Union[InferenceRequestImage, str]): The type of image data provided, one of 'url' or 'base64'.
        subject_type (str): The type of subject, one of 'image' or 'text'.
        prompt (Union[List[InferenceRequestImage], InferenceRequestImage, str, List[str], Dict[str, Union[InferenceRequestImage, str]]]): The prompt for comparison.
        prompt_type (str): The type of prompt, one of 'image' or 'text'.
    """

    subject: Union[InferenceRequestImage, str] = Field(
        examples=["url"],
        description="The type of image data provided, one of 'url' or 'base64'",
    )
    subject_type: str = Field(
        default="image",
        examples=["image"],
        description="The type of subject, one of 'image' or 'text'",
    )
    prompt: Union[
        List[InferenceRequestImage],
        InferenceRequestImage,
        str,
        List[str],
        Dict[str, Union[InferenceRequestImage, str]],
    ]
    prompt_type: str = Field(
        default="text",
        examples=["text"],
        description="The type of prompt, one of 'image' or 'text'",
    )
