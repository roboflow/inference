from typing import Dict, List, Optional, Union

from pydantic import Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.exceptions import ModelNotRecognisedError
from inference.core.env import (
    PERCEPTION_ENCODER_DISALLOWED_VERSION_IDS,
    PERCEPTION_ENCODER_VERSION_ID,
)


class PerceptionEncoderInferenceRequest(BaseRequest):
    """Request for PERCEPTION_ENCODER inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        clip_version_id (Optional[str]): The version ID of PERCEPTION_ENCODER to be used for this request.
    """

    perception_encoder_version_id: Optional[str] = Field(
        default=PERCEPTION_ENCODER_VERSION_ID,
        examples=["PE-Core-L14-336"],
        description="The version ID of PERCEPTION_ENCODER to be used for this request. Must be one of RN101, RN50, RN50x16, RN50x4, RN50x64, ViT-B-16, ViT-B-32, ViT-L-14-336px, and ViT-L-14.",
    )
    model_id: Optional[str] = Field(None)

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("perception_encoder_version_id")
    def validate_version_id(cls, value):
        if value and value in PERCEPTION_ENCODER_DISALLOWED_VERSION_IDS:
            raise ModelNotRecognisedError(
                f"Perception Encoder model versions {list(PERCEPTION_ENCODER_DISALLOWED_VERSION_IDS)} have been disallowed in this inference configuration. Please try using a model version not in the list of disallowed versions."
            )
        return value

    @validator("model_id", always=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            # Extract version from model_id if provided directly
            if "/" in value:
                version_id = value.split("/")[-1]
                if version_id in PERCEPTION_ENCODER_DISALLOWED_VERSION_IDS:
                    raise ValueError(
                        f"Perception Encoder model versions {list(PERCEPTION_ENCODER_DISALLOWED_VERSION_IDS)} have been disallowed in this inference configuration. Please try using a model version not in the list of disallowed versions."
                    )
            return value
        if values.get("perception_encoder_version_id") is None:
            return None
        return f"perception_encoder/{values['perception_encoder_version_id']}"


class PerceptionEncoderImageEmbeddingRequest(PerceptionEncoderInferenceRequest):
    """Request for PERCEPTION_ENCODER image embedding.

    Attributes:
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) to be embedded.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]


class PerceptionEncoderTextEmbeddingRequest(PerceptionEncoderInferenceRequest):
    """Request for PERCEPTION_ENCODER text embedding.

    Attributes:
        text (Union[List[str], str]): A string or list of strings.
    """

    text: Union[List[str], str] = Field(
        examples=["The quick brown fox jumps over the lazy dog"],
        description="A string or list of strings",
    )


class PerceptionEncoderCompareRequest(PerceptionEncoderInferenceRequest):
    """Request for PERCEPTION_ENCODER comparison.

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
