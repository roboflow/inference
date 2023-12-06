from typing import Dict, List, Optional, Union

from pydantic import Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import COG_VERSION_ID


class CogVLMInferenceRequest(BaseRequest):
    """Request for CogVLM inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        cog_version_id (Optional[str]): The version ID of CLIP to be used for this request.
    """

    cogvlm_version_id: Optional[str] = Field(
        default=COG_VERSION_ID,
        example="cogvlm-chat-hf",
        description="The version ID of CogVLM to be used for this request. See the huggingface model repo at THUDM.",
    )
    model_id: Optional[str] = Field()
    image: Optional[InferenceRequestImage] = Field(
        description="Image for CogVLM to look at. Specify what you want it to do with this image. Don't provide an image to just use CogVLM as an LLM."
    )
    prompt: str = Field(
        description="Text to be passed to CogVLM. Use to prompt it to describe an image or provide only text to chat with the model.",
        example="Describe this image.",
    )

    @validator("model_id", always=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("cogvlm_version_id") is None:
            return None
        return f"cogvlm/{values['cogvlm_version_id']}"
