from typing import List, Optional, Tuple

from pydantic import Field, ValidationInfo, field_validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import COGVLM_VERSION_ID


class CogVLMInferenceRequest(BaseRequest):
    """Request for CogVLM inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        cog_version_id (Optional[str]): The version ID of CLIP to be used for this request.
    """

    cogvlm_version_id: Optional[str] = Field(
        default=COGVLM_VERSION_ID,
        examples=["cogvlm-chat-hf"],
        description="The version ID of CogVLM to be used for this request. See the huggingface model repo at THUDM.",
    )
    model_id: Optional[str] = Field(None)
    image: InferenceRequestImage = Field(
        description="Image for CogVLM to look at. Use prompt to specify what you want it to do with the image."
    )
    prompt: str = Field(
        description="Text to be passed to CogVLM. Use to prompt it to describe an image or provide only text to chat with the model.",
        examples=["Describe this image."],
    )
    history: Optional[List[Tuple[str, str]]] = Field(
        None,
        description="Optional chat history, formatted as a list of 2-tuples where the first entry is the user prompt"
        " and the second entry is the generated model response",
    )

    @field_validator("model_id", validate_default=True)
    @classmethod
    def validate_model_id(cls, value, info: ValidationInfo):
        if value is not None:
            return value

        cogvlm_version_id = info.data.get("cogvlm_version_id")
        if cogvlm_version_id is None:
            return None
        return f"cogvlm/{cogvlm_version_id}"
