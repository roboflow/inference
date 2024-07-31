from typing import Dict, List, Optional, Tuple, Union

from pydantic import Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)


class Florence2InferenceRequest(BaseRequest):
    """Request for Florence2 inference.

    """
    florence2_version_id: Optional[str] = Field(
        default="1",
        examples=["cogvlm-chat-hf"],
        description="The version ID of CogVLM to be used for this request. See the huggingface model repo at THUDM.",
    )
    image: InferenceRequestImage = Field(
        description="Image for Florence2 to look at. Use prompt to specify what you want it to do with the image."
    )
    prompt: str = Field(
        description="Text to be passed to Florence2. Use to prompt it to describe an image or provide only text to chat with the model.",
        examples=["Describe this image."],
    )
