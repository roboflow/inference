from typing import Dict, List, Optional, Tuple, Union

from pydantic import Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import PALIGEMMA_VERSION_ID


class PaliGemmaInferenceRequest(BaseRequest):
    """Request for PaliGemma inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        paligemma_version_id (Optional[str]): The version ID of PaliGemma to be used for this request.
    """

    paligemma_version_id: Optional[str] = Field(
        default=PALIGEMMA_VERSION_ID,
        examples=["paligemma-3b-mix-224"],
        description="The version ID of PaliGemma to be used for this request. See the huggingface model repo at https://huggingface.co/google/paligemma-3b-pt-224/blob/main/README.md.",
    )
    model_id: Optional[str] = Field(None)
    image: InferenceRequestImage = Field(
        description="Image for PaliGemma to look at. Use prompt to specify what you want it to do with the image."
    )
    prompt: str = Field(
        description="Text to be passed to PaliGemma. Use to prompt it to describe an image or provide only text to chat with the model.",
        examples=["Describe this image."],
    )

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("model_id", always=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("paligemma_version_id") is None:
            return None
        return f"paligemma/{values['paligemma_version_id']}"
