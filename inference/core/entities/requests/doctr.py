from typing import List, Optional, Union

from pydantic import Field, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)


class DoctrOCRInferenceRequest(BaseRequest):
    """
    DocTR inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    doctr_version_id: Optional[str] = "default"
    model_id: Optional[str] = Field(None)
    # flag to generate bounding box data rather than just a string, set to False for backwards compatibility
    generate_bounding_boxes: Optional[bool] = False

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("model_id", always=True, allow_reuse=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("doctr_version_id") is None:
            return None
        return f"doctr/{values['doctr_version_id']}"
