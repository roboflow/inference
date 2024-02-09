from typing import List, Optional, Union

from pydantic import Field, validator

from inference.core.entities.common import ApiKey
from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import GAZE_VERSION_ID


class GazeDetectionInferenceRequest(BaseRequest):
    """Request for gaze detection inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        gaze_version_id (Optional[str]): The version ID of Gaze to be used for this request.
        do_run_face_detection (Optional[bool]): If true, face detection will be applied; if false, face detection will be ignored and the whole input image will be used for gaze detection.
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) for inference.
    """

    gaze_version_id: Optional[str] = Field(
        default=GAZE_VERSION_ID,
        examples=["l2cs"],
        description="The version ID of Gaze to be used for this request. Must be one of l2cs.",
    )

    do_run_face_detection: Optional[bool] = Field(
        default=True,
        examples=[False],
        description="If true, face detection will be applied; if false, face detection will be ignored and the whole input image will be used for gaze detection",
    )

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    model_id: Optional[str] = Field(None)

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("model_id", always=True, allow_reuse=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("gaze_version_id") is None:
            return None
        return f"gaze/{values['gaze_version_id']}"
