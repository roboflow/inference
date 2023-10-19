from typing import List, Optional, Union

from pydantic import BaseModel, Field

from inference.core.entities.common import ApiKey
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.env import GAZE_VERSION_ID


class GazeDetectionInferenceRequest(BaseModel):
    """Request for gaze detection inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        gaze_version_id (Optional[str]): The version ID of Gaze to be used for this request.
        do_run_face_detection (Optional[bool]): If true, face detection will be applied; if false, face detection will be ignored and the whole input image will be used for gaze detection.
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) for inference.
    """

    api_key: Optional[str] = ApiKey
    gaze_version_id: Optional[str] = Field(
        default=GAZE_VERSION_ID,
        example="l2cs",
        description="The version ID of Gaze to be used for this request. Must be one of l2cs.",
    )

    do_run_face_detection: Optional[bool] = Field(
        default=True,
        example=False,
        description="If true, face detection will be applied; if false, face detection will be ignored and the whole input image will be used for gaze detection",
    )

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
