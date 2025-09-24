from typing import List, Optional

from pydantic import BaseModel, Field

from inference.core.entities.responses.inference import FaceDetectionPrediction


class GazeDetectionPrediction(BaseModel):
    """Gaze Detection prediction.

    Attributes:
        face (inference.core.entities.responses.inference.FaceDetectionPrediction): The face prediction.
        yaw (float): Yaw (radian) of the detected face.
        pitch (float): Pitch (radian) of the detected face.
    """

    face: FaceDetectionPrediction

    yaw: float = Field(description="Yaw (radian) of the detected face")
    pitch: float = Field(description="Pitch (radian) of the detected face")


class GazeDetectionInferenceResponse(BaseModel):
    """Response for gaze detection inference.

    Attributes:
        predictions (List[inference.core.entities.responses.gaze.GazeDetectionPrediction]): List of gaze detection predictions.
        time (float): The processing time (second).
    """

    predictions: List[GazeDetectionPrediction]

    time: float = Field(description="The processing time (second)")
    time_face_det: Optional[float] = Field(
        None, description="The face detection time (second)"
    )
    time_gaze_det: Optional[float] = Field(
        None, description="The gaze detection time (second)"
    )
