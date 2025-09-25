from typing import List, Optional

from pydantic import BaseModel, Field
import supervision as sv

from inference.core.interfaces.stream.entities import ObjectDetectionPrediction


class OCRInferenceResponse(BaseModel):
    """
    OCR Inference response.

    Attributes:
        result (str): The combined OCR recognition result.
        predictions (List[ObjectDetectionPrediction]): List of objects detected by OCR
        time (float): The time in seconds it took to produce the inference including preprocessing
    """

    result: str = Field(description="The combined OCR recognition result.")
    predictions: List[ObjectDetectionPrediction] = (
        Field(description="List of objects detected by OCR", default=[]),
    )
    time: float = Field(
        description="The time in seconds it took to produce the inference including preprocessing."
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )
