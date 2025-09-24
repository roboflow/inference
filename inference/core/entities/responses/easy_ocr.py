from typing import List, Optional, Tuple
from supervision import Detections
from pydantic import BaseModel, Field


class EasyOCRInferenceResponse(BaseModel):
    """
    OCR Inference response.

    Attributes:
        result (strings, boxes, confidences): The OCR recognition result.
        time: The time in seconds it took to produce the inference including preprocessing.
    """

    result: List[Tuple[List[List[int]], str, float]] = Field(description="The OCR recognition result.")
    time: float = Field(
        description="The time in seconds it took to produce the inference including preprocessing."
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )