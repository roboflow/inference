from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class OCRInferenceResponse(BaseModel):
    """
    OCR Inference response.

    Attributes:
        result (str): The combined OCR recognition result.
        strings (Optional[List[str]]): List of strings detected by OCR.
        bounding_boxes (Optional[List[List[int]]]): List of bounding boxes detected by OCR.
        confidences (Optional[List[float]]): List of confidence scores for each OCR detection
        time: The time in seconds it took to produce the inference including preprocessing.
    """

    result: str = Field(description="The combined OCR recognition result.")
    strings: Optional[List[str]] = Field(description="List of strings detected by OCR", default=None)
    bounding_boxes: Optional[List[List[int]]] = Field(description="List of bounding boxes detected by OCR", default=None)
    confidences: Optional[List[float]] = Field(description="List of confidence scores for each OCR detection", default=None)
    time: float = Field(
        description="The time in seconds it took to produce the inference including preprocessing."
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )
