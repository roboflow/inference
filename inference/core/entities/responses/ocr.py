from typing import List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

class Object(TypedDict, total=True):
    bounding_box: List[int]
    confidence: float
    string: str

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
    # fields are bounding_box:List[int], confidence:float, string:str
    objects: Optional[List[Object]] = Field(description="List of objects detected by OCR", default=None),
    time: float = Field(
        description="The time in seconds it took to produce the inference including preprocessing."
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )
