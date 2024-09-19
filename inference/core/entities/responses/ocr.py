from typing import Optional

from pydantic import BaseModel, Field


class OCRInferenceResponse(BaseModel):
    """
    OCR Inference response.

    Attributes:
        result (str): The OCR recognition result.
        time: The time in seconds it took to produce the inference including preprocessing.
    """

    result: str = Field(description="The OCR recognition result.")
    time: float = Field(
        description="The time in seconds it took to produce the inference including preprocessing."
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )
