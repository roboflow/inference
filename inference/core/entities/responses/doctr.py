from typing import Optional

from pydantic import BaseModel, Field


class DoctrOCRInferenceResponse(BaseModel):
    """
    DocTR Inference response.

    Attributes:
        result (str): The result from OCR.
        time: The time in seconds it took to produce the segmentation including preprocessing.
    """

    result: str = Field(description="The result from OCR.")
    time: float = Field(
        description="The time in seconds it took to produce the segmentation including preprocessing."
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )
