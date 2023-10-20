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
