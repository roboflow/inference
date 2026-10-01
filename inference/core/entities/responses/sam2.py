from typing import List

from pydantic import Field

from inference.core.entities.responses.inference import InferenceResponse
from inference.core.workflows.core_steps.common.segmentation_entities import (  # noqa: F401
    Sam2SegmentationPrediction,
)


class Sam2EmbeddingResponse(InferenceResponse):
    """SAM embedding response.

    Attributes:
        embeddings (Union[List[List[List[List[float]]]], Any]): The SAM embedding.
        time (float): The time in seconds it took to produce the embeddings including preprocessing.
    """

    image_id: str = Field(description="Image id embeddings are cached to")
    time: float = Field(
        description="The time in seconds it took to produce the embeddings including preprocessing"
    )


class Sam2SegmentationResponse(InferenceResponse):
    predictions: List[Sam2SegmentationPrediction] = Field()
    time: float = Field(
        description="The time in seconds it took to produce the segmentation including preprocessing"
    )
