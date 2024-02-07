from typing import Dict, List, Optional, Union

from pydantic import Field

from inference.core.entities.responses.inference import InferenceResponse


class ClipEmbeddingResponse(InferenceResponse):
    """Response for CLIP embedding.

    Attributes:
        embeddings (List[List[float]]): A list of embeddings, each embedding is a list of floats.
        time (float): The time in seconds it took to produce the embeddings including preprocessing.
    """

    embeddings: List[List[float]] = Field(
        examples=["[[0.12, 0.23, 0.34, ..., 0.43]]"],
        description="A list of embeddings, each embedding is a list of floats",
    )
    time: Optional[float] = Field(
        None,
        description="The time in seconds it took to produce the embeddings including preprocessing",
    )


class ClipCompareResponse(InferenceResponse):
    """Response for CLIP comparison.

    Attributes:
        similarity (Union[List[float], Dict[str, float]]): Similarity scores.
        time (float): The time in seconds it took to produce the similarity scores including preprocessing.
    """

    similarity: Union[List[float], Dict[str, float]]
    time: Optional[float] = Field(
        None,
        description="The time in seconds it took to produce the similarity scores including preprocessing",
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )
