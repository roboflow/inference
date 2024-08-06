from typing import Any, List, Optional, Union

from pydantic import Field, root_validator, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import SAM2_VERSION_ID


class Sam2InferenceRequest(BaseRequest):
    """SAM2 inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        sam2_version_id (Optional[str]): The version ID of SAM2 to be used for this request.
    """

    sam2_version_id: Optional[str] = Field(
        default=SAM2_VERSION_ID,
        examples=["hiera_large"],
        description="The version ID of SAM to be used for this request. Must be one of hiera_tiny, hiera_small, hiera_large, hiera_b_plus",
    )

    model_id: Optional[str] = Field(None)

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("model_id", always=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("sam_version_id") is None:
            return None
        return f"sam2/{values['sam_version_id']}"


class Sam2EmbeddingRequest(Sam2InferenceRequest):
    """SAM embedding request.

    Attributes:
        image (Optional[inference.core.entities.requests.inference.InferenceRequestImage]): The image to be embedded.
        image_id (Optional[str]): The ID of the image to be embedded used to cache the embedding.
        format (Optional[str]): The format of the response. Must be one of json or binary.
    """

    image: Optional[InferenceRequestImage] = Field(
        default=None,
        description="The image to be embedded",
    )
    image_id: Optional[str] = Field(
        default=None,
        examples=["image_id"],
        description="The ID of the image to be embedded used to cache the embedding.",
    )


class Sam2SegmentationRequest(Sam2InferenceRequest):
    """SAM segmentation request.

    Attributes:
        format (Optional[str]): The format of the response.
        image (InferenceRequestImage): The image to be segmented.
        image_id (Optional[str]): The ID of the image to be segmented used to retrieve cached embeddings.
        point_coords (Optional[List[List[float]]]): The coordinates of the interactive points used during decoding.
        point_labels (Optional[List[float]]): The labels of the interactive points used during decoding.
    """

    format: Optional[str] = Field(
        default="json",
        examples=["json"],
        description="The format of the response. Must be one of json or binary. If binary, masks are returned as binary numpy arrays. If json, masks are converted to polygons, then returned as json.",
    )
    image: InferenceRequestImage = Field(
        description="The image to be segmented.",
    )
    image_id: Optional[str] = Field(
        default=None,
        examples=["image_id"],
        description="The ID of the image to be segmented used to retrieve cached embeddings. If an embedding is cached, it will be used instead of generating a new embedding. If no embedding is cached, a new embedding will be generated and cached.",
    )
    point_coords: Optional[List[List[float]]] = Field(
        default=None,
        examples=[[[10.0, 10.0]]],
        description="The coordinates of the interactive points used during decoding. Each point (x,y pair) corresponds to a label in point_labels.",
    )
    point_labels: Optional[List[float]] = Field(
        default=None,
        examples=[[1]],
        description="The labels of the interactive points used during decoding. A 1 represents a positive point (part of the object to be segmented). A 0 represents a negative point (not part of the object to be segmented). Each label corresponds to a point in point_coords.",
    )
