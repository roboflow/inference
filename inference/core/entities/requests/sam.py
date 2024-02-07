from typing import Any, List, Optional, Union

from pydantic import Field, root_validator, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import SAM_VERSION_ID


class SamInferenceRequest(BaseRequest):
    """SAM inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        sam_version_id (Optional[str]): The version ID of SAM to be used for this request.
    """

    sam_version_id: Optional[str] = Field(
        default=SAM_VERSION_ID,
        examples=["vit_h"],
        description="The version ID of SAM to be used for this request. Must be one of vit_h, vit_l, or vit_b.",
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
        return f"sam/{values['sam_version_id']}"


class SamEmbeddingRequest(SamInferenceRequest):
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
    format: Optional[str] = Field(
        default="json",
        examples=["json"],
        description="The format of the response. Must be one of json or binary. If binary, embedding is returned as a binary numpy array.",
    )


class SamSegmentationRequest(SamInferenceRequest):
    """SAM segmentation request.

    Attributes:
        embeddings (Optional[Union[List[List[List[List[float]]]], Any]]): The embeddings to be decoded.
        embeddings_format (Optional[str]): The format of the embeddings.
        format (Optional[str]): The format of the response.
        image (Optional[InferenceRequestImage]): The image to be segmented.
        image_id (Optional[str]): The ID of the image to be segmented used to retrieve cached embeddings.
        has_mask_input (Optional[bool]): Whether or not the request includes a mask input.
        mask_input (Optional[Union[List[List[List[float]]], Any]]): The set of output masks.
        mask_input_format (Optional[str]): The format of the mask input.
        orig_im_size (Optional[List[int]]): The original size of the image used to generate the embeddings.
        point_coords (Optional[List[List[float]]]): The coordinates of the interactive points used during decoding.
        point_labels (Optional[List[float]]): The labels of the interactive points used during decoding.
        use_mask_input_cache (Optional[bool]): Whether or not to use the mask input cache.
    """

    embeddings: Optional[Union[List[List[List[List[float]]]], Any]] = Field(
        None,
        examples=["[[[[0.1, 0.2, 0.3, ...] ...] ...]]"],
        description="The embeddings to be decoded. The dimensions of the embeddings are 1 x 256 x 64 x 64. If embeddings is not provided, image must be provided.",
    )
    embeddings_format: Optional[str] = Field(
        default="json",
        examples=["json"],
        description="The format of the embeddings. Must be one of json or binary. If binary, embeddings are expected to be a binary numpy array.",
    )
    format: Optional[str] = Field(
        default="json",
        examples=["json"],
        description="The format of the response. Must be one of json or binary. If binary, masks are returned as binary numpy arrays. If json, masks are converted to polygons, then returned as json.",
    )
    image: Optional[InferenceRequestImage] = Field(
        default=None,
        description="The image to be segmented. Only required if embeddings are not provided.",
    )
    image_id: Optional[str] = Field(
        default=None,
        examples=["image_id"],
        description="The ID of the image to be segmented used to retrieve cached embeddings. If an embedding is cached, it will be used instead of generating a new embedding. If no embedding is cached, a new embedding will be generated and cached.",
    )
    has_mask_input: Optional[bool] = Field(
        default=False,
        examples=[True],
        description="Whether or not the request includes a mask input. If true, the mask input must be provided.",
    )
    mask_input: Optional[Union[List[List[List[float]]], Any]] = Field(
        default=None,
        description="The set of output masks. If request format is json, masks is a list of polygons, where each polygon is a list of points, where each point is a tuple containing the x,y pixel coordinates of the point. If request format is binary, masks is a list of binary numpy arrays. The dimensions of each mask are 256 x 256. This is the same as the output, low resolution mask from the previous inference.",
    )
    mask_input_format: Optional[str] = Field(
        default="json",
        examples=["json"],
        description="The format of the mask input. Must be one of json or binary. If binary, mask input is expected to be a binary numpy array.",
    )
    orig_im_size: Optional[List[int]] = Field(
        default=None,
        examples=[[640, 320]],
        description="The original size of the image used to generate the embeddings. This is only required if the image is not provided.",
    )
    point_coords: Optional[List[List[float]]] = Field(
        default=[[0.0, 0.0]],
        examples=[[[10.0, 10.0]]],
        description="The coordinates of the interactive points used during decoding. Each point (x,y pair) corresponds to a label in point_labels.",
    )
    point_labels: Optional[List[float]] = Field(
        default=[-1],
        examples=[[1]],
        description="The labels of the interactive points used during decoding. A 1 represents a positive point (part of the object to be segmented). A -1 represents a negative point (not part of the object to be segmented). Each label corresponds to a point in point_coords.",
    )
    use_mask_input_cache: Optional[bool] = Field(
        default=True,
        examples=[True],
        description="Whether or not to use the mask input cache. If true, the mask input cache will be used if it exists. If false, the mask input cache will not be used.",
    )
