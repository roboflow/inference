import base64
from doctest import Example
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from inference.core.env import CLIP_VERSION_ID, GAZE_VERSION_ID, SAM_VERSION_ID
from inference.core.managers.entities import ModelDescription

# This file defines the pydantic data models used internally throughout the inference server API. Descriptions are included in many cases for the purpose of auto generated docs.

ModelID = Field(example="raccoon-detector-1", description="A unique model identifier")

ModelType = Field(
    default=None,
    example="object-detection",
    description="The type of the model, usually referring to what task the model performs",
)

ApiKey = Field(
    default=None,
    description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
)


class AddModelRequest(BaseModel):
    """Request to add a model to the inference server.

    Attributes:
        model_id (str): A unique model identifier.
        model_type (Optional[str]): The type of the model, usually referring to what task the model performs.
        api_key (Optional[str]): Roboflow API Key that will be passed to the model during initialization for artifact retrieval.
    """

    model_id: str = ModelID
    model_type: Optional[str] = ModelType
    api_key: Optional[str] = ApiKey


class ClearModelRequest(BaseModel):
    """Request to clear a model from the inference server.

    Attributes:
        model_id (str): A unique model identifier.
    """

    model_id: str = ModelID


class InferenceRequest(BaseModel):
    """Base request for inference.

    Attributes:
        model_id (str): A unique model identifier.
        model_type (Optional[str]): The type of the model, usually referring to what task the model performs.
        api_key (Optional[str]): Roboflow API Key that will be passed to the model during initialization for artifact retrieval.
    """

    model_id: Optional[str] = ModelID
    model_type: Optional[str] = ModelType
    api_key: Optional[str] = ApiKey


class InferenceRequestImage(BaseModel):
    """Image data for inference request.

    Attributes:
        type (str): The type of image data provided, one of 'url', 'base64', or 'numpy'.
        value (Optional[Any]): Image data corresponding to the image type.
    """

    type: str = Field(
        example="url",
        description="The type of image data provided, one of 'url', 'base64', or 'numpy'",
    )
    value: Optional[Any] = Field(
        example="http://www.example-image-url.com",
        description="Image data corresponding to the image type, if type = 'url' then value is a string containing the url of an image, else if type = 'base64' then value is a string containing base64 encoded image data, else if type = 'numpy' then value is binary numpy data serialized using pickle.dumps(); array should 3 dimensions, channels last, with values in the range [0,255].",
    )


class ClipInferenceRequest(BaseModel):
    """Request for CLIP inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        clip_version_id (Optional[str]): The version ID of CLIP to be used for this request.
    """

    api_key: Optional[str] = ApiKey
    clip_version_id: Optional[str] = Field(
        default=CLIP_VERSION_ID,
        example="ViT-B-16",
        description="The version ID of CLIP to be used for this request. Must be one of RN101, RN50, RN50x16, RN50x4, RN50x64, ViT-B-16, ViT-B-32, ViT-L-14-336px, and ViT-L-14.",
    )


class ClipImageEmbeddingRequest(ClipInferenceRequest):
    """Request for CLIP image embedding.

    Attributes:
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) to be embedded.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]


class ClipTextEmbeddingRequest(ClipInferenceRequest):
    """Request for CLIP text embedding.

    Attributes:
        text (Union[List[str], str]): A string or list of strings.
    """

    text: Union[List[str], str] = Field(
        example="The quick brown fox jumps over the lazy dog",
        description="A string or list of strings",
    )


class ClipCompareRequest(ClipInferenceRequest):
    """Request for CLIP comparison.

    Attributes:
        subject (Union[InferenceRequestImage, str]): The type of image data provided, one of 'url' or 'base64'.
        subject_type (str): The type of subject, one of 'image' or 'text'.
        prompt (Union[List[InferenceRequestImage], InferenceRequestImage, str, List[str], Dict[str, Union[InferenceRequestImage, str]]]): The prompt for comparison.
        prompt_type (str): The type of prompt, one of 'image' or 'text'.
    """

    subject: Union[InferenceRequestImage, str] = Field(
        example="url",
        description="The type of image data provided, one of 'url' or 'base64'",
    )
    subject_type: str = Field(
        default="image",
        example="image",
        description="The type of subject, one of 'image' or 'text'",
    )
    prompt: Union[
        List[InferenceRequestImage],
        InferenceRequestImage,
        str,
        List[str],
        Dict[str, Union[InferenceRequestImage, str]],
    ]
    prompt_type: str = Field(
        default="text",
        example="text",
        description="The type of prompt, one of 'image' or 'text'",
    )


class CVInferenceRequest(InferenceRequest):
    """Computer Vision inference request.

    Attributes:
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) for inference.
        disable_preproc_auto_orient (Optional[bool]): If true, the auto orient preprocessing step is disabled for this call. Default is False.
        disable_preproc_contrast (Optional[bool]): If true, the auto contrast preprocessing step is disabled for this call. Default is False.
        disable_preproc_grayscale (Optional[bool]): If true, the grayscale preprocessing step is disabled for this call. Default is False.
        disable_preproc_static_crop (Optional[bool]): If true, the static crop preprocessing step is disabled for this call. Default is False.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    disable_preproc_auto_orient: Optional[bool] = Field(
        default=False,
        description="If true, the auto orient preprocessing step is disabled for this call.",
    )
    disable_preproc_contrast: Optional[bool] = Field(
        default=False,
        description="If true, the auto contrast preprocessing step is disabled for this call.",
    )
    disable_preproc_grayscale: Optional[bool] = Field(
        default=False,
        description="If true, the grayscale preprocessing step is disabled for this call.",
    )
    disable_preproc_static_crop: Optional[bool] = Field(
        default=False,
        description="If true, the static crop preprocessing step is disabled for this call.",
    )


class ObjectDetectionInferenceRequest(CVInferenceRequest):
    """Object Detection inference request.

    Attributes:
        class_agnostic_nms (Optional[bool]): If true, NMS is applied to all detections at once, if false, NMS is applied per class.
        class_filter (Optional[List[str]]): If provided, only predictions for the listed classes will be returned.
        confidence (Optional[float]): The confidence threshold used to filter out predictions.
        fix_batch_size (Optional[bool]): If true, the batch size will be fixed to the maximum batch size configured for this server.
        iou_threshold (Optional[float]): The IoU threshold that must be met for a box pair to be considered duplicate during NMS.
        max_detections (Optional[int]): The maximum number of detections that will be returned.
        max_candidates (Optional[int]): The maximum number of candidate detections passed to NMS.
        visualization_labels (Optional[bool]): If true, labels will be rendered on prediction visualizations.
        visualization_stroke_width (Optional[int]): The stroke width used when visualizing predictions.
        visualize_predictions (Optional[bool]): If true, the predictions will be drawn on the original image and returned as a base64 string.
    """

    class_agnostic_nms: Optional[bool] = Field(
        default=False,
        example=False,
        description="If true, NMS is applied to all detections at once, if false, NMS is applied per class",
    )
    class_filter: Optional[List[str]] = Field(
        default=None,
        example=["class-1", "class-2", "class-n"],
        description="If provided, only predictions for the listed classes will be returned",
    )
    confidence: Optional[float] = Field(
        default=0.0,
        example=0.5,
        description="The confidence threshold used to filter out predictions",
    )
    fix_batch_size: Optional[bool] = Field(
        default=False,
        example=False,
        description="If true, the batch size will be fixed to the maximum batch size configured for this server",
    )
    iou_threshold: Optional[float] = Field(
        default=1.0,
        example=0.5,
        description="The IoU threhsold that must be met for a box pair to be considered duplicate during NMS",
    )
    max_detections: Optional[int] = Field(
        default=300,
        example=300,
        description="The maximum number of detections that will be returned",
    )
    max_candidates: Optional[int] = Field(
        default=3000,
        description="The maximum number of candidate detections passed to NMS",
    )
    visualization_labels: Optional[bool] = Field(
        default=False,
        example=False,
        description="If true, labels will be rendered on prediction visualizations",
    )
    visualization_stroke_width: Optional[int] = Field(
        default=1,
        example=1,
        description="The stroke width used when visualizing predictions",
    )
    visualize_predictions: Optional[bool] = Field(
        default=False,
        example=False,
        description="If true, the predictions will be drawn on the original image and returned as a base64 string",
    )


class InstanceSegmentationInferenceRequest(ObjectDetectionInferenceRequest):
    """Instance Segmentation inference request.

    Attributes:
        mask_decode_mode (Optional[str]): The mode used to decode instance segmentation masks, one of 'accurate', 'fast', 'tradeoff'.
        tradeoff_factor (Optional[float]): The amount to tradeoff between 0='fast' and 1='accurate'.
    """

    mask_decode_mode: Optional[str] = Field(
        default="accurate",
        example="accurate",
        description="The mode used to decode instance segmentation masks, one of 'accurate', 'fast', 'tradeoff'",
    )
    tradeoff_factor: Optional[float] = Field(
        default=0.0,
        example=0.5,
        description="The amount to tradeoff between 0='fast' and 1='accurate'",
    )


class ClassificationInferenceRequest(CVInferenceRequest):
    """Classification inference request.

    Attributes:
        confidence (Optional[float]): The confidence threshold used to filter out predictions.
        visualization_stroke_width (Optional[int]): The stroke width used when visualizing predictions.
        visualize_predictions (Optional[bool]): If true, the predictions will be drawn on the original image and returned as a base64 string.
    """

    confidence: Optional[float] = Field(
        default=0.0,
        example=0.5,
        description="The confidence threshold used to filter out predictions",
    )
    visualization_stroke_width: Optional[int] = Field(
        default=1,
        example=1,
        description="The stroke width used when visualizing predictions",
    )
    visualize_predictions: Optional[bool] = Field(
        default=False,
        example=False,
        description="If true, the predictions will be drawn on the original image and returned as a base64 string",
    )


class ServerVersionInfo(BaseModel):
    """Server version information.

    Attributes:
        name (str): Server name.
        version (str): Server version.
        uuid (str): Server UUID.
    """

    name: str = Field(example="Roboflow Inference Server")
    version: str = Field(example="0.0.1")
    uuid: str = Field(example="9c18c6f4-2266-41fb-8a0f-c12ae28f6fbe")


class ModelDescriptionEntity(BaseModel):
    model_id: str = Field(
        description="Identifier of the model", example="some-project/3"
    )
    task_type: str = Field(
        description="Type of the task that the model performs", example="classification"
    )
    batch_size: Optional[Union[int, str]] = Field(
        description="Batch size accepted by the model (if registered).",
    )
    input_height: Optional[int] = Field(
        description="Image input height accepted by the model (if registered).",
    )
    input_width: Optional[int] = Field(
        description="Image input width accepted by the model (if registered).",
    )

    @classmethod
    def from_model_description(
        cls, model_description: ModelDescription
    ) -> "ModelDescriptionEntity":
        return cls(
            model_id=model_description.model_id,
            task_type=model_description.task_type,
            batch_size=model_description.batch_size,
            input_height=model_description.input_height,
            input_width=model_description.input_width,
        )


class ModelsDescriptions(BaseModel):
    models: List[ModelDescriptionEntity] = Field(
        description="List of models that are loaded by model manager.",
    )

    @classmethod
    def from_models_descriptions(
        cls, models_descriptions: List[ModelDescription]
    ) -> "ModelsDescriptions":
        return cls(
            models=[
                ModelDescriptionEntity.from_model_description(
                    model_description=model_description
                )
                for model_description in models_descriptions
            ]
        )


class ObjectDetectionPrediction(BaseModel):
    """Object Detection prediction.

    Attributes:
        x (float): The center x-axis pixel coordinate of the prediction.
        y (float): The center y-axis pixel coordinate of the prediction.
        width (float): The width of the prediction bounding box in number of pixels.
        height (float): The height of the prediction bounding box in number of pixels.
        confidence (float): The detection confidence as a fraction between 0 and 1.
        class_name (str): The predicted class label.
        class_confidence (Union[float, None]): The class label confidence as a fraction between 0 and 1.
        class_id (int): The class id of the prediction
    """

    x: float = Field(description="The center x-axis pixel coordinate of the prediction")
    y: float = Field(description="The center y-axis pixel coordinate of the prediction")
    width: float = Field(
        description="The width of the prediction bounding box in number of pixels"
    )
    height: float = Field(
        description="The height of the prediction bounding box in number of pixels"
    )
    confidence: float = Field(
        description="The detection confidence as a fraction between 0 and 1"
    )
    class_name: str = Field(alias="class", description="The predicted class label")

    class_confidence: Union[float, None] = Field(
        description="The class label confidence as a fraction between 0 and 1"
    )
    class_id: int = Field(description="The class id of the prediction")
    tracker_id: Optional[int] = Field(
        description="The tracker id of the prediction if tracking is enabled",
        default=None,
    )


class Point(BaseModel):
    """Point coordinates.

    Attributes:
        x (float): The x-axis pixel coordinate of the point.
        y (float): The y-axis pixel coordinate of the point.
    """

    x: float = Field(description="The x-axis pixel coordinate of the point")
    y: float = Field(description="The y-axis pixel coordinate of the point")


class Point3D(Point):
    """3D Point coordinates.

    Attributes:
        z (float): The z-axis pixel coordinate of the point.
    """

    z: float = Field(description="The z-axis pixel coordinate of the point")


class InstanceSegmentationPrediction(BaseModel):
    """Instance Segmentation prediction.

    Attributes:
        x (float): The center x-axis pixel coordinate of the prediction.
        y (float): The center y-axis pixel coordinate of the prediction.
        width (float): The width of the prediction bounding box in number of pixels.
        height (float): The height of the prediction bounding box in number of pixels.
        confidence (float): The detection confidence as a fraction between 0 and 1.
        class_name (str): The predicted class label.
        class_confidence (Union[float, None]): The class label confidence as a fraction between 0 and 1.
        points (List[Point]): The list of points that make up the instance polygon.
        class_id: int = Field(description="The class id of the prediction")
    """

    x: float = Field(description="The center x-axis pixel coordinate of the prediction")
    y: float = Field(description="The center y-axis pixel coordinate of the prediction")
    width: float = Field(
        description="The width of the prediction bounding box in number of pixels"
    )
    height: float = Field(
        description="The height of the prediction bounding box in number of pixels"
    )
    confidence: float = Field(
        description="The detection confidence as a fraction between 0 and 1"
    )
    class_name: str = Field(alias="class", description="The predicted class label")

    class_confidence: Union[float, None] = Field(
        description="The class label confidence as a fraction between 0 and 1"
    )
    points: List[Point] = Field(
        description="The list of points that make up the instance polygon"
    )
    class_id: int = Field(description="The class id of the prediction")


class ClassificationPrediction(BaseModel):
    """Classification prediction.

    Attributes:
        class_name (str): The predicted class label.
        class_id (int): Numeric ID associated with the class label.
        confidence (float): The class label confidence as a fraction between 0 and 1.
    """

    class_name: str = Field(alias="class", description="The predicted class label")
    class_id: int = Field(description="Numeric ID associated with the class label")
    confidence: float = Field(
        description="The class label confidence as a fraction between 0 and 1"
    )


class MultiLabelClassificationPrediction(BaseModel):
    """Multi-label Classification prediction.

    Attributes:
        confidence (float): The class label confidence as a fraction between 0 and 1.
    """

    confidence: float = Field(
        description="The class label confidence as a fraction between 0 and 1"
    )


class InferenceResponseImage(BaseModel):
    """Inference response image information.

    Attributes:
        width (int): The original width of the image used in inference.
        height (int): The original height of the image used in inference.
    """

    width: int = Field(description="The original width of the image used in inference")
    height: int = Field(
        description="The original height of the image used in inference"
    )


class InferenceResponse(BaseModel):
    """Base inference response.

    Attributes:
        frame_id (Optional[int]): The frame id of the image used in inference if the input was a video.
        time (Optional[float]): The time in seconds it took to produce the predictions including image preprocessing.
    """

    frame_id: Optional[int] = Field(
        default=None,
        description="The frame id of the image used in inference if the input was a video",
    )
    time: Optional[float] = Field(
        default=None,
        description="The time in seconds it took to produce the predictions including image preprocessing",
    )


class ClipEmbeddingResponse(InferenceResponse):
    """Response for CLIP embedding.

    Attributes:
        embeddings (List[List[float]]): A list of embeddings, each embedding is a list of floats.
        time (float): The time in seconds it took to produce the embeddings including preprocessing.
    """

    embeddings: List[List[float]] = Field(
        example="[[0.12, 0.23, 0.34, ..., 0.43]]",
        description="A list of embeddings, each embedding is a list of floats",
    )
    time: Optional[float] = Field(
        description="The time in seconds it took to produce the embeddings including preprocessing"
    )


class ClipCompareResponse(InferenceResponse):
    """Response for CLIP comparison.

    Attributes:
        similarity (Union[List[float], Dict[str, float]]): Similarity scores.
        time (float): The time in seconds it took to produce the similarity scores including preprocessing.
    """

    similarity: Union[List[float], Dict[str, float]]
    time: Optional[float] = Field(
        description="The time in seconds it took to produce the similarity scores including preprocessing"
    )


class CvInferenceResponse(InferenceResponse):
    """Computer Vision inference response.

    Attributes:
        image (Union[List[InferenceResponseImage], InferenceResponseImage]): Image(s) used in inference.
    """

    image: Union[List[InferenceResponseImage], InferenceResponseImage]


class WithVisualizationResponse(BaseModel):
    """Response with visualization.

    Attributes:
        visualization (Optional[Any]): Base64 encoded string containing prediction visualization image data.
    """

    visualization: Optional[Any] = Field(
        default=None,
        description="Base64 encoded string containing prediction visualization image data",
    )

    # We need to tell pydantic how to encode bytes as base64 strings. We don't encode the visualization field directly because we want to return it directly for the legacy infer route with content type image/jpeg.
    class Config:
        json_encoders = {bytes: lambda v: base64.b64encode(v).decode("utf-8")}


class ObjectDetectionInferenceResponse(CvInferenceResponse, WithVisualizationResponse):
    """Object Detection inference response.

    Attributes:
        predictions (List[ObjectDetectionPrediction]): List of object detection predictions.
    """

    predictions: List[ObjectDetectionPrediction]


class InstanceSegmentationInferenceResponse(
    CvInferenceResponse, WithVisualizationResponse
):
    """Instance Segmentation inference response.

    Attributes:
        predictions (List[InstanceSegmentationPrediction]): List of instance segmentation predictions.
    """

    predictions: List[InstanceSegmentationPrediction]


class ClassificationInferenceResponse(CvInferenceResponse, WithVisualizationResponse):
    """Classification inference response.

    Attributes:
        predictions (List[ClassificationPrediction]): List of classification predictions.
        top (str): The top predicted class label.
        confidence (float): The confidence of the top predicted class label.
    """

    predictions: List[ClassificationPrediction]
    top: str = Field(description="The top predicted class label")
    confidence: float = Field(
        description="The confidence of the top predicted class label"
    )


class MultiLabelClassificationInferenceResponse(
    CvInferenceResponse, WithVisualizationResponse
):
    """Multi-label Classification inference response.

    Attributes:
        predictions (Dict[str, MultiLabelClassificationPrediction]): Dictionary of multi-label classification predictions.
        predicted_classes (List[str]): The list of predicted classes.
    """

    predictions: Dict[str, MultiLabelClassificationPrediction]
    predicted_classes: List[str] = Field(description="The list of predicted classes")


class SamInferenceRequest(BaseModel):
    """SAM inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        sam_version_id (Optional[str]): The version ID of SAM to be used for this request.
    """

    api_key: Optional[str] = ApiKey
    sam_version_id: Optional[str] = Field(
        default=SAM_VERSION_ID,
        example="vit_h",
        description="The version ID of SAM to be used for this request. Must be one of vit_h, vit_l, or vit_b.",
    )


class SamEmbeddingRequest(SamInferenceRequest):
    """SAM embedding request.

    Attributes:
        image (Optional[InferenceRequestImage]): The image to be embedded.
        image_id (Optional[str]): The ID of the image to be embedded used to cache the embedding.
        format (Optional[str]): The format of the response. Must be one of json or binary.
    """

    image: Optional[InferenceRequestImage] = Field(
        default=None,
        description="The image to be embedded",
    )
    image_id: Optional[str] = Field(
        default=None,
        example="image_id",
        description="The ID of the image to be embedded used to cache the embedding.",
    )
    format: Optional[str] = Field(
        default="json",
        example="json",
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
        example="[[[[0.1, 0.2, 0.3, ...] ...] ...]]",
        description="The embeddings to be decoded. The dimensions of the embeddings are 1 x 256 x 64 x 64. If embeddings is not provided, image must be provided.",
    )
    embeddings_format: Optional[str] = Field(
        default="json",
        example="json",
        description="The format of the embeddings. Must be one of json or binary. If binary, embeddings are expected to be a binary numpy array.",
    )
    format: Optional[str] = Field(
        default="json",
        example="json",
        description="The format of the response. Must be one of json or binary. If binary, masks are returned as binary numpy arrays. If json, masks are converted to polygons, then returned as json.",
    )
    image: Optional[InferenceRequestImage] = Field(
        default=None,
        description="The image to be segmented. Only required if embeddings are not provided.",
    )
    image_id: Optional[str] = Field(
        default=None,
        example="image_id",
        description="The ID of the image to be segmented used to retrieve cached embeddings. If an embedding is cached, it will be used instead of generating a new embedding. If no embedding is cached, a new embedding will be generated and cached.",
    )
    has_mask_input: Optional[bool] = Field(
        default=False,
        example=True,
        description="Whether or not the request includes a mask input. If true, the mask input must be provided.",
    )
    mask_input: Optional[Union[List[List[List[float]]], Any]] = Field(
        default=None,
        description="The set of output masks. If request format is json, masks is a list of polygons, where each polygon is a list of points, where each point is a tuple containing the x,y pixel coordinates of the point. If request format is binary, masks is a list of binary numpy arrays. The dimensions of each mask are 256 x 256. This is the same as the output, low resolution mask from the previous inference.",
    )
    mask_input_format: Optional[str] = Field(
        default="json",
        example="json",
        description="The format of the mask input. Must be one of json or binary. If binary, mask input is expected to be a binary numpy array.",
    )
    orig_im_size: Optional[List[int]] = Field(
        default=None,
        example=[640, 320],
        description="The original size of the image used to generate the embeddings. This is only required if the image is not provided.",
    )
    point_coords: Optional[List[List[float]]] = Field(
        default=[[0.0, 0.0]],
        example=[[10.0, 10.0]],
        description="The coordinates of the interactive points used during decoding. Each point (x,y pair) corresponds to a label in point_labels.",
    )
    point_labels: Optional[List[float]] = Field(
        default=[-1],
        example=[1],
        description="The labels of the interactive points used during decoding. A 1 represents a positive point (part of the object to be segmented). A -1 represents a negative point (not part of the object to be segmented). Each label corresponds to a point in point_coords.",
    )
    use_mask_input_cache: Optional[bool] = Field(
        default=True,
        example=True,
        description="Whether or not to use the mask input cache. If true, the mask input cache will be used if it exists. If false, the mask input cache will not be used.",
    )


class SamEmbeddingResponse(BaseModel):
    """SAM embedding response.

    Attributes:
        embeddings (Union[List[List[List[List[float]]]], Any]): The SAM embedding.
        time (float): The time in seconds it took to produce the embeddings including preprocessing.
    """

    embeddings: Union[List[List[List[List[float]]]], Any] = Field(
        example="[[[[0.1, 0.2, 0.3, ...] ...] ...]]",
        description="If request format is json, embeddings is a series of nested lists representing the SAM embedding. If request format is binary, embeddings is a binary numpy array. The dimensions of the embedding are 1 x 256 x 64 x 64.",
    )
    time: float = Field(
        description="The time in seconds it took to produce the embeddings including preprocessing"
    )


class SamSegmentationResponse(BaseModel):
    """SAM segmentation response.

    Attributes:
        masks (Union[List[List[List[int]]], Any]): The set of output masks.
        low_res_masks (Union[List[List[List[int]]], Any]): The set of output low-resolution masks.
        time (float): The time in seconds it took to produce the segmentation including preprocessing.
    """

    masks: Union[List[List[List[int]]], Any] = Field(
        description="The set of output masks. If request format is json, masks is a list of polygons, where each polygon is a list of points, where each point is a tuple containing the x,y pixel coordinates of the point. If request format is binary, masks is a list of binary numpy arrays. The dimensions of each mask are the same as the dimensions of the input image.",
    )
    low_res_masks: Union[List[List[List[int]]], Any] = Field(
        description="The set of output masks. If request format is json, masks is a list of polygons, where each polygon is a list of points, where each point is a tuple containing the x,y pixel coordinates of the point. If request format is binary, masks is a list of binary numpy arrays. The dimensions of each mask are 256 x 256",
    )
    time: float = Field(
        description="The time in seconds it took to produce the segmentation including preprocessing"
    )


class FaceDetectionPrediction(ObjectDetectionPrediction):
    """Face Detection prediction.

    Attributes:
        class_name (str): fixed value "face".
        landmarks (Union[List[Point], List[Point3D]]): The detected face landmarks.
    """

    class_id: Optional[int] = Field(
        description="The class id of the prediction", default=0
    )
    class_name: str = Field(
        alias="class", default="face", description="The predicted class label"
    )
    landmarks: Union[List[Point], List[Point3D]]


class GazeDetectionPrediction(BaseModel):
    """Gaze Detection prediction.

    Attributes:
        face (FaceDetectionPrediction): The face prediction.
        yaw (float): Yaw (radian) of the detected face.
        pitch (float): Pitch (radian) of the detected face.
    """

    face: FaceDetectionPrediction

    yaw: float = Field(description="Yaw (radian) of the detected face")
    pitch: float = Field(description="Pitch (radian) of the detected face")


class GazeDetectionInferenceRequest(BaseModel):
    """Request for gaze detection inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        gaze_version_id (Optional[str]): The version ID of Gaze to be used for this request.
        do_run_face_detection (Optional[bool]): If true, face detection will be applied; if false, face detection will be ignored and the whole input image will be used for gaze detection.
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) for inference.
    """

    api_key: Optional[str] = ApiKey
    gaze_version_id: Optional[str] = Field(
        default=GAZE_VERSION_ID,
        example="l2cs",
        description="The version ID of Gaze to be used for this request. Must be one of l2cs.",
    )

    do_run_face_detection: Optional[bool] = Field(
        default=True,
        example=False,
        description="If true, face detection will be applied; if false, face detection will be ignored and the whole input image will be used for gaze detection",
    )

    image: Union[List[InferenceRequestImage], InferenceRequestImage]


class GazeDetectionInferenceResponse(BaseModel):
    """Response for gaze detection inference.

    Attributes:
        predictions (List[GazeDetectionPrediction]): List of gaze detection predictions.
        time (float): The processing time (second).
    """

    predictions: List[GazeDetectionPrediction]

    time: float = Field(description="The processing time (second)")
    time_face_det: Optional[float] = Field(
        description="The face detection time (second)"
    )
    time_gaze_det: Optional[float] = Field(
        description="The gaze detection time (second)"
    )
