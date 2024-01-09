from typing import Any, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from inference.core.entities.common import ApiKey, ModelID, ModelType


class BaseRequest(BaseModel):
    """Base request for inference.

    Attributes:
        id (str_): A unique request identifier.
        api_key (Optional[str]): Roboflow API Key that will be passed to the model during initialization for artifact retrieval.
        start (Optional[float]): start time of request
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    api_key: Optional[str] = ApiKey
    start: Optional[float] = None


class InferenceRequest(BaseRequest):
    """Base request for inference.

    Attributes:
        model_id (str): A unique model identifier.
        model_type (Optional[str]): The type of the model, usually referring to what task the model performs.
    """

    model_id: Optional[str] = ModelID
    model_type: Optional[str] = ModelType


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
    disable_active_learning: Optional[bool] = Field(
        default=False,
        example=False,
        description="If true, the predictions will be prevented from registration by Active Learning (if the functionality is enabled)",
    )


class KeypointsDetectionInferenceRequest(ObjectDetectionInferenceRequest):
    keypoint_confidence: Optional[float] = Field(
        default=0.0,
        example=0.5,
        description="The confidence threshold used to filter out non visible keypoints",
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
    disable_active_learning: Optional[bool] = Field(
        default=False,
        example=False,
        description="If true, the predictions will be prevented from registration by Active Learning (if the functionality is enabled)",
    )


def request_from_type(model_type, request_dict):
    if model_type == "classification":
        return ClassificationInferenceRequest(**request_dict)
    elif model_type == "instance-segmentation":
        return InstanceSegmentationInferenceRequest(**request_dict)
    elif model_type == "object-detection":
        return ObjectDetectionInferenceRequest(**request_dict)
    else:
        raise ValueError(f"Uknown task type {model_type}")
