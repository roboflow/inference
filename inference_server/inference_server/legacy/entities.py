import base64
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
    validator,
)
from pydantic.json_schema import SkipJsonSchema

from inference_model_manager.pipelines import (
    DISABLED_STAGE,
    default_stage_tokens,
    pipeline_model_id,
    stage_tokens,
)
from inference_server import configuration
from inference_server.legacy.prompts import Sam2Prompt, Sam2PromptSet, Sam3Prompt

Confidence = Union[float, Literal["best", "default"]]

ModelID = Field(example="raccoon-detector-1", description="A unique model identifier")
ModelType = Field(
    default=None,
    examples=["object-detection"],
    description="The type of the model, usually referring to what task the model performs",
)
ApiKey = Field(
    default=None,
    description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
)


class Point(BaseModel):
    """Point coordinates.

    Attributes:
        x (float): The x-axis pixel coordinate of the point.
        y (float): The y-axis pixel coordinate of the point.
    """

    x: float = Field(description="The x-axis pixel coordinate of the point")
    y: float = Field(description="The y-axis pixel coordinate of the point")


class InstanceSegmentationBasePrediction(BaseModel):
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
    class_id: int = Field(description="The class id of the prediction")
    detection_id: str = Field(
        description="Unique identifier of detection",
        default_factory=lambda: str(uuid4()),
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region",
        default=None,
    )


class InstanceSegmentationPrediction(InstanceSegmentationBasePrediction):
    class_confidence: Union[float, None] = Field(
        None, description="The class label confidence as a fraction between 0 and 1"
    )
    points: List[Point] = Field(
        description="The list of points that make up the instance polygon"
    )
    mask_format: Literal["polygon"] = Field(
        default="polygon",
        description="Type of mask format",
    )


class InstanceSegmentationRLEPrediction(InstanceSegmentationBasePrediction):
    rle: dict = Field(
        description="RLE-encoded mask in COCO format: {'size': [H, W], 'counts': '...'}"
    )
    mask_format: Literal["rle"] = Field(
        default="rle",
        description="Type of mask format",
    )


class Sam2SegmentationPrediction(BaseModel):
    """SAM segmentation prediction.

    Attributes:
        masks (Union[List[List[List[int]]], Dict[str, Any], Any]): Mask data - either polygon coordinates or RLE encoding.
        confidence (float): Masks confidences.
        format (Optional[str]): Format of the mask data: 'polygon' or 'rle'.
    """

    masks: Union[List[List[List[int]]], Dict[str, Any]] = Field(
        description="If polygon format, masks is a list of polygons, where each polygon is a list of points, where each point is a tuple containing the x,y pixel coordinates of the point. If rle format, masks is a dictionary with the keys 'size' and 'counts' containing the size and counts of the RLE encoding."
    )
    confidence: float = Field(description="Masks confidences")
    format: Optional[str] = Field(
        default="polygon", description="Format of the mask data: 'polygon' or 'rle'"
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


class ResolvedModel(BaseModel):
    """Identity of the model and available package details used for inference.

    Attributes:
        model_id (str): ID of the loaded model, using the canonical ID when available.
        model_package_id (Optional[str]): ID of the loaded model package.
        backend (Optional[str]): Backend of the loaded package.
        quantization (Optional[str]): Quantization of the loaded package.
    """

    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(
        description="ID of the loaded model, using the canonical ID when available."
    )
    model_package_id: Optional[str] = Field(
        default=None,
        description="ID of the package that loaded successfully and produced this result, including after a loading fallback.",
    )
    backend: Optional[str] = Field(
        default=None,
        description="Backend of the loaded package, such as onnx, trt, or torch.",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Package quantization, such as fp32 or fp16, or unknown when unavailable. This does not specify the input tensor dtype or the precision of every runtime operation.",
    )


class InferenceResponse(BaseModel):
    """Base inference response.

    Attributes:
        inference_id (Optional[str]): Unique identifier of inference
        frame_id (Optional[int]): The frame id of the image used in inference if the input was a video.
        time (Optional[float]): The time in seconds it took to produce the predictions including image preprocessing.
        resolved_model (Optional[ResolvedModel]): Model identity and available package details for this result.
    """

    model_config = ConfigDict(protected_namespaces=())
    inference_id: Optional[str] = Field(
        description="Unique identifier of inference", default=None
    )
    frame_id: Optional[int] = Field(
        default=None,
        description="The frame id of the image used in inference if the input was a video",
    )
    time: Optional[float] = Field(
        default=None,
        description="The time in seconds it took to produce the predictions including image preprocessing",
    )
    resolved_model: Optional[ResolvedModel] = Field(
        default=None,
        description="Model identity and available package details for this result.",
    )


class CvInferenceResponse(InferenceResponse):
    """Computer Vision inference response.

    Attributes:
        image (Union[List[inference.core.entities.responses.inference.InferenceResponseImage], inference.core.entities.responses.inference.InferenceResponseImage]): Image(s) used in inference.
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

    @field_serializer("visualization", when_used="json")
    def serialize_visualisation(self, visualization: Optional[Any]) -> Optional[str]:
        if visualization is None:
            return None
        return base64.b64encode(visualization).decode("utf-8")


class InstanceSegmentationInferenceResponse(
    CvInferenceResponse, WithVisualizationResponse
):
    """Instance Segmentation inference response.

    Attributes:
        predictions (List[Union[
            inference.core.entities.responses.inference.InstanceSegmentationPrediction,
            inference.core.entities.responses.inference.InstanceSegmentationRLEPrediction
        ]]): List of instance segmentation predictions.
    """

    predictions: List[
        Union[InstanceSegmentationPrediction, InstanceSegmentationRLEPrediction]
    ]


class BaseRequest(BaseModel):
    """Base request for inference.

    Attributes:
        id (str_): A unique request identifier.
        api_key (Optional[str]): Roboflow API Key that will be passed to the model during initialization for artifact retrieval.
        start (Optional[float]): start time of request
        disable_model_monitoring (Optional[bool]): If true, disables model monitoring for this request.
    """

    def __init__(self, **kwargs):
        kwargs["id"] = kwargs.get("id", str(uuid4()))
        super().__init__(**kwargs)

    model_config = ConfigDict(protected_namespaces=())
    id: str
    api_key: Optional[str] = ApiKey
    usage_billable: bool = True
    start: Optional[float] = None
    source: Optional[str] = None
    source_info: Optional[str] = None
    stream_pipeline_context_id: Optional[str] = Field(
        default=None,
        exclude=True,
        repr=False,
        description=(
            "Internal stream-pipeline frame pairing id. Not part of the public API."
        ),
    )
    disable_model_monitoring: Optional[bool] = Field(
        default=False, description="If true, disables model monitoring for this request"
    )


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
        examples=["url"],
        description="The type of image data provided, one of 'url', 'base64', or 'numpy'",
    )
    value: Optional[Any] = Field(
        None,
        examples=["http://www.example-image-url.com"],
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


class DepthEstimationRequest(InferenceRequest):
    """Request for depth estimation.

    Attributes:
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) to be estimated.
        model_id (str): The model ID to use for depth estimation.
        depth_version_id (Optional[str]): The version ID of the depth estimation model.
        depth_map_format (Literal["json", "png16", "png8"]): Serialization format
            for the normalized depth map in the response.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    model_id: Optional[str] = Field(None)
    depth_version_id: Optional[str] = Field(
        default="small",
        examples=["small"],
        description="The version ID of the depth estimation model",
    )
    depth_map_format: Literal["json", "png16", "png8"] = Field(
        default="json",
        description="Serialization format for `normalized_depth` in the response: "
        "`json` (default, wire-compatible with older clients) returns the nested "
        "float list; `png16` returns a base64 16-bit grayscale PNG (quantization "
        "step 1/65535, typically >10x smaller payload - `inference_sdk` decodes "
        "it back to a numpy array when requested via `depth_map_format='png16'`); "
        "`png8` returns a base64 8-bit grayscale PNG (256 depth levels, roughly "
        "another order of magnitude smaller - fine for visualization/thresholding, "
        "lossy for geometric use).",
    )

    @validator("model_id", always=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("depth_version_id") is None:
            return None
        return f"depth-anything-v2/{values['depth_version_id']}"


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
        examples=[False],
        description="If true, NMS is applied to all detections at once, if false, NMS is applied per class",
    )
    class_filter: Optional[List[str]] = Field(
        default=None,
        examples=[["class-1", "class-2", "class-n"]],
        description="If provided, only predictions for the listed classes will be returned",
    )
    confidence: Confidence = Field(
        default=0.4,
        examples=[0.5, "best", "default"],
        description=(
            'Confidence threshold. "best" uses model-eval thresholds, '
            '"default" uses the model built-in, or pass a float.'
        ),
    )
    fix_batch_size: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If true, the batch size will be fixed to the maximum batch size configured for this server",
    )
    iou_threshold: Optional[float] = Field(
        default=0.3,
        examples=[0.5],
        description="The IoU threhsold that must be met for a box pair to be considered duplicate during NMS",
    )
    max_detections: Optional[int] = Field(
        default=300,
        examples=[300],
        description="The maximum number of detections that will be returned",
    )
    max_candidates: Optional[int] = Field(
        default=3000,
        description="The maximum number of candidate detections passed to NMS",
    )
    visualization_labels: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If true, labels will be rendered on prediction visualizations",
    )
    visualization_stroke_width: Optional[int] = Field(
        default=1,
        examples=[1],
        description="The stroke width used when visualizing predictions",
    )
    visualize_predictions: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If true, the predictions will be drawn on the original image and returned as a base64 string",
    )
    disable_active_learning: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If true, the predictions will be prevented from registration by Active Learning (if the functionality is enabled)",
    )
    active_learning_target_dataset: Optional[str] = Field(
        default=None,
        examples=["my_dataset"],
        description="Parameter to be used when Active Learning data registration should happen against different dataset than the one pointed by model_id",
    )


class KeypointsDetectionInferenceRequest(ObjectDetectionInferenceRequest):
    keypoint_confidence: Optional[float] = Field(
        default=0.0,
        examples=[0.5],
        description="The confidence threshold used to filter out non visible keypoints",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _reject_best_confidence(cls, value: Any) -> Any:
        if value == "best":
            raise ValueError(
                'confidence="best" is not supported for keypoint detection '
                "— model eval does not yet produce per-class thresholds for "
                'this task. Use a float or "default".'
            )
        return value


class InstanceSegmentationInferenceRequest(ObjectDetectionInferenceRequest):
    """Instance Segmentation inference request.

    Attributes:
        mask_decode_mode (Optional[str]): The mode used to decode instance segmentation masks, one of 'accurate', 'fast', 'tradeoff'.
        tradeoff_factor (Optional[float]): The amount to tradeoff between 0='fast' and 1='accurate'.
    """

    mask_decode_mode: Optional[str] = Field(
        default="accurate",
        examples=["accurate"],
        description="The mode used to decode instance segmentation masks, one of 'accurate', 'fast', 'tradeoff'",
    )
    tradeoff_factor: Optional[float] = Field(
        default=0.0,
        examples=[0.5],
        description="The amount to tradeoff between 0='fast' and 1='accurate'",
    )
    response_mask_format: Literal["polygon", "rle"] = Field(
        default="polygon",
        examples=["rle"],
        description="Requested output mask format - `polygon` is the default Roboflow format, which however is "
        "not capable representing certain shapes - RLE is compact and more standard representation, yet "
        "require special decoding on the caller side - currently supported in `opt-in` mode when server is "
        "running with `USE_INFERENCE_MODELS=True` - otherwise it's ignored.",
    )
    enforce_dense_masks_in_inference_models: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="Flag to enforce dense masks in inference models. Such masks are faster than "
        "RLE but consume more memory which may be unstable in some cases. This flag cannot be tweaked "
        "when used on Roboflow serverless platform.",
    )


class SemanticSegmentationInferenceRequest(CVInferenceRequest):
    """Semantic Segmentation inference request.

    Attributes:
        response_mask_format (str): Format of the masks in the response, one of 'base64_png'.
    """

    def __init__(self, **kwargs):
        kwargs["model_type"] = "semantic-segmentation"
        super().__init__(**kwargs)

    confidence: Confidence = Field(
        default=0.4,
        examples=[0.5, "best", "default"],
        description=(
            'Confidence threshold. "best" uses model-eval thresholds, '
            '"default" uses the model built-in, or pass a float.'
        ),
    )
    response_mask_format: SkipJsonSchema[Literal["base64_png"]] = Field(
        default="base64_png",
        examples=["base64_png"],
        description=(
            "[INTERNAL USE ONLY] Format of segmentation_mask / confidence_mask in the "
            "response. 'base64_png' returns base64-encoded PNG strings."
        ),
    )


class ClassificationInferenceRequest(CVInferenceRequest):
    """Classification inference request.

    Attributes:
        confidence (Optional[float]): The confidence threshold used to filter out predictions.
        visualization_stroke_width (Optional[int]): The stroke width used when visualizing predictions.
        visualize_predictions (Optional[bool]): If true, the predictions will be drawn on the original image and returned as a base64 string.
    """

    include_anomaly_map: bool = Field(
        default=False,
        description="Include a raw anomaly heatmap for FoundAD and PatchCore models",
    )

    def __init__(self, **kwargs):
        kwargs["model_type"] = "classification"
        super().__init__(**kwargs)

    confidence: Confidence = Field(
        default=0.4,
        examples=[0.5, "best", "default"],
        description=(
            'Confidence threshold. "best" uses model-eval thresholds, '
            '"default" uses the model built-in, or pass a float.'
        ),
    )
    visualization_stroke_width: Optional[int] = Field(
        default=1,
        examples=[1],
        description="The stroke width used when visualizing predictions",
    )
    visualize_predictions: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If true, the predictions will be drawn on the original image and returned as a base64 string",
    )
    disable_active_learning: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If true, the predictions will be prevented from registration by Active Learning (if the functionality is enabled)",
    )
    active_learning_target_dataset: Optional[str] = Field(
        default=None,
        examples=["my_dataset"],
        description="Parameter to be used when Active Learning data registration should happen against different dataset than the one pointed by model_id",
    )


class LMMInferenceRequest(CVInferenceRequest):
    visualize_predictions: ClassVar = False
    prompt: Optional[str] = Field(
        default=None,
        examples=["caption"],
        description="If set, use this prompt to guide the LMM",
    )
    enable_thinking: bool = Field(
        default=False,
        description="If true, enables thinking/reasoning mode for models that support it (e.g. Qwen3.5). The model's reasoning will be included in the response.",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate. If not set, the model's default will be used.",
    )


class DynamicClassBaseInferenceRequest(CVInferenceRequest):
    """Request for zero-shot object detection models (with dynamic class lists).

    Attributes:
        text (List[str]): A list of strings.
    """

    model_id: Optional[str] = Field(None)
    text: List[str] = Field(
        examples=[["person", "dog", "cat"]],
        description="A list of strings",
    )


class YOLOWorldInferenceRequest(DynamicClassBaseInferenceRequest):
    """Request for Grounding DINO zero-shot predictions.

    Attributes:
        text (List[str]): A list of strings.
    """

    yolo_world_version_id: Optional[str] = "l"
    confidence: Optional[float] = configuration.DEFAULT_CONFIDENCE


class GroundingDINOInferenceRequest(DynamicClassBaseInferenceRequest):
    """Request for Grounding DINO zero-shot predictions.

    Attributes:
        text (List[str]): A list of strings.
    """

    box_threshold: Optional[float] = 0.5
    grounding_dino_version_id: Optional[str] = "default"
    text_threshold: Optional[float] = 0.5
    class_agnostic_nms: Optional[bool] = configuration.CLASS_AGNOSTIC_NMS


class Moondream2InferenceRequest(DynamicClassBaseInferenceRequest):
    """Request for Moondream 2 zero-shot predictions.

    Attributes:
        text (List[str]): A list of strings.
    """

    prompt: str


class TrainBox(BaseModel):
    x: int = Field(description="Center x coordinate in pixels of train box")
    y: int = Field(description="Center y coordinate in pixels of train box")
    w: int = Field(description="Width in pixels of train box")
    h: int = Field(description="Height in pixels of train box")
    cls: str = Field(description="Class name of object this box encloses")
    negative: bool = Field(
        default=False,
        description="Whether this object is a positive or negative example for this class",
    )


class TrainingImage(BaseModel):
    boxes: List[TrainBox] = Field(
        description="List of boxes and corresponding classes of examples for the model to learn from"
    )
    image: InferenceRequestImage = Field(
        description="Image data that `boxes` describes"
    )


class OwlV2InferenceRequest(BaseRequest):
    """Request for OwlV2 inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        owlv2_version_id (Optional[str]): The version ID of OwlV2 to be used for this request.
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) for inference.
        training_data (List[TrainingImage]): Training data to ground the model on
        confidence (float): Confidence threshold to filter predictions by
    """

    owlv2_version_id: Optional[str] = Field(
        default=configuration.OWLV2_VERSION_ID,
        examples=["owlv2-base-patch16-ensemble"],
        description="The version ID of owlv2 to be used for this request.",
    )
    model_id: Optional[str] = Field(
        default=None, description="Model id to be used in the request."
    )

    image: Union[List[InferenceRequestImage], InferenceRequestImage] = Field(
        description="Images to run the model on"
    )
    training_data: List[TrainingImage] = Field(
        description="Training images for the owlvit model to learn form"
    )
    confidence: Optional[float] = Field(
        default=0.99,
        examples=[0.99],
        description="Default confidence threshold for owlvit predictions. "
        "Needs to be much higher than you're used to, probably 0.99 - 0.9999",
    )
    visualization_labels: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If true, labels will be rendered on prediction visualizations",
    )
    visualization_stroke_width: Optional[int] = Field(
        default=1,
        examples=[1],
        description="The stroke width used when visualizing predictions",
    )
    visualize_predictions: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="If true, the predictions will be drawn on the original image and returned as a base64 string",
    )

    @validator("model_id", always=True, allow_reuse=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("owl2_version_id") is None:
            return None
        return f"google/{values['owl2_version_id']}"


class ClipInferenceRequest(BaseRequest):
    """Request for CLIP inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        clip_version_id (Optional[str]): The version ID of CLIP to be used for this request.
    """

    clip_version_id: Optional[str] = Field(
        default=configuration.CLIP_VERSION_ID,
        examples=["ViT-B-16"],
        description="The version ID of CLIP to be used for this request. Must be one of RN101, RN50, RN50x16, RN50x4, RN50x64, ViT-B-16, ViT-B-32, ViT-L-14-336px, and ViT-L-14.",
    )
    model_id: Optional[str] = Field(None)

    @validator("model_id", always=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("clip_version_id") is None:
            return None
        return f"clip/{values['clip_version_id']}"


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
        examples=["The quick brown fox jumps over the lazy dog"],
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
        examples=["url"],
        description="The type of image data provided, one of 'url' or 'base64'",
    )
    subject_type: str = Field(
        default="image",
        examples=["image"],
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
        examples=["text"],
        description="The type of prompt, one of 'image' or 'text'",
    )


class PerceptionEncoderInferenceRequest(BaseRequest):
    """Request for PERCEPTION_ENCODER inference.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        clip_version_id (Optional[str]): The version ID of PERCEPTION_ENCODER to be used for this request.
    """

    perception_encoder_version_id: Optional[str] = Field(
        default=configuration.PERCEPTION_ENCODER_VERSION_ID,
        examples=["PE-Core-L14-336"],
        description="The version ID of PERCEPTION_ENCODER to be used for this request. Must be one of RN101, RN50, RN50x16, RN50x4, RN50x64, ViT-B-16, ViT-B-32, ViT-L-14-336px, and ViT-L-14.",
    )
    model_id: Optional[str] = Field(None)

    @validator("model_id", always=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("perception_encoder_version_id") is None:
            return None
        return f"perception_encoder/{values['perception_encoder_version_id']}"


class PerceptionEncoderImageEmbeddingRequest(PerceptionEncoderInferenceRequest):
    """Request for PERCEPTION_ENCODER image embedding.

    Attributes:
        image (Union[List[InferenceRequestImage], InferenceRequestImage]): Image(s) to be embedded.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]


class PerceptionEncoderTextEmbeddingRequest(PerceptionEncoderInferenceRequest):
    """Request for PERCEPTION_ENCODER text embedding.

    Attributes:
        text (Union[List[str], str]): A string or list of strings.
    """

    text: Union[List[str], str] = Field(
        examples=["The quick brown fox jumps over the lazy dog"],
        description="A string or list of strings",
    )


class PerceptionEncoderCompareRequest(PerceptionEncoderInferenceRequest):
    """Request for PERCEPTION_ENCODER comparison.

    Attributes:
        subject (Union[InferenceRequestImage, str]): The type of image data provided, one of 'url' or 'base64'.
        subject_type (str): The type of subject, one of 'image' or 'text'.
        prompt (Union[List[InferenceRequestImage], InferenceRequestImage, str, List[str], Dict[str, Union[InferenceRequestImage, str]]]): The prompt for comparison.
        prompt_type (str): The type of prompt, one of 'image' or 'text'.
    """

    subject: Union[InferenceRequestImage, str] = Field(
        examples=["url"],
        description="The type of image data provided, one of 'url' or 'base64'",
    )
    subject_type: str = Field(
        default="image",
        examples=["image"],
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
        examples=["text"],
        description="The type of prompt, one of 'image' or 'text'",
    )


class DoctrOCRInferenceRequest(BaseRequest):
    """
    DocTR inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    doctr_version_id: Optional[str] = "default"
    model_id: Optional[str] = Field(None)
    generate_bounding_boxes: Optional[bool] = False

    @validator("model_id", always=True, allow_reuse=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("doctr_version_id") is None:
            return None
        return f"doctr/{values['doctr_version_id']}"


class EasyOCRInferenceRequest(BaseRequest):
    """
    EasyOCR inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    easy_ocr_version_id: Optional[str] = configuration.EASYOCR_VERSION_ID
    model_id: Optional[str] = Field(None)
    language_codes: Optional[List[str]] = Field(default=["en"])
    quantize: Optional[bool] = Field(
        default=False,
        description="Quantized models are smaller and faster, but may be less accurate and won't work correctly on all hardware.",
    )

    @validator("model_id", always=True, allow_reuse=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("easy_ocr_version_id") is None:
            return None
        return f"easy_ocr/{values['easy_ocr_version_id']}"


class TrOCRInferenceRequest(BaseRequest):
    """
    TrOCR inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    trocr_version_id: Optional[str] = "trocr-base-printed"
    model_id: Optional[str] = Field(None)

    @validator("model_id", always=True, allow_reuse=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("trocr_version_id") is None:
            return None
        return f"trocr/{values['trocr_version_id']}"


PP_OCR_FAMILY = "pp_ocr"

_STAGE_UNSET = "__unset__"


class PPOCRInferenceRequest(BaseRequest):
    """
    PP-OCR inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    text_detection: Optional[str] = _STAGE_UNSET
    text_recognition: Optional[str] = _STAGE_UNSET
    pp_ocr_version_id: Optional[str] = Field(None)
    model_id: Optional[str] = Field(None)

    @model_validator(mode="after")
    def resolve_stages_and_model_id(self) -> "PPOCRInferenceRequest":
        text_detection = self.text_detection
        text_recognition = self.text_recognition
        if text_detection is _STAGE_UNSET and text_recognition is _STAGE_UNSET:
            if self.pp_ocr_version_id:
                parts = self.pp_ocr_version_id.split("-")
                if len(parts) == 1:
                    text_detection = text_recognition = parts[0]
                elif len(parts) == 2:
                    text_detection, text_recognition = parts
                else:
                    raise ValueError(
                        f"Invalid PP-OCR pp_ocr_version_id value: {self.pp_ocr_version_id}"
                    )
        default_detection, default_recognition = default_stage_tokens(PP_OCR_FAMILY)
        det = (
            default_detection if text_detection is _STAGE_UNSET else text_detection
        ) or DISABLED_STAGE
        rec = (
            default_recognition
            if text_recognition is _STAGE_UNSET
            else text_recognition
        ) or DISABLED_STAGE
        det = det.lower()
        rec = rec.lower()
        valid_tokens = stage_tokens(PP_OCR_FAMILY)
        if det not in valid_tokens:
            raise ValueError(f"Invalid PP-OCR text_detection value: {det}")
        if rec not in valid_tokens:
            raise ValueError(f"Invalid PP-OCR text_recognition value: {rec}")
        if det == DISABLED_STAGE and rec == DISABLED_STAGE:
            raise ValueError("PP-OCR requires at least one of detection or recognition")
        self.text_detection = det
        self.text_recognition = rec
        self.pp_ocr_version_id = f"{det}-{rec}"
        self.model_id = pipeline_model_id(PP_OCR_FAMILY, (det, rec))
        return self


class SamInferenceRequest(BaseRequest):
    """SAM inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        sam_version_id (Optional[str]): The version ID of SAM to be used for this request.
    """

    sam_version_id: Optional[str] = Field(
        default=configuration.SAM_VERSION_ID,
        examples=["vit_h"],
        description="The version ID of SAM to be used for this request. Must be one of vit_h, vit_l, or vit_b.",
    )

    model_id: Optional[str] = Field(None)

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


class Sam2InferenceRequest(BaseRequest):
    """SAM2 inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        sam2_version_id (Optional[str]): The version ID of SAM2 to be used for this request.
    """

    sam2_version_id: Optional[str] = Field(
        default=configuration.SAM2_VERSION_ID,
        examples=["hiera_large"],
        description="The version ID of SAM to be used for this request. Must be one of hiera_tiny, hiera_small, hiera_large, hiera_b_plus",
    )

    model_id: Optional[str] = Field(None)

    @validator("model_id", always=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("sam2_version_id") is None:
            return None
        return f"sam2/{values['sam2_version_id']}"


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
        description="The format of the response. Must be one of 'json', 'rle', or 'binary'. If binary, masks are returned as binary numpy arrays. If json, masks are converted to polygons. If rle, masks are converted to RLE format.",
    )
    image: InferenceRequestImage = Field(
        description="The image to be segmented.",
    )
    image_id: Optional[str] = Field(
        default=None,
        examples=["image_id"],
        description="The ID of the image to be segmented used to retrieve cached embeddings. If an embedding is cached, it will be used instead of generating a new embedding. If no embedding is cached, a new embedding will be generated and cached.",
    )
    prompts: Sam2PromptSet = Field(
        default=Sam2PromptSet(prompts=None),
        example=[{"prompts": [{"points": [{"x": 100, "y": 100, "positive": True}]}]}],
        description="A list of prompts for masks to predict. Each prompt can include a bounding box and / or a set of postive or negative points. "
        "Also accepts a flat array of prompts (e.g. 'prompts': [{...}, {...}]) for convenience.",
    )
    multimask_output: bool = Field(
        default=True,
        examples=[True],
        description="If true, the model will return three masks. "
        "For ambiguous input prompts (such as a single click), this will often "
        "produce better masks than a single prediction. If only a single "
        "mask is needed, the model's predicted quality score can be used "
        "to select the best mask. For non-ambiguous prompts, such as multiple "
        "input prompts, multimask_output=False can give better results.",
    )

    @validator("prompts", pre=True, always=True)
    def _coerce_prompts(cls, value):
        """
        Accepts any of the following and coerces to Sam2PromptSet:
        - None
        - Sam2PromptSet
        - {"prompts": [...]} (nested)
        - [...] (flat list of prompts)
        - single prompt dict (wrapped to list)
        """
        if value is None:
            return Sam2PromptSet(prompts=None)
        if isinstance(value, Sam2PromptSet):
            return value
        if isinstance(value, dict):
            if "prompts" in value:
                return Sam2PromptSet(**value)
            try:
                return Sam2PromptSet(prompts=[Sam2Prompt(**value)])
            except Exception:
                return Sam2PromptSet(**value)
        if isinstance(value, list):
            prompts: List[Sam2Prompt] = []
            for item in value:
                if isinstance(item, Sam2Prompt):
                    prompts.append(item)
                elif isinstance(item, dict):
                    prompts.append(Sam2Prompt(**item))
                else:
                    raise ValueError(
                        "Invalid prompt entry; expected dict or Sam2Prompt instance"
                    )
            return Sam2PromptSet(prompts=prompts)
        return value

    save_logits_to_cache: bool = Field(
        default=False,
        description="If True, saves the low-resolution logits to the cache for potential future use. "
        "This can speed up subsequent requests with similar prompts on the same image. "
        "This feature is ignored if DISABLE_SAM2_LOGITS_CACHE env variable is set True",
    )
    load_logits_from_cache: bool = Field(
        default=False,
        description="If True, attempts to load previously cached low-resolution logits for the given image and prompt set. "
        "This can significantly speed up inference when making multiple similar requests on the same image. "
        "This feature is ignored if DISABLE_SAM2_LOGITS_CACHE env variable is set True",
    )


class Sam3InferenceRequest(BaseRequest):
    """SAM3 inference request.

    Attributes:
        model_id (Optional[str]): The model ID to be used, typically `sam3`.
    """

    model_id: Optional[str] = Field(
        default="sam3/sam3_final",
        description="The model ID of SAM3. Use 'sam3/sam3_final' to target the generic base model.",
    )


class Sam3SegmentationRequest(Sam3InferenceRequest):
    format: Optional[str] = Field(
        default="polygon",
        description="One of 'polygon', 'rle'",
    )
    image: InferenceRequestImage = Field(description="The image to be segmented.")
    image_id: Optional[str] = Field(
        default=None, description="Optional ID for caching embeddings."
    )
    output_prob_thresh: Optional[float] = Field(
        default=0.5, description="Score threshold for outputs."
    )

    prompts: List[Sam3Prompt] = Field(
        description="List of prompts (text and/or visual)", min_items=1
    )

    nms_iou_threshold: Optional[float] = Field(
        default=None,
        description="IoU threshold for cross-prompt NMS. If None, NMS is disabled. Must be in [0.0, 1.0] when set.",
    )

    @validator("nms_iou_threshold")
    def _validate_nms_iou_threshold(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("nms_iou_threshold must be between 0.0 and 1.0")
        return v

    @validator("prompts")
    def _validate_prompts(cls, prompts: List[Sam3Prompt]):
        if not prompts or len(prompts) == 0:
            raise ValueError("At least one prompt is required")
        if len(prompts) > configuration.SAM3_MAX_PROMPT_BATCH_SIZE:
            raise ValueError(
                f"Exceeded SAM3_MAX_PROMPT_BATCH_SIZE={configuration.SAM3_MAX_PROMPT_BATCH_SIZE}: got {len(prompts)}"
            )
        return prompts


class AddModelRequest(BaseModel):
    """Request to add a model to the inference server.

    Attributes:
        model_id (str): A unique model identifier.
        model_type (Optional[str]): The type of the model, usually referring to what task the model performs.
        api_key (Optional[str]): Roboflow API Key that will be passed to the model during initialization for artifact retrieval.
    """

    model_config = ConfigDict(protected_namespaces=())
    model_id: str = ModelID
    model_type: Optional[str] = ModelType
    api_key: Optional[str] = ApiKey


class ClearModelRequest(BaseModel):
    """Request to clear a model from the inference server.

    Attributes:
        model_id (str): A unique model identifier.
    """

    model_config = ConfigDict(protected_namespaces=())
    model_id: str = ModelID


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
        None, description="The class label confidence as a fraction between 0 and 1"
    )
    class_id: int = Field(description="The class id of the prediction")
    tracker_id: Optional[int] = Field(
        description="The tracker id of the prediction if tracking is enabled",
        default=None,
    )
    detection_id: str = Field(
        description="Unique identifier of detection",
        default_factory=lambda: str(uuid4()),
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )


def _mask_to_base64_png(mask: Any) -> str:
    """Encodes a uint8 numpy mask exactly like the model-side eager encoding."""
    import io

    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.asarray(mask, dtype=np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


class SemanticSegmentationPrediction(BaseModel):
    segmentation_mask: Any = Field(
        description="base64-encoded PNG of predicted class label at each pixel.",
        json_schema_extra={"type": "string"},
    )
    class_map: Dict[str, str] = Field(
        description="Map of pixel intensity value to class label"
    )
    confidence_mask: Any = Field(
        description="base64-encoded PNG of predicted class confidence at each pixel.",
        json_schema_extra={"type": "string"},
    )
    present_class_ids: Optional[List[int]] = Field(
        default=None,
        description=(
            "Sorted list of pixel values present in segmentation_mask, including "
            "background (0) when present. Optimization hint that lets consumers "
            "skip scanning the full-resolution mask; consumers must fall back to "
            "scanning when this field is absent."
        ),
    )

    @field_serializer("segmentation_mask", "confidence_mask", when_used="json")
    def _serialize_mask(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        return _mask_to_base64_png(value)


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
    class_id: int = Field(description="Numeric ID associated with the class label")


class ObjectDetectionInferenceResponse(CvInferenceResponse, WithVisualizationResponse):
    """Object Detection inference response.

    Attributes:
        predictions (List[inference.core.entities.responses.inference.ObjectDetectionPrediction]): List of object detection predictions.
    """

    predictions: List[ObjectDetectionPrediction]


class Keypoint(Point):
    confidence: float = Field(
        description="Model confidence regarding keypoint visibility."
    )
    class_id: int = Field(description="Identifier of keypoint.")
    class_name: str = Field(alias="class", description="Type of keypoint.")


class KeypointsPrediction(ObjectDetectionPrediction):
    keypoints: List[Keypoint]


class KeypointsDetectionInferenceResponse(
    CvInferenceResponse, WithVisualizationResponse
):
    predictions: List[KeypointsPrediction]


class SemanticSegmentationInferenceResponse(
    CvInferenceResponse, WithVisualizationResponse
):
    """Semantic Segmentation inference response.

    Attributes:
        predictions (inference.core.entities.responses.inference.SemanticSegmentationPrediction): Semantic segmentation predictions.
    """

    predictions: SemanticSegmentationPrediction


class ClassificationInferenceResponse(CvInferenceResponse, WithVisualizationResponse):
    """Classification inference response.

    Attributes:
        predictions (List[inference.core.entities.responses.inference.ClassificationPrediction]): List of classification predictions.
        top (str): The top predicted class label.
        confidence (float): The confidence of the top predicted class label.
    """

    predictions: List[ClassificationPrediction]
    top: str = Field(description="The top predicted class label", default="")
    confidence: float = Field(
        description="The confidence of the top predicted class label",
        default=0.0,
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )


class MultiLabelClassificationInferenceResponse(
    CvInferenceResponse, WithVisualizationResponse
):
    """Multi-label Classification inference response.

    Attributes:
        predictions (Dict[str, inference.core.entities.responses.inference.MultiLabelClassificationPrediction]): Dictionary of multi-label classification predictions.
        predicted_classes (List[str]): The list of predicted classes.
    """

    predictions: Dict[str, MultiLabelClassificationPrediction]
    predicted_classes: List[str] = Field(description="The list of predicted classes")
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )


class LMMInferenceResponse(CvInferenceResponse):
    response: Union[str, dict] = Field(
        description="Text/structured response generated by model"
    )


class DepthEstimationResponse(InferenceResponse):
    """Response for depth estimation inference.

    Attributes:
        normalized_depth (Union[str, List[List[float]]]): The per-image normalized ordinal
            depth map as a 2D array of floats between 0 and 1. Higher values
            indicate nearer predictions.
            serialized according to the request's `depth_map_format`: a 2D array of
            floats between 0 and 1 (`json`, the default) or a base64 grayscale PNG
            string (16-bit for `png16`, 8-bit for `png8`).
        depth_map_format (Literal["json", "png16", "png8"]): The serialization
            format used for `normalized_depth`.
        image (Optional[str]): Base64 encoded visualization of the depth map if visualize_predictions is True.
        time (float): The processing time in seconds.
        visualization (Optional[str]): Base64 encoded visualization of the depth map if visualize_predictions is True.
    """

    normalized_depth: Union[str, List[List[float]]] = Field(
        description="Per-image normalized ordinal depth as a 2D array of floats between "
        "0 and 1, where 1 is nearest and 0 is farthest. Values are not "
        "physical distances or directly comparable across images or model "
        "families without calibration. The normalized depth map: a 2D array of floats between 0 and 1 "
        "(`json` format, default) or a base64 grayscale PNG string (`png16`/`png8`), "
        "per the request's `depth_map_format`"
    )
    depth_map_format: Literal["json", "png16", "png8"] = Field(
        default="json",
        description="The serialization format used for `normalized_depth`",
    )
    image: Optional[str] = Field(
        None,
        description="Base64 encoded visualization of the depth map if visualize_predictions is True",
    )


class StubResponse(InferenceResponse, WithVisualizationResponse):
    is_stub: bool = Field(description="Field to mark prediction type as stub")
    model_id: str = Field(description="Identifier of a model stub that was called")
    task_type: str = Field(description="Task type of the project")


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
        default=None,
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
        default=None,
        description="The time in seconds it took to produce the similarity scores including preprocessing",
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )


class PerceptionEncoderEmbeddingResponse(InferenceResponse):
    """Response for PERCEPTION_ENCODER embedding.

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


class PerceptionEncoderCompareResponse(InferenceResponse):
    """Response for PERCEPTION_ENCODER comparison.

    Attributes:
        similarity (Union[List[float], Dict[str, float]]): Similarity scores.
        time (float): The time in seconds it took to produce the similarity scores including preprocessing.
    """

    similarity: Union[List[float], Dict[str, float]]
    time: Optional[float] = Field(
        default=None,
        description="The time in seconds it took to produce the similarity scores including preprocessing",
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )


class OCRInferenceResponse(InferenceResponse):
    """
    OCR Inference response.

    Attributes:
        result (str): The combined OCR recognition result.
        predictions (List[ObjectDetectionPrediction]): List of objects detected by OCR
        time (float): The time in seconds it took to produce the inference including preprocessing
    """

    result: str = Field(description="The combined OCR recognition result.")
    image: Optional[InferenceResponseImage] = Field(
        description="Metadata about input image dimensions", default=None
    )
    predictions: Optional[List[ObjectDetectionPrediction]] = Field(
        description="List of objects detected by OCR",
        default=None,
    )
    time: float = Field(
        description="The time in seconds it took to produce the inference including preprocessing."
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )


class SamEmbeddingResponse(InferenceResponse):
    """SAM embedding response.

    Attributes:
        embeddings (Union[List[List[List[List[float]]]], Any]): The SAM embedding.
        time (float): The time in seconds it took to produce the embeddings including preprocessing.
    """

    embeddings: Union[List[List[List[List[float]]]], Any] = Field(
        examples=["[[[[0.1, 0.2, 0.3, ...] ...] ...]]"],
        description="If request format is json, embeddings is a series of nested lists representing the SAM embedding. If request format is binary, embeddings is a binary numpy array. The dimensions of the embedding are 1 x 256 x 64 x 64.",
    )
    time: float = Field(
        description="The time in seconds it took to produce the embeddings including preprocessing"
    )


class SamSegmentationResponse(InferenceResponse):
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


class Sam3EmbeddingResponse(InferenceResponse):
    image_id: str = Field(description="Image id embeddings are cached to")
    time: float = Field(
        description="The time in seconds it took to produce the embeddings including preprocessing"
    )


class Sam3SegmentationPrediction(BaseModel):
    masks: Union[List[List[List[int]]], Dict[str, Any]] = Field(
        description="Mask data - either polygon coordinates or RLE encoding"
    )
    confidence: float = Field(description="Masks confidence")
    format: Optional[str] = Field(
        default="polygon", description="Format of the mask data: 'polygon' or 'rle'"
    )


class Sam3PromptEcho(BaseModel):
    prompt_index: int = Field()
    type: Optional[str] = Field(default=None)
    text: Optional[str] = Field(default=None)
    num_boxes: Optional[int] = Field(default=None)


class Sam3PromptResult(BaseModel):
    prompt_index: int = Field()
    echo: Sam3PromptEcho = Field()
    predictions: List[Sam3SegmentationPrediction] = Field()


class Sam3SegmentationResponse(InferenceResponse):
    prompt_results: List[Sam3PromptResult] = Field()
    time: float = Field(
        description="The time in seconds it took to produce the segmentation including preprocessing"
    )


class ServerVersionInfo(BaseModel):
    """Server version information.

    Attributes:
        name (str): Server name.
        version (str): Server version.
        uuid (str): Server UUID.
    """

    name: str = Field(examples=["Roboflow Inference Server"])
    version: str = Field(examples=["0.0.1"])
    uuid: str = Field(examples=["9c18c6f4-2266-41fb-8a0f-c12ae28f6fbe"])


class ModelDescriptionEntity(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(
        description="Identifier of the model", examples=["some-project/3"]
    )
    task_type: str = Field(
        description="Type of the task that the model performs",
        examples=["classification"],
    )
    batch_size: Optional[int] = Field(
        None,
        description="Batch size accepted by the model (if registered).",
    )
    input_height: Optional[int] = Field(
        None,
        description="Image input height accepted by the model (if registered).",
    )
    input_width: Optional[int] = Field(
        None,
        description="Image input width accepted by the model (if registered).",
    )
    vram_bytes: Optional[int] = Field(
        None,
        description="Estimated GPU VRAM consumed by this model in bytes (measured during load).",
    )
    request_aliases: List[str] = Field(
        default_factory=list,
        description="Other model IDs that resolved to this model.",
    )
    request_paths: List[str] = Field(
        default_factory=list,
        description="HTTP request paths that triggered inference on this model (e.g. /door-glyph-locator/10, /infer/object_detection).",
    )


class ModelsDescriptions(BaseModel):
    models: List[ModelDescriptionEntity] = Field(
        description="List of models that are loaded by model manager.",
    )
    total_vram_bytes: Optional[int] = Field(
        None,
        description="Total estimated VRAM consumed by all loaded models in bytes.",
    )
    gpu_memory_used: Optional[int] = Field(
        None,
        description="Current GPU memory in use in bytes (device-level, includes all runtimes).",
    )
    gpu_memory_total: Optional[int] = Field(
        None,
        description="Total GPU memory available in bytes.",
    )
    torch_cuda_allocated: Optional[int] = Field(
        None,
        description="Live tensor memory allocated by PyTorch's CUDA allocator in bytes.",
    )
    torch_cuda_reserved: Optional[int] = Field(
        None,
        description="Total memory reserved by PyTorch's CUDA allocator in bytes.",
    )
    torch_cuda_allocator_cache: Optional[int] = Field(
        None,
        description="Reserved but currently unallocated PyTorch CUDA memory in bytes.",
    )
    non_torch_gpu_memory: Optional[int] = Field(
        None,
        description=(
            "Device memory not reserved by PyTorch in bytes. This includes native "
            "runtimes, CUDA context overhead, and allocations from other processes."
        ),
    )

    @classmethod
    def from_models_descriptions(
        cls, models_descriptions: List[ModelDescriptionEntity]
    ) -> "ModelsDescriptions":
        model_entities = list(models_descriptions)
        vram_values = [m.vram_bytes for m in model_entities if m.vram_bytes is not None]
        total_vram = sum(vram_values) if vram_values else None
        (
            gpu_used,
            gpu_total,
            torch_allocated,
            torch_reserved,
        ) = _get_gpu_memory_stats()
        allocator_cache = _non_negative_difference(torch_reserved, torch_allocated)
        non_torch_memory = _non_negative_difference(gpu_used, torch_reserved)
        return cls(
            models=model_entities,
            total_vram_bytes=total_vram,
            gpu_memory_used=gpu_used,
            gpu_memory_total=gpu_total,
            torch_cuda_allocated=torch_allocated,
            torch_cuda_reserved=torch_reserved,
            torch_cuda_allocator_cache=allocator_cache,
            non_torch_gpu_memory=non_torch_memory,
        )


def _get_gpu_memory_stats() -> tuple:
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return (
                total - free,
                total,
                torch.cuda.memory_allocated(),
                torch.cuda.memory_reserved(),
            )
    except ImportError:
        pass
    except Exception:
        pass
    return None, None, None, None


def _non_negative_difference(
    minuend: Optional[int], subtrahend: Optional[int]
) -> Optional[int]:
    if minuend is None or subtrahend is None:
        return None
    return max(0, minuend - subtrahend)
