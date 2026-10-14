"""Opt-in V2 image, model, prediction, formatter and painter catalogue.

Callers select these blocks and the built-in operators explicitly::

    from roboflow_workflows.v2.blocks import create_catalogue

    catalogue = create_catalogue()

``Catalogue.from_modules(["roboflow_workflows.v2.blocks"])``
reads the same catalogue through ``WORKFLOWS_V2_CATALOGUE``. The generic V2
engine never imports this package, and the legacy V1 loader does not discover
it. Every contract detail lives on the block classes themselves.

Visualization blocks use their canonical family identifiers. Bounding boxes
and labels expose accurate and fast implementations under the same contracts.

"""

from roboflow_workflows.v2.blocks.analytics import BLOCKS as ANALYTICS_BLOCKS
from roboflow_workflows.v2.blocks.analytics import (
    BackgroundSubtractionBlock,
    DetectionEventLogBlock,
    LineCounterBlock,
    MotionDetectionBlock,
    OverlapFilterBlock,
    PathDeviationBlock,
    TimeInZoneBlock,
    VelocityBlock,
)
from roboflow_workflows.v2.blocks.analytics.data_aggregator.v1 import (
    DataAggregatorBlock,
)
from roboflow_workflows.v2.blocks.codes import BLOCKS as CODES_BLOCKS
from roboflow_workflows.v2.blocks.codes.barcode_detector.v1 import BarcodeDetectorBlock
from roboflow_workflows.v2.blocks.codes.qr_code_detector.v1 import QRCodeDetectorBlock
from roboflow_workflows.v2.blocks.codes.qr_code_generator.v1 import QRCodeGeneratorBlock
from roboflow_workflows.v2.blocks.codes.stitch_ocr_detections.v2 import (
    StitchOCRDetectionsBlock,
)
from roboflow_workflows.v2.blocks.combine import BLOCKS as COMBINE_BLOCKS
from roboflow_workflows.v2.blocks.combine import (
    DetectionsCombineBlock,
    DetectionsMergeBlock,
)
from roboflow_workflows.v2.blocks.control import (
    ContinueIfBlock,
    ContinueIfQueryBlock,
    DeltaFilterBlock,
    HoldCapacityError,
    ObservationOrderError,
    RateLimiterBlock,
    SwitchCaseBlock,
)
from roboflow_workflows.v2.blocks.formatters import BLOCKS as FORMATTER_BLOCKS
from roboflow_workflows.v2.blocks.formatters import (
    TIMESTAMP_KIND,
    CsvFormatterBlock,
    CurrentTimeBlock,
    ExpressionBlock,
    FirstNonEmptyOrDefaultBlock,
    JsonParserBlock,
    PropertyDefinitionBlock,
    StringTemplateBlock,
)
from roboflow_workflows.v2.blocks.fusion import BLOCKS as FUSION_BLOCKS
from roboflow_workflows.v2.blocks.fusion import (
    DETECTIONS_OVERLAPS_KIND,
    DetectionsConsensusBlock,
    DetectionsDifferenceBlock,
    DetectionsListRollUpBlock,
    OverlapAnalysisBlock,
)
from roboflow_workflows.v2.blocks.generation import BLOCKS as GENERATION_BLOCKS
from roboflow_workflows.v2.blocks.generation.stability_ai_image_gen.v1 import (
    StabilityAIImageGenerationBlock,
)
from roboflow_workflows.v2.blocks.generation.stability_ai_inpainting.v1 import (
    StabilityAIInpaintingBlock,
)
from roboflow_workflows.v2.blocks.generation.stability_ai_outpainting.v1 import (
    StabilityAIOutpaintingBlock,
)
from roboflow_workflows.v2.blocks.geometry import BLOCKS as GEOMETRY_BLOCKS
from roboflow_workflows.v2.blocks.geometry import (
    BoundingRectBlock,
    DetectionOffsetBlock,
    DetectionsClassesReplacementBlock,
    DetectionsStitchBlock,
    DimensionCollapseBlock,
)
from roboflow_workflows.v2.blocks.geometry.camera_calibration.v1 import (
    CameraCalibrationBlock,
)
from roboflow_workflows.v2.blocks.geometry.contours_detection.v1 import (
    ContoursDetectionBlock,
)
from roboflow_workflows.v2.blocks.geometry.detections_nearest_neighbor.v1 import (
    DetectionsNearestNeighborBlock,
)
from roboflow_workflows.v2.blocks.geometry.dynamic_zone.v1 import DynamicZoneBlock
from roboflow_workflows.v2.blocks.geometry.geotag_detection.v1 import (
    GeotagDetectionBlock,
)
from roboflow_workflows.v2.blocks.geometry.perspective_correction.v1 import (
    PerspectiveCorrectionBlock,
)
from roboflow_workflows.v2.blocks.geometry.sift.v1 import SiftBlock
from roboflow_workflows.v2.blocks.geometry.sift_comparison.v2 import SiftComparisonBlock
from roboflow_workflows.v2.blocks.geometry.template_matching.v1 import (
    TemplateMatchingBlock,
)
from roboflow_workflows.v2.blocks.image import (
    AbsoluteStaticCropBlock,
    ContrastEnhancementBlock,
    ContrastEqualizationBlock,
    ConvertGrayscaleBlock,
    CropBlock,
    DynamicCropBlock,
    HasBrightnessBlock,
    ImageBlurBlock,
    ImageCopyBlock,
    ImagePreprocessingBlock,
    ImageSlicerBlock,
    ImageThresholdBlock,
    InvertBlock,
    MaskEdgeSnapBlock,
    MorphologicalTransformationBlock,
    MosaicBlock,
    RelativeStaticCropBlock,
    ResizeBlock,
    StaticCropBlock,
)
from roboflow_workflows.v2.blocks.image.auto_rotate_on_edges.v1 import (
    AutoRotateOnEdgesBlock,
)
from roboflow_workflows.v2.blocks.image.camera_focus.v2 import CameraFocusBlock
from roboflow_workflows.v2.blocks.image.dominant_color.v1 import DominantColorBlock
from roboflow_workflows.v2.blocks.image.stitch_images.v1 import StitchImagesBlock
from roboflow_workflows.v2.blocks.industrial import BLOCKS as INDUSTRIAL_BLOCKS
from roboflow_workflows.v2.blocks.industrial import OnvifTrackingBlock
from roboflow_workflows.v2.blocks.measurements import BLOCKS as MEASUREMENT_BLOCKS
from roboflow_workflows.v2.blocks.measurements import (
    DistanceMeasurementBlock,
    MaskAreaMeasurementBlock,
    PixelColorCountBlock,
    SizeMeasurementBlock,
)
from roboflow_workflows.v2.blocks.models import BLOCKS as MODEL_BLOCKS
from roboflow_workflows.v2.blocks.models import (
    ActionRecognitionBlock,
    ClassificationBlock,
    Cosmos3EdgeBlock,
    Florence2Block,
    GoogleVisionOCRBlock,
    InstanceSegmentationBlock,
    KeypointDetectionBlock,
    Moondream2Block,
    MultiLabelClassificationBlock,
    ObjectDetectionBlock,
    QwenVLMBlock,
    SegmentAnything2Block,
    SegmentAnything2VideoBlock,
    SegmentAnything3Block,
    SegmentAnything3InteractiveBlock,
    SegmentAnything3VideoBlock,
    SemanticSegmentationBlock,
)
from roboflow_workflows.v2.blocks.models.clip.v1 import ClipEmbeddingBlock
from roboflow_workflows.v2.blocks.models.clip_comparison.v2 import ClipComparisonBlock
from roboflow_workflows.v2.blocks.models.depth_estimation.v1 import DepthEstimationBlock
from roboflow_workflows.v2.blocks.models.easy_ocr.v1 import EasyOCRBlock
from roboflow_workflows.v2.blocks.models.glm_ocr.v1 import GlmOcrBlock
from roboflow_workflows.v2.blocks.models.ocr_model.v1 import OCRModelBlock
from roboflow_workflows.v2.blocks.models.perception_encoder.v1 import (
    PerceptionEncoderEmbeddingBlock,
)
from roboflow_workflows.v2.blocks.models.pp_ocr.v1 import PPOCRBlock
from roboflow_workflows.v2.blocks.models.qwen3_5vl.v2 import Qwen35VLBlock
from roboflow_workflows.v2.blocks.models.smolvlm2.v1 import SmolVLM2Block
from roboflow_workflows.v2.blocks.models.vision_language_models import (
    BLOCKS as LANGUAGE_BLOCKS,
)
from roboflow_workflows.v2.blocks.models.vision_language_models import (
    AnthropicClaudeBlock,
    GoogleGeminiBlock,
    GoogleGemmaBlock,
    KimiOpenRouterBlock,
    MetaVlmBlock,
    OpenAIBlock,
    OpenAICompatibleBlock,
    OpenRouterBlock,
    SpaceXAIBlock,
    ZaiVlmBlock,
)
from roboflow_workflows.v2.blocks.roboflow_platform import (
    PLATFORM_BLOCKS,
    AssetLibraryAttributesBlock,
    DatasetUploadBlock,
    ModelMonitoringInferenceAggregatorBlock,
    RoboflowCustomMetadataBlock,
    RoboflowVisionEventsBlock,
    VisionEventBundleBlock,
    VisualSearchBlock,
    VisualSearchClassifierBlock,
)
from roboflow_workflows.v2.blocks.sampling import BLOCKS as SAMPLING_BLOCKS
from roboflow_workflows.v2.blocks.sampling import (
    IdentifyChangesBlock,
    IdentifyOutliersBlock,
    SamplingCapacityError,
)
from roboflow_workflows.v2.blocks.secrets import BLOCKS as SECRETS_BLOCKS
from roboflow_workflows.v2.blocks.secrets.environment_secrets_store.v1 import (
    EnvironmentSecretsStoreBlock,
)
from roboflow_workflows.v2.blocks.sinks import BLOCKS as SINKS_BLOCKS
from roboflow_workflows.v2.blocks.sinks.email_notification.v2 import (
    EmailNotificationBlock,
)
from roboflow_workflows.v2.blocks.sinks.local_file_sink.v1 import LocalFileSinkBlock
from roboflow_workflows.v2.blocks.sinks.s3_sink.v1 import S3SinkBlock
from roboflow_workflows.v2.blocks.sinks.slack_notification.v1 import (
    SlackNotificationBlock,
)
from roboflow_workflows.v2.blocks.sinks.twilio_sms_notification.v2 import (
    TwilioSmsNotificationBlock,
)
from roboflow_workflows.v2.blocks.sinks.webhook_sink.v1 import WebhookSinkBlock
from roboflow_workflows.v2.blocks.state._shared.object_cache import ObjectCache
from roboflow_workflows.v2.blocks.state.cache_get.v1 import CacheGetBlock
from roboflow_workflows.v2.blocks.state.cache_set.v1 import CacheSetBlock
from roboflow_workflows.v2.blocks.state.state_compare_and_set.v1 import (
    StateCompareAndSetBlock,
)
from roboflow_workflows.v2.blocks.state.state_get.v1 import StateGetBlock
from roboflow_workflows.v2.blocks.state.state_increment.v1 import StateIncrementBlock
from roboflow_workflows.v2.blocks.state.state_machine_set.v1 import StateMachineSetBlock
from roboflow_workflows.v2.blocks.state.state_set.v1 import StateSetBlock
from roboflow_workflows.v2.blocks.temporal.best_frame.v1 import BestFrameBlock
from roboflow_workflows.v2.blocks.temporal.frame_delay.v1 import (
    FrameDelayBlock,
    FrameHistoryCapacityError,
)
from roboflow_workflows.v2.blocks.temporal.image_stack.v1 import (
    ImageStackBlock,
    ImageStackCapacityError,
)
from roboflow_workflows.v2.blocks.temporal.top_k_brightest.v1 import TopKBrightestBlock
from roboflow_workflows.v2.blocks.tracking import (
    BoTSORTTrackerBlock,
    ByteTrackTrackerBlock,
    OCSORTTrackerBlock,
    SORTTrackerBlock,
    StabilizeDetectionsBlock,
    TrackClassLockBlock,
    TrackerCapacityError,
)
from roboflow_workflows.v2.blocks.transforms import BLOCKS as TRANSFORM_BLOCKS
from roboflow_workflows.v2.blocks.transforms import (
    DetectionsFilterBlock,
    DetectionsTransformationBlock,
    PerClassConfidenceFilterBlock,
)
from roboflow_workflows.v2.blocks.transforms.cosine_similarity.v1 import (
    CosineSimilarityBlock,
)
from roboflow_workflows.v2.blocks.visualization import BLOCKS as VISUALIZATION_BLOCKS
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BlurVisualizationBlock,
    BoundingBoxVisualizationBlock,
    CircleVisualizationBlock,
    ClassificationLabelVisualizationBlock,
    CornerVisualizationBlock,
    CropVisualizationBlock,
    DotVisualizationBlock,
    EllipseVisualizationBlock,
    HaloVisualizationBlock,
    HeatmapVisualizationBlock,
    KeypointVisualizationBlock,
    LabelVisualizationBlock,
    LineCounterVisualizationBlock,
    MaskVisualizationBlock,
    PixelateVisualizationBlock,
    PolygonVisualizationBlock,
    PolygonZoneVisualizationBlock,
    ReferencePathVisualizationBlock,
    SemanticSegmentationVisualizationBlock,
    TraceVisualizationBlock,
    TriangleVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.background_color_visualization.v1 import (
    BackgroundColorVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.color_visualization.v1 import (
    ColorVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.grid_visualization.v1 import (
    GridVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.icon_visualization.v1 import (
    IconVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.model_comparison_visualization.v1 import (
    ModelComparisonVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rich_label_visualization.v1 import (
    RichLabelVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text_display.v1 import (
    TextDisplayBlock,
)
from roboflow_workflows.v2.data.boundaries import MEDIA_WILDCARD_KIND, NUMPY_ARRAY_KIND
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import CROP_SUMMARY_KIND, IMAGE_KIND
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.data.masks.grid import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    MaskGridGeometry,
    SegmentationView,
)
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.predictions import (
    BAR_CODE_DETECTION_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    DETECTION_KIND,
    EMBEDDING_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    NATIVE_KINDS,
    NATIVE_SEMANTIC_SEGMENTATION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    QR_CODE_DETECTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
    TENSOR_KIND,
)
from roboflow_workflows.v2.data.rows import PREDICTION_ROWS_KIND
from roboflow_workflows.v2.data.secrets import SECRET_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.operators.alignment import Align
from roboflow_workflows.v2.engine.operators.window import Window
from roboflow_workflows.v2.engine.resources import Factory
from roboflow_workflows.v2.recording_codecs import NATIVE_RECORDING_CODECS


def create_catalogue() -> Catalogue:
    """Collect the opt-in V2 vision, temporal and managed-state blocks.

    Models remain caller-owned resources: model blocks read the session's
    ``models`` resource. Building the catalogue loads no model and creates
    no resource. Cache blocks receive one lazily created ObjectCache per session;
    callers can supply an object_cache resource to share it explicitly.
    The generic V2 engine does not import this catalogue; hosts
    select it explicitly.

    Returns:
        Catalogue with image blocks, crop and image-processing ports,
        caller-loaded batched model blocks, per-source tracking and track
        consumers, measurements, region and parent-mapping, prediction
        transforms, merge and combine, formatters, boolean and query gates,
        canonical visualization blocks with selectable implementations, temporal
        and managed-state blocks, alignment/window operators and native
        media kinds, including native semantic grids and opt-in
        ``PredictionRows``. Kinds that only a block's outputs name, such as
        ``TIMESTAMP_KIND``, register with that block.
        Recording codecs preserve tensors, images and native predictions as raw
        binary data. Prepared drawings and live segmentation views have no codec;
        a segmentation view's ``full_res()`` is its explicit dense boundary.
    """
    catalogue = Catalogue(
        [
            CropBlock,
            StaticCropBlock,
            ResizeBlock,
            InvertBlock,
            MosaicBlock,
            HasBrightnessBlock,
            AbsoluteStaticCropBlock,
            RelativeStaticCropBlock,
            DynamicCropBlock,
            ImagePreprocessingBlock,
            ImageSlicerBlock,
            ImageCopyBlock,
            ConvertGrayscaleBlock,
            ImageBlurBlock,
            ImageThresholdBlock,
            MaskEdgeSnapBlock,
            CameraFocusBlock,
            StitchImagesBlock,
            AutoRotateOnEdgesBlock,
            DominantColorBlock,
            MorphologicalTransformationBlock,
            ContrastEnhancementBlock,
            ContrastEqualizationBlock,
            DeltaFilterBlock,
            RateLimiterBlock,
            SwitchCaseBlock,
            ContinueIfBlock,
            ContinueIfQueryBlock,
            BestFrameBlock,
            TopKBrightestBlock,
            FrameDelayBlock,
            ImageStackBlock,
            CacheGetBlock,
            CacheSetBlock,
            StateGetBlock,
            StateSetBlock,
            StateIncrementBlock,
            StateCompareAndSetBlock,
            StateMachineSetBlock,
            *MODEL_BLOCKS,
            SORTTrackerBlock,
            ByteTrackTrackerBlock,
            OCSORTTrackerBlock,
            BoTSORTTrackerBlock,
            TrackClassLockBlock,
            StabilizeDetectionsBlock,
            *ANALYTICS_BLOCKS,
            *GEOMETRY_BLOCKS,
            *TRANSFORM_BLOCKS,
            *COMBINE_BLOCKS,
            *FUSION_BLOCKS,
            *SAMPLING_BLOCKS,
            *FORMATTER_BLOCKS,
            *VISUALIZATION_BLOCKS,
            *PLATFORM_BLOCKS,
            *MEASUREMENT_BLOCKS,
            *SINKS_BLOCKS,
            *SECRETS_BLOCKS,
            *GENERATION_BLOCKS,
            *CODES_BLOCKS,
            *LANGUAGE_BLOCKS,
            *INDUSTRIAL_BLOCKS,
        ],
        operators=[Align, Window],
        kinds=[
            IMAGE_KIND,
            CROP_SUMMARY_KIND,
            *NATIVE_KINDS,
            PREDICTION_ROWS_KIND,
            INSTANCE_SEGMENTATION_VIEW_KIND,
            MEDIA_WILDCARD_KIND,
            NUMPY_ARRAY_KIND,
            LANGUAGE_MODEL_OUTPUT_KIND,
            SECRET_KIND,
        ],
        codecs=NATIVE_RECORDING_CODECS,
        providers={"object_cache": Factory(ObjectCache)},
    )

    return catalogue


WORKFLOWS_V2_CATALOGUE = create_catalogue

__all__ = [
    "Cosmos3EdgeBlock",
    "Florence2Block",
    "GoogleVisionOCRBlock",
    "Moondream2Block",
    "QwenVLMBlock",
    "AbsoluteStaticCropBlock",
    "ActionRecognitionBlock",
    "Align",
    "AssetLibraryAttributesBlock",
    "BAR_CODE_DETECTION_KIND",
    "BestFrameBlock",
    "BoTSORTTrackerBlock",
    "BlurVisualizationBlock",
    "BoundingBoxVisualizationBlock",
    "BoundingRectBlock",
    "ByteTrackTrackerBlock",
    "CacheGetBlock",
    "CacheSetBlock",
    "CLASSIFICATION_PREDICTION_KIND",
    "CROP_SUMMARY_KIND",
    "ClassificationBlock",
    "CircleVisualizationBlock",
    "ClassificationLabelVisualizationBlock",
    "ContrastEnhancementBlock",
    "ContrastEqualizationBlock",
    "ConvertGrayscaleBlock",
    "CornerVisualizationBlock",
    "ContinueIfBlock",
    "ContinueIfQueryBlock",
    "CropBlock",
    "CropVisualizationBlock",
    "CsvFormatterBlock",
    "CurrentTimeBlock",
    "DETECTION_KIND",
    "DETECTIONS_OVERLAPS_KIND",
    "DatasetUploadBlock",
    "DeltaFilterBlock",
    "DetectionOffsetBlock",
    "DetectionsClassesReplacementBlock",
    "DetectionsCombineBlock",
    "DetectionsConsensusBlock",
    "DetectionsDifferenceBlock",
    "DetectionsFilterBlock",
    "DetectionsListRollUpBlock",
    "DetectionsMergeBlock",
    "DetectionsStitchBlock",
    "DetectionsTransformationBlock",
    "DimensionCollapseBlock",
    "DistanceMeasurementBlock",
    "DotVisualizationBlock",
    "DynamicCropBlock",
    "EMBEDDING_KIND",
    "EllipseVisualizationBlock",
    "ExpressionBlock",
    "FirstNonEmptyOrDefaultBlock",
    "FrameDelayBlock",
    "FrameHistoryCapacityError",
    "HaloVisualizationBlock",
    "HasBrightnessBlock",
    "HoldCapacityError",
    "IMAGE_KIND",
    "INSTANCE_SEGMENTATION_PREDICTION_KIND",
    "INSTANCE_SEGMENTATION_VIEW_KIND",
    "ImageData",
    "ImageBlurBlock",
    "ImageCopyBlock",
    "IdentifyChangesBlock",
    "IdentifyOutliersBlock",
    "ImagePreprocessingBlock",
    "ImageSlicerBlock",
    "ImageStackBlock",
    "ImageStackCapacityError",
    "ImageThresholdBlock",
    "InstanceSegmentationBlock",
    "InvertBlock",
    "JsonParserBlock",
    "KEYPOINT_DETECTION_PREDICTION_KIND",
    "KeypointDetectionBlock",
    "KeypointVisualizationBlock",
    "LabelVisualizationBlock",
    "MEDIA_WILDCARD_KIND",
    "MaskGridGeometry",
    "MaskVisualizationBlock",
    "MaskAreaMeasurementBlock",
    "MorphologicalTransformationBlock",
    "ModelMonitoringInferenceAggregatorBlock",
    "MosaicBlock",
    "MultiLabelClassificationBlock",
    "OCSORTTrackerBlock",
    "NATIVE_KINDS",
    "NATIVE_RECORDING_CODECS",
    "NATIVE_SEMANTIC_SEGMENTATION_KIND",
    "NUMPY_ARRAY_KIND",
    "OBJECT_DETECTION_PREDICTION_KIND",
    "ObjectDetectionBlock",
    "ObjectCache",
    "ObservationOrderError",
    "OverlapAnalysisBlock",
    "PREDICTION_ROWS_KIND",
    "PerClassConfidenceFilterBlock",
    "PixelColorCountBlock",
    "PixelateVisualizationBlock",
    "PolygonVisualizationBlock",
    "PropertyDefinitionBlock",
    "QR_CODE_DETECTION_KIND",
    "RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND",
    "RateLimiterBlock",
    "RelativeStaticCropBlock",
    "ResizeBlock",
    "RoboflowCustomMetadataBlock",
    "RoboflowVisionEventsBlock",
    "SEMANTIC_SEGMENTATION_PREDICTION_KIND",
    "SegmentationView",
    "SamplingCapacityError",
    "SemanticSegmentationBlock",
    "SemanticSegmentationVisualizationBlock",
    "SizeMeasurementBlock",
    "SORTTrackerBlock",
    "StabilizeDetectionsBlock",
    "StateCompareAndSetBlock",
    "StateGetBlock",
    "StateIncrementBlock",
    "StateMachineSetBlock",
    "StateSetBlock",
    "StaticCropBlock",
    "StringTemplateBlock",
    "SwitchCaseBlock",
    "TENSOR_KIND",
    "TIMESTAMP_KIND",
    "TopKBrightestBlock",
    "TrackClassLockBlock",
    "TrackerCapacityError",
    "TriangleVisualizationBlock",
    "VelocityBlock",
    "VisionEventBundleBlock",
    "VisualSearchBlock",
    "VisualSearchClassifierBlock",
    "WORKFLOWS_V2_CATALOGUE",
    "Window",
    "create_catalogue",
    "select_predictions",
    "SegmentAnything2Block",
    "SegmentAnything2VideoBlock",
    "SegmentAnything3Block",
    "SegmentAnything3InteractiveBlock",
    "SegmentAnything3VideoBlock",
    "MaskEdgeSnapBlock",
    "HeatmapVisualizationBlock",
    "TraceVisualizationBlock",
    "PolygonZoneVisualizationBlock",
    "LineCounterVisualizationBlock",
    "ReferencePathVisualizationBlock",
    "Qwen35VLBlock",
    "OpenRouterBlock",
    "KimiOpenRouterBlock",
    "MetaVlmBlock",
    "ZaiVlmBlock",
    "GoogleGemmaBlock",
    "SpaceXAIBlock",
    "OpenAIBlock",
    "OpenAICompatibleBlock",
    "AnthropicClaudeBlock",
    "GoogleGeminiBlock",
    "LineCounterBlock",
    "TimeInZoneBlock",
    "PathDeviationBlock",
    "DetectionEventLogBlock",
    "OverlapFilterBlock",
    "BackgroundSubtractionBlock",
    "MotionDetectionBlock",
    "OnvifTrackingBlock",
    "BackgroundColorVisualizationBlock",
    "ColorVisualizationBlock",
    "GridVisualizationBlock",
    "LANGUAGE_MODEL_OUTPUT_KIND",
    "SECRET_KIND",
    "IconVisualizationBlock",
    "RichLabelVisualizationBlock",
    "ModelComparisonVisualizationBlock",
    "ClipComparisonBlock",
    "ClipEmbeddingBlock",
    "PerceptionEncoderEmbeddingBlock",
    "DepthEstimationBlock",
    "CosineSimilarityBlock",
    "GlmOcrBlock",
    "OCRModelBlock",
    "EasyOCRBlock",
    "PPOCRBlock",
    "SmolVLM2Block",
    "CameraFocusBlock",
    "StitchImagesBlock",
    "AutoRotateOnEdgesBlock",
    "StitchOCRDetectionsBlock",
    "SiftBlock",
    "PerspectiveCorrectionBlock",
    "TemplateMatchingBlock",
    "GeotagDetectionBlock",
    "DetectionsNearestNeighborBlock",
    "SiftComparisonBlock",
    "CameraCalibrationBlock",
    "DominantColorBlock",
    "QRCodeGeneratorBlock",
    "BarcodeDetectorBlock",
    "QRCodeDetectorBlock",
    "ContoursDetectionBlock",
    "DynamicZoneBlock",
    "StabilityAIImageGenerationBlock",
    "StabilityAIInpaintingBlock",
    "StabilityAIOutpaintingBlock",
    "DataAggregatorBlock",
    "EmailNotificationBlock",
    "EnvironmentSecretsStoreBlock",
    "LocalFileSinkBlock",
    "S3SinkBlock",
    "SlackNotificationBlock",
    "TextDisplayBlock",
    "TwilioSmsNotificationBlock",
    "WebhookSinkBlock",
]
