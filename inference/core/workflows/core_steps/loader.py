from typing import List, Type

from inference.core.cache import cache
from inference.core.env import (
    ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES,
    ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE,
    API_KEY,
    WORKFLOW_BLOCKS_WRITE_DIRECTORY,
    WORKFLOWS_STEP_EXECUTION_MODE,
)
from inference.core.workflows.core_steps.analytics.data_aggregator.v1 import (
    DataAggregatorBlockV1,
)
from inference.core.workflows.core_steps.analytics.line_counter.v1 import (
    LineCounterBlockV1,
)
from inference.core.workflows.core_steps.analytics.line_counter.v2 import (
    LineCounterBlockV2,
)
from inference.core.workflows.core_steps.analytics.path_deviation.v1 import (
    PathDeviationAnalyticsBlockV1,
)
from inference.core.workflows.core_steps.analytics.path_deviation.v2 import (
    PathDeviationAnalyticsBlockV2,
)
from inference.core.workflows.core_steps.analytics.time_in_zone.v1 import (
    TimeInZoneBlockV1,
)
from inference.core.workflows.core_steps.analytics.time_in_zone.v2 import (
    TimeInZoneBlockV2,
)
from inference.core.workflows.core_steps.analytics.velocity.v1 import VelocityBlockV1
from inference.core.workflows.core_steps.cache.cache_get.v1 import CacheGetBlockV1
from inference.core.workflows.core_steps.cache.cache_set.v1 import CacheSetBlockV1
from inference.core.workflows.core_steps.classical_cv.camera_focus.v1 import (
    CameraFocusBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.contours.v1 import (
    ImageContoursDetectionBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.convert_grayscale.v1 import (
    ConvertGrayscaleBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.distance_measurement.v1 import (
    DistanceMeasurementBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.dominant_color.v1 import (
    DominantColorBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.image_blur.v1 import (
    ImageBlurBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.image_preprocessing.v1 import (
    ImagePreprocessingBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.pixel_color_count.v1 import (
    PixelationCountBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.sift.v1 import SIFTBlockV1
from inference.core.workflows.core_steps.classical_cv.sift_comparison.v1 import (
    SIFTComparisonBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.sift_comparison.v2 import (
    SIFTComparisonBlockV2,
)
from inference.core.workflows.core_steps.classical_cv.size_measurement.v1 import (
    SizeMeasurementBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.template_matching.v1 import (
    TemplateMatchingBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.threshold.v1 import (
    ImageThresholdBlockV1,
)
from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_boolean_kind,
    deserialize_bytes_kind,
    deserialize_classification_prediction_kind,
    deserialize_detections_kind,
    deserialize_dictionary_kind,
    deserialize_float_kind,
    deserialize_float_zero_to_one_kind,
    deserialize_image_kind,
    deserialize_integer_kind,
    deserialize_list_of_values_kind,
    deserialize_numpy_array,
    deserialize_optional_string_kind,
    deserialize_point_kind,
    deserialize_rgb_color_kind,
    deserialize_string_kind,
    deserialize_video_metadata_kind,
    deserialize_zone_kind,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.serializers import (
    serialise_image,
    serialise_sv_detections,
    serialize_secret,
    serialize_video_metadata_kind,
    serialize_wildcard_kind,
)
from inference.core.workflows.core_steps.flow_control.continue_if.v1 import (
    ContinueIfBlockV1,
)
from inference.core.workflows.core_steps.flow_control.delta_filter.v1 import (
    DeltaFilterBlockV1,
)
from inference.core.workflows.core_steps.flow_control.rate_limiter.v1 import (
    RateLimiterBlockV1,
)
from inference.core.workflows.core_steps.formatters.csv.v1 import CSVFormatterBlockV1
from inference.core.workflows.core_steps.formatters.expression.v1 import (
    ExpressionBlockV1,
)
from inference.core.workflows.core_steps.formatters.first_non_empty_or_default.v1 import (
    FirstNonEmptyOrDefaultBlockV1,
)
from inference.core.workflows.core_steps.formatters.json_parser.v1 import (
    JSONParserBlockV1,
)
from inference.core.workflows.core_steps.formatters.property_definition.v1 import (
    PropertyDefinitionBlockV1,
)
from inference.core.workflows.core_steps.formatters.vlm_as_classifier.v1 import (
    VLMAsClassifierBlockV1,
)
from inference.core.workflows.core_steps.formatters.vlm_as_classifier.v2 import (
    VLMAsClassifierBlockV2,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.v1 import (
    VLMAsDetectorBlockV1,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.v2 import (
    VLMAsDetectorBlockV2,
)
from inference.core.workflows.core_steps.fusion.buffer.v1 import BufferBlockV1
from inference.core.workflows.core_steps.fusion.detections_classes_replacement.v1 import (
    DetectionsClassesReplacementBlockV1,
)
from inference.core.workflows.core_steps.fusion.detections_consensus.v1 import (
    DetectionsConsensusBlockV1,
)
from inference.core.workflows.core_steps.fusion.detections_stitch.v1 import (
    DetectionsStitchBlockV1,
)
from inference.core.workflows.core_steps.fusion.dimension_collapse.v1 import (
    DimensionCollapseBlockV1,
)
from inference.core.workflows.core_steps.math.cosine_similarity.v1 import (
    CosineSimilarityBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v1 import (
    AnthropicClaudeBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.clip.v1 import (
    ClipModelBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.clip_comparison.v1 import (
    ClipComparisonBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.clip_comparison.v2 import (
    ClipComparisonBlockV2,
)
from inference.core.workflows.core_steps.models.foundation.cog_vlm.v1 import (
    CogVLMBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.florence2.v1 import (
    Florence2BlockV1,
)
from inference.core.workflows.core_steps.models.foundation.florence2.v2 import (
    Florence2BlockV2,
)
from inference.core.workflows.core_steps.models.foundation.gaze.v1 import GazeBlockV1
from inference.core.workflows.core_steps.models.foundation.google_gemini.v1 import (
    GoogleGeminiBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.google_vision_ocr.v1 import (
    GoogleVisionOCRBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.llama_vision.v1 import (
    LlamaVisionBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.lmm.v1 import LMMBlockV1
from inference.core.workflows.core_steps.models.foundation.lmm_classifier.v1 import (
    LMMForClassificationBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.ocr.v1 import OCRModelBlockV1
from inference.core.workflows.core_steps.models.foundation.openai.v1 import (
    OpenAIBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.openai.v2 import (
    OpenAIBlockV2,
)
from inference.core.workflows.core_steps.models.foundation.qwen.v1 import (
    Qwen25VLBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
    SegmentAnything2BlockV1,
)
from inference.core.workflows.core_steps.models.foundation.stability_ai.image_gen.v1 import (
    StabilityAIImageGenBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.stability_ai.inpainting.v1 import (
    StabilityAIInpaintingBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.yolo_world.v1 import (
    YoloWorldModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v1 import (
    RoboflowInstanceSegmentationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v2 import (
    RoboflowInstanceSegmentationModelBlockV2,
)
from inference.core.workflows.core_steps.models.roboflow.keypoint_detection.v1 import (
    RoboflowKeypointDetectionModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.keypoint_detection.v2 import (
    RoboflowKeypointDetectionModelBlockV2,
)
from inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v1 import (
    RoboflowClassificationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v2 import (
    RoboflowClassificationModelBlockV2,
)
from inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v1 import (
    RoboflowMultiLabelClassificationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v2 import (
    RoboflowMultiLabelClassificationModelBlockV2,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v1 import (
    RoboflowObjectDetectionModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v2 import (
    RoboflowObjectDetectionModelBlockV2,
)
from inference.core.workflows.core_steps.models.third_party.barcode_detection.v1 import (
    BarcodeDetectorBlockV1,
)
from inference.core.workflows.core_steps.models.third_party.qr_code_detection.v1 import (
    QRCodeDetectorBlockV1,
)
from inference.core.workflows.core_steps.sampling.identify_changes.v1 import (
    IdentifyChangesBlockV1,
)
from inference.core.workflows.core_steps.sampling.identify_outliers.v1 import (
    IdentifyOutliersBlockV1,
)
from inference.core.workflows.core_steps.secrets_providers.environment_secrets_store.v1 import (
    EnvironmentSecretsStoreBlockV1,
)
from inference.core.workflows.core_steps.sinks.email_notification.v1 import (
    EmailNotificationBlockV1,
)
from inference.core.workflows.core_steps.sinks.local_file.v1 import LocalFileSinkBlockV1
from inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1 import (
    RoboflowCustomMetadataBlockV1,
)
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v1 import (
    RoboflowDatasetUploadBlockV1,
)
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v2 import (
    RoboflowDatasetUploadBlockV2,
)
from inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1 import (
    ModelMonitoringInferenceAggregatorBlockV1,
)
from inference.core.workflows.core_steps.sinks.slack.notification.v1 import (
    SlackNotificationBlockV1,
)
from inference.core.workflows.core_steps.sinks.twilio.sms.v1 import (
    TwilioSMSNotificationBlockV1,
)
from inference.core.workflows.core_steps.sinks.webhook.v1 import WebhookSinkBlockV1
from inference.core.workflows.core_steps.transformations.absolute_static_crop.v1 import (
    AbsoluteStaticCropBlockV1,
)
from inference.core.workflows.core_steps.transformations.bounding_rect.v1 import (
    BoundingRectBlockV1,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v1 import (
    ByteTrackerBlockV1,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v2 import (
    ByteTrackerBlockV2,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v3 import (
    ByteTrackerBlockV3,
)
from inference.core.workflows.core_steps.transformations.camera_calibration.v1 import (
    CameraCalibrationBlockV1,
)
from inference.core.workflows.core_steps.transformations.detection_offset.v1 import (
    DetectionOffsetBlockV1,
)
from inference.core.workflows.core_steps.transformations.detections_filter.v1 import (
    DetectionsFilterBlockV1,
)
from inference.core.workflows.core_steps.transformations.detections_merge.v1 import (
    DetectionsMergeBlockV1,
)
from inference.core.workflows.core_steps.transformations.detections_transformation.v1 import (
    DetectionsTransformationBlockV1,
)
from inference.core.workflows.core_steps.transformations.dynamic_crop.v1 import (
    DynamicCropBlockV1,
)
from inference.core.workflows.core_steps.transformations.dynamic_zones.v1 import (
    DynamicZonesBlockV1,
)
from inference.core.workflows.core_steps.transformations.image_slicer.v1 import (
    ImageSlicerBlockV1,
)
from inference.core.workflows.core_steps.transformations.image_slicer.v2 import (
    ImageSlicerBlockV2,
)
from inference.core.workflows.core_steps.transformations.perspective_correction.v1 import (
    PerspectiveCorrectionBlockV1,
)
from inference.core.workflows.core_steps.transformations.relative_static_crop.v1 import (
    RelativeStaticCropBlockV1,
)
from inference.core.workflows.core_steps.transformations.stabilize_detections.v1 import (
    StabilizeTrackedDetectionsBlockV1,
)
from inference.core.workflows.core_steps.transformations.stitch_images.v1 import (
    StitchImagesBlockV1,
)
from inference.core.workflows.core_steps.transformations.stitch_ocr_detections.v1 import (
    StitchOCRDetectionsBlockV1,
)

# Visualizers
from inference.core.workflows.core_steps.visualizations.background_color.v1 import (
    BackgroundColorVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.blur.v1 import (
    BlurVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.bounding_box.v1 import (
    BoundingBoxVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.circle.v1 import (
    CircleVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.classification_label.v1 import (
    ClassificationLabelVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.color.v1 import (
    ColorVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.corner.v1 import (
    CornerVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.crop.v1 import (
    CropVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.dot.v1 import (
    DotVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.ellipse.v1 import (
    EllipseVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.grid.v1 import (
    GridVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.halo.v1 import (
    HaloVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.keypoint.v1 import (
    KeypointVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.label.v1 import (
    LabelVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.line_zone.v1 import (
    LineCounterZoneVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.mask.v1 import (
    MaskVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.model_comparison.v1 import (
    ModelComparisonVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.pixelate.v1 import (
    PixelateVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.polygon.v1 import (
    PolygonVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.polygon_zone.v1 import (
    PolygonZoneVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.reference_path.v1 import (
    ReferencePathVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.trace.v1 import (
    TraceVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.triangle.v1 import (
    TriangleVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.types import (
    BAR_CODE_DETECTION_KIND,
    BOOLEAN_KIND,
    BYTES_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    CONTOURS_KIND,
    DETECTION_KIND,
    DICTIONARY_KIND,
    EMBEDDING_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KEYPOINTS_KIND,
    IMAGE_KIND,
    IMAGE_METADATA_KIND,
    INFERENCE_ID_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    LIST_OF_VALUES_KIND,
    NUMPY_ARRAY_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PARENT_ID_KIND,
    POINT_KIND,
    PREDICTION_TYPE_KIND,
    QR_CODE_DETECTION_KIND,
    RGB_COLOR_KIND,
    ROBOFLOW_API_KEY_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    SECRET_KIND,
    SERIALISED_PAYLOADS_KIND,
    STRING_KIND,
    TOP_CLASS_KIND,
    VIDEO_METADATA_KIND,
    WILDCARD_KIND,
    ZONE_KIND,
    Kind,
)
from inference.core.workflows.prototypes.block import WorkflowBlock

REGISTERED_INITIALIZERS = {
    "api_key": API_KEY,
    "cache": cache,
    "step_execution_mode": StepExecutionMode(WORKFLOWS_STEP_EXECUTION_MODE),
    "background_tasks": None,
    "thread_pool_executor": None,
    "allow_access_to_file_system": ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE,
    "allowed_write_directory": WORKFLOW_BLOCKS_WRITE_DIRECTORY,
    "allow_access_to_environmental_variables": ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES,
}

KINDS_SERIALIZERS = {
    IMAGE_KIND.name: serialise_image,
    VIDEO_METADATA_KIND.name: serialize_video_metadata_kind,
    OBJECT_DETECTION_PREDICTION_KIND.name: serialise_sv_detections,
    INSTANCE_SEGMENTATION_PREDICTION_KIND.name: serialise_sv_detections,
    KEYPOINT_DETECTION_PREDICTION_KIND.name: serialise_sv_detections,
    QR_CODE_DETECTION_KIND.name: serialise_sv_detections,
    BAR_CODE_DETECTION_KIND.name: serialise_sv_detections,
    SECRET_KIND.name: serialize_secret,
    WILDCARD_KIND.name: serialize_wildcard_kind,
}
KINDS_DESERIALIZERS = {
    IMAGE_KIND.name: deserialize_image_kind,
    VIDEO_METADATA_KIND.name: deserialize_video_metadata_kind,
    OBJECT_DETECTION_PREDICTION_KIND.name: deserialize_detections_kind,
    INSTANCE_SEGMENTATION_PREDICTION_KIND.name: deserialize_detections_kind,
    KEYPOINT_DETECTION_PREDICTION_KIND.name: deserialize_detections_kind,
    QR_CODE_DETECTION_KIND.name: deserialize_detections_kind,
    BAR_CODE_DETECTION_KIND.name: deserialize_detections_kind,
    NUMPY_ARRAY_KIND.name: deserialize_numpy_array,
    ROBOFLOW_MODEL_ID_KIND.name: deserialize_string_kind,
    ROBOFLOW_PROJECT_KIND.name: deserialize_string_kind,
    ROBOFLOW_API_KEY_KIND.name: deserialize_optional_string_kind,
    FLOAT_ZERO_TO_ONE_KIND.name: deserialize_float_zero_to_one_kind,
    LIST_OF_VALUES_KIND.name: deserialize_list_of_values_kind,
    BOOLEAN_KIND.name: deserialize_boolean_kind,
    INTEGER_KIND.name: deserialize_integer_kind,
    STRING_KIND.name: deserialize_string_kind,
    TOP_CLASS_KIND.name: deserialize_string_kind,
    FLOAT_KIND.name: deserialize_float_kind,
    DICTIONARY_KIND.name: deserialize_dictionary_kind,
    CLASSIFICATION_PREDICTION_KIND.name: deserialize_classification_prediction_kind,
    POINT_KIND.name: deserialize_point_kind,
    ZONE_KIND.name: deserialize_zone_kind,
    RGB_COLOR_KIND.name: deserialize_rgb_color_kind,
    LANGUAGE_MODEL_OUTPUT_KIND.name: deserialize_string_kind,
    PREDICTION_TYPE_KIND.name: deserialize_string_kind,
    PARENT_ID_KIND.name: deserialize_string_kind,
    BYTES_KIND.name: deserialize_bytes_kind,
    INFERENCE_ID_KIND.name: deserialize_string_kind,
}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        AbsoluteStaticCropBlockV1,
        DynamicCropBlockV1,
        DetectionsFilterBlockV1,
        DetectionOffsetBlockV1,
        ByteTrackerBlockV1,
        RelativeStaticCropBlockV1,
        DetectionsTransformationBlockV1,
        RoboflowDatasetUploadBlockV1,
        ContinueIfBlockV1,
        RateLimiterBlockV1,
        PerspectiveCorrectionBlockV1,
        DeltaFilterBlockV1,
        CameraCalibrationBlockV1,
        DynamicZonesBlockV1,
        SizeMeasurementBlockV1,
        BufferBlockV1,
        DetectionsClassesReplacementBlockV1,
        ExpressionBlockV1,
        PropertyDefinitionBlockV1,
        DimensionCollapseBlockV1,
        FirstNonEmptyOrDefaultBlockV1,
        AnthropicClaudeBlockV1,
        CosineSimilarityBlockV1,
        BackgroundColorVisualizationBlockV1,
        BarcodeDetectorBlockV1,
        BlurVisualizationBlockV1,
        BoundingBoxVisualizationBlockV1,
        BoundingRectBlockV1,
        DetectionsMergeBlockV1,
        ByteTrackerBlockV2,
        CacheGetBlockV1,
        CacheSetBlockV1,
        CameraFocusBlockV1,
        CircleVisualizationBlockV1,
        ClipComparisonBlockV1,
        ClipComparisonBlockV2,
        ClipModelBlockV1,
        CogVLMBlockV1,
        ColorVisualizationBlockV1,
        ConvertGrayscaleBlockV1,
        CornerVisualizationBlockV1,
        CropVisualizationBlockV1,
        DetectionsConsensusBlockV1,
        DetectionsStitchBlockV1,
        DistanceMeasurementBlockV1,
        DominantColorBlockV1,
        DotVisualizationBlockV1,
        EllipseVisualizationBlockV1,
        Florence2BlockV1,
        Florence2BlockV2,
        GoogleGeminiBlockV1,
        GoogleVisionOCRBlockV1,
        GridVisualizationBlockV1,
        HaloVisualizationBlockV1,
        ImageBlurBlockV1,
        ImageContoursDetectionBlockV1,
        ImagePreprocessingBlockV1,
        ImageSlicerBlockV1,
        ImageThresholdBlockV1,
        JSONParserBlockV1,
        LMMBlockV1,
        LMMForClassificationBlockV1,
        LabelVisualizationBlockV1,
        ClassificationLabelVisualizationBlockV1,
        LineCounterBlockV1,
        LineCounterBlockV2,
        LineCounterZoneVisualizationBlockV1,
        MaskVisualizationBlockV1,
        ModelComparisonVisualizationBlockV1,
        OCRModelBlockV1,
        OpenAIBlockV1,
        OpenAIBlockV2,
        PathDeviationAnalyticsBlockV1,
        PathDeviationAnalyticsBlockV2,
        PixelateVisualizationBlockV1,
        PixelationCountBlockV1,
        PolygonVisualizationBlockV1,
        PolygonZoneVisualizationBlockV1,
        QRCodeDetectorBlockV1,
        RoboflowClassificationModelBlockV1,
        RoboflowCustomMetadataBlockV1,
        ModelMonitoringInferenceAggregatorBlockV1,
        RoboflowDatasetUploadBlockV2,
        RoboflowInstanceSegmentationModelBlockV1,
        RoboflowKeypointDetectionModelBlockV1,
        RoboflowMultiLabelClassificationModelBlockV1,
        RoboflowObjectDetectionModelBlockV1,
        SIFTBlockV1,
        SIFTComparisonBlockV1,
        SIFTComparisonBlockV2,
        SegmentAnything2BlockV1,
        StabilityAIInpaintingBlockV1,
        StabilityAIImageGenBlockV1,
        StabilizeTrackedDetectionsBlockV1,
        StitchImagesBlockV1,
        StitchOCRDetectionsBlockV1,
        TemplateMatchingBlockV1,
        TimeInZoneBlockV1,
        TimeInZoneBlockV2,
        TriangleVisualizationBlockV1,
        VLMAsClassifierBlockV1,
        VLMAsDetectorBlockV1,
        YoloWorldModelBlockV1,
        KeypointVisualizationBlockV1,
        DataAggregatorBlockV1,
        CSVFormatterBlockV1,
        EmailNotificationBlockV1,
        LocalFileSinkBlockV1,
        TraceVisualizationBlockV1,
        ReferencePathVisualizationBlockV1,
        ByteTrackerBlockV3,
        WebhookSinkBlockV1,
        VelocityBlockV1,
        RoboflowInstanceSegmentationModelBlockV2,
        RoboflowKeypointDetectionModelBlockV2,
        RoboflowClassificationModelBlockV2,
        RoboflowMultiLabelClassificationModelBlockV2,
        RoboflowObjectDetectionModelBlockV2,
        VLMAsClassifierBlockV2,
        VLMAsDetectorBlockV2,
        IdentifyOutliersBlockV1,
        IdentifyChangesBlockV1,
        EnvironmentSecretsStoreBlockV1,
        SlackNotificationBlockV1,
        TwilioSMSNotificationBlockV1,
        GazeBlockV1,
        LlamaVisionBlockV1,
        ImageSlicerBlockV2,
        Qwen25VLBlockV1,
    ]


def load_kinds() -> List[Kind]:
    return [
        WILDCARD_KIND,
        IMAGE_KIND,
        VIDEO_METADATA_KIND,
        ROBOFLOW_MODEL_ID_KIND,
        ROBOFLOW_PROJECT_KIND,
        ROBOFLOW_API_KEY_KIND,
        FLOAT_ZERO_TO_ONE_KIND,
        LIST_OF_VALUES_KIND,
        SERIALISED_PAYLOADS_KIND,
        BOOLEAN_KIND,
        INTEGER_KIND,
        STRING_KIND,
        TOP_CLASS_KIND,
        FLOAT_KIND,
        DICTIONARY_KIND,
        DETECTION_KIND,
        CLASSIFICATION_PREDICTION_KIND,
        POINT_KIND,
        ZONE_KIND,
        OBJECT_DETECTION_PREDICTION_KIND,
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        KEYPOINT_DETECTION_PREDICTION_KIND,
        RGB_COLOR_KIND,
        IMAGE_KEYPOINTS_KIND,
        CONTOURS_KIND,
        LANGUAGE_MODEL_OUTPUT_KIND,
        NUMPY_ARRAY_KIND,
        QR_CODE_DETECTION_KIND,
        BAR_CODE_DETECTION_KIND,
        PREDICTION_TYPE_KIND,
        PARENT_ID_KIND,
        IMAGE_METADATA_KIND,
        BYTES_KIND,
        INFERENCE_ID_KIND,
        SECRET_KIND,
        EMBEDDING_KIND,
    ]
