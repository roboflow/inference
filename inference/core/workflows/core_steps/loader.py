from typing import List, Type

from inference.core.cache import cache
from inference.core.env import API_KEY, WORKFLOWS_STEP_EXECUTION_MODE
from inference.core.workflows.core_steps.analytics.line_counter.v1 import (
    LineCounterBlockV1,
)
from inference.core.workflows.core_steps.analytics.path_deviation.v1 import (
    PathDeviationAnalyticsBlockV1,
)
from inference.core.workflows.core_steps.analytics.time_in_zone.v1 import (
    TimeInZoneBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.camera_focus.v1 import (
    CameraFocusBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.contours.v1 import (
    ImageContoursDetectionBlockV1,
)
from inference.core.workflows.core_steps.classical_cv.convert_grayscale.v1 import (
    ConvertGrayscaleBlockV1,
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
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.flow_control.continue_if.v1 import (
    ContinueIfBlockV1,
)
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
from inference.core.workflows.core_steps.formatters.vlm_as_detector.v1 import (
    VLMAsDetectorBlockV1,
)
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
from inference.core.workflows.core_steps.models.foundation.anthropic_claude.v1 import (
    AntropicClaudeBlockV1,
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
from inference.core.workflows.core_steps.models.foundation.google_gemini.v1 import (
    GoogleGeminiBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.google_vision_ocr.v1 import (
    GoogleVisionOCRBlockV1,
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
from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
    SegmentAnything2BlockV1,
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
from inference.core.workflows.core_steps.models.roboflow.keypoint_detection.v1 import (
    RoboflowKeypointDetectionModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v1 import (
    RoboflowClassificationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v1 import (
    RoboflowMultiLabelClassificationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.v1 import (
    RoboflowObjectDetectionModelBlockV1,
)
from inference.core.workflows.core_steps.models.third_party.barcode_detection.v1 import (
    BarcodeDetectorBlockV1,
)
from inference.core.workflows.core_steps.models.third_party.qr_code_detection.v1 import (
    QRCodeDetectorBlockV1,
)
from inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1 import (
    RoboflowCustomMetadataBlockV1,
)
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v1 import (
    RoboflowDatasetUploadBlockV1,
)
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v2 import (
    RoboflowDatasetUploadBlockV2,
)
from inference.core.workflows.core_steps.transformations.absolute_static_crop.v1 import (
    AbsoluteStaticCropBlockV1,
)
from inference.core.workflows.core_steps.transformations.bounding_rect.v1 import (
    BoundingRectBlockV1,
)
from inference.core.workflows.core_steps.transformations.byte_tracker.v1 import (
    ByteTrackerBlockV1,
)
from inference.core.workflows.core_steps.transformations.detection_offset.v1 import (
    DetectionOffsetBlockV1,
)
from inference.core.workflows.core_steps.transformations.detections_filter.v1 import (
    DetectionsFilterBlockV1,
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
from inference.core.workflows.core_steps.transformations.perspective_correction.v1 import (
    PerspectiveCorrectionBlockV1,
)
from inference.core.workflows.core_steps.transformations.relative_static_crop.v1 import (
    RelativeStaticCropBlockV1,
)
from inference.core.workflows.core_steps.transformations.stitch_images.v1 import (
    StitchImagesBlockV1,
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
from inference.core.workflows.core_steps.visualizations.halo.v1 import (
    HaloVisualizationBlockV1,
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
from inference.core.workflows.core_steps.visualizations.pixelate.v1 import (
    PixelateVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.polygon.v1 import (
    PolygonVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.polygon_zone.v1 import (
    PolygonZoneVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.triangle.v1 import (
    TriangleVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.types import (
    BAR_CODE_DETECTION_KIND,
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    CONTOURS_KIND,
    DETECTION_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KEYPOINTS_KIND,
    IMAGE_KIND,
    IMAGE_METADATA_KIND,
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
}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        TimeInZoneBlockV1,
        BoundingRectBlockV1,
        SegmentAnything2BlockV1,
        DetectionsConsensusBlockV1,
        ClipComparisonBlockV1,
        LMMBlockV1,
        LMMForClassificationBlockV1,
        OpenAIBlockV1,
        CogVLMBlockV1,
        OCRModelBlockV1,
        YoloWorldModelBlockV1,
        RoboflowInstanceSegmentationModelBlockV1,
        RoboflowKeypointDetectionModelBlockV1,
        RoboflowClassificationModelBlockV1,
        RoboflowMultiLabelClassificationModelBlockV1,
        RoboflowObjectDetectionModelBlockV1,
        BarcodeDetectorBlockV1,
        QRCodeDetectorBlockV1,
        AbsoluteStaticCropBlockV1,
        DynamicCropBlockV1,
        DetectionsFilterBlockV1,
        DetectionOffsetBlockV1,
        ByteTrackerBlockV1,
        RelativeStaticCropBlockV1,
        DetectionsTransformationBlockV1,
        RoboflowDatasetUploadBlockV1,
        ContinueIfBlockV1,
        PerspectiveCorrectionBlockV1,
        DynamicZonesBlockV1,
        SizeMeasurementBlockV1,
        DetectionsClassesReplacementBlockV1,
        ExpressionBlockV1,
        PropertyDefinitionBlockV1,
        DimensionCollapseBlockV1,
        FirstNonEmptyOrDefaultBlockV1,
        BackgroundColorVisualizationBlockV1,
        BlurVisualizationBlockV1,
        BoundingBoxVisualizationBlockV1,
        CircleVisualizationBlockV1,
        ColorVisualizationBlockV1,
        CornerVisualizationBlockV1,
        CropVisualizationBlockV1,
        DotVisualizationBlockV1,
        EllipseVisualizationBlockV1,
        HaloVisualizationBlockV1,
        LabelVisualizationBlockV1,
        MaskVisualizationBlockV1,
        PixelateVisualizationBlockV1,
        PolygonVisualizationBlockV1,
        LineCounterZoneVisualizationBlockV1,
        TriangleVisualizationBlockV1,
        RoboflowCustomMetadataBlockV1,
        DetectionsStitchBlockV1,
        ImageSlicerBlockV1,
        DominantColorBlockV1,
        PixelationCountBlockV1,
        SIFTComparisonBlockV1,
        SIFTComparisonBlockV2,
        SIFTBlockV1,
        TemplateMatchingBlockV1,
        ImageBlurBlockV1,
        ConvertGrayscaleBlockV1,
        ImageThresholdBlockV1,
        ImageContoursDetectionBlockV1,
        ClipComparisonBlockV2,
        CameraFocusBlockV1,
        RoboflowDatasetUploadBlockV2,
        StitchImagesBlockV1,
        OpenAIBlockV2,
        JSONParserBlockV1,
        VLMAsClassifierBlockV1,
        GoogleGeminiBlockV1,
        GoogleVisionOCRBlockV1,
        VLMAsDetectorBlockV1,
        AntropicClaudeBlockV1,
        LineCounterBlockV1,
        PolygonZoneVisualizationBlockV1,
        Florence2BlockV1,
        StabilityAIInpaintingBlockV1,
        ImagePreprocessingBlockV1,
        PathDeviationAnalyticsBlockV1,
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
    ]
