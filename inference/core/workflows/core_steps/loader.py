from typing import List, Type

from inference.core.cache import cache
from inference.core.env import API_KEY, WORKFLOWS_STEP_EXECUTION_MODE
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
from inference.core.workflows.core_steps.formatters.property_definition.v1 import (
    PropertyDefinitionBlockV1,
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
from inference.core.workflows.core_steps.models.foundation.clip_comparison.v1 import (
    ClipComparisonBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.cog_vlm.v1 import (
    CogVLMBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.lmm.v1 import LMMBlockV1
from inference.core.workflows.core_steps.models.foundation.lmm_classifier.v1 import (
    LMMForClassificationBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.ocr.v1 import OCRModelBlockV1
from inference.core.workflows.core_steps.models.foundation.openai.v1 import (
    OpenAIBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
    SegmentAnything2BlockV1,
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
from inference.core.workflows.core_steps.transformations.absolute_static_crop.v1 import (
    AbsoluteStaticCropBlockV1,
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
from inference.core.workflows.core_steps.visualizations.background_color.v1 import (
    BackgroundColorVisualizationBlockV1,
)

# Visualizers
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
from inference.core.workflows.core_steps.visualizations.mask.v1 import (
    MaskVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.pixelate.v1 import (
    PixelateVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.polygon.v1 import (
    PolygonVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.triangle.v1 import (
    TriangleVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_BAR_CODE_DETECTION_KIND,
    BATCH_OF_BOOLEAN_KIND,
    BATCH_OF_CLASSIFICATION_PREDICTION_KIND,
    BATCH_OF_DICTIONARY_KIND,
    BATCH_OF_IMAGE_METADATA_KIND,
    BATCH_OF_IMAGES_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_PARENT_ID_KIND,
    BATCH_OF_PREDICTION_TYPE_KIND,
    BATCH_OF_QR_CODE_DETECTION_KIND,
    BATCH_OF_SERIALISED_PAYLOADS_KIND,
    BATCH_OF_STRING_KIND,
    BATCH_OF_TOP_CLASS_KIND,
    BOOLEAN_KIND,
    DETECTION_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    POINT_KIND,
    ROBOFLOW_API_KEY_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
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
        RelativeStaticCropBlockV1,
        DetectionsTransformationBlockV1,
        RoboflowDatasetUploadBlockV1,
        ContinueIfBlockV1,
        PerspectiveCorrectionBlockV1,
        DynamicZonesBlockV1,
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
        TriangleVisualizationBlockV1,
        RoboflowCustomMetadataBlockV1,
        DetectionsStitchBlockV1,
        ImageSlicerBlockV1,
    ]


def load_kinds() -> List[Kind]:
    return [
        WILDCARD_KIND,
        IMAGE_KIND,
        BATCH_OF_IMAGES_KIND,
        ROBOFLOW_MODEL_ID_KIND,
        ROBOFLOW_PROJECT_KIND,
        ROBOFLOW_API_KEY_KIND,
        FLOAT_ZERO_TO_ONE_KIND,
        LIST_OF_VALUES_KIND,
        BATCH_OF_SERIALISED_PAYLOADS_KIND,
        BOOLEAN_KIND,
        BATCH_OF_BOOLEAN_KIND,
        INTEGER_KIND,
        STRING_KIND,
        BATCH_OF_STRING_KIND,
        BATCH_OF_TOP_CLASS_KIND,
        FLOAT_KIND,
        DICTIONARY_KIND,
        BATCH_OF_DICTIONARY_KIND,
        BATCH_OF_CLASSIFICATION_PREDICTION_KIND,
        DETECTION_KIND,
        POINT_KIND,
        ZONE_KIND,
        OBJECT_DETECTION_PREDICTION_KIND,
        BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        KEYPOINT_DETECTION_PREDICTION_KIND,
        BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
        BATCH_OF_QR_CODE_DETECTION_KIND,
        BATCH_OF_BAR_CODE_DETECTION_KIND,
        BATCH_OF_PREDICTION_TYPE_KIND,
        BATCH_OF_PARENT_ID_KIND,
        BATCH_OF_IMAGE_METADATA_KIND,
    ]
