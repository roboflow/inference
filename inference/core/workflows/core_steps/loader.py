from typing import List, Type

from inference.core.cache import cache
from inference.core.env import API_KEY, WORKFLOWS_STEP_EXECUTION_MODE
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.flow_control.continue_if.version_1 import (
    ContinueIfBlockV1,
)
from inference.core.workflows.core_steps.formatters.expression.version_1 import (
    ExpressionBlockV1,
)
from inference.core.workflows.core_steps.formatters.first_non_empty_or_default.version_1 import (
    FirstNonEmptyOrDefaultBlockV1,
)
from inference.core.workflows.core_steps.formatters.property_definition.version_1 import (
    PropertyDefinitionBlockV1,
)
from inference.core.workflows.core_steps.fusion.detections_classes_replacement.version_1 import (
    DetectionsClassesReplacementBlockV1,
)
from inference.core.workflows.core_steps.fusion.detections_consensus.version_1 import (
    DetectionsConsensusBlockV1,
)
from inference.core.workflows.core_steps.fusion.dimension_collapse.version_1 import (
    DimensionCollapseBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.clip_comparison.version_1 import (
    ClipComparisonBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.cog_vlm.version_1 import (
    CogVLMBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.lmm.version_1 import (
    LMMBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.lmm_classifier.version_1 import (
    LMMForClassificationBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.ocr.version_1 import (
    OCRModelBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.openai.version_1 import (
    OpenAIBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.yolo_world.version_1 import (
    YoloWorldModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.version_1 import (
    RoboflowInstanceSegmentationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.keypoint_detection.version_1 import (
    RoboflowKeypointDetectionModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.multi_class_classification.version_1 import (
    RoboflowClassificationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.multi_label_classification.version_1 import (
    RoboflowMultiLabelClassificationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection.version_1 import (
    RoboflowObjectDetectionModelBlockV1,
)
from inference.core.workflows.core_steps.models.third_party.barcode_detection.version_1 import (
    BarcodeDetectorBlockV1,
)
from inference.core.workflows.core_steps.models.third_party.qr_code_detection.version_1 import (
    QRCodeDetectorBlockV1,
)
from inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.version_1 import (
    RoboflowCustomMetadataBlockV1,
)
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.version_1 import (
    RoboflowDatasetUploadBlockV1,
)
from inference.core.workflows.core_steps.transformations.absolute_static_crop.version_1 import (
    AbsoluteStaticCropBlockV1,
)
from inference.core.workflows.core_steps.transformations.detection_offset.version_1 import (
    DetectionOffsetBlockV1,
)
from inference.core.workflows.core_steps.transformations.detections_filter.version_1 import (
    DetectionsFilterBlockV1,
)
from inference.core.workflows.core_steps.transformations.detections_transformation.version_1 import (
    DetectionsTransformationBlockV1,
)
from inference.core.workflows.core_steps.transformations.dynamic_crop.version_1 import (
    DynamicCropBlockV1,
)
from inference.core.workflows.core_steps.transformations.dynamic_zones.version_1 import (
    DynamicZonesBlockV1,
)
from inference.core.workflows.core_steps.transformations.perspective_correction.version_1 import (
    PerspectiveCorrectionBlockV1,
)
from inference.core.workflows.core_steps.transformations.relative_static_crop.version_1 import (
    RelativeStaticCropBlockV1,
)
from inference.core.workflows.core_steps.visualizations.background_color.version_1 import (
    BackgroundColorVisualizationBlockV1,
)

# Visualizers
from inference.core.workflows.core_steps.visualizations.blur.version_1 import (
    BlurVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.bounding_box.version_1 import (
    BoundingBoxVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.circle.version_1 import (
    CircleVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.color.version_1 import (
    ColorVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.corner.version_1 import (
    CornerVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.crop.version_1 import (
    CropVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.dot.version_1 import (
    DotVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.ellipse.version_1 import (
    EllipseVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.halo.version_1 import (
    HaloVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.label.version_1 import (
    LabelVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.mask.version_1 import (
    MaskVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.pixelate.version_1 import (
    PixelateVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.polygon.version_1 import (
    PolygonVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.triangle.version_1 import (
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
