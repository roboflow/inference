from typing import Callable, List, Tuple, Type, Union

from inference.core.cache import cache
from inference.core.env import API_KEY, WORKFLOWS_STEP_EXECUTION_MODE
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.flow_control.continue_if import ContinueIfBlock
from inference.core.workflows.core_steps.formatters.expression import ExpressionBlock
from inference.core.workflows.core_steps.formatters.first_non_empty_or_default import (
    FirstNonEmptyOrDefaultBlock,
)
from inference.core.workflows.core_steps.formatters.property_definition import (
    PropertyDefinitionBlock,
)
from inference.core.workflows.core_steps.fusion.detections_classes_replacement import (
    DetectionsClassesReplacementBlock,
)
from inference.core.workflows.core_steps.fusion.detections_consensus import (
    DetectionsConsensusBlock,
)
from inference.core.workflows.core_steps.fusion.dimension_collapse import (
    DimensionCollapseBlock,
)
from inference.core.workflows.core_steps.models.foundation.clip_comparison import (
    ClipComparisonBlock,
)
from inference.core.workflows.core_steps.models.foundation.lmm import LMMBlock
from inference.core.workflows.core_steps.models.foundation.lmm_classifier import (
    LMMForClassificationBlock,
)
from inference.core.workflows.core_steps.models.foundation.ocr import OCRModelBlock
from inference.core.workflows.core_steps.models.foundation.yolo_world import (
    YoloWorldModelBlock,
)
from inference.core.workflows.core_steps.models.roboflow.instance_segmentation import (
    RoboflowInstanceSegmentationModelBlock,
)
from inference.core.workflows.core_steps.models.roboflow.keypoint_detection import (
    RoboflowKeypointDetectionModelBlock,
)
from inference.core.workflows.core_steps.models.roboflow.multi_class_classification import (
    RoboflowClassificationModelBlock,
)
from inference.core.workflows.core_steps.models.roboflow.multi_label_classification import (
    RoboflowMultiLabelClassificationModelBlock,
)
from inference.core.workflows.core_steps.models.roboflow.object_detection import (
    RoboflowObjectDetectionModelBlock,
)
from inference.core.workflows.core_steps.models.third_party.barcode_detection import (
    BarcodeDetectorBlock,
)
from inference.core.workflows.core_steps.models.third_party.qr_code_detection import (
    QRCodeDetectorBlock,
)
from inference.core.workflows.core_steps.sinks.roboflow.roboflow_dataset_upload import (
    RoboflowDatasetUploadBlock,
)
from inference.core.workflows.core_steps.transformations.absolute_static_crop import (
    AbsoluteStaticCropBlock,
)
from inference.core.workflows.core_steps.transformations.detection_offset import (
    DetectionOffsetBlock,
)
from inference.core.workflows.core_steps.transformations.detections_filter import (
    DetectionsFilterBlock,
)
from inference.core.workflows.core_steps.transformations.detections_transformation import (
    DetectionsTransformationBlock,
)
from inference.core.workflows.core_steps.transformations.dynamic_crop import (
    DynamicCropBlock,
)
from inference.core.workflows.core_steps.transformations.dynamic_zones import (
    DynamicZonesBlock,
)
from inference.core.workflows.core_steps.transformations.perspective_correction import (
    PerspectiveCorrectionBlock,
)
from inference.core.workflows.core_steps.transformations.relative_static_crop import (
    RelativeStaticCropBlock,
)

# Visualizers
from inference.core.workflows.core_steps.visualizations.blur import (
    BlurVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.bounding_box import (
    BoundingBoxVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.circle import (
    CircleVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.color import (
    ColorVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.corner import (
    CornerVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.crop import (
    CropVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.dot import DotVisualizationBlock
from inference.core.workflows.core_steps.visualizations.ellipse import (
    EllipseVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.halo import (
    HaloVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.label import (
    LabelVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.mask import (
    MaskVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.pixelate import (
    PixelateVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.polygon import (
    PolygonVisualizationBlock,
)
from inference.core.workflows.core_steps.visualizations.triangle import (
    TriangleVisualizationBlock,
)
from inference.core.workflows.entities.types import (
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
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

REGISTERED_INITIALIZERS = {
    "api_key": API_KEY,
    "cache": cache,
    "step_execution_mode": StepExecutionMode(WORKFLOWS_STEP_EXECUTION_MODE),
}


def load_blocks() -> List[
    Union[
        Type[WorkflowBlock],
        Tuple[
            Type[WorkflowBlockManifest],
            Callable[[Type[WorkflowBlockManifest]], WorkflowBlock],
        ],
    ]
]:
    return [
        DetectionsConsensusBlock,
        ClipComparisonBlock,
        LMMBlock,
        LMMForClassificationBlock,
        OCRModelBlock,
        YoloWorldModelBlock,
        RoboflowInstanceSegmentationModelBlock,
        RoboflowKeypointDetectionModelBlock,
        RoboflowClassificationModelBlock,
        RoboflowMultiLabelClassificationModelBlock,
        RoboflowObjectDetectionModelBlock,
        BarcodeDetectorBlock,
        QRCodeDetectorBlock,
        AbsoluteStaticCropBlock,
        DynamicCropBlock,
        DetectionsFilterBlock,
        DetectionOffsetBlock,
        RelativeStaticCropBlock,
        DetectionsTransformationBlock,
        RoboflowDatasetUploadBlock,
        ContinueIfBlock,
        PerspectiveCorrectionBlock,
        DynamicZonesBlock,
        DetectionsClassesReplacementBlock,
        ExpressionBlock,
        PropertyDefinitionBlock,
        DimensionCollapseBlock,
        FirstNonEmptyOrDefaultBlock,
        BlurVisualizationBlock,
        BoundingBoxVisualizationBlock,
        CircleVisualizationBlock,
        ColorVisualizationBlock,
        CornerVisualizationBlock,
        CropVisualizationBlock,
        DotVisualizationBlock,
        EllipseVisualizationBlock,
        HaloVisualizationBlock,
        LabelVisualizationBlock,
        MaskVisualizationBlock,
        PixelateVisualizationBlock,
        PolygonVisualizationBlock,
        TriangleVisualizationBlock,
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
