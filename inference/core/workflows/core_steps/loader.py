from typing import List, Type

from inference.core.workflows.core_steps.flow_control.continue_if import ContinueIfBlock
from inference.core.workflows.core_steps.fusion.detections_consensus import (
    DetectionsConsensusBlock,
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
from inference.core.workflows.prototypes.block import WorkflowBlock


def load_blocks() -> List[Type[WorkflowBlock]]:
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
    ]
