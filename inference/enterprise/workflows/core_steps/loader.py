from inference.enterprise.workflows.core_steps.flow_control.condition import (
    ConditionBlock,
)
from inference.enterprise.workflows.core_steps.fusion.detections_consensus import (
    DetectionsConsensusBlock,
)
from inference.enterprise.workflows.core_steps.models.foundation.clip_comparison import (
    ClipComparisonBlock,
)
from inference.enterprise.workflows.core_steps.models.foundation.lmm import LMMBlock
from inference.enterprise.workflows.core_steps.models.foundation.lmm_classifier import (
    LMMForClassificationBlock,
)
from inference.enterprise.workflows.core_steps.models.foundation.ocr import (
    OCRModelBlock,
)
from inference.enterprise.workflows.core_steps.models.foundation.yolo_world import (
    YoloWorldModelBlock,
)
from inference.enterprise.workflows.core_steps.models.roboflow.instance_segmentation import (
    RoboflowInstanceSegmentationBlock,
)
from inference.enterprise.workflows.core_steps.models.roboflow.keypoint_detection import (
    RoboflowKeypointDetectionBlock,
)
from inference.enterprise.workflows.core_steps.models.roboflow.multi_class_classification import (
    RoboflowClassificationBlock,
)
from inference.enterprise.workflows.core_steps.models.roboflow.multi_label_classification import (
    RoboflowMultiLabelClassificationBlock,
)
from inference.enterprise.workflows.core_steps.models.roboflow.object_detection import (
    RoboflowObjectDetectionBlock,
)
from inference.enterprise.workflows.core_steps.models.third_party.barcode_detection import (
    BarcodeDetectorBlock,
)
from inference.enterprise.workflows.core_steps.models.third_party.qr_code_detection import (
    QRCodeDetectorBlock,
)
from inference.enterprise.workflows.core_steps.sinks.active_learning.data_collector import (
    ActiveLearningDataCollectorBlock,
)
from inference.enterprise.workflows.core_steps.transformations.absolute_static_crop import (
    AbsoluteStaticCropBlock,
)
from inference.enterprise.workflows.core_steps.transformations.crop import CropBlock
from inference.enterprise.workflows.core_steps.transformations.detection_filter import (
    DetectionFilterBlock,
)
from inference.enterprise.workflows.core_steps.transformations.detection_offset import (
    DetectionOffsetBlock,
)
from inference.enterprise.workflows.core_steps.transformations.relative_static_crop import (
    RelativeStaticCropBlock,
)


def load_blocks_classes() -> list:
    return [
        ConditionBlock,
        DetectionsConsensusBlock,
        ClipComparisonBlock,
        LMMBlock,
        LMMForClassificationBlock,
        OCRModelBlock,
        YoloWorldModelBlock,
        RoboflowInstanceSegmentationBlock,
        RoboflowKeypointDetectionBlock,
        RoboflowClassificationBlock,
        RoboflowMultiLabelClassificationBlock,
        RoboflowObjectDetectionBlock,
        BarcodeDetectorBlock,
        QRCodeDetectorBlock,
        ActiveLearningDataCollectorBlock,
        AbsoluteStaticCropBlock,
        CropBlock,
        DetectionFilterBlock,
        DetectionOffsetBlock,
        RelativeStaticCropBlock,
    ]
