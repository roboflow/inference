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
    RoboflowInstanceSegmentationModelBlock,
)
from inference.enterprise.workflows.core_steps.models.roboflow.keypoint_detection import (
    RoboflowKeypointDetectionModelBlock,
)
from inference.enterprise.workflows.core_steps.models.roboflow.multi_class_classification import (
    RoboflowClassificationModelBlock,
)
from inference.enterprise.workflows.core_steps.models.roboflow.multi_label_classification import (
    RoboflowMultiLabelClassificationModelBlock,
)
from inference.enterprise.workflows.core_steps.models.roboflow.object_detection import (
    RoboflowObjectDetectionModelBlock,
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
        RoboflowInstanceSegmentationModelBlock,
        RoboflowKeypointDetectionModelBlock,
        RoboflowClassificationModelBlock,
        RoboflowMultiLabelClassificationModelBlock,
        RoboflowObjectDetectionModelBlock,
        BarcodeDetectorBlock,
        QRCodeDetectorBlock,
        ActiveLearningDataCollectorBlock,
        AbsoluteStaticCropBlock,
        CropBlock,
        DetectionFilterBlock,
        DetectionOffsetBlock,
        RelativeStaticCropBlock,
    ]
