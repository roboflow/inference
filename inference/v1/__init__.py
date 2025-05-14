import os
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# PUBLIC INTERFACE LISTED BELOW
from .models.base.classification import (
    ClassificationModel,
    ClassificationPrediction,
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
)
from .models.base.instance_segmentation import (
    InstanceDetections,
    InstanceSegmentationModel,
)
from .models.base.keypoints_detection import KeyPoints, KeyPointsDetectionModel
from .models.base.object_detection import Detections, ObjectDetectionModel
