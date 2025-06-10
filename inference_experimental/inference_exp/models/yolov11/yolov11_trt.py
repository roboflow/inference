from inference_exp.models.yolov8 import (
    YOLOv8ForInstanceSegmentationTRT,
)
from inference_exp.models.yolov8 import (
    YOLOv8ForKeyPointsDetectionTRT,
)
from inference_exp.models.yolov8 import (
    YOLOv8ForObjectDetectionTRT,
)


class YOLOv11ForObjectDetectionTRT(YOLOv8ForObjectDetectionTRT):
    pass


class YOLOv11ForInstanceSegmentationTRT(YOLOv8ForInstanceSegmentationTRT):
    pass


class YOLOv11ForForKeyPointsDetectionTRT(YOLOv8ForKeyPointsDetectionTRT):
    pass
