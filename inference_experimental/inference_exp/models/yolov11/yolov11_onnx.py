from inference_exp.models.yolov8 import (
    YOLOv8ForInstanceSegmentationOnnx,
)
from inference_exp.models.yolov8 import (
    YOLOv8ForKeyPointsDetectionOnnx,
)
from inference_exp.models.yolov8 import (
    YOLOv8ForObjectDetectionOnnx,
)


class YOLOv11ForObjectDetectionOnnx(YOLOv8ForObjectDetectionOnnx):
    pass


class YOLOv11ForInstanceSegmentationOnnx(YOLOv8ForInstanceSegmentationOnnx):
    pass


class YOLOv11ForForKeyPointsDetectionOnnx(YOLOv8ForKeyPointsDetectionOnnx):
    pass
