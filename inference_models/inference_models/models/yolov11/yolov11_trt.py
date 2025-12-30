from inference_models.models.yolov8.yolov8_instance_segmentation_trt import (
    YOLOv8ForInstanceSegmentationTRT,
)
from inference_models.models.yolov8.yolov8_key_points_detection_trt import (
    YOLOv8ForKeyPointsDetectionTRT,
)
from inference_models.models.yolov8.yolov8_object_detection_trt import (
    YOLOv8ForObjectDetectionTRT,
)


class YOLOv11ForObjectDetectionTRT(YOLOv8ForObjectDetectionTRT):
    pass


class YOLOv11ForInstanceSegmentationTRT(YOLOv8ForInstanceSegmentationTRT):
    pass


class YOLOv11ForForKeyPointsDetectionTRT(YOLOv8ForKeyPointsDetectionTRT):
    pass
