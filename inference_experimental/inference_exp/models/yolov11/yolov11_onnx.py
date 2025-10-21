from inference_exp.models.yolov8.yolov8_classification_onnx import (
    YOLOv8ForClassificationOnnx,
)
from inference_exp.models.yolov8.yolov8_instance_segmentation_onnx import (
    YOLOv8ForInstanceSegmentationOnnx,
)
from inference_exp.models.yolov8.yolov8_key_points_detection_onnx import (
    YOLOv8ForKeyPointsDetectionOnnx,
)
from inference_exp.models.yolov8.yolov8_object_detection_onnx import (
    YOLOv8ForObjectDetectionOnnx,
)


class YOLOv11ForObjectDetectionOnnx(YOLOv8ForObjectDetectionOnnx):
    pass


class YOLOv11ForInstanceSegmentationOnnx(YOLOv8ForInstanceSegmentationOnnx):
    pass


class YOLOv11ForForKeyPointsDetectionOnnx(YOLOv8ForKeyPointsDetectionOnnx):
    pass


class YOLOv11ForClassificationOnnx(YOLOv8ForClassificationOnnx):
    pass
