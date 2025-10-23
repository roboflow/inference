from inference.models.yolov8.yolov8_classification import YOLOv8Classification
from inference.models.yolov8.yolov8_instance_segmentation import (
    YOLOv8InstanceSegmentation,
)
from inference.models.yolov8.yolov8_keypoints_detection import YOLOv8KeypointsDetection

from inference.core.env import USE_INFERENCE_EXP_MODELS

if USE_INFERENCE_EXP_MODELS:
    from inference.models.yolov8.yolov8_object_detection_ext import (
        Yolo8ODExperimentalModel as YOLOv8ObjectDetection,
    )
else:
    from inference.models.yolov8.yolov8_object_detection import YOLOv8ObjectDetection
