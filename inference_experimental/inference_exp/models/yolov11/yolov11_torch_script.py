from inference_exp.models.yolov8.yolov8_instance_segmentation_torch_script import (
    YOLOv8ForInstanceSegmentationTorchScript,
)
from inference_exp.models.yolov8.yolov8_key_points_detection_torch_script import (
    YOLOv8ForKeyPointsDetectionTorchScript,
)
from inference_exp.models.yolov8.yolov8_object_detection_torch_script import (
    YOLOv8ForObjectDetectionTorchScript,
)


class YOLOv11ForObjectDetectionTorchScript(YOLOv8ForObjectDetectionTorchScript):
    pass


class YOLOv11ForInstanceSegmentationTorchScript(
    YOLOv8ForInstanceSegmentationTorchScript
):
    pass


class YOLOv11ForForKeyPointsDetectionTorchScript(
    YOLOv8ForKeyPointsDetectionTorchScript
):
    pass
