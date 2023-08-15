from inference.models.vit.vit_classification import (
    VitClassificationOnnxRoboflowInferenceModel,
)
from inference.models.yolact.yolact_instance_segmentation import (
    YOLACTInstanceSegmentationOnnxRoboflowInferenceModel,
)
from inference.models.yolov5.yolov5_instance_segmentation import (
    YOLOv5InstanceSegmentationOnnxRoboflowInferenceModel,
)
from inference.models.yolov5.yolov5_object_detection import (
    YOLOv5ObjectDetectionOnnxRoboflowInferenceModel,
)
from inference.models.yolov7.yolov7_instance_segmentation import (
    YOLOv7InstanceSegmentationOnnxRoboflowInferenceModel,
)
from inference.models.yolov8.yolov8_classification import (
    YOLOv8ClassificationOnnxRoboflowInferenceModel,
)
from inference.models.yolov8.yolov8_instance_segmentation import (
    YOLOv8InstanceSegmentationOnnxRoboflowInferenceModel,
)
from inference.models.yolov8.yolov8_object_detection import (
    YOLOv8ObjectDetectionOnnxRoboflowInferenceModel,
)

ROBOFLOW_MODEL_TYPES = {
    ("classification", "vit"): VitClassificationOnnxRoboflowInferenceModel,
    ("classification", "yolov8n"): YOLOv8ClassificationOnnxRoboflowInferenceModel,
    ("classification", "yolov8s"): YOLOv8ClassificationOnnxRoboflowInferenceModel,
    ("classification", "yolov8m"): YOLOv8ClassificationOnnxRoboflowInferenceModel,
    ("classification", "yolov8l"): YOLOv8ClassificationOnnxRoboflowInferenceModel,
    ("classification", "yolov8x"): YOLOv8ClassificationOnnxRoboflowInferenceModel,
    ("object-detection", "yolov5"): YOLOv5ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov5v2s"): YOLOv5ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov5v6n"): YOLOv5ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov5v6s"): YOLOv5ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov5v6m"): YOLOv5ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov5v6l"): YOLOv5ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov5v6x"): YOLOv5ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov8"): YOLOv8ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov8s"): YOLOv8ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov8n"): YOLOv8ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov8s"): YOLOv8ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov8m"): YOLOv8ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov8l"): YOLOv8ObjectDetectionOnnxRoboflowInferenceModel,
    ("object-detection", "yolov8x"): YOLOv8ObjectDetectionOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov5-seg",
    ): YOLOv5InstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov5n-seg",
    ): YOLOv5InstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov5s-seg",
    ): YOLOv5InstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov5m-seg",
    ): YOLOv5InstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov5l-seg",
    ): YOLOv5InstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov5x-seg",
    ): YOLOv5InstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolact",
    ): YOLACTInstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov7-seg",
    ): YOLOv7InstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov8n",
    ): YOLOv8InstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov8s",
    ): YOLOv8InstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov8m",
    ): YOLOv8InstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov8l",
    ): YOLOv8InstanceSegmentationOnnxRoboflowInferenceModel,
    (
        "instance-segmentation",
        "yolov8x",
    ): YOLOv8InstanceSegmentationOnnxRoboflowInferenceModel,
}

try:
    from models.sam.segment_anything import SegmentAnythingRoboflowCoreModel

    ROBOFLOW_MODEL_TYPES[("embed", "sam")] = SegmentAnythingRoboflowCoreModel
except:
    pass

try:
    from inference.models.clip.clip import ClipOnnxRoboflowCoreModel

    ROBOFLOW_MODEL_TYPES[("embed", "clip")] = ClipOnnxRoboflowCoreModel
except:
    pass
