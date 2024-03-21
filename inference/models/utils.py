from inference.core.env import API_KEY, API_KEY_ENV_NAMES
from inference.core.exceptions import MissingApiKeyError
from inference.core.models.base import Model
from inference.core.models.stubs import (
    ClassificationModelStub,
    InstanceSegmentationModelStub,
    KeypointsDetectionModelStub,
    ObjectDetectionModelStub,
)
from inference.core.registries.roboflow import get_model_type
from inference.core.utils.function import deprecated
from inference.models import (
    YOLACT,
    VitClassification,
    YOLONASObjectDetection,
    YOLOv5InstanceSegmentation,
    YOLOv5ObjectDetection,
    YOLOv7InstanceSegmentation,
    YOLOv8Classification,
    YOLOv8InstanceSegmentation,
    YOLOv8ObjectDetection,
    YOLOv9ObjectDetection,
)
from inference.models.yolov8.yolov8_keypoints_detection import YOLOv8KeypointsDetection

ROBOFLOW_MODEL_TYPES = {
    ("classification", "stub"): ClassificationModelStub,
    ("classification", "vit"): VitClassification,
    ("classification", "yolov8n"): YOLOv8Classification,
    ("classification", "yolov8s"): YOLOv8Classification,
    ("classification", "yolov8m"): YOLOv8Classification,
    ("classification", "yolov8l"): YOLOv8Classification,
    ("classification", "yolov8x"): YOLOv8Classification,
    ("object-detection", "stub"): ObjectDetectionModelStub,
    ("object-detection", "yolov5"): YOLOv5ObjectDetection,
    ("object-detection", "yolov5v2s"): YOLOv5ObjectDetection,
    ("object-detection", "yolov5v6n"): YOLOv5ObjectDetection,
    ("object-detection", "yolov5v6s"): YOLOv5ObjectDetection,
    ("object-detection", "yolov5v6m"): YOLOv5ObjectDetection,
    ("object-detection", "yolov5v6l"): YOLOv5ObjectDetection,
    ("object-detection", "yolov5v6x"): YOLOv5ObjectDetection,
    ("object-detection", "yolov9"): YOLOv9ObjectDetection,
    ("object-detection", "yolov8"): YOLOv8ObjectDetection,
    ("object-detection", "yolov8s"): YOLOv8ObjectDetection,
    ("object-detection", "yolov8n"): YOLOv8ObjectDetection,
    ("object-detection", "yolov8s"): YOLOv8ObjectDetection,
    ("object-detection", "yolov8m"): YOLOv8ObjectDetection,
    ("object-detection", "yolov8l"): YOLOv8ObjectDetection,
    ("object-detection", "yolov8x"): YOLOv8ObjectDetection,
    ("object-detection", "yolo_nas_s"): YOLONASObjectDetection,
    ("object-detection", "yolo_nas_m"): YOLONASObjectDetection,
    ("object-detection", "yolo_nas_l"): YOLONASObjectDetection,
    ("instance-segmentation", "stub"): InstanceSegmentationModelStub,
    (
        "instance-segmentation",
        "yolov5-seg",
    ): YOLOv5InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov5n-seg",
    ): YOLOv5InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov5s-seg",
    ): YOLOv5InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov5m-seg",
    ): YOLOv5InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov5l-seg",
    ): YOLOv5InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov5x-seg",
    ): YOLOv5InstanceSegmentation,
    (
        "instance-segmentation",
        "yolact",
    ): YOLACT,
    (
        "instance-segmentation",
        "yolov7-seg",
    ): YOLOv7InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov8n",
    ): YOLOv8InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov8s",
    ): YOLOv8InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov8m",
    ): YOLOv8InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov8l",
    ): YOLOv8InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov8x",
    ): YOLOv8InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov8n-seg",
    ): YOLOv8InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov8s-seg",
    ): YOLOv8InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov8m-seg",
    ): YOLOv8InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov8l-seg",
    ): YOLOv8InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov8x-seg",
    ): YOLOv8InstanceSegmentation,
    (
        "instance-segmentation",
        "yolov8-seg",
    ): YOLOv8InstanceSegmentation,
    ("keypoint-detection", "stub"): KeypointsDetectionModelStub,
    ("keypoint-detection", "yolov8n"): YOLOv8KeypointsDetection,
    ("keypoint-detection", "yolov8s"): YOLOv8KeypointsDetection,
    ("keypoint-detection", "yolov8m"): YOLOv8KeypointsDetection,
    ("keypoint-detection", "yolov8l"): YOLOv8KeypointsDetection,
    ("keypoint-detection", "yolov8x"): YOLOv8KeypointsDetection,
    ("keypoint-detection", "yolov8n-pose"): YOLOv8KeypointsDetection,
    ("keypoint-detection", "yolov8s-pose"): YOLOv8KeypointsDetection,
    ("keypoint-detection", "yolov8m-pose"): YOLOv8KeypointsDetection,
    ("keypoint-detection", "yolov8l-pose"): YOLOv8KeypointsDetection,
    ("keypoint-detection", "yolov8x-pose"): YOLOv8KeypointsDetection,
}

try:
    from inference.models import SegmentAnything

    ROBOFLOW_MODEL_TYPES[("embed", "sam")] = SegmentAnything
except:
    pass

try:
    from inference.models import Clip

    ROBOFLOW_MODEL_TYPES[("embed", "clip")] = Clip
except:
    pass

try:
    from inference.models import Gaze

    ROBOFLOW_MODEL_TYPES[("gaze", "l2cs")] = Gaze
except:
    pass

try:
    from inference.models import DocTR

    ROBOFLOW_MODEL_TYPES[("ocr", "doctr")] = DocTR
except:
    pass

try:
    from inference.models import GroundingDINO

    ROBOFLOW_MODEL_TYPES[("object-detection", "grounding-dino")] = GroundingDINO
except:
    pass

try:
    from inference.models import CogVLM

    ROBOFLOW_MODEL_TYPES[("llm", "cogvlm")] = CogVLM
except:
    pass

try:
    from inference.models import YOLOWorld

    ROBOFLOW_MODEL_TYPES[("object-detection", "yolo-world")] = YOLOWorld
except:
    pass


def get_model(model_id, api_key=API_KEY, **kwargs) -> Model:
    task, model = get_model_type(model_id, api_key=api_key)
    return ROBOFLOW_MODEL_TYPES[(task, model)](model_id, api_key=api_key, **kwargs)


def get_roboflow_model(*args, **kwargs):
    return get_model(*args, **kwargs)
