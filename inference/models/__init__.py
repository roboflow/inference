try:
    from inference.models.clip import Clip
except:
    pass

try:
    from inference.models.gaze import Gaze
except:
    pass

try:
    from inference.models.sam import SegmentAnything
except:
    pass

from inference.models.vit import VitClassification
from inference.models.yolact import YOLACT
from inference.models.yolov5 import YOLOv5InstanceSegmentation, YOLOv5ObjectDetection
from inference.models.yolov7 import YOLOv7InstanceSegmentation
from inference.models.yolov8 import (
    YOLOv8Classification,
    YOLOv8InstanceSegmentation,
    YOLOv8ObjectDetection,
)
