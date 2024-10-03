from inference.core.env import (
    CORE_MODEL_CLIP_ENABLED,
    CORE_MODEL_COGVLM_ENABLED,
    CORE_MODEL_DOCTR_ENABLED,
    CORE_MODEL_GAZE_ENABLED,
    CORE_MODEL_GROUNDINGDINO_ENABLED,
    CORE_MODEL_SAM2_ENABLED,
    CORE_MODEL_SAM_ENABLED,
    CORE_MODEL_YOLO_WORLD_ENABLED,
    CORE_MODELS_ENABLED,
)

if CORE_MODELS_ENABLED:
    if CORE_MODEL_CLIP_ENABLED:
        try:
            from inference.models.clip import Clip
        except:
            pass

    if CORE_MODEL_GAZE_ENABLED:
        try:
            from inference.models.gaze import Gaze
        except:
            pass

    if CORE_MODEL_SAM_ENABLED:
        try:
            from inference.models.sam import SegmentAnything
        except:
            pass

    if CORE_MODEL_SAM2_ENABLED:
        try:
            from inference.models.sam2 import SegmentAnything2
        except:
            pass

    if CORE_MODEL_DOCTR_ENABLED:
        try:
            from inference.models.doctr import DocTR
        except:
            pass

    if CORE_MODEL_GROUNDINGDINO_ENABLED:
        try:
            from inference.models.grounding_dino import GroundingDINO
        except:
            pass

    if CORE_MODEL_COGVLM_ENABLED:
        try:
            from inference.models.cogvlm import CogVLM
        except:
            pass

    if CORE_MODEL_YOLO_WORLD_ENABLED:
        try:
            from inference.models.yolo_world import YOLOWorld
        except:
            pass

try:
    from inference.models.paligemma import LoRAPaliGemma, PaliGemma
except:
    pass

try:
    from inference.models.florence2 import Florence2, LoRAFlorence2
except:
    pass

try:
    from inference.models.trocr import TrOCR
except:
    pass

from inference.models.vit import VitClassification
from inference.models.yolact import YOLACT
from inference.models.yolonas import YOLONASObjectDetection
from inference.models.yolov5 import YOLOv5InstanceSegmentation, YOLOv5ObjectDetection
from inference.models.yolov7 import YOLOv7InstanceSegmentation
from inference.models.yolov8 import (
    YOLOv8Classification,
    YOLOv8InstanceSegmentation,
    YOLOv8KeypointsDetection,
    YOLOv8ObjectDetection,
)
from inference.models.yolov9 import YOLOv9ObjectDetection
from inference.models.yolov10 import YOLOv10ObjectDetection
from inference.models.yolov11 import (
    YOLOv11InstanceSegmentation,
    YOLOv11KeypointsDetection,
    YOLOv11ObjectDetection,
)
