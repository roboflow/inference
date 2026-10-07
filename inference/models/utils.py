from inference.core.env import (
    API_KEY,
    CORE_MODEL_CLIP_ENABLED,
    CORE_MODEL_DOCTR_ENABLED,
    CORE_MODEL_EASYOCR_ENABLED,
    CORE_MODEL_GAZE_ENABLED,
    CORE_MODEL_GROUNDINGDINO_ENABLED,
    CORE_MODEL_OWLV2_ENABLED,
    CORE_MODEL_PE_ENABLED,
    CORE_MODEL_PPOCR_ENABLED,
    CORE_MODEL_SAM2_ENABLED,
    CORE_MODEL_SAM3_ENABLED,
    CORE_MODEL_SAM_ENABLED,
    CORE_MODEL_TROCR_ENABLED,
    CORE_MODEL_YOLO_WORLD_ENABLED,
    CORE_MODELS_ENABLED,
    COSMOS3_ENABLED,
    DEPTH_ESTIMATION_ENABLED,
    FLORENCE2_ENABLED,
    GLM_OCR_ENABLED,
    MOONDREAM2_ENABLED,
    PALIGEMMA_ENABLED,
    QWEN_2_5_ENABLED,
    QWEN_3_5_ENABLED,
    QWEN_3_8_ENABLED,
    QWEN_3_ENABLED,
    SAM3_3D_OBJECTS_ENABLED,
    SMOLVLM2_ENABLED,
    USE_INFERENCE_MODELS,
)
from inference.core.models.base import Model
from inference.core.registries.lazy import _LazyModelClass, _LazyModelRegistry
from inference.core.registries.roboflow import (
    LOCAL_INFERENCE_MODELS_MODEL_TYPE,
    get_model_type,
)
from inference.core.warnings import ModelDependencyMissing
from inference.models.vllm_proxy import VLLM_PROXY_ENABLED
from inference.usage_tracking.model_types import bind_usage_model_descriptor

_PALIGEMMA_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support PaliGemma model. "
    "Use pip install 'inference[transformers]' to install missing requirements."
    "To suppress this warning, set PALIGEMMA_ENABLED to False."
)

_FLORENCE2_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support Florence2 model. "
    "Use pip install 'inference[transformers]' to install missing requirements."
    "To suppress this warning, set FLORENCE2_ENABLED to False."
)

_QWEN_2_5_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support Qwen2.5-VL model. "
    "Use pip install 'inference[transformers]' to install missing requirements."
    "To suppress this warning, set QWEN_2_5_ENABLED to False."
)

_COSMOS3_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support the Cosmos 3 model. "
    "Since inference 1.3.6 was shipped when downstream model dependencies were not "
    "released yet, "
    "we have enabled the model in selected builds only. Installation guide will be "
    "provided "
    "in following releases. To suppress this warning, set COSMOS3_ENABLED to False."
)

_CORE_MODEL_SAM_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support SAM model. "
    "Use pip install 'inference[sam]' to install missing requirements."
    "To suppress this warning, set CORE_MODEL_SAM_ENABLED to False."
)

_CORE_MODEL_SAM2_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support SAM2 model. "
    "Use pip install 'inference[sam]' to install missing requirements."
    "To suppress this warning, set CORE_MODEL_SAM2_ENABLED to False."
)

_CORE_MODEL_SAM3_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support SAM3 model. "
    "Install SAM3 dependencies and set CORE_MODEL_SAM3_ENABLED to True."
)

_CORE_MODEL_CLIP_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support CLIP model. "
    "Use pip install 'inference[clip]' to install missing requirements."
    "To suppress this warning, set CORE_MODEL_CLIP_ENABLED to False."
)

_CORE_MODEL_OWLV2_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support OWLv2 model. "
    "Use pip install 'inference[transformers]' to install missing requirements."
    "To suppress this warning, set CORE_MODEL_OWLV2_ENABLED to False."
)

_CORE_MODEL_GAZE_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support Gaze Detection model. "
    "This model got deprecated and remaining left-overs will be removed end of Q2 2026."
    "To suppress this warning, set CORE_MODEL_GAZE_ENABLED to False."
)

_SMOLVLM2_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support SmolVLM2."
    "Use pip install 'inference[transformers]' to install missing requirements."
    "To suppress this warning, set SMOLVLM2_ENABLED to False."
)

_DEPTH_ESTIMATION_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support Depth Estimation."
    "Use pip install 'inference[transformers]' to install missing requirements."
    "To suppress this warning, set DEPTH_ESTIMATION_ENABLED to False."
)

_MOONDREAM2_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support Moondream2."
    "Use pip install 'inference[transformers]' to install missing requirements."
    "To suppress this warning, set MOONDREAM2_ENABLED to False."
)

_SAM3_3D_OBJECTS_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support SAM3_3D_Objects model. "
    "Use pip install 'inference[sam3_3d]' to install missing requirements."
    "To suppress this warning, set SAM3_3D_OBJECTS_ENABLED to False."
)

_CORE_MODEL_TROCR_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support TrOCR model. "
    "Use pip install 'inference[transformers]' to install missing requirements."
    "To suppress this warning, set CORE_MODEL_TROCR_ENABLED to False."
)

_CORE_MODEL_PPOCR_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support PP-OCR model. "
    "Use pip install 'inference[inference-models]' to install missing requirements."
    "To suppress this warning, set CORE_MODEL_PPOCR_ENABLED to False."
)

_CORE_MODEL_GROUNDINGDINO_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support GroundingDINO model. "
    "Use pip install 'inference[grounding-dino]' to install missing requirements."
    "To suppress this warning, set CORE_MODEL_GROUNDINGDINO_ENABLED to False."
)

_CORE_MODEL_YOLO_WORLD_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support YoloWorld model. "
    "Use pip install 'inference[yolo-world]' to install missing requirements."
    "To suppress this warning, set CORE_MODEL_YOLO_WORLD_ENABLED to False."
)

_CORE_MODEL_PE_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support Perception Encoder."
    "Use pip install 'inference[transformers]' to install missing requirements."
    "To suppress this warning, set CORE_MODEL_PE_ENABLED to False."
)

_QWEN_3_DEPENDENCY_WARNING = (
    "Your `inference` configuration does not support Qwen3-VL model. "
    "Use pip install 'inference[transformers]' to install missing requirements."
    "To suppress this warning, set QWEN_3_ENABLED to False."
)

_QWEN_3_8_DEPENDENCY_WARNING = (
    "qwen3_8 models disabled: installed inference_models has "
    "no Qwen38HF (upgrade inference-models to enable)."
)

ClassificationModelStub = _LazyModelClass(
    "inference.core.models.stubs:ClassificationModelStub"
)
InstanceSegmentationModelStub = _LazyModelClass(
    "inference.core.models.stubs:InstanceSegmentationModelStub"
)
KeypointsDetectionModelStub = _LazyModelClass(
    "inference.core.models.stubs:KeypointsDetectionModelStub"
)
ObjectDetectionModelStub = _LazyModelClass(
    "inference.core.models.stubs:ObjectDetectionModelStub"
)
YOLACT = _LazyModelClass("inference.models:YOLACT")
DeepLabV3PlusSemanticSegmentation = _LazyModelClass(
    "inference.models:DeepLabV3PlusSemanticSegmentation"
)
DinoV3Classification = _LazyModelClass("inference.models:DinoV3Classification")
ResNetClassification = _LazyModelClass("inference.models:ResNetClassification")
RFDETRInstanceSegmentation = _LazyModelClass(
    "inference.models:RFDETRInstanceSegmentation"
)
RFDETRNasInstanceSegmentation = _LazyModelClass(
    "inference.models:RFDETRNasInstanceSegmentation"
)
RFDETRNasObjectDetection = _LazyModelClass("inference.models:RFDETRNasObjectDetection")
RFDETRObjectDetection = _LazyModelClass("inference.models:RFDETRObjectDetection")
VitClassification = _LazyModelClass("inference.models:VitClassification")
YOLO26InstanceSegmentation = _LazyModelClass(
    "inference.models:YOLO26InstanceSegmentation"
)
YOLO26ObjectDetection = _LazyModelClass("inference.models:YOLO26ObjectDetection")
YOLONASObjectDetection = _LazyModelClass("inference.models:YOLONASObjectDetection")
YOLOv5InstanceSegmentation = _LazyModelClass(
    "inference.models:YOLOv5InstanceSegmentation"
)
YOLOv5ObjectDetection = _LazyModelClass("inference.models:YOLOv5ObjectDetection")
YOLOv7InstanceSegmentation = _LazyModelClass(
    "inference.models:YOLOv7InstanceSegmentation"
)
YOLOv8Classification = _LazyModelClass("inference.models:YOLOv8Classification")
YOLOv8InstanceSegmentation = _LazyModelClass(
    "inference.models:YOLOv8InstanceSegmentation"
)
YOLOv8ObjectDetection = _LazyModelClass("inference.models:YOLOv8ObjectDetection")
YOLOv9ObjectDetection = _LazyModelClass("inference.models:YOLOv9ObjectDetection")
YOLOv10ObjectDetection = _LazyModelClass("inference.models:YOLOv10ObjectDetection")
YOLOv11InstanceSegmentation = _LazyModelClass(
    "inference.models:YOLOv11InstanceSegmentation"
)
YOLOv11ObjectDetection = _LazyModelClass("inference.models:YOLOv11ObjectDetection")
YOLOv12ObjectDetection = _LazyModelClass("inference.models:YOLOv12ObjectDetection")
YOLO26KeypointsDetection = _LazyModelClass(
    "inference.models.yolo26.yolo26_keypoints_detection:YOLO26KeypointsDetection"
)
YOLOv8KeypointsDetection = _LazyModelClass(
    "inference.models.yolov8.yolov8_keypoints_detection:YOLOv8KeypointsDetection"
)
YOLOv11KeypointsDetection = _LazyModelClass(
    "inference.models.yolov11.yolov11_keypoints_detection:YOLOv11KeypointsDetection"
)

ROBOFLOW_MODEL_TYPES = _LazyModelRegistry(
    {
        ("classification", "stub"): ClassificationModelStub,
        ("classification", "vit"): VitClassification,
        ("classification", "dinov3"): DinoV3Classification,
        ("classification", "dinov3_probe"): DinoV3Classification,
        ("classification", "resnet"): ResNetClassification,
        ("classification", "resnet18"): ResNetClassification,
        ("classification", "resnet34"): ResNetClassification,
        ("classification", "resnet50"): ResNetClassification,
        ("classification", "resnet101"): ResNetClassification,
        ("classification", "yolov8"): YOLOv8Classification,
        ("classification", "yolov8n"): YOLOv8Classification,
        ("classification", "yolov8s"): YOLOv8Classification,
        ("classification", "yolov8m"): YOLOv8Classification,
        ("classification", "yolov8l"): YOLOv8Classification,
        ("classification", "yolov8x"): YOLOv8Classification,
        ("object-detection", "stub"): ObjectDetectionModelStub,
        ("object-detection", "yolov5"): YOLOv5ObjectDetection,
        ("instance-segmentation", "yolov5"): YOLOv5InstanceSegmentation,
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
        ("object-detection", "yolonas"): YOLONASObjectDetection,
        ("object-detection", "yolo_nas_s"): YOLONASObjectDetection,
        ("object-detection", "yolo_nas_m"): YOLONASObjectDetection,
        ("object-detection", "yolo_nas_l"): YOLONASObjectDetection,
        ("object-detection", "yolov10"): YOLOv10ObjectDetection,
        ("object-detection", "yolov10s"): YOLOv10ObjectDetection,
        ("object-detection", "yolov10n"): YOLOv10ObjectDetection,
        ("object-detection", "yolov10b"): YOLOv10ObjectDetection,
        ("object-detection", "yolov10m"): YOLOv10ObjectDetection,
        ("object-detection", "yolov10l"): YOLOv10ObjectDetection,
        ("object-detection", "yolov10x"): YOLOv10ObjectDetection,
        ("object-detection", "yolov11"): YOLOv11ObjectDetection,
        ("object-detection", "yolov11s"): YOLOv11ObjectDetection,
        ("object-detection", "yolov11n"): YOLOv11ObjectDetection,
        ("object-detection", "yolov11b"): YOLOv11ObjectDetection,
        ("object-detection", "yolov11m"): YOLOv11ObjectDetection,
        ("object-detection", "yolov11l"): YOLOv11ObjectDetection,
        ("object-detection", "yolov11x"): YOLOv11ObjectDetection,
        ("object-detection", "yolov12"): YOLOv12ObjectDetection,
        ("object-detection", "yolov12s"): YOLOv12ObjectDetection,
        ("object-detection", "yolov12n"): YOLOv12ObjectDetection,
        ("object-detection", "yolov12m"): YOLOv12ObjectDetection,
        ("object-detection", "yolov12l"): YOLOv12ObjectDetection,
        ("object-detection", "yolov12x"): YOLOv12ObjectDetection,
        ("object-detection", "yolo26"): YOLO26ObjectDetection,
        ("object-detection", "yolo26s"): YOLO26ObjectDetection,
        ("object-detection", "yolo26n"): YOLO26ObjectDetection,
        ("object-detection", "yolo26b"): YOLO26ObjectDetection,
        ("object-detection", "yolo26m"): YOLO26ObjectDetection,
        ("object-detection", "yolo26l"): YOLO26ObjectDetection,
        ("object-detection", "yolo26x"): YOLO26ObjectDetection,
        ("object-detection", "rfdetr"): RFDETRObjectDetection,
        ("object-detection", "rfdetr-base"): RFDETRObjectDetection,
        ("object-detection", "rfdetr-nano"): RFDETRObjectDetection,
        ("object-detection", "rfdetr-small"): RFDETRObjectDetection,
        ("object-detection", "rfdetr-medium"): RFDETRObjectDetection,
        ("object-detection", "rfdetr-large"): RFDETRObjectDetection,
        ("object-detection", "rfdetr-xlarge"): RFDETRObjectDetection,
        ("object-detection", "rfdetr-2xlarge"): RFDETRObjectDetection,
        ("object-detection", "rfdetr-nas"): RFDETRNasObjectDetection,
        ("instance-segmentation", "rfdetr"): RFDETRInstanceSegmentation,
        ("instance-segmentation", "rfdetr-seg-preview"): RFDETRInstanceSegmentation,
        ("instance-segmentation", "rfdetr-seg-nano"): RFDETRInstanceSegmentation,
        ("instance-segmentation", "rfdetr-seg-small"): RFDETRInstanceSegmentation,
        ("instance-segmentation", "rfdetr-seg-medium"): RFDETRInstanceSegmentation,
        ("instance-segmentation", "rfdetr-seg-large"): RFDETRInstanceSegmentation,
        ("instance-segmentation", "rfdetr-seg-xlarge"): RFDETRInstanceSegmentation,
        ("instance-segmentation", "rfdetr-seg-xxlarge"): RFDETRInstanceSegmentation,
        ("instance-segmentation", "rfdetr-seg-2xlarge"): RFDETRInstanceSegmentation,
        ("instance-segmentation", "rfdetr-nas-seg"): RFDETRNasInstanceSegmentation,
        (
            "instance-segmentation",
            "yolov11n",
        ): YOLOv11InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov11s",
        ): YOLOv11InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov11m",
        ): YOLOv11InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov11l",
        ): YOLOv11InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov11x",
        ): YOLOv11InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov11n-seg",
        ): YOLOv11InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov11s-seg",
        ): YOLOv11InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov11m-seg",
        ): YOLOv11InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov11l-seg",
        ): YOLOv11InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov11x-seg",
        ): YOLOv11InstanceSegmentation,
        (
            "instance-segmentation",
            "yolo26",
        ): YOLO26InstanceSegmentation,
        (
            "instance-segmentation",
            "yolo26n",
        ): YOLO26InstanceSegmentation,
        (
            "instance-segmentation",
            "yolo26s",
        ): YOLO26InstanceSegmentation,
        (
            "instance-segmentation",
            "yolo26m",
        ): YOLO26InstanceSegmentation,
        (
            "instance-segmentation",
            "yolo26l",
        ): YOLO26InstanceSegmentation,
        (
            "instance-segmentation",
            "yolo26x",
        ): YOLO26InstanceSegmentation,
        (
            "instance-segmentation",
            "yolo26n-seg",
        ): YOLO26InstanceSegmentation,
        (
            "instance-segmentation",
            "yolo26s-seg",
        ): YOLO26InstanceSegmentation,
        (
            "instance-segmentation",
            "yolo26m-seg",
        ): YOLO26InstanceSegmentation,
        (
            "instance-segmentation",
            "yolo26l-seg",
        ): YOLO26InstanceSegmentation,
        (
            "instance-segmentation",
            "yolo26x-seg",
        ): YOLO26InstanceSegmentation,
        ("keypoint-detection", "yolov11n"): YOLOv11KeypointsDetection,
        ("keypoint-detection", "yolov11s"): YOLOv11KeypointsDetection,
        ("keypoint-detection", "yolov11m"): YOLOv11KeypointsDetection,
        ("keypoint-detection", "yolov11l"): YOLOv11KeypointsDetection,
        ("keypoint-detection", "yolov11x"): YOLOv11KeypointsDetection,
        ("keypoint-detection", "yolov11n-pose"): YOLOv11KeypointsDetection,
        ("keypoint-detection", "yolov11s-pose"): YOLOv11KeypointsDetection,
        ("keypoint-detection", "yolov11m-pose"): YOLOv11KeypointsDetection,
        ("keypoint-detection", "yolov11l-pose"): YOLOv11KeypointsDetection,
        ("keypoint-detection", "yolov11x-pose"): YOLOv11KeypointsDetection,
        ("keypoint-detection", "yolo26"): YOLO26KeypointsDetection,
        ("keypoint-detection", "yolo26n"): YOLO26KeypointsDetection,
        ("keypoint-detection", "yolo26s"): YOLO26KeypointsDetection,
        ("keypoint-detection", "yolo26m"): YOLO26KeypointsDetection,
        ("keypoint-detection", "yolo26l"): YOLO26KeypointsDetection,
        ("keypoint-detection", "yolo26x"): YOLO26KeypointsDetection,
        ("keypoint-detection", "yolo26n-pose"): YOLO26KeypointsDetection,
        ("keypoint-detection", "yolo26s-pose"): YOLO26KeypointsDetection,
        ("keypoint-detection", "yolo26m-pose"): YOLO26KeypointsDetection,
        ("keypoint-detection", "yolo26l-pose"): YOLO26KeypointsDetection,
        ("keypoint-detection", "yolo26x-pose"): YOLO26KeypointsDetection,
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
            "yolov7",
        ): YOLOv7InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov7-seg",
        ): YOLOv7InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov7s-seg",
        ): YOLOv7InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov8n",
        ): YOLOv8InstanceSegmentation,
        (
            "instance-segmentation",
            "yolov8",
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
        ("keypoint-detection", "yolov8"): YOLOv8KeypointsDetection,
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
        ("semantic-segmentation", "deeplabv3plus"): DeepLabV3PlusSemanticSegmentation,
    }
)

if PALIGEMMA_ENABLED:
    LoRAPaliGemma = _LazyModelClass(
        "inference.models:LoRAPaliGemma",
        optional=True,
        warning_message=_PALIGEMMA_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )
    PaliGemma = _LazyModelClass(
        "inference.models:PaliGemma",
        optional=True,
        warning_message=_PALIGEMMA_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    paligemma_models = {
        (
            "object-detection",
            "paligemma-3b-pt-224",
        ): PaliGemma,  # TODO: change when we have a new project type
        ("object-detection", "paligemma-3b-pt-448"): PaliGemma,
        ("object-detection", "paligemma-3b-pt-896"): PaliGemma,
        (
            "instance-segmentation",
            "paligemma-3b-pt-224",
        ): PaliGemma,  # TODO: change when we have a new project type
        ("instance-segmentation", "paligemma-3b-pt-448"): PaliGemma,
        ("instance-segmentation", "paligemma-3b-pt-896"): PaliGemma,
        (
            "object-detection",
            "paligemma-3b-pt-224-peft",
        ): LoRAPaliGemma,  # TODO: change when we have a new project type
        ("object-detection", "paligemma-3b-pt-448-peft"): LoRAPaliGemma,
        ("object-detection", "paligemma-3b-pt-896-peft"): LoRAPaliGemma,
        (
            "instance-segmentation",
            "paligemma-3b-pt-224-peft",
        ): LoRAPaliGemma,  # TODO: change when we have a new project type
        ("instance-segmentation", "paligemma-3b-pt-448-peft"): LoRAPaliGemma,
        ("instance-segmentation", "paligemma-3b-pt-896-peft"): LoRAPaliGemma,
        ("text-image-pairs", "paligemma2-3b-pt-224"): PaliGemma,
        ("text-image-pairs", "paligemma2-3b-pt-448"): PaliGemma,
        ("text-image-pairs", "paligemma2-3b-pt-896"): PaliGemma,
        ("text-image-pairs", "paligemma2-3b-pt-224-peft"): LoRAPaliGemma,
        ("text-image-pairs", "paligemma2-3b-pt-448-peft"): LoRAPaliGemma,
        ("text-image-pairs", "paligemma2-3b-pt-896-peft"): LoRAPaliGemma,
    }
    ROBOFLOW_MODEL_TYPES.update(paligemma_models)

if FLORENCE2_ENABLED:
    Florence2 = _LazyModelClass(
        "inference.models:Florence2",
        optional=True,
        warning_message=_FLORENCE2_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )
    LoRAFlorence2 = _LazyModelClass(
        "inference.models:LoRAFlorence2",
        optional=True,
        warning_message=_FLORENCE2_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    florence2_models = {
        (
            "object-detection",
            "florence-2-base",
        ): Florence2,  # TODO: change when we have a new project type
        ("object-detection", "florence-2-large"): Florence2,
        (
            "instance-segmentation",
            "florence-2-base",
        ): Florence2,  # TODO: change when we have a new project type
        ("instance-segmentation", "florence-2-large"): Florence2,
        (
            "object-detection",
            "florence-2-base-peft",
        ): LoRAFlorence2,  # TODO: change when we have a new project type
        (
            "text-image-pairs",
            "florence-2-base",
        ): Florence2,  # TODO: change when we have a new project type
        ("text-image-pairs", "florence-2-large"): Florence2,
        ("object-detection", "florence-2-large-peft"): LoRAFlorence2,
        (
            "instance-segmentation",
            "florence-2-base-peft",
        ): LoRAFlorence2,  # TODO: change when we have a new project type
        ("instance-segmentation", "florence-2-large-peft"): LoRAFlorence2,
        (
            "text-image-pairs",
            "florence-2-base-peft",
        ): LoRAFlorence2,
        ("text-image-pairs", "florence-2-large-peft"): LoRAFlorence2,
    }
    ROBOFLOW_MODEL_TYPES.update(florence2_models)

if QWEN_2_5_ENABLED:
    LoRAQwen25VL = _LazyModelClass(
        "inference.models:LoRAQwen25VL",
        optional=True,
        warning_message=_QWEN_2_5_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )
    Qwen25VL = _LazyModelClass(
        "inference.models:Qwen25VL",
        optional=True,
        warning_message=_QWEN_2_5_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    qwen25vl_models = {
        ("text-image-pairs", "qwen25-vl-7b"): Qwen25VL,
        ("text-image-pairs", "qwen25-vl-7b-peft"): LoRAQwen25VL,
    }
    ROBOFLOW_MODEL_TYPES.update(qwen25vl_models)

if QWEN_3_ENABLED:
    if VLLM_PROXY_ENABLED:
        Qwen3VLVLLMProxy = _LazyModelClass(
            "inference.models.vllm_proxy.qwen3vl_vllm:Qwen3VLVLLMProxy",
            optional=True,
            warning_message=_QWEN_3_DEPENDENCY_WARNING,
            warning_category=ModelDependencyMissing,
        )

        qwen3vl_models = {
            ("text-image-pairs", "qwen3vl-2b-instruct"): Qwen3VLVLLMProxy,
            ("text-image-pairs", "qwen3vl-2b-instruct-peft"): Qwen3VLVLLMProxy,
        }
    else:
        LoRAQwen3VL = _LazyModelClass(
            "inference.models:LoRAQwen3VL",
            optional=True,
            warning_message=_QWEN_3_DEPENDENCY_WARNING,
            warning_category=ModelDependencyMissing,
        )
        Qwen3VL = _LazyModelClass(
            "inference.models:Qwen3VL",
            optional=True,
            warning_message=_QWEN_3_DEPENDENCY_WARNING,
            warning_category=ModelDependencyMissing,
        )

        qwen3vl_models = {
            ("text-image-pairs", "qwen3vl-2b-instruct"): Qwen3VL,
            ("text-image-pairs", "qwen3vl-2b-instruct-peft"): LoRAQwen3VL,
        }
    ROBOFLOW_MODEL_TYPES.update(qwen3vl_models)

if COSMOS3_ENABLED and USE_INFERENCE_MODELS:
    InferenceModelsActionRecognitionAdapter = _LazyModelClass(
        "inference.core.models.inference_models_adapters:"
        "InferenceModelsActionRecognitionAdapter",
        optional=True,
        warning_message=_COSMOS3_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )
    InferenceModelsCosmos3ReasonerAdapter = _LazyModelClass(
        "inference.models.cosmos3.cosmos3_reasoner_inference_models:"
        "InferenceModelsCosmos3ReasonerAdapter",
        optional=True,
        warning_message=_COSMOS3_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    cosmos3_models = {
        (
            "text-image-pairs",
            "cosmos-3-edge",
        ): InferenceModelsCosmos3ReasonerAdapter,
        ("vlm", "cosmos-3-edge"): InferenceModelsCosmos3ReasonerAdapter,
        # Roboflow fine-tunes carry the platform's model type as their
        # architecture.
        ("text-image-pairs", "cosmos3-edge"): InferenceModelsCosmos3ReasonerAdapter,
        ("vlm", "cosmos3-edge"): InferenceModelsCosmos3ReasonerAdapter,
        # Action recognition fine-tunes ship under the dash-less trainer
        # slug; the hosted base keeps the dash and is wrapped for the task
        # on load.
        (
            "action-recognition",
            "cosmos3-edge",
        ): InferenceModelsActionRecognitionAdapter,
        (
            "action-recognition",
            "cosmos-3-edge",
        ): InferenceModelsActionRecognitionAdapter,
    }
    ROBOFLOW_MODEL_TYPES.update(cosmos3_models)


if CORE_MODELS_ENABLED and CORE_MODEL_SAM_ENABLED:
    SegmentAnything = _LazyModelClass(
        "inference.models:SegmentAnything",
        optional=True,
        warning_message=_CORE_MODEL_SAM_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("embed", "sam")] = SegmentAnything
if CORE_MODELS_ENABLED and CORE_MODEL_SAM2_ENABLED:
    SegmentAnything2 = _LazyModelClass(
        "inference.models:SegmentAnything2",
        optional=True,
        warning_message=_CORE_MODEL_SAM2_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("embed", "sam2")] = SegmentAnything2

if CORE_MODELS_ENABLED and CORE_MODEL_SAM3_ENABLED:
    Sam3ForInteractiveImageSegmentation = _LazyModelClass(
        "inference.models:Sam3ForInteractiveImageSegmentation",
        optional=True,
        warning_message=_CORE_MODEL_SAM3_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )
    SegmentAnything3 = _LazyModelClass(
        "inference.models:SegmentAnything3",
        optional=True,
        warning_message=_CORE_MODEL_SAM3_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("embed", "sam3")] = SegmentAnything3
    ROBOFLOW_MODEL_TYPES[("instance-segmentation", "sam3-large")] = SegmentAnything3
    ROBOFLOW_MODEL_TYPES[("interactive-segmentation", "sam3")] = (
        Sam3ForInteractiveImageSegmentation
    )

if CORE_MODELS_ENABLED and CORE_MODEL_CLIP_ENABLED:
    Clip = _LazyModelClass(
        "inference.models:Clip",
        optional=True,
        warning_message=_CORE_MODEL_CLIP_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("embed", "clip")] = Clip

if CORE_MODEL_OWLV2_ENABLED:
    OwlV2 = _LazyModelClass(
        "inference.models.owlv2.owlv2:OwlV2",
        optional=True,
        warning_message=_CORE_MODEL_OWLV2_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )
    SerializedOwlV2 = _LazyModelClass(
        "inference.models.owlv2.owlv2:SerializedOwlV2",
        optional=True,
        warning_message=_CORE_MODEL_OWLV2_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("object-detection", "owlv2")] = OwlV2
    ROBOFLOW_MODEL_TYPES[("object-detection", "owlv2-finetuned")] = SerializedOwlV2

if CORE_MODELS_ENABLED and CORE_MODEL_GAZE_ENABLED:
    Gaze = _LazyModelClass(
        "inference.models:Gaze",
        optional=True,
        warning_message=_CORE_MODEL_GAZE_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("gaze", "l2cs")] = Gaze

if SMOLVLM2_ENABLED:
    LoRASmolVLM = _LazyModelClass(
        "inference.models.smolvlm.smolvlm:LoRASmolVLM",
        optional=True,
        warning_message=_SMOLVLM2_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )
    SmolVLM = _LazyModelClass(
        "inference.models.smolvlm.smolvlm:SmolVLM",
        optional=True,
        warning_message=_SMOLVLM2_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("lmm", "smolvlm-2.2b-instruct")] = SmolVLM
    ROBOFLOW_MODEL_TYPES[("text-image-pairs", "smolvlm2-peft")] = LoRASmolVLM
    ROBOFLOW_MODEL_TYPES[("text-image-pairs", "smolvlm-256m-peft")] = LoRASmolVLM


if DEPTH_ESTIMATION_ENABLED:
    DepthAnythingV2 = _LazyModelClass(
        "inference.models.depth_anything_v2.depth_anything_v2:DepthAnythingV2",
        optional=True,
        warning_message=_DEPTH_ESTIMATION_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )
    DepthAnythingV3 = _LazyModelClass(
        "inference.models.depth_anything_v3.depth_anything_v3:DepthAnythingV3",
        optional=True,
        warning_message=_DEPTH_ESTIMATION_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("depth-estimation", "depth-anything-v2")] = DepthAnythingV2
    ROBOFLOW_MODEL_TYPES[("depth-estimation", "depth-anything-v3")] = DepthAnythingV3


if MOONDREAM2_ENABLED:
    Moondream2 = _LazyModelClass(
        "inference.models.moondream2.moondream2:Moondream2",
        optional=True,
        warning_message=_MOONDREAM2_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("lmm", "moondream2")] = Moondream2

if SAM3_3D_OBJECTS_ENABLED:
    SegmentAnything3_3D_Objects = _LazyModelClass(
        "inference.models.sam3_3d.segment_anything_3d:SegmentAnything3_3D_Objects",
        optional=True,
        warning_message=_SAM3_3D_OBJECTS_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("3d-reconstruction", "sam3-3d-objects")] = (
        SegmentAnything3_3D_Objects
    )

if CORE_MODELS_ENABLED and CORE_MODEL_DOCTR_ENABLED:
    DocTR = _LazyModelClass("inference.models:DocTR", optional=True)

    ROBOFLOW_MODEL_TYPES[("ocr", "doctr")] = DocTR

if CORE_MODEL_EASYOCR_ENABLED:
    EasyOCR = _LazyModelClass("inference.models:EasyOCR", optional=True)

    ROBOFLOW_MODEL_TYPES[("ocr", "easy_ocr")] = EasyOCR

if CORE_MODEL_TROCR_ENABLED:
    InferenceModelsTrOCRAdapter = _LazyModelClass(
        "inference.models.trocr.trocr_inference_models:InferenceModelsTrOCRAdapter",
        optional=True,
        warning_message=_CORE_MODEL_TROCR_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("ocr", "trocr")] = InferenceModelsTrOCRAdapter

if CORE_MODEL_PPOCR_ENABLED:
    InferenceModelsPPOCRAdapter = _LazyModelClass(
        "inference.models.pp_ocr.pp_ocr_inference_models:InferenceModelsPPOCRAdapter",
        optional=True,
        warning_message=_CORE_MODEL_PPOCR_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("ocr", "pp_ocr")] = InferenceModelsPPOCRAdapter

if CORE_MODELS_ENABLED and CORE_MODEL_GROUNDINGDINO_ENABLED:
    GroundingDINO = _LazyModelClass(
        "inference.models:GroundingDINO",
        optional=True,
        warning_message=_CORE_MODEL_GROUNDINGDINO_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("object-detection", "grounding-dino")] = GroundingDINO

if CORE_MODELS_ENABLED and CORE_MODEL_YOLO_WORLD_ENABLED:
    YOLOWorld = _LazyModelClass(
        "inference.models:YOLOWorld",
        optional=True,
        warning_message=_CORE_MODEL_YOLO_WORLD_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("object-detection", "yolo-world")] = YOLOWorld


if CORE_MODEL_PE_ENABLED:
    PerceptionEncoder = _LazyModelClass(
        "inference.models:PerceptionEncoder",
        optional=True,
        warning_message=_CORE_MODEL_PE_DEPENDENCY_WARNING,
        warning_category=ModelDependencyMissing,
    )

    ROBOFLOW_MODEL_TYPES[("embed", "perception_encoder")] = PerceptionEncoder


def get_model(model_id, api_key=API_KEY, **kwargs) -> Model:
    task, model = get_model_type(model_id, api_key=api_key)
    instance = ROBOFLOW_MODEL_TYPES[(task, model)](model_id, api_key=api_key, **kwargs)
    bind_usage_model_descriptor(instance, model_id)
    return instance


def get_roboflow_model(*args, **kwargs):
    return get_model(*args, **kwargs)


if USE_INFERENCE_MODELS:
    # Select adapters without importing their implementations
    InferenceModelsClassificationAdapter = _LazyModelClass(
        "inference.core.models.inference_models_adapters:"
        "InferenceModelsClassificationAdapter"
    )
    InferenceModelsInstanceSegmentationAdapter = _LazyModelClass(
        "inference.core.models.inference_models_adapters:"
        "InferenceModelsInstanceSegmentationAdapter"
    )
    InferenceModelsKeyPointsDetectionAdapter = _LazyModelClass(
        "inference.core.models.inference_models_adapters:"
        "InferenceModelsKeyPointsDetectionAdapter"
    )
    InferenceModelsObjectDetectionAdapter = _LazyModelClass(
        "inference.core.models.inference_models_adapters:"
        "InferenceModelsObjectDetectionAdapter"
    )
    InferenceModelsSemanticSegmentationAdapter = _LazyModelClass(
        "inference.core.models.inference_models_adapters:"
        "InferenceModelsSemanticSegmentationAdapter"
    )

    # Inspect registration metadata while constructing the registry; public key
    # iteration checks optional availability and is reserved for callers.
    tasks_and_variants = list(ROBOFLOW_MODEL_TYPES._entries)
    for task, variant in tasks_and_variants:
        if task == "object-detection" and variant.startswith("rfdetr"):
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsObjectDetectionAdapter
            )
        elif task == "object-detection" and variant.startswith("yolov"):
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsObjectDetectionAdapter
            )
        elif task == "object-detection" and variant.startswith("yolo26"):
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsObjectDetectionAdapter
            )
        elif task == "object-detection" and (
            variant.startswith("yolo_nas") or variant.startswith("yolonas")
        ):
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsObjectDetectionAdapter
            )
        elif task == "instance-segmentation" and variant.startswith("rfdetr"):
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsInstanceSegmentationAdapter
            )
        elif task == "instance-segmentation" and variant.startswith("yolov"):
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsInstanceSegmentationAdapter
            )
        elif task == "instance-segmentation" and variant.startswith("yolo26"):
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsInstanceSegmentationAdapter
            )
        elif task == "instance-segmentation" and variant.startswith("yolact"):
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsInstanceSegmentationAdapter
            )
        elif task == "keypoint-detection" and variant.startswith("yolov"):
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsKeyPointsDetectionAdapter
            )
        elif task == "keypoint-detection" and variant.startswith("yolo26"):
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsKeyPointsDetectionAdapter
            )
        elif task == "classification" and (
            variant.startswith("yolov")
            or variant.startswith("dinov3")
            or variant.startswith("resnet")
            or variant.startswith("vit")
        ):
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsClassificationAdapter
            )
        elif variant.startswith("paligemma-") or variant.startswith("paligemma2-"):
            InferenceModelsPaligemmaAdapter = _LazyModelClass(
                "inference.models.paligemma.paligemma_inference_models:"
                "InferenceModelsPaligemmaAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsPaligemmaAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("vlm", "paligemma-2"), InferenceModelsPaligemmaAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("vlm", "paligemma"), InferenceModelsPaligemmaAdapter
            )
        elif variant.startswith("florence-2"):
            InferenceModelsFlorence2Adapter = _LazyModelClass(
                "inference.models.florence2.florence2_inference_models:"
                "InferenceModelsFlorence2Adapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsFlorence2Adapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("vlm", "florence-2"), InferenceModelsFlorence2Adapter
            )
        elif variant.startswith("qwen25-vl"):
            InferenceModelsQwen25VLAdapter = _LazyModelClass(
                "inference.models.qwen25vl.qwen25vl_inference_models:"
                "InferenceModelsQwen25VLAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsQwen25VLAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("vlm", "qwen25vl"), InferenceModelsQwen25VLAdapter
            )
        elif variant.startswith("qwen3vl-"):
            if VLLM_PROXY_ENABLED:
                _Qwen3VLModelClass = _LazyModelClass(
                    "inference.models.vllm_proxy.qwen3vl_vllm:Qwen3VLVLLMProxy"
                )
            else:
                _Qwen3VLModelClass = _LazyModelClass(
                    "inference.models.qwen3vl.qwen3vl_inference_models:"
                    "InferenceModelsQwen3VLAdapter"
                )

            ROBOFLOW_MODEL_TYPES.set_adapter((task, variant), _Qwen3VLModelClass)
            ROBOFLOW_MODEL_TYPES.set_adapter(("vlm", "qwen3vl"), _Qwen3VLModelClass)
        elif variant.startswith("qwen3_5"):
            if VLLM_PROXY_ENABLED:
                _Qwen35ModelClass = _LazyModelClass(
                    "inference.models.vllm_proxy.qwen3_5_vllm:Qwen35VLLMProxy"
                )
            else:
                _Qwen35ModelClass = _LazyModelClass(
                    "inference.models.qwen3_5vl.qwen3_5vl_inference_models:"
                    "InferenceModelsQwen35VLAdapter"
                )

            ROBOFLOW_MODEL_TYPES.set_adapter((task, variant), _Qwen35ModelClass)
            ROBOFLOW_MODEL_TYPES.set_adapter(("vlm", "qwen_3_5"), _Qwen35ModelClass)
            ROBOFLOW_MODEL_TYPES.set_adapter(("vlm", "qwen3_5"), _Qwen35ModelClass)
        elif variant.startswith("qwen3_8"):
            if VLLM_PROXY_ENABLED:
                _Qwen38ModelClass = _LazyModelClass(
                    "inference.models.vllm_proxy.qwen3_8_vllm:Qwen38VLLMProxy"
                )
            else:
                _Qwen38ModelClass = _LazyModelClass(
                    "inference.models.qwen3_8vl.qwen3_8vl_inference_models:"
                    "InferenceModelsQwen38VLAdapter"
                )

            ROBOFLOW_MODEL_TYPES.set_adapter((task, variant), _Qwen38ModelClass)
            ROBOFLOW_MODEL_TYPES.set_adapter(("vlm", "qwen3_8"), _Qwen38ModelClass)
        elif task == "embed" and variant == "sam":
            InferenceModelsSAMAdapter = _LazyModelClass(
                "inference.models.sam.segment_anything_inference_models:"
                "InferenceModelsSAMAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter((task, variant), InferenceModelsSAMAdapter)
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("interactive-instance-segmentation", "sam"), InferenceModelsSAMAdapter
            )
        elif task == "embed" and variant == "sam2":
            InferenceModelsSAM2Adapter = _LazyModelClass(
                "inference.models.sam2.segment_anything2_inference_models:"
                "InferenceModelsSAM2Adapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsSAM2Adapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("interactive-instance-segmentation", "sam2"),
                InferenceModelsSAM2Adapter,
            )
        elif task == "embed" and variant == "sam3":
            InferenceModelsSAM3Adapter = _LazyModelClass(
                "inference.models.sam3.segment_anything3_inference_models:"
                "InferenceModelsSAM3Adapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsSAM3Adapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("instance-segmentation", "sam3"), InferenceModelsSAM3Adapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("instance-segmentation", "sam3-large"), InferenceModelsSAM3Adapter
            )
        elif task == "interactive-segmentation" and variant == "sam3":
            InferenceModelsSAM3InteractiveAdapter = _LazyModelClass(
                "inference.models.sam3.visual_segmentation_inference_models:"
                "InferenceModelsSAM3InteractiveAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsSAM3InteractiveAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("interactive-instance-segmentation", "sam3"),
                InferenceModelsSAM3InteractiveAdapter,
            )
        elif task == "embed" and variant == "clip":
            InferenceModelsClipAdapter = _LazyModelClass(
                "inference.models.clip.clip_inference_models:InferenceModelsClipAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsClipAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("embedding", "clip"), InferenceModelsClipAdapter
            )
        elif task == "object-detection" and variant == "owlv2":
            InferenceModelsOwlV2Adapter = _LazyModelClass(
                "inference.models.owlv2.owlv2_inference_models:"
                "InferenceModelsOwlV2Adapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsOwlV2Adapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("open-vocabulary-object-detection", variant),
                InferenceModelsOwlV2Adapter,
            )
        elif task == "object-detection" and variant == "owlv2-finetuned":
            InferenceModelsRFInstantModelAdapter = _LazyModelClass(
                "inference.models.owlv2.rf_instant_inference_models:"
                "InferenceModelsRFInstantModelAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsRFInstantModelAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, "roboflow-instant"), InferenceModelsRFInstantModelAdapter
            )
        elif task == "gaze" and variant == "l2cs":
            InferenceModelsGazeAdapter = _LazyModelClass(
                "inference.models.gaze.gaze_inference_models:InferenceModelsGazeAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsGazeAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("gaze-detection", "l2cs-net"), InferenceModelsGazeAdapter
            )
        elif task in {"lmm", "text-image-pairs"} and (
            variant.startswith("smolvlm-2.2b")
            or variant.startswith("smolvlm2")
            or variant.startswith("smolvlm-256m")
        ):
            InferenceModelsSmolVLMAdapter = _LazyModelClass(
                "inference.models.smolvlm.smolvlm_inference_models:"
                "InferenceModelsSmolVLMAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsSmolVLMAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("vlm", "smolvlm-v2"), InferenceModelsSmolVLMAdapter
            )
        elif task == "depth-estimation" and variant == "depth-anything-v2":
            InferenceModelsDepthAnythingV2Adapter = _LazyModelClass(
                "inference.models.depth_anything_v2.depth_anything_v2_inference_model"
                "s:InferenceModelsDepthAnythingV2Adapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsDepthAnythingV2Adapter
            )
        elif task == "depth-estimation" and variant == "depth-anything-v3":
            InferenceModelsDepthAnythingV3Adapter = _LazyModelClass(
                "inference.models.depth_anything_v3.depth_anything_v3_inference_model"
                "s:InferenceModelsDepthAnythingV3Adapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsDepthAnythingV3Adapter
            )
        elif task == "lmm" and variant == "moondream2":
            InferenceModelsMoondream2Adapter = _LazyModelClass(
                "inference.models.moondream2.moondream2_inference_models:"
                "InferenceModelsMoondream2Adapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsMoondream2Adapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("vlm", "moondream2"), InferenceModelsMoondream2Adapter
            )
        elif task == "ocr" and variant == "doctr":
            InferenceModelsDocTRAdapter = _LazyModelClass(
                "inference.models.doctr.doctr_model_inference_models:"
                "InferenceModelsDocTRAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsDocTRAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("structured-ocr", "doctr"), InferenceModelsDocTRAdapter
            )
        elif task == "ocr" and variant == "easy_ocr":
            InferenceModelsEasyOCRAdapter = _LazyModelClass(
                "inference.models.easy_ocr.easy_ocr_inference_models:"
                "InferenceModelsEasyOCRAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsEasyOCRAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("structured-ocr", "easy-ocr"), InferenceModelsEasyOCRAdapter
            )
        elif task == "ocr" and variant == "trocr":
            InferenceModelsTrOCRAdapter = _LazyModelClass(
                "inference.models.trocr.trocr_inference_models:"
                "InferenceModelsTrOCRAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsTrOCRAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("text-only-ocr", "tr-ocr"), InferenceModelsTrOCRAdapter
            )
        elif task == "object-detection" and variant == "grounding-dino":
            InferenceModelsGroundingDINOAdapter = _LazyModelClass(
                "inference.models.grounding_dino.grounding_dino_inference_models:"
                "InferenceModelsGroundingDINOAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsGroundingDINOAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("open-vocabulary-object-detection", "grounding-dino"),
                InferenceModelsGroundingDINOAdapter,
            )
        elif task == "embed" and variant == "perception_encoder":
            InferenceModelsPerceptionEncoderAdapter = _LazyModelClass(
                "inference.models.perception_encoder.perception_encoder_inference_mod"
                "els:InferenceModelsPerceptionEncoderAdapter"
            )

            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsPerceptionEncoderAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                ("embedding", "perception-encoder"),
                InferenceModelsPerceptionEncoderAdapter,
            )
        elif task == "semantic-segmentation" and variant == "deeplabv3plus":
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, variant), InferenceModelsSemanticSegmentationAdapter
            )
            ROBOFLOW_MODEL_TYPES.set_adapter(
                (task, "deep-lab-v3-plus"), InferenceModelsSemanticSegmentationAdapter
            )

    # Exact (taskType, modelArchitecture) tuples returned by
    # /models/v1/external/stat for entries backed by generic inference-models
    # adapters. These complement the legacy variant aliases above.
    ROBOFLOW_MODEL_TYPES.update(
        {
            ("object-detection", "rfdetr"): InferenceModelsObjectDetectionAdapter,
            ("object-detection", "yolo26"): InferenceModelsObjectDetectionAdapter,
            ("object-detection", "yololite"): InferenceModelsObjectDetectionAdapter,
            ("object-detection", "yolonas"): InferenceModelsObjectDetectionAdapter,
            ("object-detection", "yolov10"): InferenceModelsObjectDetectionAdapter,
            ("object-detection", "yolov11"): InferenceModelsObjectDetectionAdapter,
            ("object-detection", "yolov12"): InferenceModelsObjectDetectionAdapter,
            ("object-detection", "yolov5"): InferenceModelsObjectDetectionAdapter,
            ("object-detection", "yolov8"): InferenceModelsObjectDetectionAdapter,
            ("object-detection", "yolov9"): InferenceModelsObjectDetectionAdapter,
            ("instance-segmentation", "rfdetr"): (
                InferenceModelsInstanceSegmentationAdapter
            ),
            ("instance-segmentation", "segment-anything-2-rt"): (
                InferenceModelsInstanceSegmentationAdapter
            ),
            ("instance-segmentation", "yolact"): (
                InferenceModelsInstanceSegmentationAdapter
            ),
            ("instance-segmentation", "yolo26"): (
                InferenceModelsInstanceSegmentationAdapter
            ),
            ("instance-segmentation", "yolov11"): (
                InferenceModelsInstanceSegmentationAdapter
            ),
            ("instance-segmentation", "yolov5"): (
                InferenceModelsInstanceSegmentationAdapter
            ),
            ("instance-segmentation", "yolov7"): (
                InferenceModelsInstanceSegmentationAdapter
            ),
            ("instance-segmentation", "yolov8"): (
                InferenceModelsInstanceSegmentationAdapter
            ),
            ("keypoint-detection", "rfdetr"): (
                InferenceModelsKeyPointsDetectionAdapter
            ),
            ("keypoint-detection", "yolo26"): (
                InferenceModelsKeyPointsDetectionAdapter
            ),
            ("keypoint-detection", "yolov11"): (
                InferenceModelsKeyPointsDetectionAdapter
            ),
            ("keypoint-detection", "yolov8"): (
                InferenceModelsKeyPointsDetectionAdapter
            ),
            ("semantic-segmentation", "deep-lab-v3-plus"): (
                InferenceModelsSemanticSegmentationAdapter
            ),
            ("semantic-segmentation", "yolo26"): (
                InferenceModelsSemanticSegmentationAdapter
            ),
            ("classification", "dinov3_probe"): InferenceModelsClassificationAdapter,
            ("classification", "resnet"): InferenceModelsClassificationAdapter,
            ("classification", "vit"): InferenceModelsClassificationAdapter,
            ("classification", "yolov11"): InferenceModelsClassificationAdapter,
            ("classification", "yolov8"): InferenceModelsClassificationAdapter,
            ("multi-label-classification", "dinov3_probe"): (
                InferenceModelsClassificationAdapter
            ),
            ("multi-label-classification", "resnet"): (
                InferenceModelsClassificationAdapter
            ),
            ("multi-label-classification", "vit"): (
                InferenceModelsClassificationAdapter
            ),
        }
    )

    # YOLO26 semantic segmentation is inference_models-only (no legacy implementation),
    # so we add entries directly rather than swapping existing ones.
    for variant in [
        "yolo26",
        "yolo26n-sem",
        "yolo26s-sem",
        "yolo26m-sem",
        "yolo26l-sem",
        "yolo26x-sem",
    ]:
        ROBOFLOW_MODEL_TYPES[("semantic-segmentation", variant)] = (
            InferenceModelsSemanticSegmentationAdapter
        )

    # YOLO26 depth estimation is inference_models-only (no legacy implementation),
    # so we add entries directly rather than swapping existing ones.
    if DEPTH_ESTIMATION_ENABLED:
        InferenceModelsDepthEstimationAdapter = _LazyModelClass(
            "inference.core.models.inference_models_adapters:"
            "InferenceModelsDepthEstimationAdapter"
        )

        for variant in [
            "yolo26",
            "yolo26n-depth",
            "yolo26s-depth",
            "yolo26m-depth",
            "yolo26l-depth",
            "yolo26x-depth",
        ]:
            ROBOFLOW_MODEL_TYPES[("depth-estimation", variant)] = (
                InferenceModelsDepthEstimationAdapter
            )

    # RFDETR keypoint detection is inference_models-only (no legacy implementation),
    # so we add entries directly rather than swapping existing ones.
    ROBOFLOW_MODEL_TYPES[("keypoint-detection", "rfdetr-keypoint-preview")] = (
        InferenceModelsKeyPointsDetectionAdapter
    )
    ROBOFLOW_MODEL_TYPES[("keypoint-detection", "rfdetr-keypoint-two-stage")] = (
        InferenceModelsKeyPointsDetectionAdapter
    )
    ROBOFLOW_MODEL_TYPES[("keypoint-detection", "rfdetr-keypoint-stage2")] = (
        InferenceModelsKeyPointsDetectionAdapter
    )

    # PatchCore and FoundAD anomaly detection are inference_models-only
    # (no legacy implementation), so we add entries directly.
    InferenceModelsAnomalyDetectionAdapter = _LazyModelClass(
        "inference.core.models.inference_models_adapters:"
        "InferenceModelsAnomalyDetectionAdapter"
    )

    for variant in ["patchcore", "foundad"]:
        ROBOFLOW_MODEL_TYPES[("classification", variant)] = (
            InferenceModelsAnomalyDetectionAdapter
        )

    # YOLOLite is inference_models-only (no legacy implementation),
    # so we add entries directly rather than swapping existing ones.
    for variant in [
        "yololite",
        "yololite-n",
        "yololite-s",
        "yololite-m",
        "yololite-l",
        "yololite-xl",
        "yololite-edge-n",
        "yololite-edge-s",
        "yololite-edge-m",
        "yololite-edge-l",
        "yololite-edge-xl",
    ]:
        ROBOFLOW_MODEL_TYPES[("object-detection", variant)] = (
            InferenceModelsObjectDetectionAdapter
        )

    # inference-models only, needs to be added here
    if QWEN_3_5_ENABLED:
        if VLLM_PROXY_ENABLED:
            _Qwen35ExplicitModelClass = _LazyModelClass(
                "inference.models.vllm_proxy.qwen3_5_vllm:Qwen35VLLMProxy"
            )
        else:
            _Qwen35ExplicitModelClass = _LazyModelClass(
                "inference.models.qwen3_5vl.qwen3_5vl_inference_models:"
                "InferenceModelsQwen35VLAdapter"
            )

        for variant in [
            "qwen3_5-0.8b",
            "qwen3_5-2b",
            "qwen3_5-4b",
            "qwen3_5-0.8b-peft",
            "qwen3_5-2b-peft",
        ]:
            ROBOFLOW_MODEL_TYPES[("lmm", variant)] = _Qwen35ExplicitModelClass
            ROBOFLOW_MODEL_TYPES[("text-image-pairs", variant)] = (
                _Qwen35ExplicitModelClass
            )
        ROBOFLOW_MODEL_TYPES[("vlm", "qwen_3_5")] = _Qwen35ExplicitModelClass
        ROBOFLOW_MODEL_TYPES[("vlm", "qwen3_5")] = _Qwen35ExplicitModelClass

    if QWEN_3_8_ENABLED:
        _Qwen38ExplicitModelClass = None
        if VLLM_PROXY_ENABLED:
            _Qwen38ExplicitModelClass = _LazyModelClass(
                "inference.models.vllm_proxy.qwen3_8_vllm:Qwen38VLLMProxy"
            )
        else:
            _Qwen38ExplicitModelClass = _LazyModelClass(
                "inference.models.qwen3_8vl.qwen3_8vl_inference_models:"
                "InferenceModelsQwen38VLAdapter",
                optional=True,
                warning_message=_QWEN_3_8_DEPENDENCY_WARNING,
                warning_category=UserWarning,
            )

        if _Qwen38ExplicitModelClass is not None:
            for variant in [
                "qwen3_8-27b",
            ]:
                ROBOFLOW_MODEL_TYPES[("lmm", variant)] = _Qwen38ExplicitModelClass
                ROBOFLOW_MODEL_TYPES[("text-image-pairs", variant)] = (
                    _Qwen38ExplicitModelClass
                )
            ROBOFLOW_MODEL_TYPES[("vlm", "qwen3_8")] = _Qwen38ExplicitModelClass

    if GLM_OCR_ENABLED:
        InferenceModelsGLMOCRAdapter = _LazyModelClass(
            "inference.models.glm_ocr.glm_ocr_inference_models:"
            "InferenceModelsGLMOCRAdapter"
        )

        ROBOFLOW_MODEL_TYPES[("vlm", "glm-ocr")] = InferenceModelsGLMOCRAdapter

    # Models loaded directly from a local directory
    # (ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES).
    # Task type is read from the local model_config.json; the adapter forwards the path
    # to
    # AutoModel.from_pretrained with allow_direct_local_storage_loading=True.
    for local_task, local_adapter in [
        ("object-detection", InferenceModelsObjectDetectionAdapter),
        ("instance-segmentation", InferenceModelsInstanceSegmentationAdapter),
        ("keypoint-detection", InferenceModelsKeyPointsDetectionAdapter),
        ("classification", InferenceModelsClassificationAdapter),
        ("semantic-segmentation", InferenceModelsSemanticSegmentationAdapter),
    ]:
        ROBOFLOW_MODEL_TYPES[(local_task, LOCAL_INFERENCE_MODELS_MODEL_TYPE)] = (
            local_adapter
        )


_MODEL_CLASS_EXPORTS = {
    name: value
    for name, value in list(globals().items())
    if not name.startswith("_") and isinstance(value, _LazyModelClass)
}
for _export_name in _MODEL_CLASS_EXPORTS:
    del globals()[_export_name]

__all__ = sorted(
    {name for name in globals() if not name.startswith("_")}
    | _MODEL_CLASS_EXPORTS.keys()
)


def __getattr__(name: str):
    """Resolve a model-class export on first attribute access."""
    reference = _MODEL_CLASS_EXPORTS.get(name)
    if reference is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    model_class = reference._resolve()
    return model_class


def __dir__():
    """Include deferred model-class exports in module introspection."""
    names = sorted(set(globals()) | _MODEL_CLASS_EXPORTS.keys())
    return names
