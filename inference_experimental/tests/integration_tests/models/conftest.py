import os.path
import zipfile

import pytest
import requests
from filelock import FileLock

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
MODELS_DIR = os.path.join(ASSETS_DIR, "models")
CLIP_RN50_TORCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/clip_packages/RN50/torch/model.pt"
CLIP_RN50_ONNX_VISUAL = "https://storage.googleapis.com/roboflow-tests-assets/clip_packages/RN50/onnx/visual.onnx"
CLIP_RN50_ONNX_TEXTUAL = "https://storage.googleapis.com/roboflow-tests-assets/clip_packages/RN50/onnx/textual.onnx"
PE_MODEL_URL = "https://storage.googleapis.com/roboflow-tests-assets/perception-encoder/pe-core-b16-224/model.pt"
PE_CONFIG_URL = "https://storage.googleapis.com/roboflow-tests-assets/perception-encoder/pe-core-b16-224/config.json"
FLORENCE2_BASE_FT_URL = "https://storage.googleapis.com/roboflow-tests-assets/florence2/florence-2-base-converted-for-transformers-056.zip"
FLORENCE2_LARGE_FT_URL = "https://storage.googleapis.com/roboflow-tests-assets/florence2/florence-2-large-converted-for-transformers-056.zip"
QWEN25VL_3B_FT_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/qwen/qwen25vl-3b.zip"
)
PALIGEMMA_BASE_FT_URL = "https://storage.googleapis.com/roboflow-tests-assets/paligemma/paligemma2-3b-pt-224.zip"
SMOLVLM_BASE_FT_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/smolvlm/smolvlm-256m.zip"
)
MOONDREAM2_BASE_FT_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/moondream2/moondream2-2b.zip"
)
COIN_COUNTING_RFDETR_NANO_TORCH_CS_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/coin-counting-rfdetr-nano-torch-cs-stretch-640.zip"
COIN_COUNTING_RFDETR_NANO_ONNX_CS_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-onnx-cs-stretch-640.zip"
COIN_COUNTING_RFDETR_NANO_ONNX_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-onnx-static-crop-letterbox-640.zip"
COIN_COUNTING_RFDETR_NANO_TORCH_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-torch-static-crop-letterbox-640.zip"

COIN_COUNTING_RFDETR_NANO_ONNX_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-onnx-center-crop-640.zip"
COIN_COUNTING_RFDETR_NANO_TORCH_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-torch-center-crop-640.zip"
COIN_COUNTING_RFDETR_NANO_ONNX_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-onnx-static-crop-center-crop-640.zip"
COIN_COUNTING_RFDETR_NANO_TORCH_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-nano-torch-static-crop-center-crop-640.zip"
OG_RFDETR_WEIGHTS_URL = "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth"

COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-letterbox.zip"
COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_LETTERBOX_FUSED_NMS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-letterbox-fused-nms.zip"
COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-static-bs-letterbox.zip"
COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-letterbox.zip"
COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_LETTERBOX_FUSED_NMS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-letterbox-fused-nms.zip"
COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-static-crop-stretch.zip"
COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_NMS_FUSED_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-static-crop-stretch-nms-fused.zip"
COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-static-bs-static-crop-stretch.zip"
COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-static-crop-stretch.zip"
COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_STATIC_CROP_STRETCH_NMS_FUSED_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-static-crop-stretch-nms-fused.zip"
COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-center-crop.zip"
COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_CENTER_CROP_NMS_FUSED_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-dynamic-bs-center-crop-fused-nms.zip"
COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-onnx-static-bs-center-crop.zip"
COIN_COUNTING_YOLOV8N_TORCHSCRIPT_STATIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-center-crop.zip"
COIN_COUNTING_YOLOV8N_TORCHSCRIPT_STATIC_BS_CENTER_CROP_FUSED_NMS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-torchscript-static-bs-center-crop-fused-nms.zip"
COIN_COUNTING_YOLO5_ONNX_STATIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov5-od-static-bs-letterbox-onnx.zip"
COIN_COUNTING_YOLO5_ONNX_DYNAMIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov5-od-dynamic-bs-letterbox-onnx.zip"

COIN_COUNTING_YOLO_NAS_ONNX_DYNAMIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-dynamic-bs-letterbox.zip"
COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-static-bs-letterbox.zip"
COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-static-bs-static-crop-letterbox.zip"
COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-static-bs-static-crop-stretch.zip"
COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-static-bs-static-crop-center-crop.zip"
COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolo-nas-onnx-static-bs-center-crop.zip"

COIN_COUNTING_YOLACT_ONNX_STATIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolact-static-bs-letterbox-onnx.zip"
COIN_COUNTING_YOLACT_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolact-static-bs-static-crop-stretch-onnx.zip"
COIN_COUNTING_YOLACT_ONNX_STATIC_BS_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolact-static-bs-stretch-onnx.zip"

ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-onnx-dynamic-bs-stretch.zip"
ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STRETCH_FUSED_NMS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-onnx-dynamic-bs-stretch-fused-nms.zip"
ASL_YOLOV8N_SEG_ONNX_STATIC_BS_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-onnx-static-bs-stretch.zip"
ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-torchscript-static-bs-stretch.zip"
ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STRETCH_FUSED_NMS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-torchscript-static-bs-stretch-fused-nms.zip"
ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-onnx-dynamic-bs-static-crop-stretch.zip"
ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_FUSED_NMS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-onnx-dynamic-bs-static-crop-stretch-fused-nms.zip"
ASL_YOLOV8N_SEG_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-onnx-static-bs-static-crop-stretch.zip"
ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-torchscript-static-bs-static-crop-stretch.zip"
ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STATIC_CROP_STRETCH_FUSED_NMS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-torchscript-static-bs-static-crop-stretch-fused-nms.zip"
ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-onnx-dynamic-bs-static-crop-center-crop.zip"
ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STATIC_CROP_CENTER_CROP_FUSED_NMS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-onnx-dynamic-bs-static-crop-center-crop-fused-nms.zip"
ASL_YOLOV8N_SEG_ONNX_STATIC_BS_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-onnx-static-bs-static-crop-center-crop.zip"
ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-torchscript-static-bs-static-crop-center-crop.zip"
ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-onnx-dynamic-bs-center-crop.zip"
ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-torchscript-static-bs-center-crop.zip"
ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-onnx-dynamic-bs-static-crop-letterbox.zip"
ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-seg-torchscript-static-bs-static-crop-letterbox.zip"
ASL_YOLOv5_SEG_ONNX_STATIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov5-seg-static-bs-letterbox-onnx.zip"
ASL_YOLOv7_SEG_ONNX_STATIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov7-seg-static-bs-letterbox-onnx.zip"

DEEP_LAB_V3_SEGMENTATION_ONNX_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/deep-lab-v3-plus-segmentation-stretch-onnx.zip"
DEEP_LAB_V3_SEGMENTATION_TORCH_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/deep-lab-v3-plus-segmentation-stretch-torch.zip"
DEEP_LAB_V3_SEGMENTATION_ONNX_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/deep-lab-v3-plus-segmentation-center-crop-letterbox-onnx.zip"
DEEP_LAB_V3_SEGMENTATION_TORCH_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/deep-lab-v3-plus-segmentation-center-crop-letterbox-torch.zip"
DEEP_LAB_V3_SEGMENTATION_ONNX_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/deep-lab-v3-plus-segmentation-static-crop-center-crop-onnx.zip"
DEEP_LAB_V3_SEGMENTATION_TORCH_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/deep-lab-v3-plus-segmentation-static-crop-center-crop-torch.zip"

FLOWERS_MULTI_LABEL_VIT_HF_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/vit-multi-label-hugging-face.zip"
FLOWERS_MULTI_LABEL_VIT_ONNX_DYNAMIC_BS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/vit-multi-label-onnx-dynamic-bs.zip"
FLOWERS_MULTI_LABEL_VIT_ONNX_STATIC_BS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/vit-multi-label-onnx-static-bs.zip"

FLOWERS_MULTI_LABEL_RES_NET_TORCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/res-net-multi-label-torch.zip"
FLOWERS_MULTI_LABEL_RES_NET_ONNX_DYNAMIC_BS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/res-net-multi-label-onnx-dynamic-bs.zip"
FLOWERS_MULTI_LABEL_RES_NET_ONNX_STATIC_BS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/res-net-multi-label-onnx-static-bs.zip"

VEHICLES_MULTI_CLASS_VIT_HF_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/vit-multi-class-hugging-face.zip"
VEHICLES_MULTI_CLASS_VIT_ONNX_DYNAMIC_BS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/vit-multi-class-onnx-dynamic-bs.zip"
VEHICLES_MULTI_CLASS_VIT_ONNX_STATIC_BS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/vit-multi-class-onnx-static-bs.zip"

VEHICLES_MULTI_CLASS_RES_NET_TORCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/res-net-multi-class-torch.zip"
VEHICLES_MULTI_CLASS_RES_NET_ONNX_DYNAMIC_BS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/res-net-multi-class-onnx-dynamic-bs.zip"
VEHICLES_MULTI_CLASS_RES_NET_ONNX_STATIC_BS_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/res-net-multi-class-onnx-static-bs.zip"

YOLOV8N_POSE_ONNX_STATIC_CENTER_CROP_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-onnx-static-center-crop.zip"
YOLOV8N_POSE_ONNX_STATIC_STATIC_CROP_CENTER_CROP_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-onnx-static-static-crop-center-crop.zip"
YOLOV8N_POSE_ONNX_STATIC_STATIC_CROP_LETTERBOX_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-onnx-static-static-crop-letterbox.zip"
YOLOV8N_POSE_ONNX_STATIC_STATIC_CROP_STRETCH_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-onnx-static-static-crop-stretch.zip"
YOLOV8N_POSE_ONNX_DYNAMIC_CENTER_CROP_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-onnx-dynamic-center-crop.zip"
YOLOV8N_POSE_ONNX_DYNAMIC_STATIC_CROP_CENTER_CROP_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-onnx-dynamic-static-crop-center-crop.zip"
YOLOV8N_POSE_ONNX_DYNAMIC_STATIC_CROP_LETTERBOX_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-onnx-dynamic-static-crop-letterbox.zip"
YOLOV8N_POSE_ONNX_DYNAMIC_STATIC_CROP_STRETCH_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-onnx-dynamic-static-crop-stretch.zip"
YOLOV8N_POSE_ONNX_DYNAMIC_NMS_FUSED_CENTER_CROP_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-onnx-dynamic-nms-fused-center-crop.zip"
YOLOV8N_POSE_ONNX_DYNAMIC_NMS_FUSED_STATIC_CROP_CENTER_CROP_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-onnx-dynamic-nms-fused-static-crop-center-crop.zip"
YOLOV8N_POSE_TORCHSCRIPT_STATIC_CENTER_CROP_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-torchscript-static-center-crop.zip"
YOLOV8N_POSE_TORCHSCRIPT_STATIC_STATIC_CROP_CENTER_CROP_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-torchscript-static-static-crop-center-crop.zip"
YOLOV8N_POSE_TORCHSCRIPT_STATIC_STATIC_CROP_LETTERBOX_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-torchscript-static-static-crop-letterbox.zip"
YOLOV8N_POSE_TORCHSCRIPT_STATIC_STATIC_CROP_STRETCH_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-torchscript-static-static-crop-stretch.zip"
YOLOV8N_POSE_TORCHSCRIPT_STATIC_NMS_FUSED_CENTER_CROP_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-torchscript-static-nms-fused-center-crop.zip"
YOLOV8N_POSE_TORCHSCRIPT_STATIC_NMS_FUSED_STATIC_CROP_CENTER_CROP_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8n-pose-torchscript-static-nms-fused-static-crop-center-crop.zip"

YOLOV8_CLS_ONNX_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/yolov8-cls-onnx-static-bs.zip"

SNAKES_RFDETR_SEG_TORCH_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-torch-stretch.zip"
SNAKES_RFDETR_SEG_ONNX_STATIC_BS_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-onnx-static-bs-stretch.zip"
SNAKES_RFDETR_SEG_TORCH_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-torch-letterbox.zip"
SNAKES_RFDETR_SEG_ONNX_STATIC_BS_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-onnx-static-bs-letterbox.zip"
SNAKES_RFDETR_SEG_TORCH_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-torch-center-crop.zip"
SNAKES_RFDETR_SEG_ONNX_STATIC_BS_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-onnx-static-bs-center-crop.zip"
SNAKES_RFDETR_SEG_TORCH_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-torch-static-crop-stretch.zip"
SNAKES_RFDETR_SEG_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-onnx-static-bs-static-crop-stretch.zip"
SNAKES_RFDETR_SEG_TORCH_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-torch-static-crop-letterbox.zip"
SNAKES_RFDETR_SEG_ONNX_STATIC_BS_STATIC_CROP_LETTERBOX_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-onnx-static-bs-static-crop-letterbox.zip"
SNAKES_RFDETR_SEG_TORCH_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-torch-static-crop-center-crop.zip"
SNAKES_RFDETR_SEG_ONNX_STATIC_BS_STATIC_CROP_CENTER_CROP_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/rfdetr-seg-onnx-static-bs-static-crop-center-crop.zip"

DINOV3_CLASSIFICATION_ONNX_STATIC_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/dinov3-classification-onnx.zip"
DINOV3_MULTI_LABEL_ONNX_STATIC_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/dinov3-multi-label-onnx.zip"
DINOV3_CLASSIFICATION_TORCH_STATIC_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/dinov3-classification-torch.zip"
DINOV3_MULTI_LABEL_TORCH_STATIC_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/dinov3-multi-label-torch.zip"

DEPTH_ANYTHING_V2_SMALL_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/depth-anything-v2.zip"
DOCTR_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/doctr-dbnet-rn50-crnn-vgg16.zip"
EASY_OCR_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/easy-ocr-english.zip"
TROCR_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/tr-ocr-small-printed.zip"
MEDIAPIPE_FACE_DETECTOR_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/mediapipe-face-detector.zip"
L2CS_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/l2cs-net.zip"
OWLv2_PACKAGE_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/owlv2.zip"
)
INSTANT_MODEL_COIN_COUNTING_PACKAGE_URL = "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/instant-model-coin-counting.zip"
SAM_PACKAGE_URL = (
    "https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/sam.zip"
)


@pytest.fixture(scope="module")
def original_clip_download_dir() -> str:
    clip_dir = os.path.join(MODELS_DIR, "clip_original")
    os.makedirs(clip_dir, exist_ok=True)
    return clip_dir


@pytest.fixture(scope="module")
def clip_rn50_pytorch_path() -> str:
    package_path = os.path.join(MODELS_DIR, "clip_rn50", "torch")
    os.makedirs(package_path, exist_ok=True)
    model_path = os.path.join(package_path, "model.pt")
    _download_if_not_exists(file_path=model_path, url=CLIP_RN50_TORCH_URL)
    return package_path


@pytest.fixture(scope="module")
def clip_rn50_onnx_path() -> str:
    package_path = os.path.join(MODELS_DIR, "clip_rn50", "onnx")
    os.makedirs(package_path, exist_ok=True)
    visual_path = os.path.join(package_path, "visual.onnx")
    textual_path = os.path.join(package_path, "textual.onnx")
    _download_if_not_exists(file_path=visual_path, url=CLIP_RN50_ONNX_VISUAL)
    _download_if_not_exists(file_path=textual_path, url=CLIP_RN50_ONNX_TEXTUAL)
    return package_path


@pytest.fixture(scope="module")
def perception_encoder_path() -> str:
    package_path = os.path.join(MODELS_DIR, "perception_encoder")
    os.makedirs(package_path, exist_ok=True)
    model_path = os.path.join(package_path, "model.pt")
    config_path = os.path.join(package_path, "config.json")
    _download_if_not_exists(file_path=model_path, url=PE_MODEL_URL)
    _download_if_not_exists(file_path=config_path, url=PE_CONFIG_URL)
    return package_path


def _download_if_not_exists(file_path: str, url: str, lock_timeout: int = 180) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lock_path = f"{file_path}.lock"
    with FileLock(lock_file=lock_path, timeout=lock_timeout):
        if os.path.exists(file_path):
            return None
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)


@pytest.fixture(scope="module")
def florence2_base_ft_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "florence2")
    unzipped_package_path = os.path.join(package_dir, "florence-2-base")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "base-ft.zip")
    _download_if_not_exists(file_path=zip_path, url=FLORENCE2_BASE_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=180):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(package_dir)
    return unzipped_package_path


@pytest.fixture(scope="module")
def florence2_large_ft_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "florence2")
    unzipped_package_path = os.path.join(package_dir, "florence-2-base")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "large-ft.zip")
    _download_if_not_exists(file_path=zip_path, url=FLORENCE2_LARGE_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=180):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(package_dir)
    return unzipped_package_path


@pytest.fixture(scope="module")
def qwen25vl_3b_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "qwen25vl-3b")
    unzipped_package_path = os.path.join(package_dir, "weights")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "qwen25vl-3b.zip")
    _download_if_not_exists(file_path=zip_path, url=QWEN25VL_3B_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=180):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(package_dir)
    return unzipped_package_path


@pytest.fixture(scope="module")
def paligemma_3b_224_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "paligemma2-3b-pt-224")
    unzipped_package_path = os.path.join(package_dir, "weights")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "paligemma2-3b-pt-224.zip")
    _download_if_not_exists(file_path=zip_path, url=PALIGEMMA_BASE_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=180):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(package_dir)
    return unzipped_package_path


@pytest.fixture(scope="module")
def smolvlm_256m_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "smolvlm-256m")
    unzipped_package_path = os.path.join(package_dir, "weights")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "smolvlm-256m.zip")
    _download_if_not_exists(file_path=zip_path, url=SMOLVLM_BASE_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=180):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(package_dir)
    return unzipped_package_path


@pytest.fixture(scope="module")
def moondream2_path() -> str:
    package_dir = os.path.join(MODELS_DIR, "moondream2")
    unzipped_package_path = os.path.join(package_dir, "moondream2-2b")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "moondream2-2b.zip")
    _download_if_not_exists(file_path=zip_path, url=MOONDREAM2_BASE_FT_URL)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=180):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(unzipped_package_path)
    return unzipped_package_path


def download_model_package(
    model_package_zip_url: str,
    package_name: str,
) -> str:
    package_dir = os.path.join(MODELS_DIR, package_name)
    unzipped_package_path = os.path.join(package_dir, "unpacked")
    os.makedirs(package_dir, exist_ok=True)
    zip_path = os.path.join(package_dir, "package.zip")
    _download_if_not_exists(file_path=zip_path, url=model_package_zip_url)
    lock_path = f"{unzipped_package_path}.lock"
    with FileLock(lock_path, timeout=180):
        if not os.path.exists(unzipped_package_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(unzipped_package_path)
    return unzipped_package_path


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_torch_cs_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_TORCH_CS_STRETCH_URL,
        package_name="coin-counting-rfdetr-nano-torch-cs-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_onnx_cs_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_ONNX_CS_STRETCH_URL,
        package_name="coin-counting-rfdetr-nano-onnx-cs-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_ONNX_STATIC_CROP_LETTERBOX_URL,
        package_name="coin-counting-rfdetr-nano-onnx-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_torch_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_TORCH_STATIC_CROP_LETTERBOX_URL,
        package_name="coin-counting-rfdetr-nano-torch-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_onnx_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_ONNX_CENTER_CROP_URL,
        package_name="coin-counting-rfdetr-nano-onnx-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_torch_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_TORCH_CENTER_CROP_URL,
        package_name="coin-counting-rfdetr-nano-torch-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_ONNX_STATIC_CROP_CENTER_CROP_URL,
        package_name="coin-counting-rfdetr-nano-onnx-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_rfdetr_nano_torch_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_RFDETR_NANO_TORCH_STATIC_CROP_CENTER_CROP_URL,
        package_name="coin-counting-rfdetr-nano-torch-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def og_rfdetr_base_weights() -> str:
    package_path = os.path.join(MODELS_DIR, "og-rfdetr-base")
    os.makedirs(package_path, exist_ok=True)
    model_path = os.path.join(package_path, "model.pt")
    _download_if_not_exists(file_path=model_path, url=OG_RFDETR_WEIGHTS_URL)
    return model_path


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_LETTERBOX_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_letterbox_fused_nms_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_LETTERBOX_FUSED_NMS_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-letterbox-fused-nms",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_static_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_LETTERBOX_URL,
        package_name="coin-counting-yolov8n-onnx-static-bs-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_torch_script_static_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_LETTERBOX_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_LETTERBOX_FUSED_NMS_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-fused-nms-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_static_crop_stretch_nms_fused_package() -> (
    str
):
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_NMS_FUSED_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-static-crop-stretch-nms-fused",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_static_bs_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="coin-counting-yolov8n-onnx-static-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package() -> (
    str
):
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCH_SCRIPT_STATIC_BS_STATIC_CROP_STRETCH_NMS_FUSED_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-static-crop-stretch-fused-nms",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_center_crop_package() -> str:
    # THIS MODEL IS KIND OF SHITTY IN TERMS OF OUTPUTS, IT'S HERE JUST TO VERIFY PRE- / POST- processing
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_CENTER_CROP_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_dynamic_bs_center_crop_fused_nms_package() -> str:
    # THIS MODEL IS KIND OF SHITTY IN TERMS OF OUTPUTS, IT'S HERE JUST TO VERIFY PRE- / POST- processing
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_DYNAMIC_BS_CENTER_CROP_NMS_FUSED_URL,
        package_name="coin-counting-yolov8n-onnx-dynamic-bs-center-crop-fused-nms",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_onnx_static_bs_center_crop_package() -> str:
    # THIS MODEL IS KIND OF SHITTY IN TERMS OF OUTPUTS, IT'S HERE JUST TO VERIFY PRE- / POST- processing
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_ONNX_STATIC_BS_CENTER_CROP_URL,
        package_name="coin-counting-yolov8n-onnx-static-bs-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_toch_script_static_bs_center_crop_package() -> str:
    # THIS MODEL IS KIND OF SHITTY IN TERMS OF OUTPUTS, IT'S HERE JUST TO VERIFY PRE- / POST- processing
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCHSCRIPT_STATIC_BS_CENTER_CROP_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package() -> str:
    # THIS MODEL IS KIND OF SHITTY IN TERMS OF OUTPUTS, IT'S HERE JUST TO VERIFY PRE- / POST- processing
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLOV8N_TORCHSCRIPT_STATIC_BS_CENTER_CROP_FUSED_NMS_URL,
        package_name="coin-counting-yolov8n-torchscript-static-bs-center-fused-nms-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_dynamic_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_DYNAMIC_BS_LETTERBOX_URL,
        package_name="coin-counting-yolo-nas-dynamic-bs-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_static_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_LETTERBOX_URL,
        package_name="coin-counting-yolo-nas-static-bs-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_LETTERBOX_URL,
        package_name="coin-counting-yolo-nas-static-bs-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="coin-counting-yolo-nas-static-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_STATIC_CROP_CENTER_CROP_URL,
        package_name="coin-counting-yolo-nas-static-bs-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def coin_counting_yolo_nas_onnx_static_bs_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO_NAS_ONNX_STATIC_BS_CENTER_CROP_URL,
        package_name="coin-counting-yolo-nas-static-bs-center-crop",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_onnx_seg_dynamic_bs_stretch() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STRETCH_URL,
        package_name="asl-yolov8n-seg-dynamic-bs-stretch",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_onnx_seg_dynamic_bs_stretch_fused_nms() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STRETCH_FUSED_NMS_URL,
        package_name="asl-yolov8n-seg-dynamic-bs-stretch-fused-nms",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_onnx_seg_static_bs_stretch() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_ONNX_STATIC_BS_STRETCH_URL,
        package_name="asl-yolov8n-seg-static-bs-stretch",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_torchscript_seg_static_bs_stretch() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STRETCH_URL,
        package_name="asl-yolov8n-seg-torchscript-static-bs-stretch",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_torchscript_seg_static_bs_stretch_fused_nms() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STRETCH_FUSED_NMS_URL,
        package_name="asl-yolov8n-seg-torchscript-static-bs-stretch-fused-nms",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="asl-yolov8n-seg-dynamic-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_onnx_seg_dynamic_bs_static_crop_stretch_fused_nms() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STATIC_CROP_STRETCH_FUSED_NMS_URL,
        package_name="asl-yolov8n-seg-dynamic-bs-static-crop-stretch-fused-nms",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_onnx_seg_static_bs_static_crop_stretch() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="asl-yolov8n-seg-static-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="asl-yolov8n-seg-torchscript-static-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_torchscript_seg_static_bs_static_crop_stretch_fused_nms() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STATIC_CROP_STRETCH_FUSED_NMS_URL,
        package_name="asl-yolov8n-seg-torchscript-static-bs-static-crop-stretch-fised-nms",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STATIC_CROP_CENTER_CROP_URL,
        package_name="asl-yolov8n-seg-dynamic-bs-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_onnx_seg_dynamic_bs_static_crop_center_crop_fused_nms() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STATIC_CROP_CENTER_CROP_FUSED_NMS_URL,
        package_name="asl-yolov8n-seg-dynamic-bs-static-crop-center-crop-fused-nms",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_onnx_seg_static_bs_static_crop_center_crop() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_ONNX_STATIC_BS_STATIC_CROP_CENTER_CROP_URL,
        package_name="asl-yolov8n-seg-static-bs-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_torchscript_seg_static_bs_static_crop_center_crop() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STATIC_CROP_CENTER_CROP_URL,
        package_name="asl-yolov8n-seg-torchscript-static-bs-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_onnx_seg_dynamic_bs_center_crop() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_CENTER_CROP_URL,
        package_name="asl-yolov8n-seg-dynamic-bs-center-crop",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_torchscript_seg_static_bs_center_crop() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_CENTER_CROP_URL,
        package_name="asl-yolov8n-seg-torchscript-static-bs-center-crop",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_onnx_seg_dynamic_bs_static_crop_letterbox() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_ONNX_DYNAMIC_BS_STATIC_CROP_LETTERBOX_URL,
        package_name="asl-yolov8n-seg-dynamic-bs-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def asl_yolov8n_torchscript_seg_static_bs_static_crop_letterbox() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOV8N_SEG_TORCHSCRIPT_STATIC_BS_STATIC_CROP_LETTERBOX_URL,
        package_name="asl-yolov8n-seg-torchscript-static-bs-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def balloons_deep_lab_v3_onnx_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=DEEP_LAB_V3_SEGMENTATION_ONNX_STRETCH_URL,
        package_name="balloons-deep-lab-v3-onnx-stretch",
    )


@pytest.fixture(scope="module")
def balloons_deep_lab_v3_torch_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=DEEP_LAB_V3_SEGMENTATION_TORCH_STRETCH_URL,
        package_name="balloons-deep-lab-v3-torch-stretch",
    )


@pytest.fixture(scope="module")
def balloons_deep_lab_v3_onnx_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=DEEP_LAB_V3_SEGMENTATION_ONNX_STATIC_CROP_LETTERBOX_URL,
        package_name="balloons-deep-lab-v3-onnx-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def balloons_deep_lab_v3_torch_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=DEEP_LAB_V3_SEGMENTATION_TORCH_STATIC_CROP_LETTERBOX_URL,
        package_name="balloons-deep-lab-v3-torch-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def balloons_deep_lab_v3_onnx_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=DEEP_LAB_V3_SEGMENTATION_ONNX_STATIC_CROP_CENTER_CROP_URL,
        package_name="balloons-deep-lab-v3-onnx-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def balloons_deep_lab_v3_torch_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=DEEP_LAB_V3_SEGMENTATION_TORCH_STATIC_CROP_CENTER_CROP_URL,
        package_name="balloons-deep-lab-v3-torch-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def flowers_multi_label_vit_hf_package() -> str:
    return download_model_package(
        model_package_zip_url=FLOWERS_MULTI_LABEL_VIT_HF_URL,
        package_name="flowers-multi-label-vit-hf",
    )


@pytest.fixture(scope="module")
def flowers_multi_label_vit_onnx_static_bs_package() -> str:
    return download_model_package(
        model_package_zip_url=FLOWERS_MULTI_LABEL_VIT_ONNX_STATIC_BS_URL,
        package_name="flowers-multi-label-vit-onnx-static-bs",
    )


@pytest.fixture(scope="module")
def flowers_multi_label_vit_onnx_dynamic_bs_package() -> str:
    return download_model_package(
        model_package_zip_url=FLOWERS_MULTI_LABEL_VIT_ONNX_DYNAMIC_BS_URL,
        package_name="flowers-multi-label-vit-onnx-dynamic-bs",
    )


@pytest.fixture(scope="module")
def flowers_multi_label_resnet_torch_package() -> str:
    return download_model_package(
        model_package_zip_url=FLOWERS_MULTI_LABEL_RES_NET_TORCH_URL,
        package_name="flowers-multi-label-resnet-torch",
    )


@pytest.fixture(scope="module")
def flowers_multi_label_resnet_onnx_static_bs_package() -> str:
    return download_model_package(
        model_package_zip_url=FLOWERS_MULTI_LABEL_RES_NET_ONNX_STATIC_BS_URL,
        package_name="flowers-multi-label-resnet-onnx-static-bs",
    )


@pytest.fixture(scope="module")
def flowers_multi_label_resnet_onnx_dynamic_bs_package() -> str:
    return download_model_package(
        model_package_zip_url=FLOWERS_MULTI_LABEL_RES_NET_ONNX_DYNAMIC_BS_URL,
        package_name="flowers-multi-label-resnet-onnx-dynamic-bs",
    )


@pytest.fixture(scope="module")
def vehicles_multi_class_vit_hf_package() -> str:
    return download_model_package(
        model_package_zip_url=VEHICLES_MULTI_CLASS_VIT_HF_URL,
        package_name="vehicles-multi-class-vit-hf",
    )


@pytest.fixture(scope="module")
def vehicles_multi_class_vit_onnx_static_bs_package() -> str:
    return download_model_package(
        model_package_zip_url=VEHICLES_MULTI_CLASS_VIT_ONNX_STATIC_BS_URL,
        package_name="vehicles-multi-class-vit-onnx-static-bs",
    )


@pytest.fixture(scope="module")
def vehicles_multi_class_vit_onnx_dynamic_bs_package() -> str:
    return download_model_package(
        model_package_zip_url=VEHICLES_MULTI_CLASS_VIT_ONNX_DYNAMIC_BS_URL,
        package_name="vehicles-multi-class-vit-onnx-dynamic-bs",
    )


@pytest.fixture(scope="module")
def vehicles_multi_class_resenet_torch_package() -> str:
    return download_model_package(
        model_package_zip_url=VEHICLES_MULTI_CLASS_RES_NET_TORCH_URL,
        package_name="vehicles-multi-class-resnet-torch",
    )


@pytest.fixture(scope="module")
def vehicles_multi_class_resenet_onnx_static_bs_package() -> str:
    return download_model_package(
        model_package_zip_url=VEHICLES_MULTI_CLASS_RES_NET_ONNX_STATIC_BS_URL,
        package_name="vehicles-multi-class-resnet-onnx-static-bs",
    )


@pytest.fixture(scope="module")
def vehicles_multi_class_resenet_onnx_dynamic_bs_package() -> str:
    return download_model_package(
        model_package_zip_url=VEHICLES_MULTI_CLASS_RES_NET_ONNX_DYNAMIC_BS_URL,
        package_name="vehicles-multi-class-resnet-onnx-dynamic-bs",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_onnx_static_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_ONNX_STATIC_CENTER_CROP_PACKAGE_URL,
        package_name="yolov8n-pose-onnx-static-center-crop",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_onnx_static_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_ONNX_STATIC_STATIC_CROP_CENTER_CROP_PACKAGE_URL,
        package_name="yolov8n-pose-onnx-static-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_onnx_static_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_ONNX_STATIC_STATIC_CROP_LETTERBOX_PACKAGE_URL,
        package_name="yolov8n-pose-onnx-static-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_onnx_static_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_ONNX_STATIC_STATIC_CROP_STRETCH_PACKAGE_URL,
        package_name="yolov8n-pose-onnx-static-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_onnx_dynamic_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_ONNX_DYNAMIC_CENTER_CROP_PACKAGE_URL,
        package_name="yolov8n-pose-onnx-dynamic-center-crop",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_onnx_dynamic_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_ONNX_DYNAMIC_STATIC_CROP_CENTER_CROP_PACKAGE_URL,
        package_name="yolov8n-pose-onnx-dynamic-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_onnx_dynamic_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_ONNX_DYNAMIC_STATIC_CROP_LETTERBOX_PACKAGE_URL,
        package_name="yolov8n-pose-onnx-dynamic-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_onnx_dynamic_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_ONNX_DYNAMIC_STATIC_CROP_STRETCH_PACKAGE_URL,
        package_name="yolov8n-pose-onnx-dynamic-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_ONNX_DYNAMIC_NMS_FUSED_CENTER_CROP_PACKAGE_URL,
        package_name="yolov8n-pose-onnx-dynamic-nms-fused-center-crop",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_ONNX_DYNAMIC_NMS_FUSED_STATIC_CROP_CENTER_CROP_PACKAGE_URL,
        package_name="yolov8n-pose-onnx-dynamic-nms-fused-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_torchscript_static_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_TORCHSCRIPT_STATIC_CENTER_CROP_PACKAGE_URL,
        package_name="yolov8n-pose-torchscript-static-center-crop",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_torchscript_static_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_TORCHSCRIPT_STATIC_STATIC_CROP_CENTER_CROP_PACKAGE_URL,
        package_name="yolov8n-pose-torchscript-static-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_torchscript_static_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_TORCHSCRIPT_STATIC_STATIC_CROP_LETTERBOX_PACKAGE_URL,
        package_name="yolov8n-pose-torchscript-static-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_torchscript_static_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_TORCHSCRIPT_STATIC_STATIC_CROP_STRETCH_PACKAGE_URL,
        package_name="yolov8n-pose-torchscript-static-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_torchscript_static_nms_fused_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_TORCHSCRIPT_STATIC_NMS_FUSED_CENTER_CROP_PACKAGE_URL,
        package_name="yolov8n-pose-torchscript-static-nms-fused-center-crop",
    )


@pytest.fixture(scope="module")
def yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8N_POSE_TORCHSCRIPT_STATIC_NMS_FUSED_STATIC_CROP_CENTER_CROP_PACKAGE_URL,
        package_name="yolov8n-pose-torchscript-static-nms-fused-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def yolov8_cls_static_bs_onnx_package() -> str:
    return download_model_package(
        model_package_zip_url=YOLOV8_CLS_ONNX_PACKAGE_URL,
        package_name="yolov8-cls-static-onnx",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov5_onnx_static_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO5_ONNX_STATIC_BS_CENTER_CROP_URL,
        package_name="coin-counting-yolov5-onnx-static-bs-letterbox",
    )


@pytest.fixture(scope="module")
def coin_counting_yolov5_onnx_dynamic_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLO5_ONNX_DYNAMIC_BS_CENTER_CROP_URL,
        package_name="coin-counting-yolov5-onnx-dynamic-bs-letterbox",
    )


@pytest.fixture(scope="module")
def asl_yolov5_onnx_seg_static_bs_letterbox() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOv5_SEG_ONNX_STATIC_BS_LETTERBOX_URL,
        package_name="asl-yolov5-onnx-static-bs-letterbox",
    )


@pytest.fixture(scope="module")
def asl_yolov7_onnx_seg_static_bs_letterbox() -> str:
    return download_model_package(
        model_package_zip_url=ASL_YOLOv7_SEG_ONNX_STATIC_BS_LETTERBOX_URL,
        package_name="asl-yolov7-onnx-static-bs-letterbox",
    )


@pytest.fixture(scope="module")
def asl_yolact_onnx_seg_static_bs_letterbox() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLACT_ONNX_STATIC_BS_LETTERBOX_URL,
        package_name="asl-yolact-onnx-static-bs-letterbox",
    )


@pytest.fixture(scope="module")
def asl_yolact_onnx_seg_static_bs_stretch() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLACT_ONNX_STATIC_BS_STRETCH_URL,
        package_name="asl-yolact-onnx-static-bs-stretch",
    )


@pytest.fixture(scope="module")
def asl_yolact_onnx_seg_static_bs_static_crop_stretch() -> str:
    return download_model_package(
        model_package_zip_url=COIN_COUNTING_YOLACT_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="asl-yolact-onnx-static-bs-static_crop-stretch",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_torch_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_TORCH_STRETCH_URL,
        package_name="snakes-rfdetr-seg-torch-stretch",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_onnx_static_bs_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_ONNX_STATIC_BS_STRETCH_URL,
        package_name="snakes-rfdetr-seg-onnx-static-bs-stretch",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_torch_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_TORCH_LETTERBOX_URL,
        package_name="snakes-rfdetr-seg-torch-letterbox",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_onnx_static_bs_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_ONNX_STATIC_BS_LETTERBOX_URL,
        package_name="snakes-rfdetr-seg-onnx-static-bs-letterbox",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_torch_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_TORCH_CENTER_CROP_URL,
        package_name="snakes-rfdetr-seg-torch-center-crop",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_onnx_static_bs_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_ONNX_STATIC_BS_CENTER_CROP_URL,
        package_name="snakes-rfdetr-seg-onnx-static-bs-center-crop",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_torch_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_TORCH_STATIC_CROP_STRETCH_URL,
        package_name="snakes-rfdetr-seg-torch-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_onnx_static_bs_static_crop_stretch_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_ONNX_STATIC_BS_STATIC_CROP_STRETCH_URL,
        package_name="snakes-rfdetr-seg-onnx-static-bs-static-crop-stretch",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_torch_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_TORCH_STATIC_CROP_LETTERBOX_URL,
        package_name="snakes-rfdetr-seg-torch-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_onnx_static_bs_static_crop_letterbox_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_ONNX_STATIC_BS_STATIC_CROP_LETTERBOX_URL,
        package_name="snakes-rfdetr-seg-onnx-static-bs-static-crop-letterbox",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_torch_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_TORCH_STATIC_CROP_CENTER_CROP_URL,
        package_name="snakes-rfdetr-seg-torch-center-static-crop-crop",
    )


@pytest.fixture(scope="module")
def snakes_rfdetr_seg_onnx_static_bs_static_crop_center_crop_package() -> str:
    return download_model_package(
        model_package_zip_url=SNAKES_RFDETR_SEG_ONNX_STATIC_BS_STATIC_CROP_CENTER_CROP_URL,
        package_name="snakes-rfdetr-seg-onnx-static-bs-static-crop-center-crop",
    )


@pytest.fixture(scope="module")
def depth_anything_v2_small_package() -> str:
    return download_model_package(
        model_package_zip_url=DEPTH_ANYTHING_V2_SMALL_PACKAGE_URL,
        package_name="depth-anything-v2-small",
    )


@pytest.fixture(scope="module")
def doctr_package() -> str:
    return download_model_package(
        model_package_zip_url=DOCTR_PACKAGE_URL, package_name="doctr"
    )


@pytest.fixture(scope="module")
def easy_ocr_package() -> str:
    return download_model_package(
        model_package_zip_url=EASY_OCR_PACKAGE_URL, package_name="easy-ocr"
    )


@pytest.fixture(scope="module")
def tr_ocr_package() -> str:
    return download_model_package(
        model_package_zip_url=TROCR_PACKAGE_URL, package_name="tr-ocr"
    )


@pytest.fixture(scope="module")
def mediapipe_face_detector_package() -> str:
    return download_model_package(
        model_package_zip_url=MEDIAPIPE_FACE_DETECTOR_PACKAGE_URL,
        package_name="mediapipe-face-detector",
    )


@pytest.fixture(scope="module")
def l2cs_package() -> str:
    return download_model_package(
        model_package_zip_url=L2CS_PACKAGE_URL,
        package_name="l2cs",
    )


@pytest.fixture(scope="module")
def dinov3_classification_onnx_static_package() -> str:
    return download_model_package(
        model_package_zip_url=DINOV3_CLASSIFICATION_ONNX_STATIC_URL,
        package_name="dinov3-classification-onnx-static",
    )


@pytest.fixture(scope="module")
def dinov3_multi_label_onnx_static_package() -> str:
    return download_model_package(
        model_package_zip_url=DINOV3_MULTI_LABEL_ONNX_STATIC_URL,
        package_name="dinov3-multi-label-onnx-static",
    )


@pytest.fixture(scope="module")
def dinov3_classification_torch_static_package() -> str:
    return download_model_package(
        model_package_zip_url=DINOV3_CLASSIFICATION_TORCH_STATIC_URL,
        package_name="dinov3-classification-torch-static",
    )


@pytest.fixture(scope="module")
def dinov3_multi_label_torch_static_package() -> str:
    return download_model_package(
        model_package_zip_url=DINOV3_MULTI_LABEL_TORCH_STATIC_URL,
        package_name="dinov3-multi-label-torch-static",
    )


@pytest.fixture(scope="module")
def owlv2_package() -> str:
    return download_model_package(
        model_package_zip_url=OWLv2_PACKAGE_URL,
        package_name="owl-v2",
    )


@pytest.fixture(scope="module")
def rf_instant_model_coin_counting_package() -> str:
    return download_model_package(
        model_package_zip_url=INSTANT_MODEL_COIN_COUNTING_PACKAGE_URL,
        package_name="rf-instant-coin-counting",
    )


@pytest.fixture(scope="module")
def sam_package() -> str:
    return download_model_package(
        model_package_zip_url=SAM_PACKAGE_URL,
        package_name="sam",
    )
