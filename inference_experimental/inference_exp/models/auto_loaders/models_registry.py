from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple, Union

from inference_exp.errors import ModelImplementationLoaderError
from inference_exp.models.auto_loaders.entities import ModelArchitecture, TaskType
from inference_exp.utils.imports import LazyClass
from inference_exp.weights_providers.entities import BackendType

OBJECT_DETECTION_TASK = "object-detection"
INSTANCE_SEGMENTATION_TASK = "instance-segmentation"
SEMANTIC_SEGMENTATION_TASK = "semantic-segmentation"
KEYPOINT_DETECTION_TASK = "keypoint-detection"
VLM_TASK = "vlm"
EMBEDDING_TASK = "embedding"
CLASSIFICATION_TASK = "classification"
MULTI_LABEL_CLASSIFICATION_TASK = "multi-label-classification"
DEPTH_ESTIMATION_TASK = "depth-estimation"
STRUCTURED_OCR_TASK = "structured-ocr"
TEXT_ONLY_OCR_TASK = "text-only-ocr"
GAZE_DETECTION_TASK = "gaze-detection"
OPEN_VOCABULARY_OBJECT_DETECTION_TASK = "open-vocabulary-object-detection"


@dataclass(frozen=True)
class RegistryEntry:
    model_class: LazyClass
    supported_model_features: Optional[Set[str]] = field(default=None)


REGISTERED_MODELS: Dict[
    Tuple[ModelArchitecture, TaskType, BackendType], Union[LazyClass, RegistryEntry]
] = {
    ("yolonas", OBJECT_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolonas.yolonas_object_detection_onnx",
        class_name="YOLONasForObjectDetectionOnnx",
    ),
    ("yolonas", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolonas.yolonas_object_detection_trt",
        class_name="YOLONasForObjectDetectionTRT",
    ),
    ("yolov5", OBJECT_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov5.yolov5_object_detection_onnx",
        class_name="YOLOv5ForObjectDetectionOnnx",
    ),
    ("yolov5", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov5.yolov5_object_detection_trt",
        class_name="YOLOv5ForObjectDetectionTRT",
    ),
    ("yolov5", INSTANCE_SEGMENTATION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov5.yolov5_instance_segmentation_onnx",
        class_name="YOLOv5ForInstanceSegmentationOnnx",
    ),
    ("yolov5", INSTANCE_SEGMENTATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov5.yolov5_instance_segmentation_trt",
        class_name="YOLOv5ForInstanceSegmentationTRT",
    ),
    ("yolov7", INSTANCE_SEGMENTATION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov7.yolov7_instance_segmentation_onnx",
        class_name="YOLOv7ForInstanceSegmentationOnnx",
    ),
    ("yolov7", INSTANCE_SEGMENTATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov7.yolov7_instance_segmentation_trt",
        class_name="YOLOv7ForInstanceSegmentationTRT",
    ),
    ("yolov8", CLASSIFICATION_TASK, BackendType.ONNX): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov8.yolov8_classification_onnx",
            class_name="YOLOv8ForClassificationOnnx",
        ),
    ),
    ("yolov8", OBJECT_DETECTION_TASK, BackendType.ONNX): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov8.yolov8_object_detection_onnx",
            class_name="YOLOv8ForObjectDetectionOnnx",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov8", OBJECT_DETECTION_TASK, BackendType.TORCH_SCRIPT): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov8.yolov8_object_detection_torch_script",
            class_name="YOLOv8ForObjectDetectionTorchScript",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov8", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov8.yolov8_object_detection_trt",
        class_name="YOLOv8ForObjectDetectionTRT",
    ),
    ("yolov8", KEYPOINT_DETECTION_TASK, BackendType.ONNX): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov8.yolov8_key_points_detection_onnx",
            class_name="YOLOv8ForKeyPointsDetectionOnnx",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov8", KEYPOINT_DETECTION_TASK, BackendType.TORCH_SCRIPT): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov8.yolov8_key_points_detection_torch_script",
            class_name="YOLOv8ForKeyPointsDetectionTorchScript",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov8", KEYPOINT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov8.yolov8_key_points_detection_trt",
        class_name="YOLOv8ForKeyPointsDetectionTRT",
    ),
    ("yolov8", INSTANCE_SEGMENTATION_TASK, BackendType.ONNX): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov8.yolov8_instance_segmentation_onnx",
            class_name="YOLOv8ForInstanceSegmentationOnnx",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov8", INSTANCE_SEGMENTATION_TASK, BackendType.TORCH_SCRIPT): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov8.yolov8_instance_segmentation_torch_script",
            class_name="YOLOv8ForInstanceSegmentationTorchScript",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov8", INSTANCE_SEGMENTATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov8.yolov8_instance_segmentation_trt",
        class_name="YOLOv8ForInstanceSegmentationTRT",
    ),
    ("yolov9", OBJECT_DETECTION_TASK, BackendType.ONNX): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov9.yolov9_onnx",
            class_name="YOLOv9ForObjectDetectionOnnx",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov9", OBJECT_DETECTION_TASK, BackendType.TORCH_SCRIPT): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov9.yolov9_torch_script",
            class_name="YOLOv9ForObjectDetectionTorchScript",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov9", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov9.yolov9_trt",
        class_name="YOLOv9ForObjectDetectionTRT",
    ),
    ("yolov10", OBJECT_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov10.yolov10_object_detection_onnx",
        class_name="YOLOv10ForObjectDetectionOnnx",
    ),
    ("yolov10", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov10.yolov10_object_detection_trt",
        class_name="YOLOv10ForObjectDetectionTRT",
    ),
    ("yolov11", CLASSIFICATION_TASK, BackendType.ONNX): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov11.yolov11_onnx",
            class_name="YOLOv11ForClassificationOnnx",
        ),
    ),
    ("yolov11", OBJECT_DETECTION_TASK, BackendType.ONNX): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov11.yolov11_onnx",
            class_name="YOLOv11ForObjectDetectionOnnx",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov11", OBJECT_DETECTION_TASK, BackendType.TORCH_SCRIPT): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov11.yolov11_torch_script",
            class_name="YOLOv11ForObjectDetectionTorchScript",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov11", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov11.yolov11_trt",
        class_name="YOLOv11ForObjectDetectionTRT",
    ),
    ("yolov11", KEYPOINT_DETECTION_TASK, BackendType.ONNX): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov11.yolov11_onnx",
            class_name="YOLOv11ForForKeyPointsDetectionOnnx",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov11", KEYPOINT_DETECTION_TASK, BackendType.TORCH_SCRIPT): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov11.yolov11_torch_script",
            class_name="YOLOv11ForForKeyPointsDetectionTorchScript",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov11", KEYPOINT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov11.yolov11_trt",
        class_name="YOLOv11ForForKeyPointsDetectionTRT",
    ),
    ("yolov11", INSTANCE_SEGMENTATION_TASK, BackendType.ONNX): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov11.yolov11_onnx",
            class_name="YOLOv11ForInstanceSegmentationOnnx",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov11", INSTANCE_SEGMENTATION_TASK, BackendType.TORCH_SCRIPT): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov11.yolov11_torch_script",
            class_name="YOLOv11ForInstanceSegmentationTorchScript",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov11", INSTANCE_SEGMENTATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov11.yolov11_trt",
        class_name="YOLOv11ForInstanceSegmentationTRT",
    ),
    ("yolov12", OBJECT_DETECTION_TASK, BackendType.ONNX): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov12.yolov12_onnx",
            class_name="YOLOv12ForObjectDetectionOnnx",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov12", OBJECT_DETECTION_TASK, BackendType.TORCH_SCRIPT): RegistryEntry(
        model_class=LazyClass(
            module_name="inference_exp.models.yolov12.yolov12_torch_script",
            class_name="YOLOv12ForObjectDetectionTorchScript",
        ),
        supported_model_features={"nms_fused"},
    ),
    ("yolov12", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov12.yolov12_trt",
        class_name="YOLOv12ForObjectDetectionTRT",
    ),
    ("paligemma-2", VLM_TASK, BackendType.HF): LazyClass(
        module_name="inference_exp.models.paligemma.paligemma_hf",
        class_name="PaliGemmaHF",
    ),
    ("paligemma", VLM_TASK, BackendType.HF): LazyClass(
        module_name="inference_exp.models.paligemma.paligemma_hf",
        class_name="PaliGemmaHF",
    ),
    ("smolvlm-v2", VLM_TASK, BackendType.HF): LazyClass(
        module_name="inference_exp.models.smolvlm.smolvlm_hf",
        class_name="SmolVLMHF",
    ),
    ("qwen25vl", VLM_TASK, BackendType.HF): LazyClass(
        module_name="inference_exp.models.qwen25vl.qwen25vl_hf",
        class_name="Qwen25VLHF",
    ),
    ("florence-2", VLM_TASK, BackendType.HF): LazyClass(
        module_name="inference_exp.models.florence2.florence2_hf",
        class_name="Florence2HF",
    ),
    ("clip", EMBEDDING_TASK, BackendType.TORCH): LazyClass(
        module_name="inference_exp.models.clip.clip_pytorch",
        class_name="ClipTorch",
    ),
    ("clip", EMBEDDING_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.clip.clip_onnx",
        class_name="ClipOnnx",
    ),
    ("perception-encoder", EMBEDDING_TASK, BackendType.TORCH): LazyClass(
        module_name="inference_exp.models.perception_encoder.perception_encoder_pytorch",
        class_name="PerceptionEncoderTorch",
    ),
    ("rfdetr", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.rfdetr.rfdetr_object_detection_trt",
        class_name="RFDetrForObjectDetectionTRT",
    ),
    ("rfdetr", OBJECT_DETECTION_TASK, BackendType.TORCH): LazyClass(
        module_name="inference_exp.models.rfdetr.rfdetr_object_detection_pytorch",
        class_name="RFDetrForObjectDetectionTorch",
    ),
    ("rfdetr", OBJECT_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.rfdetr.rfdetr_object_detection_onnx",
        class_name="RFDetrForObjectDetectionONNX",
    ),
    ("rfdetr", INSTANCE_SEGMENTATION_TASK, BackendType.TORCH): LazyClass(
        module_name="inference_exp.models.rfdetr.rfdetr_instance_segmentation_pytorch",
        class_name="RFDetrForInstanceSegmentationTorch",
    ),
    ("rfdetr", INSTANCE_SEGMENTATION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.rfdetr.rfdetr_instance_segmentation_onnx",
        class_name="RFDetrForInstanceSegmentationOnnx",
    ),
    ("rfdetr", INSTANCE_SEGMENTATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.rfdetr.rfdetr_instance_segmentation_trt",
        class_name="RFDetrForInstanceSegmentationTRT",
    ),
    ("moondream2", VLM_TASK, BackendType.HF): LazyClass(
        module_name="inference_exp.models.moondream2.moondream2_hf",
        class_name="MoonDream2HF",
    ),
    ("vit", CLASSIFICATION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.vit.vit_classification_onnx",
        class_name="VITForClassificationOnnx",
    ),
    ("vit", MULTI_LABEL_CLASSIFICATION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.vit.vit_classification_onnx",
        class_name="VITForMultiLabelClassificationOnnx",
    ),
    ("vit", CLASSIFICATION_TASK, BackendType.HF): LazyClass(
        module_name="inference_exp.models.vit.vit_classification_huggingface",
        class_name="VITForClassificationHF",
    ),
    ("vit", MULTI_LABEL_CLASSIFICATION_TASK, BackendType.HF): LazyClass(
        module_name="inference_exp.models.vit.vit_classification_huggingface",
        class_name="VITForMultiLabelClassificationHF",
    ),
    ("vit", CLASSIFICATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.vit.vit_classification_trt",
        class_name="VITForClassificationTRT",
    ),
    ("vit", MULTI_LABEL_CLASSIFICATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.vit.vit_classification_trt",
        class_name="VITForMultiLabelClassificationTRT",
    ),
    ("resnet", CLASSIFICATION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.resnet.resnet_classification_onnx",
        class_name="ResNetForClassificationOnnx",
    ),
    ("resnet", MULTI_LABEL_CLASSIFICATION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.resnet.resnet_classification_onnx",
        class_name="ResNetForMultiLabelClassificationOnnx",
    ),
    ("resnet", CLASSIFICATION_TASK, BackendType.TORCH): LazyClass(
        module_name="inference_exp.models.resnet.resnet_classification_torch",
        class_name="ResNetForClassificationTorch",
    ),
    ("resnet", MULTI_LABEL_CLASSIFICATION_TASK, BackendType.TORCH): LazyClass(
        module_name="inference_exp.models.resnet.resnet_classification_torch",
        class_name="ResNetForMultiLabelClassificationTorch",
    ),
    ("resnet", CLASSIFICATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.resnet.resnet_classification_trt",
        class_name="ResNetForClassificationTRT",
    ),
    ("resnet", MULTI_LABEL_CLASSIFICATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.resnet.resnet_classification_trt",
        class_name="ResNetForMultiLabelClassificationTRT",
    ),
    ("segment-anything-2-rt", INSTANCE_SEGMENTATION_TASK, BackendType.TORCH): LazyClass(
        module_name="inference_exp.models.sam2_rt.sam2_pytorch",
        class_name="SAM2ForStream",
    ),
    ("deep-lab-v3-plus", SEMANTIC_SEGMENTATION_TASK, BackendType.TORCH): LazyClass(
        module_name="inference_exp.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_torch",
        class_name="DeepLabV3PlusForSemanticSegmentationTorch",
    ),
    ("deep-lab-v3-plus", SEMANTIC_SEGMENTATION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx",
        class_name="DeepLabV3PlusForSemanticSegmentationOnnx",
    ),
    ("deep-lab-v3-plus", SEMANTIC_SEGMENTATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_trt",
        class_name="DeepLabV3PlusForSemanticSegmentationTRT",
    ),
    ("yolact", INSTANCE_SEGMENTATION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolact.yolact_instance_segmentation_onnx",
        class_name="YOLOACTForInstanceSegmentationOnnx",
    ),
    ("yolact", INSTANCE_SEGMENTATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolact.yolact_instance_segmentation_trt",
        class_name="YOLOACTForInstanceSegmentationTRT",
    ),
    ("depth-anything-v2", DEPTH_ESTIMATION_TASK, BackendType.HF): LazyClass(
        module_name="inference_exp.models.depth_anything_v2.depth_anything_v2_hf",
        class_name="DepthAnythingV2HF",
    ),
    ("doctr", STRUCTURED_OCR_TASK, BackendType.TORCH): LazyClass(
        module_name="inference_exp.models.doctr.doctr_torch", class_name="DocTR"
    ),
    ("easy-ocr", STRUCTURED_OCR_TASK, BackendType.TORCH): LazyClass(
        module_name="inference_exp.models.easy_ocr.easy_ocr_torch",
        class_name="EasyOCRTorch",
    ),
    ("tr-ocr", TEXT_ONLY_OCR_TASK, BackendType.HF): LazyClass(
        module_name="inference_exp.models.trocr.trocr_hf",
        class_name="TROcrHF",
    ),
    (
        "mediapipe-face-detector",
        KEYPOINT_DETECTION_TASK,
        BackendType.MEDIAPIPE,
    ): LazyClass(
        module_name="inference_exp.models.mediapipe_face_detection.face_detection",
        class_name="MediaPipeFaceDetector",
    ),
    ("l2cs-net", GAZE_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.l2cs.l2cs_onnx",
        class_name="L2CSNetOnnx",
    ),
    (
        "grounding-dino",
        OPEN_VOCABULARY_OBJECT_DETECTION_TASK,
        BackendType.TORCH,
    ): LazyClass(
        module_name="inference_exp.models.grounding_dino.grounding_dino_torch",
        class_name="GroundingDinoForObjectDetectionTorch",
    ),
    (
        "dinov3_probe",
        MULTI_LABEL_CLASSIFICATION_TASK,
        BackendType.ONNX,
    ): LazyClass(
        module_name="inference_exp.models.dinov3.dinov3_classification_onnx",
        class_name="DinoV3ForMultiLabelClassificationOnnx",
    ),
    (
        "dinov3_probe",
        CLASSIFICATION_TASK,
        BackendType.ONNX,
    ): LazyClass(
        module_name="inference_exp.models.dinov3.dinov3_classification_onnx",
        class_name="DinoV3ForClassificationOnnx",
    ),
}


def resolve_model_class(
    model_architecture: ModelArchitecture,
    task_type: TaskType,
    backend: BackendType,
    model_features: Optional[Set[str]] = None,
) -> type:
    if not model_implementation_exists(
        model_architecture=model_architecture,
        task_type=task_type,
        backend=backend,
        model_features=model_features,
    ):
        raise ModelImplementationLoaderError(
            message=f"Did not find implementation for model with architecture: {model_architecture}, "
            f"task type: {task_type} backend: {backend} and model features: {model_features}",
            help_url="https://todo",
        )
    matched_model = REGISTERED_MODELS[(model_architecture, task_type, backend)]
    if isinstance(matched_model, RegistryEntry):
        return matched_model.model_class.resolve()
    return matched_model.resolve()


def model_implementation_exists(
    model_architecture: ModelArchitecture,
    task_type: TaskType,
    backend: BackendType,
    model_features: Optional[Set[str]] = None,
) -> bool:
    lookup_key = (model_architecture, task_type, backend)
    if lookup_key not in REGISTERED_MODELS:
        return False
    if not model_features:
        return True
    matched_model = REGISTERED_MODELS[(model_architecture, task_type, backend)]
    if not isinstance(matched_model, RegistryEntry):
        # features requested, but no supported features manifested
        return False
    return all(f in matched_model.supported_model_features for f in model_features)
