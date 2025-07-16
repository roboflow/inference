from typing import Dict, Tuple

from inference_exp.errors import ModelImplementationLoaderError
from inference_exp.models.auto_loaders.entities import ModelArchitecture, TaskType
from inference_exp.utils.imports import LazyClass
from inference_exp.weights_providers.entities import BackendType

OBJECT_DETECTION_TASK = "object-detection"
INSTANCE_SEGMENTATION_TASK = "instance-segmentation"
KEYPOINT_DETECTION_TASK = "keypoint-detection"
VLM_TASK = "vlm"
EMBEDDING_TASK = "embedding"


REGISTERED_MODELS: Dict[Tuple[ModelArchitecture, TaskType, BackendType], LazyClass] = {
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
    ("yolov8", OBJECT_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov8.yolov8_object_detection_onnx",
        class_name="YOLOv8ForObjectDetectionOnnx",
    ),
    ("yolov8", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov8.yolov8_object_detection_trt",
        class_name="YOLOv8ForObjectDetectionTRT",
    ),
    ("yolov8", KEYPOINT_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov8.yolov8_key_points_detection_onnx",
        class_name="YOLOv8ForKeyPointsDetectionOnnx",
    ),
    ("yolov8", KEYPOINT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov8.yolov8_key_points_detection_trt",
        class_name="YOLOv8ForKeyPointsDetectionTRT",
    ),
    ("yolov8", INSTANCE_SEGMENTATION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov8.yolov8_instance_segmentation_onnx",
        class_name="YOLOv8ForInstanceSegmentationOnnx",
    ),
    ("yolov8", INSTANCE_SEGMENTATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov8.yolov8_instance_segmentation_trt",
        class_name="YOLOv8ForInstanceSegmentationTRT",
    ),
    ("yolov9", OBJECT_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov9.yolov9_onnx",
        class_name="YOLOv9ForObjectDetectionOnnx",
    ),
    ("yolov9", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov9.yolov9_trt",
        class_name="YOLOv9ForObjectDetectionOnnx",
    ),
    ("yolov10", OBJECT_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov10.yolov10_object_detection_onnx",
        class_name="YOLOv10ForObjectDetectionOnnx",
    ),
    ("yolov10", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov10.yolov10_object_detection_trt",
        class_name="YOLOv10ForObjectDetectionTRT",
    ),
    ("yolov11", OBJECT_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov11.yolov11_onnx",
        class_name="YOLOv11ForObjectDetectionOnnx",
    ),
    ("yolov11", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov11.yolov11_trt",
        class_name="YOLOv11ForObjectDetectionTRT",
    ),
    ("yolov11", KEYPOINT_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov11.yolov11_onnx",
        class_name="YOLOv11ForForKeyPointsDetectionOnnx",
    ),
    ("yolov11", KEYPOINT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov11.yolov11_trt",
        class_name="YOLOv11ForForKeyPointsDetectionTRT",
    ),
    ("yolov11", INSTANCE_SEGMENTATION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov11.yolov11_onnx",
        class_name="YOLOv11ForInstanceSegmentationOnnx",
    ),
    ("yolov11", INSTANCE_SEGMENTATION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov11.yolov11_trt",
        class_name="YOLOv11ForInstanceSegmentationTRT",
    ),
    ("yolov12", OBJECT_DETECTION_TASK, BackendType.ONNX): LazyClass(
        module_name="inference_exp.models.yolov12.yolov12_onnx",
        class_name="YOLOv12ForObjectDetectionOnnx",
    ),
    ("yolov12", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.yolov12.yolov12_trt",
        class_name="YOLOv12ForObjectDetectionTRT",
    ),
    ("paligemma", VLM_TASK, BackendType.HF): LazyClass(
        module_name="inference_exp.models.paligemma.paligemma_hf",
        class_name="PaliGemmaHF",
    ),
    ("perception_encoder", EMBEDDING_TASK, BackendType.TORCH): LazyClass(
        module_name="inference_exp.models.perception_encoder.perception_encoder_pytorch",
        class_name="PerceptionEncoderTorch",
    ),
    ("rfdetr", OBJECT_DETECTION_TASK, BackendType.TRT): LazyClass(
        module_name="inference_exp.models.rfdetr.rfdetr_object_detection_trt",
        class_name="RFDetrForObjectDetectionTRT",
    ),
}


def resolve_model_class(
    model_architecture: ModelArchitecture,
    task_type: TaskType,
    backend: BackendType,
) -> type:
    if not model_implementation_exists(
        model_architecture=model_architecture,
        task_type=task_type,
        backend=backend,
    ):
        raise ModelImplementationLoaderError(
            message=f"Did not find implementation for model with architecture: {model_architecture}, "
            f"task type: {task_type} and backend: {backend}",
            help_url="https://todo",
        )
    return REGISTERED_MODELS[(model_architecture, task_type, backend)].resolve()


def model_implementation_exists(
    model_architecture: ModelArchitecture,
    task_type: TaskType,
    backend: BackendType,
) -> bool:
    lookup_key = (model_architecture, task_type, backend)
    return lookup_key in REGISTERED_MODELS
