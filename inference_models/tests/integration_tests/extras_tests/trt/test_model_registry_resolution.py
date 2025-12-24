import pytest

from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.models_registry import resolve_model_class


@pytest.mark.trt_extras
def test_yolonas_object_detection_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolonas",
        task_type="object-detection",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov5_object_detection_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov5",
        task_type="object-detection",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov5_instance_segmentation_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov5",
        task_type="instance-segmentation",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov7_instance_segmentation_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov7",
        task_type="instance-segmentation",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov8_object_detection_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov8",
        task_type="object-detection",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov8_instance_segmentation_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov8",
        task_type="instance-segmentation",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov8_keypoint_detection_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov8",
        task_type="keypoint-detection",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov9_object_detection_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov9",
        task_type="object-detection",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov10_object_detection_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov10",
        task_type="object-detection",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov11_object_detection_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov11",
        task_type="object-detection",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov11_instance_segmentation_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov11",
        task_type="instance-segmentation",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov11_keypoint_detection_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov11",
        task_type="keypoint-detection",
        backend=BackendType.TRT,
    )


@pytest.mark.trt_extras
def test_yolov12_object_detection_trt_resolution() -> None:
    _ = resolve_model_class(
        model_architecture="yolov12",
        task_type="object-detection",
        backend=BackendType.TRT,
    )
