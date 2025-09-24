import numpy as np
import pytest
import torch
from inference_exp.models.yolov8.yolov8_key_points_detection_onnx import (
    YOLOv8ForKeyPointsDetectionOnnx,
)
from inference_exp.models.yolov8.yolov8_key_points_detection_torch_script import (
    YOLOv8ForKeyPointsDetectionTorchScript,
)


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_center_crop_package_numpy(
    yolov8n_pose_onnx_static_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_center_crop_package_numpy_custom_size(
    yolov8n_pose_onnx_static_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_center_crop_package_batch_numpy(
    yolov8n_pose_onnx_static_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_center_crop_package_torch(
    yolov8n_pose_onnx_static_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_center_crop_package_torch_custom_size(
    yolov8n_pose_onnx_static_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_center_crop_package_batch_torch(
    yolov8n_pose_onnx_static_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_center_crop_package_list_torch(
    yolov8n_pose_onnx_static_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_center_crop_package_numpy(
    yolov8n_pose_onnx_static_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_center_crop_package_numpy_custom_size(
    yolov8n_pose_onnx_static_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_center_crop_package_batch_numpy(
    yolov8n_pose_onnx_static_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_center_crop_package_torch(
    yolov8n_pose_onnx_static_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_center_crop_package_torch_custom_size(
    yolov8n_pose_onnx_static_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_center_crop_package_batch_torch(
    yolov8n_pose_onnx_static_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_center_crop_package_list_torch(
    yolov8n_pose_onnx_static_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_letterbox_package_numpy(
    yolov8n_pose_onnx_static_static_crop_letterbox_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_letterbox_package_numpy_custom_size(
    yolov8n_pose_onnx_static_static_crop_letterbox_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_letterbox_package_batch_numpy(
    yolov8n_pose_onnx_static_static_crop_letterbox_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_letterbox_package_torch(
    yolov8n_pose_onnx_static_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_letterbox_package_torch_custom_size(
    yolov8n_pose_onnx_static_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_letterbox_package_batch_torch(
    yolov8n_pose_onnx_static_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_letterbox_package_list_torch(
    yolov8n_pose_onnx_static_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_stretch_package_numpy(
    yolov8n_pose_onnx_static_static_crop_stretch_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_stretch_package_numpy_custom_size(
    yolov8n_pose_onnx_static_static_crop_stretch_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_stretch_package_batch_numpy(
    yolov8n_pose_onnx_static_static_crop_stretch_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_stretch_package_torch(
    yolov8n_pose_onnx_static_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_stretch_package_torch_custom_size(
    yolov8n_pose_onnx_static_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_stretch_package_batch_torch(
    yolov8n_pose_onnx_static_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_static_static_crop_stretch_package_list_torch(
    yolov8n_pose_onnx_static_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_center_crop_package_numpy(
    yolov8n_pose_onnx_dynamic_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_center_crop_package_numpy_custom_size(
    yolov8n_pose_onnx_dynamic_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_center_crop_package_batch_numpy(
    yolov8n_pose_onnx_dynamic_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_center_crop_package_torch(
    yolov8n_pose_onnx_dynamic_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_center_crop_package_torch_custom_size(
    yolov8n_pose_onnx_dynamic_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_center_crop_package_batch_torch(
    yolov8n_pose_onnx_dynamic_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_center_crop_package_list_torch(
    yolov8n_pose_onnx_dynamic_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_center_crop_package_numpy(
    yolov8n_pose_onnx_dynamic_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_center_crop_package_numpy_custom_size(
    yolov8n_pose_onnx_dynamic_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_center_crop_package_batch_numpy(
    yolov8n_pose_onnx_dynamic_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_center_crop_package_torch(
    yolov8n_pose_onnx_dynamic_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_center_crop_package_torch_custom_size(
    yolov8n_pose_onnx_dynamic_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_center_crop_package_batch_torch(
    yolov8n_pose_onnx_dynamic_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_center_crop_package_list_torch(
    yolov8n_pose_onnx_dynamic_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_letterbox_package_numpy(
    yolov8n_pose_onnx_dynamic_static_crop_letterbox_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_letterbox_package_numpy_custom_size(
    yolov8n_pose_onnx_dynamic_static_crop_letterbox_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_letterbox_package_batch_numpy(
    yolov8n_pose_onnx_dynamic_static_crop_letterbox_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_letterbox_package_torch(
    yolov8n_pose_onnx_dynamic_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_letterbox_package_torch_custom_size(
    yolov8n_pose_onnx_dynamic_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_letterbox_package_batch_torch(
    yolov8n_pose_onnx_dynamic_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_letterbox_package_list_torch(
    yolov8n_pose_onnx_dynamic_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_stretch_package_numpy(
    yolov8n_pose_onnx_dynamic_static_crop_stretch_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_stretch_package_numpy_custom_size(
    yolov8n_pose_onnx_dynamic_static_crop_stretch_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_stretch_package_batch_numpy(
    yolov8n_pose_onnx_dynamic_static_crop_stretch_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_stretch_package_torch(
    yolov8n_pose_onnx_dynamic_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_stretch_package_torch_custom_size(
    yolov8n_pose_onnx_dynamic_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_stretch_package_batch_torch(
    yolov8n_pose_onnx_dynamic_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_static_crop_stretch_package_list_torch(
    yolov8n_pose_onnx_dynamic_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package_numpy(
    yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package_numpy_custom_size(
    yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package_batch_numpy(
    yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package_torch(
    yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package_torch_custom_size(
    yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package_batch_torch(
    yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package_list_torch(
    yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package_numpy(
    yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package_numpy_custom_size(
    yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package_batch_numpy(
    yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package_torch(
    yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package_torch_custom_size(
    yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package_batch_torch(
    yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package_list_torch(
    yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolov8n_pose_onnx_dynamic_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_center_crop_package_numpy(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_center_crop_package_numpy_custom_size(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_center_crop_package_batch_numpy(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_center_crop_package_torch(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_center_crop_package_torch_custom_size(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_center_crop_package_batch_torch(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_center_crop_package_list_torch(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_numpy(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_numpy_custom_size(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_batch_numpy(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_torch(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_torch_custom_size(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_batch_torch(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_list_torch(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_numpy(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_numpy_custom_size(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_batch_numpy(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_torch(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_torch_custom_size(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_batch_torch(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_list_torch(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_numpy(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_numpy_custom_size(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_batch_numpy(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_torch(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_torch_custom_size(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_batch_torch(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_list_torch(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_numpy(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_numpy_custom_size(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_batch_numpy(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_torch(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_torch_custom_size(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_batch_torch(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_list_torch(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_numpy(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_numpy_custom_size(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_batch_numpy(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_torch(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_torch_custom_size(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_batch_torch(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )


@pytest.mark.slow
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_list_torch(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        onnx_execution_providers=["CPUExecutionProvider"],
    )
