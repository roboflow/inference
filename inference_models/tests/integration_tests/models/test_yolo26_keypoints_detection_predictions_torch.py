import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.yolo26.yolo26_key_points_detection_torch_script import (
    YOLO26ForKeyPointsDetectionTorchScript,
)


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_letterbox_numpy(
    yolo26n_pose_basketball_letterbox_torch_script_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(basketball_image_numpy)

    print("=== test_yolo26n_pose_torchscript_letterbox_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_letterbox_batch_numpy(
    yolo26n_pose_basketball_letterbox_torch_script_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([basketball_image_numpy, basketball_image_numpy])

    print("=== test_yolo26n_pose_torchscript_letterbox_batch_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[0][1].xy: {predictions[0][1].xy.cpu().tolist()}")
    print(f"predictions[0][1].confidence: {predictions[0][1].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")
    print(f"predictions[1][1].xyxy: {predictions[1][1].xyxy.cpu().tolist()}")
    print(f"predictions[1][1].confidence: {predictions[1][1].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_letterbox_torch(
    yolo26n_pose_basketball_letterbox_torch_script_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(basketball_image_torch)

    print("=== test_yolo26n_pose_torchscript_letterbox_torch ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_stretch_numpy(
    yolo26n_pose_basketball_stretch_torch_script_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(basketball_image_numpy)

    print("=== test_yolo26n_pose_torchscript_stretch_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_stretch_batch_numpy(
    yolo26n_pose_basketball_stretch_torch_script_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([basketball_image_numpy, basketball_image_numpy])

    print("=== test_yolo26n_pose_torchscript_stretch_batch_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[0][1].xy: {predictions[0][1].xy.cpu().tolist()}")
    print(f"predictions[0][1].confidence: {predictions[0][1].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")
    print(f"predictions[1][1].xyxy: {predictions[1][1].xyxy.cpu().tolist()}")
    print(f"predictions[1][1].confidence: {predictions[1][1].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_stretch_torch(
    yolo26n_pose_basketball_stretch_torch_script_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(basketball_image_torch)

    print("=== test_yolo26n_pose_torchscript_stretch_torch ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")
