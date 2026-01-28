import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_letterbox_numpy(
    yolo26n_pose_basketball_letterbox_onnx_static_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(basketball_image_numpy)

    print("=== test_yolo26n_pose_onnx_static_letterbox_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_letterbox_batch_numpy(
    yolo26n_pose_basketball_letterbox_onnx_static_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([basketball_image_numpy, basketball_image_numpy])

    print("=== test_yolo26n_pose_onnx_static_letterbox_batch_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[0][1].xy: {predictions[0][1].xy.cpu().tolist()}")
    print(f"predictions[0][1].confidence: {predictions[0][1].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")
    print(f"predictions[1][1].xyxy: {predictions[1][1].xyxy.cpu().tolist()}")
    print(f"predictions[1][1].confidence: {predictions[1][1].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_letterbox_torch(
    yolo26n_pose_basketball_letterbox_onnx_static_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(basketball_image_torch)

    print("=== test_yolo26n_pose_onnx_static_letterbox_torch ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_letterbox_numpy(
    yolo26n_pose_basketball_letterbox_onnx_dynamic_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(basketball_image_numpy)

    print("=== test_yolo26n_pose_onnx_dynamic_letterbox_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_letterbox_batch_numpy(
    yolo26n_pose_basketball_letterbox_onnx_dynamic_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([basketball_image_numpy, basketball_image_numpy])

    print("=== test_yolo26n_pose_onnx_dynamic_letterbox_batch_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[0][1].xy: {predictions[0][1].xy.cpu().tolist()}")
    print(f"predictions[0][1].confidence: {predictions[0][1].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")
    print(f"predictions[1][1].xyxy: {predictions[1][1].xyxy.cpu().tolist()}")
    print(f"predictions[1][1].confidence: {predictions[1][1].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_letterbox_torch(
    yolo26n_pose_basketball_letterbox_onnx_dynamic_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(basketball_image_torch)

    print("=== test_yolo26n_pose_onnx_dynamic_letterbox_torch ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_stretch_numpy(
    yolo26n_pose_basketball_stretch_onnx_static_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(basketball_image_numpy)

    print("=== test_yolo26n_pose_onnx_static_stretch_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_stretch_batch_numpy(
    yolo26n_pose_basketball_stretch_onnx_static_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([basketball_image_numpy, basketball_image_numpy])

    print("=== test_yolo26n_pose_onnx_static_stretch_batch_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[0][1].xy: {predictions[0][1].xy.cpu().tolist()}")
    print(f"predictions[0][1].confidence: {predictions[0][1].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")
    print(f"predictions[1][1].xyxy: {predictions[1][1].xyxy.cpu().tolist()}")
    print(f"predictions[1][1].confidence: {predictions[1][1].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_stretch_torch(
    yolo26n_pose_basketball_stretch_onnx_static_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(basketball_image_torch)

    print("=== test_yolo26n_pose_onnx_static_stretch_torch ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_stretch_numpy(
    yolo26n_pose_basketball_stretch_onnx_dynamic_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(basketball_image_numpy)

    print("=== test_yolo26n_pose_onnx_dynamic_stretch_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_stretch_batch_numpy(
    yolo26n_pose_basketball_stretch_onnx_dynamic_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([basketball_image_numpy, basketball_image_numpy])

    print("=== test_yolo26n_pose_onnx_dynamic_stretch_batch_numpy ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[0][1].xy: {predictions[0][1].xy.cpu().tolist()}")
    print(f"predictions[0][1].confidence: {predictions[0][1].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")
    print(f"predictions[1][1].xyxy: {predictions[1][1].xyxy.cpu().tolist()}")
    print(f"predictions[1][1].confidence: {predictions[1][1].confidence.cpu().tolist()}")


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_stretch_torch(
    yolo26n_pose_basketball_stretch_onnx_dynamic_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(basketball_image_torch)

    print("=== test_yolo26n_pose_onnx_dynamic_stretch_torch ===")
    print(f"predictions[0][0].xy: {predictions[0][0].xy.cpu().tolist()}")
    print(f"predictions[0][0].confidence: {predictions[0][0].confidence.cpu().tolist()}")
    print(f"predictions[1][0].xyxy: {predictions[1][0].xyxy.cpu().tolist()}")
    print(f"predictions[1][0].confidence: {predictions[1][0].confidence.cpu().tolist()}")
