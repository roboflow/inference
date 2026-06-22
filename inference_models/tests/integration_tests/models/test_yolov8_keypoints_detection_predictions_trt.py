import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    yolov8_pose_trt_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_key_points_detection_trt import (
        YOLOv8ForKeyPointsDetectionTRT,
    )

    model = YOLOv8ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_pose_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(people_walking_image_numpy)

    # then
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.8787, 0.8724]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[351, 124, 540, 756], [619, 120, 824, 767]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert abs(predictions[0][0].confidence.sum().item() - 26.07147979736328) < 1e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy(
    yolov8_pose_trt_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_key_points_detection_trt import (
        YOLOv8ForKeyPointsDetectionTRT,
    )

    model = YOLOv8ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_pose_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([people_walking_image_numpy, people_walking_image_numpy])

    # then
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.8787, 0.8724]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[351, 124, 540, 756], [619, 120, 824, 767]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.8787, 0.8724]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[351, 124, 540, 756], [619, 120, 824, 767]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert abs(predictions[0][0].confidence.sum().item() - 26.07147979736328) < 1e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch(
    yolov8_pose_trt_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_key_points_detection_trt import (
        YOLOv8ForKeyPointsDetectionTRT,
    )

    model = YOLOv8ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_pose_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(people_walking_image_torch)

    # then
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.8783, 0.8719]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[351, 124, 540, 756], [619, 120, 824, 767]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert abs(predictions[0][0].confidence.sum().item() - 26.07147979736328) < 1e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_multiple_predictions_in_row(
    yolov8_pose_trt_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_key_points_detection_trt import (
        YOLOv8ForKeyPointsDetectionTRT,
    )

    model = YOLOv8ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_pose_trt_package,
        engine_host_code_allowed=True,
    )

    for _ in range(8):
        # when
        predictions = model(people_walking_image_torch)

        # then
        assert torch.allclose(
            predictions[1][0].confidence.cpu(),
            torch.tensor([0.8783, 0.8719]).cpu(),
            atol=0.01,
        )
        assert torch.allclose(
            predictions[1][0].class_id.cpu(),
            torch.tensor([0, 0], dtype=torch.int32).cpu(),
        )
        expected_xyxy = torch.tensor(
            [[351, 124, 540, 756], [619, 120, 824, 767]],
            dtype=torch.int32,
        )
        assert torch.allclose(
            predictions[1][0].xyxy.cpu(),
            expected_xyxy.cpu(),
            atol=5,
        )
        assert abs(predictions[0][0].confidence.sum().item() - 26.07147979736328) < 1e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_list(
    yolov8_pose_trt_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_key_points_detection_trt import (
        YOLOv8ForKeyPointsDetectionTRT,
    )

    model = YOLOv8ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_pose_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([people_walking_image_torch, people_walking_image_torch])

    # then
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.8783, 0.8719]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[351, 124, 540, 756], [619, 120, 824, 767]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.8783, 0.8719]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[351, 124, 540, 756], [619, 120, 824, 767]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert abs(predictions[0][0].confidence.sum().item() - 26.07147979736328) < 1e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_batch(
    yolov8_pose_trt_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_key_points_detection_trt import (
        YOLOv8ForKeyPointsDetectionTRT,
    )

    model = YOLOv8ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_pose_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(
        torch.stack([people_walking_image_torch, people_walking_image_torch], dim=0)
    )

    # then
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.8783, 0.8719]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[351, 124, 540, 756], [619, 120, 824, 767]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.8783, 0.8719]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[351, 124, 540, 756], [619, 120, 824, 767]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert abs(predictions[0][0].confidence.sum().item() - 26.07147979736328) < 1e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_per_class_confidence_filters_detections(
    yolov8_pose_trt_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolov8.yolov8_key_points_detection_trt import (
        YOLOv8ForKeyPointsDetectionTRT,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = YOLOv8ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_pose_trt_package,
        engine_host_code_allowed=True,
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.25,
        per_class_confidence={class_names[0]: 1.01},
    )
    _, predictions_det = model(people_walking_image_numpy, confidence="best")
    assert predictions_det[0].class_id.numel() == 0
