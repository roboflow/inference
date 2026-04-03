import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    yolo26_pose_trt_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_key_points_detection_trt import (
        YOLO26ForKeyPointsDetectionTRT,
    )

    model = YOLO26ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_pose_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(people_walking_image_numpy)

    # then
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9267, 0.9230]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[353, 129, 539, 758], [618, 123, 822, 771]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert abs(predictions[0][0].confidence.sum().item() - 26.268831253051758) < 1e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy(
    yolo26_pose_trt_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_key_points_detection_trt import (
        YOLO26ForKeyPointsDetectionTRT,
    )

    model = YOLO26ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_pose_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([people_walking_image_numpy, people_walking_image_numpy])

    # then
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9267, 0.9230]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[353, 129, 539, 758], [618, 123, 822, 771]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.9267, 0.9230]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[353, 129, 539, 758], [618, 123, 822, 771]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert abs(predictions[0][0].confidence.sum().item() - 26.268831253051758) < 1e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch(
    yolo26_pose_trt_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_key_points_detection_trt import (
        YOLO26ForKeyPointsDetectionTRT,
    )

    model = YOLO26ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_pose_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(people_walking_image_torch)

    # then
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9271, 0.9230]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[353, 129, 539, 758], [618, 123, 822, 771]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert abs(predictions[0][0].confidence.sum().item() - 26.268831253051758) < 1e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_multiple_predictions_in_row(
    yolo26_pose_trt_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_key_points_detection_trt import (
        YOLO26ForKeyPointsDetectionTRT,
    )

    model = YOLO26ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_pose_trt_package,
        engine_host_code_allowed=True,
    )

    for _ in range(8):
        # when
        predictions = model(people_walking_image_torch)

        # then
        assert torch.allclose(
            predictions[1][0].confidence.cpu(),
            torch.tensor([0.9271, 0.9230]).cpu(),
            atol=0.01,
        )
        assert torch.allclose(
            predictions[1][0].class_id.cpu(),
            torch.tensor([0, 0], dtype=torch.int32).cpu(),
        )
        expected_xyxy = torch.tensor(
            [[353, 129, 539, 758], [618, 123, 822, 771]],
            dtype=torch.int32,
        )
        assert torch.allclose(
            predictions[1][0].xyxy.cpu(),
            expected_xyxy.cpu(),
            atol=5,
        )
        assert (
            abs(predictions[0][0].confidence.sum().item() - 26.268831253051758) < 1e-2
        )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_list(
    yolo26_pose_trt_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_key_points_detection_trt import (
        YOLO26ForKeyPointsDetectionTRT,
    )

    model = YOLO26ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_pose_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([people_walking_image_torch, people_walking_image_torch])

    # then
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9271, 0.9230]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[353, 129, 539, 758], [618, 123, 822, 771]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.9271, 0.9230]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[353, 129, 539, 758], [618, 123, 822, 771]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert abs(predictions[0][0].confidence.sum().item() - 26.268831253051758) < 1e-2


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_batch(
    yolo26_pose_trt_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_key_points_detection_trt import (
        YOLO26ForKeyPointsDetectionTRT,
    )

    model = YOLO26ForKeyPointsDetectionTRT.from_pretrained(
        model_name_or_path=yolo26_pose_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(
        torch.stack([people_walking_image_torch, people_walking_image_torch], dim=0)
    )

    # then
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9271, 0.9230]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[353, 129, 539, 758], [618, 123, 822, 771]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.9271, 0.9230]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].class_id.cpu(),
        torch.tensor([0, 0], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[353, 129, 539, 758], [618, 123, 822, 771]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert abs(predictions[0][0].confidence.sum().item() - 26.268831253051758) < 1e-2
