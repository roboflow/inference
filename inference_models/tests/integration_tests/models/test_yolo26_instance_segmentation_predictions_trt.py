import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    yolo26_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(asl_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16600


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy(
    yolo26_seg_asl_trt_package: str,
    asl_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([asl_image_numpy, asl_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16600
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[1].mask.cpu().sum().item() <= 16600


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch(
    yolo26_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(asl_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16600


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_multiple_predictions_in_row(
    yolo26_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    for _ in range(8):
        predictions = model(asl_image_torch)

        # then
        assert torch.allclose(
            predictions[0].confidence.cpu(),
            torch.tensor([0.9671]).cpu(),
            atol=0.01,
        )
        assert torch.allclose(
            predictions[0].class_id.cpu(),
            torch.tensor([20], dtype=torch.int32).cpu(),
        )
        expected_xyxy = torch.tensor(
            [[63, 174, 186, 368]],
            dtype=torch.int32,
        )
        assert torch.allclose(
            predictions[0].xyxy.cpu(),
            expected_xyxy.cpu(),
            atol=5,
        )
        assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16600


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_list(
    yolo26_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([asl_image_torch, asl_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16600
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[1].mask.cpu().sum().item() <= 16600


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_batch(
    yolo26_seg_asl_trt_package: str,
    asl_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_trt import (
        YOLO26ForInstanceSegmentationTRT,
    )

    model = YOLO26ForInstanceSegmentationTRT.from_pretrained(
        model_name_or_path=yolo26_seg_asl_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(torch.stack([asl_image_torch, asl_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[0].mask.cpu().sum().item() <= 16600
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9671]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([20], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[63, 174, 186, 368]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert 16100 <= predictions[1].mask.cpu().sum().item() <= 16600
