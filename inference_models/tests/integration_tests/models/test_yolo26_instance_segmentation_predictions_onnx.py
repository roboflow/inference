import numpy as np
import pytest
import torch

CONFIDENCE_ATOL = 0.01
XYXY_ATOL = 2


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_static_bs_numpy(
    yolo26n_seg_snakes_stretch_onnx_static_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(snake_image_numpy)

    print(f"confidence: {predictions[0].confidence.cpu()}")
    print(f"class_id: {predictions[0].class_id.cpu()}")
    print(f"xyxy: {predictions[0].xyxy.cpu()}")
    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )
    print(f"mask_region_sum: {mask_region_sum}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9645]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[128, 326, 1263, 558]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 210000 <= mask_region_sum <= 210200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_static_bs_batch_numpy(
    yolo26n_seg_snakes_stretch_onnx_static_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([snake_image_numpy, snake_image_numpy])

    print(f"predictions[0].confidence: {predictions[0].confidence.cpu()}")
    print(f"predictions[1].confidence: {predictions[1].confidence.cpu()}")
    print(f"predictions[0].class_id: {predictions[0].class_id.cpu()}")
    print(f"predictions[1].class_id: {predictions[1].class_id.cpu()}")
    print(f"predictions[0].xyxy: {predictions[0].xyxy.cpu()}")
    print(f"predictions[1].xyxy: {predictions[1].xyxy.cpu()}")
    xyxy_0 = predictions[0].xyxy.cpu().tolist()[0]
    xyxy_1 = predictions[1].xyxy.cpu().tolist()[0]
    mask_region_sum_0 = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy_0[1] : xyxy_0[3], xyxy_0[0] : xyxy_0[2]]
        .sum()
    )
    mask_region_sum_1 = (
        predictions[1]
        .to_supervision()
        .mask[0, xyxy_1[1] : xyxy_1[3], xyxy_1[0] : xyxy_1[2]]
        .sum()
    )
    print(f"mask_region_sum_0: {mask_region_sum_0}")
    print(f"mask_region_sum_1: {mask_region_sum_1}")

    assert len(predictions) == 2
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9645]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9645]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[128, 326, 1263, 558]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 210000 <= mask_region_sum_0 <= 210200
    assert 210000 <= mask_region_sum_1 <= 210200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_static_bs_torch(
    yolo26n_seg_snakes_stretch_onnx_static_package: str,
    snake_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(snake_image_torch)

    print(f"confidence: {predictions[0].confidence.cpu()}")
    print(f"class_id: {predictions[0].class_id.cpu()}")
    print(f"xyxy: {predictions[0].xyxy.cpu()}")
    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )
    print(f"mask_region_sum: {mask_region_sum}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9641]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[128, 326, 1263, 558]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 210000 <= mask_region_sum <= 210200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_dynamic_bs_numpy(
    yolo26n_seg_snakes_stretch_onnx_dynamic_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(snake_image_numpy)

    print(f"confidence: {predictions[0].confidence.cpu()}")
    print(f"class_id: {predictions[0].class_id.cpu()}")
    print(f"xyxy: {predictions[0].xyxy.cpu()}")
    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )
    print(f"mask_region_sum: {mask_region_sum}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9645]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[128, 326, 1263, 558]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 210000 <= mask_region_sum <= 210200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_dynamic_bs_batch_numpy(
    yolo26n_seg_snakes_stretch_onnx_dynamic_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([snake_image_numpy, snake_image_numpy])

    print(f"predictions[0].confidence: {predictions[0].confidence.cpu()}")
    print(f"predictions[1].confidence: {predictions[1].confidence.cpu()}")
    print(f"predictions[0].class_id: {predictions[0].class_id.cpu()}")
    print(f"predictions[1].class_id: {predictions[1].class_id.cpu()}")
    print(f"predictions[0].xyxy: {predictions[0].xyxy.cpu()}")
    print(f"predictions[1].xyxy: {predictions[1].xyxy.cpu()}")
    xyxy_0 = predictions[0].xyxy.cpu().tolist()[0]
    xyxy_1 = predictions[1].xyxy.cpu().tolist()[0]
    mask_region_sum_0 = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy_0[1] : xyxy_0[3], xyxy_0[0] : xyxy_0[2]]
        .sum()
    )
    mask_region_sum_1 = (
        predictions[1]
        .to_supervision()
        .mask[0, xyxy_1[1] : xyxy_1[3], xyxy_1[0] : xyxy_1[2]]
        .sum()
    )
    print(f"mask_region_sum_0: {mask_region_sum_0}")
    print(f"mask_region_sum_1: {mask_region_sum_1}")

    assert len(predictions) == 2
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9646]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9646]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[128, 326, 1263, 558]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 210000 <= mask_region_sum_0 <= 210200
    assert 210000 <= mask_region_sum_1 <= 210200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_dynamic_bs_torch(
    yolo26n_seg_snakes_stretch_onnx_dynamic_package: str,
    snake_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(snake_image_torch)

    print(f"confidence: {predictions[0].confidence.cpu()}")
    print(f"class_id: {predictions[0].class_id.cpu()}")
    print(f"xyxy: {predictions[0].xyxy.cpu()}")
    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )
    print(f"mask_region_sum: {mask_region_sum}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9641]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[128, 326, 1263, 558]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 210000 <= mask_region_sum <= 210200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_static_bs_numpy(
    yolo26n_seg_snakes_letterbox_onnx_static_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(snake_image_numpy)

    print(f"confidence: {predictions[0].confidence.cpu()}")
    print(f"class_id: {predictions[0].class_id.cpu()}")
    print(f"xyxy: {predictions[0].xyxy.cpu()}")
    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )
    print(f"mask_region_sum: {mask_region_sum}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.275]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[107, 325, 1293, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 221000 <= mask_region_sum <= 221300


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_static_bs_batch_numpy(
    yolo26n_seg_snakes_letterbox_onnx_static_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([snake_image_numpy, snake_image_numpy])

    print(f"predictions[0].confidence: {predictions[0].confidence.cpu()}")
    print(f"predictions[1].confidence: {predictions[1].confidence.cpu()}")
    print(f"predictions[0].class_id: {predictions[0].class_id.cpu()}")
    print(f"predictions[1].class_id: {predictions[1].class_id.cpu()}")
    print(f"predictions[0].xyxy: {predictions[0].xyxy.cpu()}")
    print(f"predictions[1].xyxy: {predictions[1].xyxy.cpu()}")
    xyxy_0 = predictions[0].xyxy.cpu().tolist()[0]
    xyxy_1 = predictions[1].xyxy.cpu().tolist()[0]
    mask_region_sum_0 = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy_0[1] : xyxy_0[3], xyxy_0[0] : xyxy_0[2]]
        .sum()
    )
    mask_region_sum_1 = (
        predictions[1]
        .to_supervision()
        .mask[0, xyxy_1[1] : xyxy_1[3], xyxy_1[0] : xyxy_1[2]]
        .sum()
    )
    print(f"mask_region_sum_0: {mask_region_sum_0}")
    print(f"mask_region_sum_1: {mask_region_sum_1}")

    assert len(predictions) == 2
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.275]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.275]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[107, 325, 1293, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 221000 <= mask_region_sum_0 <= 221300
    assert 221000 <= mask_region_sum_1 <= 221300


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_static_bs_torch(
    yolo26n_seg_snakes_letterbox_onnx_static_package: str,
    snake_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(snake_image_torch)

    print(f"confidence: {predictions[0].confidence.cpu()}")
    print(f"class_id: {predictions[0].class_id.cpu()}")
    print(f"xyxy: {predictions[0].xyxy.cpu()}")
    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )
    print(f"mask_region_sum: {mask_region_sum}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.271]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[107, 325, 1293, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 221000 <= mask_region_sum <= 221300


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_dynamic_bs_numpy(
    yolo26n_seg_snakes_letterbox_onnx_dynamic_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(snake_image_numpy)

    print(f"confidence: {predictions[0].confidence.cpu()}")
    print(f"class_id: {predictions[0].class_id.cpu()}")
    print(f"xyxy: {predictions[0].xyxy.cpu()}")
    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )
    print(f"mask_region_sum: {mask_region_sum}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.275]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[107, 325, 1293, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 221000 <= mask_region_sum <= 221300


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_dynamic_bs_batch_numpy(
    yolo26n_seg_snakes_letterbox_onnx_dynamic_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([snake_image_numpy, snake_image_numpy])

    print(f"predictions[0].confidence: {predictions[0].confidence.cpu()}")
    print(f"predictions[1].confidence: {predictions[1].confidence.cpu()}")
    print(f"predictions[0].class_id: {predictions[0].class_id.cpu()}")
    print(f"predictions[1].class_id: {predictions[1].class_id.cpu()}")
    print(f"predictions[0].xyxy: {predictions[0].xyxy.cpu()}")
    print(f"predictions[1].xyxy: {predictions[1].xyxy.cpu()}")
    xyxy_0 = predictions[0].xyxy.cpu().tolist()[0]
    xyxy_1 = predictions[1].xyxy.cpu().tolist()[0]
    mask_region_sum_0 = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy_0[1] : xyxy_0[3], xyxy_0[0] : xyxy_0[2]]
        .sum()
    )
    mask_region_sum_1 = (
        predictions[1]
        .to_supervision()
        .mask[0, xyxy_1[1] : xyxy_1[3], xyxy_1[0] : xyxy_1[2]]
        .sum()
    )
    print(f"mask_region_sum_0: {mask_region_sum_0}")
    print(f"mask_region_sum_1: {mask_region_sum_1}")

    assert len(predictions) == 2
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.274]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.274]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[107, 325, 1293, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 221000 <= mask_region_sum_0 <= 221300
    assert 221000 <= mask_region_sum_1 <= 221300


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_dynamic_bs_torch(
    yolo26n_seg_snakes_letterbox_onnx_dynamic_package: str,
    snake_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(snake_image_torch)

    print(f"confidence: {predictions[0].confidence.cpu()}")
    print(f"class_id: {predictions[0].class_id.cpu()}")
    print(f"xyxy: {predictions[0].xyxy.cpu()}")
    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )
    print(f"mask_region_sum: {mask_region_sum}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.271]),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor([[107, 325, 1293, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 221000 <= mask_region_sum <= 221300
