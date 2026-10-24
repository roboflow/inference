import numpy as np
import pytest
import supervision as sv
import torch

from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask
from inference_models.models.common.roboflow.model_packages import StaticCrop

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

    predictions = model(snake_image_numpy, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

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
    assert 209000 <= mask_region_sum <= 210200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_static_bs_numpy_rle_variant(
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

    predictions = model(snake_image_numpy, confidence=0.25, mask_format="rle")
    predictions_ref = model(snake_image_numpy, confidence=0.25)
    decoded_mask = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )
    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

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
    assert 209000 <= mask_region_sum <= 210200
    assert np.allclose(
        decoded_mask.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )


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

    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.25)

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
    assert 209000 <= mask_region_sum_0 <= 210200
    assert 209000 <= mask_region_sum_1 <= 210200


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_static_bs_batch_numpy_rle_variant(
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

    predictions = model(
        [snake_image_numpy, snake_image_numpy], confidence=0.25, mask_format="rle"
    )
    predictions_ref = model([snake_image_numpy, snake_image_numpy], confidence=0.25)
    decoded_mask_1 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[0].mask, device=torch.device("cpu")
    )
    decoded_mask_2 = coco_rle_masks_to_torch_mask(
        instances_masks=predictions[1].mask, device=torch.device("cpu")
    )

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
    assert 209000 <= mask_region_sum_0 <= 210200
    assert 209000 <= mask_region_sum_1 <= 210200
    assert np.allclose(
        decoded_mask_1.cpu().numpy(), predictions_ref[0].mask.cpu().numpy()
    )
    assert np.allclose(
        decoded_mask_2.cpu().numpy(), predictions_ref[1].mask.cpu().numpy()
    )


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

    predictions = model(snake_image_torch, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

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
    assert 209000 <= mask_region_sum <= 210200


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

    predictions = model(snake_image_numpy, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

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
    assert 209000 <= mask_region_sum <= 210200


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

    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.25)

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
    assert 209000 <= mask_region_sum_0 <= 210200
    assert 209000 <= mask_region_sum_1 <= 210200


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

    predictions = model(snake_image_torch, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

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
    assert 209000 <= mask_region_sum <= 210200


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

    predictions = model(snake_image_numpy, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

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
    expected_xyxy = torch.tensor([[107, 325, 1280, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 219500 <= mask_region_sum <= 221300


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

    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.25)

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
    expected_xyxy = torch.tensor([[107, 325, 1280, 562]], dtype=torch.int32)
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
    assert 219500 <= mask_region_sum_0 <= 221300
    assert 219500 <= mask_region_sum_1 <= 221300


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

    predictions = model(snake_image_torch, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

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
    expected_xyxy = torch.tensor([[107, 325, 1280, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 219500 <= mask_region_sum <= 221300


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

    predictions = model(snake_image_numpy, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

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
    expected_xyxy = torch.tensor([[107, 325, 1280, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 219500 <= mask_region_sum <= 221300


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

    predictions = model([snake_image_numpy, snake_image_numpy], confidence=0.25)

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
    expected_xyxy = torch.tensor([[107, 325, 1280, 562]], dtype=torch.int32)
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
    assert 219500 <= mask_region_sum_0 <= 221300
    assert 219500 <= mask_region_sum_1 <= 221300


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

    predictions = model(snake_image_torch, confidence=0.25)

    xyxy = predictions[0].xyxy.cpu().tolist()[0]
    mask_region_sum = (
        predictions[0]
        .to_supervision()
        .mask[0, xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        .sum()
    )

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
    expected_xyxy = torch.tensor([[107, 325, 1280, 562]], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=XYXY_ATOL,
    )
    assert 219500 <= mask_region_sum <= 221300


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_per_class_confidence_blocks_snake_class(
    yolo26n_seg_snakes_stretch_onnx_static_package: str,
    snake_image_numpy: np.ndarray,
) -> None:
    """Baseline (see `test_onnx_package_stretch_static_bs_numpy` above) returns
    1 detection of class 0 (snake) at conf 0.9645. Setting a 0.99 per-class
    threshold on class 0 leaves no detections."""
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.25,
        per_class_confidence={class_names[0]: 0.99},
    )
    predictions = model(snake_image_numpy, confidence="best")
    assert predictions[0].class_id.numel() == 0


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.parametrize(
    "factor,expected_grid,expected_area",
    [(1.0, (890, 1280), 211140), (0.5, (501, 696), 211025), (0.0, (112, 112), 208787)],
)
@pytest.mark.parametrize(
    "crop_start", [None, 0, 25], ids=["uncropped", "origin-crop", "offset-crop"]
)
def test_mask_resolution_and_crop_round_trip(
    yolo26n_seg_snakes_stretch_onnx_static_package: str,
    snake_image_numpy: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    factor: float,
    crop_start: int | None,
    expected_grid: tuple,
    expected_area: int,
) -> None:
    # given
    from inference_models.models.yolo26.yolo26_instance_segmentation_onnx import (
        YOLO26ForInstanceSegmentationOnnx,
    )

    model = YOLO26ForInstanceSegmentationOnnx.from_pretrained(
        model_name_or_path=yolo26n_seg_snakes_stretch_onnx_static_package,
        onnx_execution_providers=["CPUExecutionProvider"],
        device=torch.device("cpu"),
    )
    image = np.pad(
        snake_image_numpy,
        (
            (0, snake_image_numpy.shape[0] % 2),
            (0, snake_image_numpy.shape[1] % 2),
            (0, 0),
        ),
    )
    height, width = image.shape[:2]
    scene = image
    offset_x = offset_y = 0
    crop_config = None
    if crop_start is not None:
        offset_y = 0 if crop_start == 0 else height // 2
        offset_x = 0 if crop_start == 0 else width // 2
        scene = np.zeros((2 * height, 2 * width, 3), dtype=image.dtype)
        scene[offset_y : offset_y + height, offset_x : offset_x + width] = image
        crop_config = StaticCrop(
            enabled=True,
            x_min=crop_start,
            y_min=crop_start,
            x_max=crop_start + 50,
            y_max=crop_start + 50,
        )
    monkeypatch.setattr(
        model._inference_config.image_pre_processing, "static_crop", crop_config
    )

    # when
    dense = model(
        scene, confidence=0.5, mask_format="dense", masks_resolution_factor=factor
    )[0]
    rle = model(
        scene, confidence=0.5, mask_format="rle", masks_resolution_factor=factor
    )[0]
    decoded = coco_rle_masks_to_torch_mask(
        instances_masks=rle.mask, device=torch.device("cpu")
    )
    detections = dense.to_supervision()

    # then
    offset = np.array([offset_x, offset_y, offset_x, offset_y])
    expected_boxes = np.array([[126, 326, 1263, 558]]) + offset
    np.testing.assert_allclose(detections.xyxy, expected_boxes, atol=2, rtol=0)
    np.testing.assert_array_equal(detections.class_id, [0])
    np.testing.assert_allclose(detections.confidence, [0.9657], atol=0.01, rtol=0)
    assert dense.image_size == scene.shape[:2]
    assert dense.mask_frame_size == scene.shape[:2]
    assert rle.image_size == scene.shape[:2]
    assert rle.mask_frame_size == scene.shape[:2]
    canvas_multiplier = 1 if crop_start is None else 2
    assert dense.mask_size == tuple(size * canvas_multiplier for size in expected_grid)
    assert rle.mask_size == dense.mask_size

    grid_scale = np.array(
        [dense.mask_size[1] / scene.shape[1], dense.mask_size[0] / scene.shape[0]] * 2
    )
    np.testing.assert_allclose(
        dense.xyxy.cpu().numpy(),
        expected_boxes * grid_scale,
        atol=2 * max(grid_scale),
        rtol=0,
    )
    if dense.mask_size != scene.shape[:2]:
        assert dense.xyxy.is_floating_point()
    assert detections.mask.shape == (1, *scene.shape[:2])
    # Areas were checked against overlays of this fixture at all three factors.
    assert detections.mask.sum() == pytest.approx(expected_area, rel=0.02)
    expected_mask_boxes = np.array([[120, 322, 1259, 559]]) + offset
    pixel_tolerance = (
        np.ceil(
            max(
                scene.shape[0] / dense.mask_size[0], scene.shape[1] / dense.mask_size[1]
            )
        )
        + 2
    )
    np.testing.assert_allclose(
        sv.mask_to_xyxy(detections.mask),
        expected_mask_boxes,
        atol=pixel_tolerance,
        rtol=0,
    )
    crop_masks = detections.mask[
        :, offset_y : offset_y + height, offset_x : offset_x + width
    ]
    assert crop_masks.sum() == detections.mask.sum()
    torch.testing.assert_close(dense.xyxy.cpu(), rle.xyxy.cpu())
    torch.testing.assert_close(dense.class_id.cpu(), rle.class_id.cpu())
    torch.testing.assert_close(dense.confidence.cpu(), rle.confidence.cpu())
    np.testing.assert_array_equal(decoded.numpy(), dense.mask.cpu().numpy())
    np.testing.assert_array_equal(detections.mask, rle.to_supervision().mask)
    sv.MaskAnnotator().annotate(scene.copy(), detections)
