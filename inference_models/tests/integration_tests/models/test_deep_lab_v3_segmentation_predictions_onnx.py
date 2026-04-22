import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_stretch_numpy(
    balloons_deep_lab_v3_onnx_stretch_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(balloons_image_numpy, confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 16700
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_stretch_batch_numpy(
    balloons_deep_lab_v3_onnx_stretch_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([balloons_image_numpy, balloons_image_numpy], confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 16700
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 16700
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_stretch_torch(
    balloons_deep_lab_v3_onnx_stretch_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(balloons_image_torch, confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 16700
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_stretch_batch_torch(
    balloons_deep_lab_v3_onnx_stretch_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([balloons_image_torch, balloons_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 16700
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 16700
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_stretch_batch_torch_list(
    balloons_deep_lab_v3_onnx_stretch_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([balloons_image_torch, balloons_image_torch], confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 16700
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.9646).cpu(),
        atol=0.001,
    )
    assert (
        245000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 246000
    )
    assert (
        16600 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 16700
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_letterbox_numpy(
    balloons_deep_lab_v3_onnx_static_crop_letterbox_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(balloons_image_numpy, confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 15000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_letterbox_batch_numpy(
    balloons_deep_lab_v3_onnx_static_crop_letterbox_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([balloons_image_numpy, balloons_image_numpy], confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 15000
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 15000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_letterbox_torch(
    balloons_deep_lab_v3_onnx_static_crop_letterbox_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(balloons_image_torch, confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 15000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_letterbox_batch_torch(
    balloons_deep_lab_v3_onnx_static_crop_letterbox_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([balloons_image_torch, balloons_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 15000
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 15000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_letterbox_batch_torch_list(
    balloons_deep_lab_v3_onnx_static_crop_letterbox_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([balloons_image_torch, balloons_image_torch], confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 15000
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2423).cpu(),
        atol=0.001,
    )
    assert (
        247000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 248000
    )
    assert (
        14800 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 15000
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_center_crop_numpy(
    balloons_deep_lab_v3_onnx_static_crop_center_crop_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(balloons_image_numpy, confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 13900
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_center_crop_batch_numpy(
    balloons_deep_lab_v3_onnx_static_crop_center_crop_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([balloons_image_numpy, balloons_image_numpy], confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 13900
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 13900
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_center_crop_torch(
    balloons_deep_lab_v3_onnx_static_crop_center_crop_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(balloons_image_torch, confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 13900
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_center_crop_batch_torch(
    balloons_deep_lab_v3_onnx_static_crop_center_crop_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([balloons_image_torch, balloons_image_torch], dim=0), confidence=0.5
    )

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 13900
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 13900
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_center_crop_batch_torch_list(
    balloons_deep_lab_v3_onnx_static_crop_center_crop_package: str,
    balloons_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([balloons_image_torch, balloons_image_torch], confidence=0.5)

    # then
    assert sorted(torch.unique(predictions[0].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[0].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[0].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[0].segmentation_map.cpu() == 3).item() <= 13900
    )
    assert sorted(torch.unique(predictions[1].segmentation_map).cpu().tolist()) == [
        0,
        1,
        3,
    ]
    assert torch.allclose(
        torch.mean(predictions[1].confidence).cpu(),
        torch.tensor(0.2461).cpu(),
        atol=0.001,
    )
    assert (
        248000 <= torch.sum(predictions[1].segmentation_map.cpu() == 0).item() <= 249000
    )
    assert (
        13700 <= torch.sum(predictions[1].segmentation_map.cpu() == 3).item() <= 13900
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_per_class_confidence_reduces_balloon_pixels(
    balloons_deep_lab_v3_onnx_stretch_package: str,
    balloons_image_numpy: np.ndarray,
) -> None:
    """Baseline (see `test_onnx_package_with_stretch_numpy` above) has class 3
    (balloon) occupying ~16600 pixels. Setting a 0.99 per-class threshold on
    class 3 reassigns below-threshold pixels to background, reducing the
    class 3 pixel count."""
    from inference_models.models.deep_lab_v3_plus.deep_lab_v3_plus_segmentation_onnx import (
        DeepLabV3PlusForSemanticSegmentationOnnx,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = DeepLabV3PlusForSemanticSegmentationOnnx.from_pretrained(
        model_name_or_path=balloons_deep_lab_v3_onnx_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.5,
        per_class_confidence={class_names[3]: 0.99},
    )
    predictions = model(balloons_image_numpy, confidence="best")
    balloon_pixels = torch.sum(predictions[0].segmentation_map.cpu() == 3).item()
    assert balloon_pixels < 16600
