import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_stretch_resize_and_contrast_stretching_numpy(
    coin_counting_rfdetr_nano_onnx_cs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_cs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.8575,
                0.8568,
                0.8105,
                0.7940,
                0.7364,
                0.6872,
                0.6419,
                0.5810,
                0.5425,
                0.5340,
                0.5133,
            ]
        ).cpu(),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1704, 2567, 1894, 2756],
            [1741, 2292, 1921, 2468],
            [1460, 2297, 1630, 2468],
            [1500, 1875, 1730, 2096],
            [2673, 792, 2876, 978],
            [1158, 2620, 1383, 2849],
            [1247, 2055, 1430, 2227],
            [1087, 2342, 1264, 2524],
            [919, 1835, 1102, 2009],
            [919, 1835, 1102, 2009],
            [1247, 2055, 1430, 2227],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_stretch_resize_and_contrast_stretching_batch_numpy(
    coin_counting_rfdetr_nano_onnx_cs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_cs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [coins_counting_image_numpy, coins_counting_image_numpy], confidence=0.5
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.8575,
                0.8568,
                0.8105,
                0.7940,
                0.7364,
                0.6872,
                0.6419,
                0.5810,
                0.5425,
                0.5340,
                0.5133,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.8575,
                0.8568,
                0.8105,
                0.7940,
                0.7364,
                0.6872,
                0.6419,
                0.5810,
                0.5425,
                0.5340,
                0.5133,
            ]
        ).cpu(),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1704, 2567, 1894, 2756],
            [1741, 2292, 1921, 2468],
            [1460, 2297, 1630, 2468],
            [1500, 1875, 1730, 2096],
            [2673, 792, 2876, 978],
            [1158, 2620, 1383, 2849],
            [1247, 2055, 1430, 2227],
            [1087, 2342, 1264, 2524],
            [919, 1835, 1102, 2009],
            [919, 1835, 1102, 2009],
            [1247, 2055, 1430, 2227],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_stretch_resize_and_contrast_stretching_torch(
    coin_counting_rfdetr_nano_onnx_cs_stretch_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_cs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.8575,
                0.8568,
                0.8105,
                0.7940,
                0.7364,
                0.6872,
                0.6419,
                0.5810,
                0.5425,
                0.5340,
                0.5133,
            ]
        ).cpu(),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1704, 2567, 1894, 2756],
            [1741, 2292, 1921, 2468],
            [1460, 2297, 1630, 2468],
            [1500, 1875, 1730, 2096],
            [2673, 792, 2876, 978],
            [1158, 2620, 1383, 2849],
            [1247, 2055, 1430, 2227],
            [1087, 2342, 1264, 2524],
            [919, 1835, 1102, 2009],
            [919, 1835, 1102, 2009],
            [1247, 2055, 1430, 2227],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_stretch_resize_and_contrast_stretching_torch_batch(
    coin_counting_rfdetr_nano_onnx_cs_stretch_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_cs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch] * 2, dim=0), confidence=0.5
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.8575,
                0.8568,
                0.8105,
                0.7940,
                0.7364,
                0.6872,
                0.6419,
                0.5810,
                0.5425,
                0.5340,
                0.5133,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.8575,
                0.8568,
                0.8105,
                0.7940,
                0.7364,
                0.6872,
                0.6419,
                0.5810,
                0.5425,
                0.5340,
                0.5133,
            ]
        ).cpu(),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1704, 2567, 1894, 2756],
            [1741, 2292, 1921, 2468],
            [1460, 2297, 1630, 2468],
            [1500, 1875, 1730, 2096],
            [2673, 792, 2876, 978],
            [1158, 2620, 1383, 2849],
            [1247, 2055, 1430, 2227],
            [1087, 2342, 1264, 2524],
            [919, 1835, 1102, 2009],
            [919, 1835, 1102, 2009],
            [1247, 2055, 1430, 2227],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_letterbox_numpy(
    coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [0.9481, 0.9406, 0.9182, 0.9177, 0.9111, 0.8997, 0.8429, 0.7885]
        ).cpu(),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1253, 2056, 1424, 2227],
            [1460, 2302, 1625, 2469],
            [1743, 2295, 1914, 2467],
            [1094, 2348, 1256, 2524],
            [925, 1840, 1091, 2003],
            [1505, 1880, 1721, 2092],
            [1706, 2570, 1889, 2755],
            [1165, 2628, 1376, 2847],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_letterbox_numpy_batch(
    coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [coins_counting_image_numpy, coins_counting_image_numpy], confidence=0.5
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [0.9481, 0.9406, 0.9182, 0.9177, 0.9111, 0.8997, 0.8429, 0.7885]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [0.9481, 0.9406, 0.9182, 0.9177, 0.9111, 0.8997, 0.8429, 0.7885]
        ).cpu(),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1253, 2056, 1424, 2227],
            [1460, 2302, 1625, 2469],
            [1743, 2295, 1914, 2467],
            [1094, 2348, 1256, 2524],
            [925, 1840, 1091, 2003],
            [1505, 1880, 1721, 2092],
            [1706, 2570, 1889, 2755],
            [1165, 2628, 1376, 2847],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_onnx_package_with_static_crop_letterbox_torch(
    coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [0.9481, 0.9406, 0.9182, 0.9177, 0.9111, 0.8997, 0.8429, 0.7885]
        ).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1253, 2056, 1424, 2227],
            [1460, 2302, 1625, 2469],
            [1743, 2295, 1914, 2467],
            [925, 1840, 1091, 2003],
            [1094, 2348, 1256, 2524],
            [1505, 1880, 1721, 2092],
            [1706, 2570, 1889, 2755],
            [1165, 2628, 1376, 2847],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_onnx_package_with_static_crop_letterbox_torch_batch(
    coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch] * 2, dim=0), confidence=0.5
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [0.9481, 0.9406, 0.9182, 0.9177, 0.9111, 0.8997, 0.8429, 0.7885]
        ).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [0.9481, 0.9406, 0.9182, 0.9177, 0.9111, 0.8997, 0.8429, 0.7885]
        ).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1253, 2056, 1424, 2227],
            [1460, 2302, 1625, 2469],
            [1743, 2295, 1914, 2467],
            [925, 1840, 1091, 2003],
            [1094, 2348, 1256, 2524],
            [1505, 1880, 1721, 2092],
            [1706, 2570, 1889, 2755],
            [1165, 2628, 1376, 2847],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_onnx_package_with_static_crop_letterbox_torch_list(
    coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [coins_counting_image_torch, coins_counting_image_torch], confidence=0.5
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [0.9481, 0.9406, 0.9182, 0.9177, 0.9111, 0.8997, 0.8429, 0.7885]
        ).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [0.9481, 0.9406, 0.9182, 0.9177, 0.9111, 0.8997, 0.8429, 0.7885]
        ).cpu(),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1253, 2056, 1424, 2227],
            [1460, 2302, 1625, 2469],
            [1743, 2295, 1914, 2467],
            [925, 1840, 1091, 2003],
            [1094, 2348, 1256, 2524],
            [1505, 1880, 1721, 2092],
            [1706, 2570, 1889, 2755],
            [1165, 2628, 1376, 2847],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_center_crop_numpy(
    coin_counting_rfdetr_nano_onnx_center_crop_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy, confidence=0.55)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9746, 0.9664]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[1507, 1878, 1722, 2090], [1252, 2057, 1426, 2229]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_center_crop_batch_numpy(
    coin_counting_rfdetr_nano_onnx_center_crop_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [coins_counting_image_numpy, coins_counting_image_numpy], confidence=0.55
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9746, 0.9664]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9746, 0.9664]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[1507, 1878, 1722, 2090], [1252, 2057, 1426, 2229]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_center_crop_torch(
    coin_counting_rfdetr_nano_onnx_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch, confidence=0.55)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9746, 0.9664]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[1507, 1878, 1722, 2090], [1252, 2057, 1426, 2229]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_center_crop_batch_torch(
    coin_counting_rfdetr_nano_onnx_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0),
        confidence=0.55,
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9746, 0.9664]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9746, 0.9664]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[1507, 1878, 1722, 2090], [1252, 2057, 1426, 2229]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_center_crop_list_of_torch(
    coin_counting_rfdetr_nano_onnx_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [coins_counting_image_torch, coins_counting_image_torch], confidence=0.55
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9746, 0.9664]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9746, 0.9664]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[1507, 1878, 1722, 2090], [1252, 2057, 1426, 2229]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_and_center_crop_numpy(
    coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9750122, 0.96309197]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[1506, 1879, 1720, 2089], [1252, 2057, 1427, 2229]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_and_center_crop_numpy_when_image_smaller_than_center_crop(
    coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        coins_counting_image_numpy[2000:2300, 1250:1450], confidence=0.5
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(), torch.tensor([0.7778]).cpu(), atol=0.01
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[13, 59, 181, 229]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_and_center_crop_batch_numpy(
    coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [coins_counting_image_numpy, coins_counting_image_numpy], confidence=0.5
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9750122, 0.96309197]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9750122, 0.96309197]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[1506, 1879, 1720, 2089], [1252, 2057, 1427, 2229]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_and_center_crop_torch(
    coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9750122, 0.96309197]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[1506, 1879, 1720, 2089], [1252, 2057, 1427, 2229]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_crop_and_center_crop_batch_torch(
    coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0),
        confidence=0.5,
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.9750122, 0.96309197]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.9750122, 0.96309197]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[1506, 1879, 1720, 2089], [1252, 2057, 1427, 2229]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


_NONSQUARE_LETTERBOX_ONNX_EXPECTED_CONFIDENCE_NUMPY = torch.tensor(
    [
        0.9049550294876099,
        0.899626612663269,
        0.8745410442352295,
        0.8741258382797241,
        0.8696562647819519,
        0.8625761270523071,
        0.8569862246513367,
        0.8495969772338867,
        0.8333320617675781,
    ]
)
_NONSQUARE_LETTERBOX_ONNX_EXPECTED_CONFIDENCE_TORCH = torch.tensor(
    [
        0.905476987361908,
        0.8992284536361694,
        0.8746137022972107,
        0.8734182715415955,
        0.8694590926170349,
        0.862565279006958,
        0.8569447994232178,
        0.8493640422821045,
        0.832780659198761,
    ]
)
_NONSQUARE_LETTERBOX_ONNX_EXPECTED_CLASS_ID = torch.tensor(
    [0, 2, 0, 0, 0, 0, 2, 0, 2], dtype=torch.int32
)
_NONSQUARE_LETTERBOX_ONNX_EXPECTED_XYXY = torch.tensor(
    [
        [1464, 2300, 1636, 2479],
        [1706, 2574, 1898, 2769],
        [1501, 1881, 1722, 2104],
        [1178, 2627, 1368, 2854],
        [1089, 2357, 1254, 2529],
        [930, 1844, 1101, 2014],
        [1739, 2296, 1920, 2480],
        [2688, 809, 2850, 977],
        [1256, 2064, 1425, 2238],
    ],
    dtype=torch.int32,
)


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_nonsquare_letterbox_numpy(
    coin_counting_rfdetr_nano_onnx_static_bs_nonsquare_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_bs_nonsquare_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        _NONSQUARE_LETTERBOX_ONNX_EXPECTED_CONFIDENCE_NUMPY,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(), _NONSQUARE_LETTERBOX_ONNX_EXPECTED_CLASS_ID
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(), _NONSQUARE_LETTERBOX_ONNX_EXPECTED_XYXY, atol=2
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_nonsquare_letterbox_numpy_batch(
    coin_counting_rfdetr_nano_onnx_static_bs_nonsquare_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_bs_nonsquare_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [coins_counting_image_numpy, coins_counting_image_numpy], confidence=0.5
    )

    # then
    for pred in predictions:
        assert torch.allclose(
            pred.confidence.cpu(),
            _NONSQUARE_LETTERBOX_ONNX_EXPECTED_CONFIDENCE_NUMPY,
            atol=0.01,
        )
        assert torch.allclose(
            pred.class_id.cpu(), _NONSQUARE_LETTERBOX_ONNX_EXPECTED_CLASS_ID
        )
        assert torch.allclose(
            pred.xyxy.cpu(), _NONSQUARE_LETTERBOX_ONNX_EXPECTED_XYXY, atol=2
        )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_onnx_package_with_nonsquare_letterbox_torch(
    coin_counting_rfdetr_nano_onnx_static_bs_nonsquare_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_bs_nonsquare_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch, confidence=0.5)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        _NONSQUARE_LETTERBOX_ONNX_EXPECTED_CONFIDENCE_TORCH,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(), _NONSQUARE_LETTERBOX_ONNX_EXPECTED_CLASS_ID
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(), _NONSQUARE_LETTERBOX_ONNX_EXPECTED_XYXY, atol=2
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_onnx_package_with_nonsquare_letterbox_torch_batch(
    coin_counting_rfdetr_nano_onnx_static_bs_nonsquare_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_bs_nonsquare_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch] * 2, dim=0), confidence=0.5
    )

    # then
    for pred in predictions:
        assert torch.allclose(
            pred.confidence.cpu(),
            _NONSQUARE_LETTERBOX_ONNX_EXPECTED_CONFIDENCE_TORCH,
            atol=0.01,
        )
        assert torch.allclose(
            pred.class_id.cpu(), _NONSQUARE_LETTERBOX_ONNX_EXPECTED_CLASS_ID
        )
        assert torch.allclose(
            pred.xyxy.cpu(), _NONSQUARE_LETTERBOX_ONNX_EXPECTED_XYXY, atol=2
        )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_onnx_package_with_nonsquare_letterbox_torch_list(
    coin_counting_rfdetr_nano_onnx_static_bs_nonsquare_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_static_bs_nonsquare_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        [coins_counting_image_torch, coins_counting_image_torch], confidence=0.5
    )

    # then
    for pred in predictions:
        assert torch.allclose(
            pred.confidence.cpu(),
            _NONSQUARE_LETTERBOX_ONNX_EXPECTED_CONFIDENCE_TORCH,
            atol=0.01,
        )
        assert torch.allclose(
            pred.class_id.cpu(), _NONSQUARE_LETTERBOX_ONNX_EXPECTED_CLASS_ID
        )
        assert torch.allclose(
            pred.xyxy.cpu(), _NONSQUARE_LETTERBOX_ONNX_EXPECTED_XYXY, atol=2
        )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_per_class_confidence_blocks_specific_class(
    coin_counting_rfdetr_nano_onnx_cs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    """RFDETR OD: Group B direct-indexing + DETR no-object handling.
    Other tests in this file (e.g. center-crop runs) return all class_id=1 for
    coin counting. Setting a 0.99 per-class threshold on class 1 leaves no
    detections."""
    from inference_models.models.rfdetr.rfdetr_object_detection_onnx import (
        RFDetrForObjectDetectionONNX,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_cs_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.3,
        per_class_confidence={class_names[1]: 0.99},
    )
    predictions = model(coins_counting_image_numpy, confidence="best")
    assert 1 not in predictions[0].class_id.cpu().tolist()
