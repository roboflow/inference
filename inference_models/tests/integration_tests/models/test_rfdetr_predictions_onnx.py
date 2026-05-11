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
        torch.tensor([
            0.9444,
            0.9384,
            0.9369,
            0.9279,
            0.902,
            0.898,
            0.8813,
            0.879,
            0.7601,
            0.5973,
        ]),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1743, 2295, 1919, 2465],
            [1090, 2348, 1261, 2521],
            [1460, 2301, 1629, 2468],
            [1252, 2059, 1425, 2228],
            [924, 1843, 1098, 2002],
            [1708, 2574, 1890, 2754],
            [2677, 801, 2868, 975],
            [1506, 1881, 1726, 2091],
            [1167, 2628, 1380, 2846],
            [1303, 538, 3019, 1955],
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
        torch.tensor([
            0.9444,
            0.9384,
            0.9369,
            0.9279,
            0.902,
            0.898,
            0.8813,
            0.879,
            0.7601,
            0.5973,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([
            0.9444,
            0.9384,
            0.9369,
            0.9279,
            0.902,
            0.898,
            0.8813,
            0.879,
            0.7601,
            0.5973,
        ]),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1743, 2295, 1919, 2465],
            [1090, 2348, 1261, 2521],
            [1460, 2301, 1629, 2468],
            [1252, 2059, 1425, 2228],
            [924, 1843, 1098, 2002],
            [1708, 2574, 1890, 2754],
            [2677, 801, 2868, 975],
            [1506, 1881, 1726, 2091],
            [1167, 2628, 1380, 2846],
            [1303, 538, 3019, 1955],
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
        torch.tensor([
            0.9346,
            0.9344,
            0.9145,
            0.9036,
            0.892,
            0.8903,
            0.8553,
            0.7684,
            0.6861,
            0.5013,
        ]),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1460, 2300, 1630, 2468],
            [1089, 2348, 1262, 2521],
            [1742, 2294, 1920, 2469],
            [1707, 2573, 1891, 2757],
            [1250, 2058, 1426, 2228],
            [1504, 1881, 1725, 2092],
            [921, 1841, 1101, 2002],
            [2675, 800, 2872, 975],
            [1165, 2629, 1381, 2846],
            [1300, 533, 3025, 1959],
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
        torch.tensor([
            0.9346,
            0.9344,
            0.9145,
            0.9036,
            0.892,
            0.8903,
            0.8553,
            0.7684,
            0.6861,
            0.5013,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([
            0.9346,
            0.9344,
            0.9145,
            0.9036,
            0.892,
            0.8903,
            0.8553,
            0.7684,
            0.6861,
            0.5013,
        ]),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1460, 2300, 1630, 2468],
            [1089, 2348, 1262, 2521],
            [1742, 2294, 1920, 2469],
            [1707, 2573, 1891, 2757],
            [1250, 2058, 1426, 2228],
            [1504, 1881, 1725, 2092],
            [921, 1841, 1101, 2002],
            [2675, 800, 2872, 975],
            [1165, 2629, 1381, 2846],
            [1300, 533, 3025, 1959],
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
        torch.tensor([
            0.9123,
            0.9047,
            0.8818,
            0.8336,
            0.8227,
            0.783,
            0.7721,
            0.5093,
        ]),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1094, 2352, 1257, 2523],
            [1742, 2297, 1914, 2464],
            [1253, 2058, 1423, 2226],
            [1171, 2633, 1367, 2844],
            [926, 1843, 1090, 2001],
            [1463, 2305, 1625, 2467],
            [1709, 2573, 1887, 2755],
            [1506, 1885, 1721, 2090],
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
        torch.tensor([
            0.9123,
            0.9047,
            0.8818,
            0.8336,
            0.8227,
            0.783,
            0.7721,
            0.5093,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([
            0.9123,
            0.9047,
            0.8818,
            0.8336,
            0.8227,
            0.783,
            0.7721,
            0.5093,
        ]),
        atol=0.01,
    )
    expected_xyxy = torch.tensor(
        [
            [1094, 2352, 1257, 2523],
            [1742, 2297, 1914, 2464],
            [1253, 2058, 1423, 2226],
            [1171, 2633, 1367, 2844],
            [926, 1843, 1090, 2001],
            [1463, 2305, 1625, 2467],
            [1709, 2573, 1887, 2755],
            [1506, 1885, 1721, 2090],
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
        torch.tensor([
            0.9333,
            0.8997,
            0.899,
            0.886,
            0.8851,
            0.8332,
            0.6993,
        ]),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [1093, 2352, 1257, 2522],
            [1462, 2304, 1625, 2467],
            [1170, 2632, 1370, 2844],
            [1742, 2296, 1914, 2467],
            [1710, 2571, 1886, 2758],
            [925, 1843, 1091, 2001],
            [1252, 2059, 1424, 2227],
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
        torch.tensor([
            0.9333,
            0.8997,
            0.899,
            0.886,
            0.8851,
            0.8332,
            0.6993,
        ]),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([
            0.9333,
            0.8997,
            0.899,
            0.886,
            0.8851,
            0.8332,
            0.6993,
        ]),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [1093, 2352, 1257, 2522],
            [1462, 2304, 1625, 2467],
            [1170, 2632, 1370, 2844],
            [1742, 2296, 1914, 2467],
            [1710, 2571, 1886, 2758],
            [925, 1843, 1091, 2001],
            [1252, 2059, 1424, 2227],
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
        torch.tensor([
            0.9333,
            0.8997,
            0.899,
            0.886,
            0.8851,
            0.8332,
            0.6993,
        ]),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([
            0.9333,
            0.8997,
            0.899,
            0.886,
            0.8851,
            0.8332,
            0.6993,
        ]),
        atol=0.02,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [1093, 2352, 1257, 2522],
            [1462, 2304, 1625, 2467],
            [1170, 2632, 1370, 2844],
            [1742, 2296, 1914, 2467],
            [1710, 2571, 1886, 2758],
            [925, 1843, 1091, 2001],
            [1252, 2059, 1424, 2227],
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
        torch.tensor([
            0.9417,
            0.9379,
            0.9327,
            0.9145,
            0.9115,
            0.9024,
            0.8951,
            0.8713,
            0.7543,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            1,
            1,
            1,
            3,
            3,
            6,
            3,
            6,
            3,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [1249, 2059, 1432, 2228],
            [1740, 2297, 1922, 2463],
            [1700, 2572, 1897, 2752],
            [916, 1830, 1105, 2009],
            [2671, 793, 2877, 981],
            [1161, 2622, 1386, 2850],
            [1087, 2340, 1266, 2527],
            [1501, 1874, 1732, 2092],
            [1457, 2291, 1635, 2469],
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
        torch.tensor([
            0.9417,
            0.9379,
            0.9327,
            0.9145,
            0.9115,
            0.9024,
            0.8951,
            0.8713,
            0.7543,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([
            0.9417,
            0.9379,
            0.9327,
            0.9145,
            0.9115,
            0.9024,
            0.8951,
            0.8713,
            0.7543,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            1,
            1,
            1,
            3,
            3,
            6,
            3,
            6,
            3,
        ], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([
            1,
            1,
            1,
            3,
            3,
            6,
            3,
            6,
            3,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [1249, 2059, 1432, 2228],
            [1740, 2297, 1922, 2463],
            [1700, 2572, 1897, 2752],
            [916, 1830, 1105, 2009],
            [2671, 793, 2877, 981],
            [1161, 2622, 1386, 2850],
            [1087, 2340, 1266, 2527],
            [1501, 1874, 1732, 2092],
            [1457, 2291, 1635, 2469],
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
        torch.tensor([
            0.9045,
            0.9044,
            0.8649,
            0.8351,
            0.7959,
            0.7359,
            0.7029,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            3,
            3,
            3,
            3,
            3,
            6,
            3,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [2671, 794, 2878, 981],
            [915, 1832, 1105, 2010],
            [1739, 2285, 1925, 2472],
            [1087, 2340, 1265, 2525],
            [1248, 2049, 1433, 2235],
            [1160, 2622, 1387, 2849],
            [1456, 2293, 1635, 2468],
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
        torch.tensor([
            0.9045,
            0.9044,
            0.8649,
            0.8351,
            0.7959,
            0.7359,
            0.7029,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([
            0.9045,
            0.9044,
            0.8649,
            0.8351,
            0.7959,
            0.7359,
            0.7029,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            3,
            3,
            3,
            3,
            3,
            6,
            3,
        ], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([
            3,
            3,
            3,
            3,
            3,
            6,
            3,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [2671, 794, 2878, 981],
            [915, 1832, 1105, 2010],
            [1739, 2285, 1925, 2472],
            [1087, 2340, 1265, 2525],
            [1248, 2049, 1433, 2235],
            [1160, 2622, 1387, 2849],
            [1456, 2293, 1635, 2468],
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
        torch.tensor([
            0.9045,
            0.9044,
            0.8649,
            0.8351,
            0.7959,
            0.7359,
            0.7029,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([
            0.9045,
            0.9044,
            0.8649,
            0.8351,
            0.7959,
            0.7359,
            0.7029,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            3,
            3,
            3,
            3,
            3,
            6,
            3,
        ], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([
            3,
            3,
            3,
            3,
            3,
            6,
            3,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [2671, 794, 2878, 981],
            [915, 1832, 1105, 2010],
            [1739, 2285, 1925, 2472],
            [1087, 2340, 1265, 2525],
            [1248, 2049, 1433, 2235],
            [1160, 2622, 1387, 2849],
            [1456, 2293, 1635, 2468],
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
        torch.tensor([
            0.9569,
            0.9367,
            0.921,
            0.9165,
            0.8757,
            0.715,
            0.6055,
            0.6008,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            1,
            1,
            1,
            3,
            3,
            5,
            3,
            6,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [1250, 2059, 1431, 2226],
            [1739, 2298, 1923, 2465],
            [1702, 2568, 1898, 2756],
            [918, 1835, 1103, 2007],
            [1085, 2344, 1267, 2526],
            [1162, 2624, 1385, 2851],
            [1456, 2297, 1634, 2469],
            [1162, 2624, 1385, 2851],
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
        predictions[0].confidence.cpu(), torch.tensor([0.7049]), atol=0.01
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [21, 60, 177, 230],
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
        torch.tensor([
            0.9569,
            0.9367,
            0.921,
            0.9165,
            0.8757,
            0.715,
            0.6055,
            0.6008,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([
            0.9569,
            0.9367,
            0.921,
            0.9165,
            0.8757,
            0.715,
            0.6055,
            0.6008,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            1,
            1,
            1,
            3,
            3,
            5,
            3,
            6,
        ], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([
            1,
            1,
            1,
            3,
            3,
            5,
            3,
            6,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [1250, 2059, 1431, 2226],
            [1739, 2298, 1923, 2465],
            [1702, 2568, 1898, 2756],
            [918, 1835, 1103, 2007],
            [1085, 2344, 1267, 2526],
            [1162, 2624, 1385, 2851],
            [1456, 2297, 1634, 2469],
            [1162, 2624, 1385, 2851],
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
        torch.tensor([
            0.905,
            0.8226,
            0.8017,
            0.7009,
            0.6046,
            0.5421,
            0.5292,
            0.5212,
            0.5194,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            3,
            1,
            3,
            3,
            5,
            3,
            3,
            1,
            1,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [917, 1835, 1103, 2006],
            [1456, 2301, 1634, 2467],
            [1087, 2347, 1267, 2523],
            [1738, 2290, 1922, 2471],
            [1161, 2626, 1386, 2849],
            [1248, 2055, 1432, 2229],
            [1701, 2563, 1898, 2758],
            [1248, 2055, 1432, 2229],
            [1501, 1876, 1729, 2091],
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
        torch.tensor([
            0.905,
            0.8226,
            0.8017,
            0.7009,
            0.6046,
            0.5421,
            0.5292,
            0.5212,
            0.5194,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([
            0.905,
            0.8226,
            0.8017,
            0.7009,
            0.6046,
            0.5421,
            0.5292,
            0.5212,
            0.5194,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([
            3,
            1,
            3,
            3,
            5,
            3,
            3,
            1,
            1,
        ], dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([
            3,
            1,
            3,
            3,
            5,
            3,
            3,
            1,
            1,
        ], dtype=torch.int32),
    )
    expected_xyxy = torch.tensor(
        [
            [917, 1835, 1103, 2006],
            [1456, 2301, 1634, 2467],
            [1087, 2347, 1267, 2523],
            [1738, 2290, 1922, 2471],
            [1161, 2626, 1386, 2849],
            [1248, 2055, 1432, 2229],
            [1701, 2563, 1898, 2758],
            [1248, 2055, 1432, 2229],
            [1501, 1876, 1729, 2091],
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
        torch.tensor([
            0.9022,
            0.896,
            0.8829,
            0.8745,
            0.8734,
            0.8663,
            0.8585,
            0.8466,
            0.8416,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(), torch.tensor([
            2,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            2,
        ], dtype=torch.int32)
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(), torch.tensor(
        [
            [1704, 2575, 1899, 2769],
            [1464, 2300, 1636, 2479],
            [1502, 1881, 1721, 2104],
            [1090, 2358, 1257, 2529],
            [1179, 2627, 1368, 2854],
            [931, 1844, 1103, 2013],
            [1740, 2296, 1920, 2480],
            [2688, 811, 2852, 977],
            [1256, 2064, 1426, 2238],
        ],
        dtype=torch.int32,
    ), atol=2
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
            torch.tensor([
            0.9022,
            0.896,
            0.8829,
            0.8745,
            0.8734,
            0.8663,
            0.8585,
            0.8466,
            0.8416,
        ]),
            atol=0.01,
        )
        assert torch.allclose(
            pred.class_id.cpu(), torch.tensor([
            2,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            2,
        ], dtype=torch.int32)
        )
        assert torch.allclose(
            pred.xyxy.cpu(), torch.tensor(
        [
            [1704, 2575, 1899, 2769],
            [1464, 2300, 1636, 2479],
            [1502, 1881, 1721, 2104],
            [1090, 2358, 1257, 2529],
            [1179, 2627, 1368, 2854],
            [931, 1844, 1103, 2013],
            [1740, 2296, 1920, 2480],
            [2688, 811, 2852, 977],
            [1256, 2064, 1426, 2238],
        ],
        dtype=torch.int32,
    ), atol=2
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
        torch.tensor([
            0.8914,
            0.8833,
            0.8738,
            0.8713,
            0.8623,
            0.8522,
            0.8473,
            0.8245,
            0.7112,
        ]),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(), torch.tensor([
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ], dtype=torch.int32)
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(), torch.tensor(
        [
            [1255, 2065, 1421, 2240],
            [1709, 2577, 1888, 2764],
            [1090, 2360, 1253, 2530],
            [1464, 2300, 1637, 2482],
            [1735, 2298, 1912, 2479],
            [933, 1845, 1101, 2014],
            [2688, 811, 2857, 977],
            [1500, 1882, 1723, 2107],
            [1175, 2628, 1367, 2854],
        ],
        dtype=torch.int32,
    ), atol=2
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
            torch.tensor([
            0.8914,
            0.8833,
            0.8738,
            0.8713,
            0.8623,
            0.8522,
            0.8473,
            0.8245,
            0.7112,
        ]),
            atol=0.01,
        )
        assert torch.allclose(
            pred.class_id.cpu(), torch.tensor([
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ], dtype=torch.int32)
        )
        assert torch.allclose(
            pred.xyxy.cpu(), torch.tensor(
        [
            [1255, 2065, 1421, 2240],
            [1709, 2577, 1888, 2764],
            [1090, 2360, 1253, 2530],
            [1464, 2300, 1637, 2482],
            [1735, 2298, 1912, 2479],
            [933, 1845, 1101, 2014],
            [2688, 811, 2857, 977],
            [1500, 1882, 1723, 2107],
            [1175, 2628, 1367, 2854],
        ],
        dtype=torch.int32,
    ), atol=2
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
            torch.tensor([
            0.8914,
            0.8833,
            0.8738,
            0.8713,
            0.8623,
            0.8522,
            0.8473,
            0.8245,
            0.7112,
        ]),
            atol=0.01,
        )
        assert torch.allclose(
            pred.class_id.cpu(), torch.tensor([
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ], dtype=torch.int32)
        )
        assert torch.allclose(
            pred.xyxy.cpu(), torch.tensor(
        [
            [1255, 2065, 1421, 2240],
            [1709, 2577, 1888, 2764],
            [1090, 2360, 1253, 2530],
            [1464, 2300, 1637, 2482],
            [1735, 2298, 1912, 2479],
            [933, 1845, 1101, 2014],
            [2688, 811, 2857, 977],
            [1500, 1882, 1723, 2107],
            [1175, 2628, 1367, 2854],
        ],
        dtype=torch.int32,
    ), atol=2
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
