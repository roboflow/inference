import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_letterbox_numpy(
    coin_counting_yolo_nas_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9593847,
                0.9211226,
                0.91721797,
                0.91207564,
                0.90716594,
                0.8921168,
                0.89186037,
                0.8840609,
                0.87438107,
                0.8660693,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1296, 481, 3024, 2081],
            [1701, 2559, 1892, 2769],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1386, 2861],
            [921, 1836, 1094, 2008],
            [2671, 792, 2874, 977],
            [1247, 2049, 1431, 2236],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2292, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_dynamic_batch_size_and_letterbox_torch(
    coin_counting_yolo_nas_onnx_dynamic_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_dynamic_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9593847,
                0.9211226,
                0.91721797,
                0.91207564,
                0.90716594,
                0.8921168,
                0.89186037,
                0.8840609,
                0.87438107,
                0.8660693,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1296, 481, 3024, 2081],
            [1701, 2559, 1892, 2769],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1386, 2861],
            [921, 1836, 1094, 2008],
            [1247, 2049, 1431, 2236],
            [2671, 792, 2874, 977],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2292, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_letterbox_numpy(
    coin_counting_yolo_nas_onnx_static_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9593847,
                0.9211226,
                0.91721797,
                0.91207564,
                0.90716594,
                0.8921168,
                0.89186037,
                0.8840609,
                0.87438107,
                0.8660693,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1296, 481, 3024, 2081],
            [1701, 2559, 1892, 2769],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1386, 2861],
            [921, 1836, 1094, 2008],
            [2671, 792, 2874, 977],
            [1247, 2049, 1431, 2236],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2292, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_letterbox_batch_numpy(
    coin_counting_yolo_nas_onnx_static_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9593847,
                0.9211226,
                0.91721797,
                0.91207564,
                0.90716594,
                0.8921168,
                0.89186037,
                0.8840609,
                0.87438107,
                0.8660693,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9593847,
                0.9211226,
                0.91721797,
                0.91207564,
                0.90716594,
                0.8921168,
                0.89186037,
                0.8840609,
                0.87438107,
                0.8660693,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1296, 481, 3024, 2081],
            [1701, 2559, 1892, 2769],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1386, 2861],
            [921, 1836, 1094, 2008],
            [2671, 792, 2874, 977],
            [1247, 2049, 1431, 2236],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2292, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_letterbox_torch(
    coin_counting_yolo_nas_onnx_static_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9593847,
                0.9211226,
                0.91721797,
                0.91207564,
                0.90716594,
                0.8921168,
                0.89186037,
                0.8840609,
                0.87438107,
                0.8660693,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1296, 481, 3024, 2081],
            [1701, 2559, 1892, 2769],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1386, 2861],
            [921, 1836, 1094, 2008],
            [1247, 2049, 1431, 2236],
            [2671, 792, 2874, 977],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2292, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_letterbox_batch_torch(
    coin_counting_yolo_nas_onnx_static_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0)
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9593847,
                0.9211226,
                0.91721797,
                0.91207564,
                0.90716594,
                0.8921168,
                0.89186037,
                0.8840609,
                0.87438107,
                0.8660693,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9593847,
                0.9211226,
                0.91721797,
                0.91207564,
                0.90716594,
                0.8921168,
                0.89186037,
                0.8840609,
                0.87438107,
                0.8660693,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1296, 481, 3024, 2081],
            [1701, 2559, 1892, 2769],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1386, 2861],
            [921, 1836, 1094, 2008],
            [1247, 2049, 1431, 2236],
            [2671, 792, 2874, 977],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2292, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_letterbox_batch_torch_list(
    coin_counting_yolo_nas_onnx_static_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9593847,
                0.9211226,
                0.91721797,
                0.91207564,
                0.90716594,
                0.8921168,
                0.89186037,
                0.8840609,
                0.87438107,
                0.8660693,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9593847,
                0.9211226,
                0.91721797,
                0.91207564,
                0.90716594,
                0.8921168,
                0.89186037,
                0.8840609,
                0.87438107,
                0.8660693,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1296, 481, 3024, 2081],
            [1701, 2559, 1892, 2769],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1386, 2861],
            [921, 1836, 1094, 2008],
            [1247, 2049, 1431, 2236],
            [2671, 792, 2874, 977],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2292, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_center_crop_numpy(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8236, 0.3160]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0, 4], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1193, 1696, 1830, 2335],
            [1503, 2304, 1576, 2336],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_center_crop_batch_numpy(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8236, 0.3160]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8236, 0.3160]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0, 4], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([0, 4], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1193, 1696, 1830, 2335],
            [1503, 2304, 1576, 2336],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_center_crop_torch(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8236, 0.3160]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0, 4], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1193, 1696, 1830, 2335],
            [1503, 2304, 1576, 2336],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_center_crop_torch_batch(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0)
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8236, 0.3160]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8236, 0.3160]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0, 4], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([0, 4], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1193, 1696, 1830, 2335],
            [1503, 2304, 1576, 2336],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_center_crop_torch_list(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.8236, 0.3160]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.8236, 0.3160]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([0, 4], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([0, 4], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1193, 1696, 1830, 2335],
            [1503, 2304, 1576, 2336],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_stretch_numpy(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9422,
                0.9151,
                0.9015,
                0.8959,
                0.8703,
                0.8614,
                0.8597,
                0.8486,
                0.8437,
                0.8300,
                0.3981,
                0.3623,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1275, 221, 3014, 2175],
            [1706, 2552, 1891, 2779],
            [922, 1825, 1104, 2015],
            [1247, 2039, 1426, 2243],
            [1160, 2613, 1379, 2867],
            [1739, 2281, 1914, 2483],
            [1461, 2288, 1632, 2486],
            [1499, 1863, 1723, 2111],
            [2672, 786, 2870, 991],
            [1086, 2338, 1268, 2538],
            [180, 1202, 263, 1296],
            [350, 911, 431, 997],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_stretch_batch_numpy(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9422,
                0.9151,
                0.9015,
                0.8959,
                0.8703,
                0.8614,
                0.8597,
                0.8486,
                0.8437,
                0.8300,
                0.3981,
                0.3623,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9422,
                0.9151,
                0.9015,
                0.8959,
                0.8703,
                0.8614,
                0.8597,
                0.8486,
                0.8437,
                0.8300,
                0.3981,
                0.3623,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1275, 221, 3014, 2175],
            [1706, 2552, 1891, 2779],
            [922, 1825, 1104, 2015],
            [1247, 2039, 1426, 2243],
            [1160, 2613, 1379, 2867],
            [1739, 2281, 1914, 2483],
            [1461, 2288, 1632, 2486],
            [1499, 1863, 1723, 2111],
            [2672, 786, 2870, 991],
            [1086, 2338, 1268, 2538],
            [180, 1202, 263, 1296],
            [350, 911, 431, 997],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_onnx_package_with_static_batch_size_and_static_crop_stretch_torch(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch)
    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.94191,
                0.91431,
                0.90124,
                0.89497,
                0.86867,
                0.86162,
                0.85914,
                0.84799,
                0.83728,
                0.83039,
                0.38431,
                0.34926,
            ]
        ).cpu(),
        atol=0.025,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1273, 213, 3015, 2175],
            [1706, 2552, 1891, 2778],
            [922, 1825, 1104, 2015],
            [1247, 2039, 1426, 2243],
            [1160, 2613, 1379, 2867],
            [1739, 2281, 1914, 2483],
            [1461, 2288, 1632, 2486],
            [1499, 1862, 1723, 2111],
            [2672, 786, 2870, 991],
            [1086, 2338, 1268, 2538],
            [179, 1202, 263, 1296],
            [350, 911, 430, 997],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_onnx_package_with_static_batch_size_and_static_crop_stretch_batch_torch(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0)
    )
    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.94191,
                0.91431,
                0.90124,
                0.89497,
                0.86867,
                0.86162,
                0.85914,
                0.84799,
                0.83728,
                0.83039,
                0.38431,
                0.34926,
            ]
        ).cpu(),
        atol=0.025,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.94191,
                0.91431,
                0.90124,
                0.89497,
                0.86867,
                0.86162,
                0.85914,
                0.84799,
                0.83728,
                0.83039,
                0.38431,
                0.34926,
            ]
        ).cpu(),
        atol=0.025,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1273, 213, 3015, 2175],
            [1706, 2552, 1891, 2778],
            [922, 1825, 1104, 2015],
            [1247, 2039, 1426, 2243],
            [1160, 2613, 1379, 2867],
            [1739, 2281, 1914, 2483],
            [1461, 2288, 1632, 2486],
            [1499, 1862, 1723, 2111],
            [2672, 786, 2870, 991],
            [1086, 2338, 1268, 2538],
            [179, 1202, 263, 1296],
            [350, 911, 430, 997],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.cpu_only
def test_onnx_package_with_static_batch_size_and_static_crop_stretch_list_of_torch(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_stretch_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])
    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.94191,
                0.91431,
                0.90124,
                0.89497,
                0.86867,
                0.86162,
                0.85914,
                0.84799,
                0.83728,
                0.83039,
                0.38431,
                0.34926,
            ]
        ).cpu(),
        atol=0.025,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.94191,
                0.91431,
                0.90124,
                0.89497,
                0.86867,
                0.86162,
                0.85914,
                0.84799,
                0.83728,
                0.83039,
                0.38431,
                0.34926,
            ]
        ).cpu(),
        atol=0.025,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1273, 213, 3015, 2175],
            [1706, 2552, 1891, 2778],
            [922, 1825, 1104, 2015],
            [1247, 2039, 1426, 2243],
            [1160, 2613, 1379, 2867],
            [1739, 2281, 1914, 2483],
            [1461, 2288, 1632, 2486],
            [1499, 1862, 1723, 2111],
            [2672, 786, 2870, 991],
            [1086, 2338, 1268, 2538],
            [179, 1202, 263, 1296],
            [350, 911, 430, 997],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_letterbox_numpy(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.95909,
                0.92093,
                0.91757,
                0.91166,
                0.90718,
                0.89292,
                0.89128,
                0.88446,
                0.87536,
                0.86531,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1295, 482, 3024, 2081],
            [1701, 2560, 1892, 2768],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1385, 2861],
            [921, 1836, 1095, 2008],
            [2671, 792, 2874, 977],
            [1247, 2049, 1431, 2236],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2293, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_letterbox_batch_numpy(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.95909,
                0.92093,
                0.91757,
                0.91166,
                0.90718,
                0.89292,
                0.89128,
                0.88446,
                0.87536,
                0.86531,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.95909,
                0.92093,
                0.91757,
                0.91166,
                0.90718,
                0.89292,
                0.89128,
                0.88446,
                0.87536,
                0.86531,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1295, 482, 3024, 2081],
            [1701, 2560, 1892, 2768],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1385, 2861],
            [921, 1836, 1095, 2008],
            [2671, 792, 2874, 977],
            [1247, 2049, 1431, 2236],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2293, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_letterbox_torch(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.95909,
                0.92093,
                0.91757,
                0.91166,
                0.90718,
                0.89292,
                0.89128,
                0.88446,
                0.87536,
                0.86531,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1295, 482, 3024, 2081],
            [1701, 2560, 1892, 2768],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1385, 2861],
            [921, 1836, 1095, 2008],
            [1247, 2049, 1431, 2236],
            [2671, 792, 2874, 977],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2293, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_letterbox_batch_torch(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0)
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.95909,
                0.92093,
                0.91757,
                0.91166,
                0.90718,
                0.89292,
                0.89128,
                0.88446,
                0.87536,
                0.86531,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1295, 482, 3024, 2081],
            [1701, 2560, 1892, 2768],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1385, 2861],
            [921, 1836, 1095, 2008],
            [1247, 2049, 1431, 2236],
            [2671, 792, 2874, 977],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2293, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_static_crop_letterbox_list_torch(
    coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_static_crop_letterbox_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.95909,
                0.92093,
                0.91757,
                0.91166,
                0.90718,
                0.89292,
                0.89128,
                0.88446,
                0.87536,
                0.86531,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1295, 482, 3024, 2081],
            [1701, 2560, 1892, 2768],
            [1735, 2288, 1920, 2477],
            [1161, 2616, 1385, 2861],
            [921, 1836, 1095, 2008],
            [1247, 2049, 1431, 2236],
            [2671, 792, 2874, 977],
            [1086, 2341, 1270, 2533],
            [1497, 1870, 1730, 2104],
            [1455, 2293, 1632, 2485],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_center_crop_numpy(
    coin_counting_yolo_nas_onnx_static_bs_center_crop_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.96446, 0.91637, 0.86476, 0.66761, 0.61128]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 4, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1245, 2053, 1429, 2233],
            [1493, 1871, 1726, 2097],
            [1253, 1695, 1831, 1987],
            [1455, 2297, 1651, 2336],
            [1739, 2290, 1834, 2336],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_center_crop_batch_numpy(
    coin_counting_yolo_nas_onnx_static_bs_center_crop_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.96446, 0.91637, 0.86476, 0.66761, 0.61128]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.96446, 0.91637, 0.86476, 0.66761, 0.61128]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 4, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 4, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1245, 2053, 1429, 2233],
            [1493, 1871, 1726, 2097],
            [1253, 1695, 1831, 1987],
            [1455, 2297, 1651, 2336],
            [1739, 2290, 1834, 2336],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_center_crop_torch(
    coin_counting_yolo_nas_onnx_static_bs_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.96446, 0.91637, 0.86476, 0.66761, 0.61128]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 4, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1245, 2053, 1429, 2233],
            [1493, 1871, 1726, 2097],
            [1253, 1695, 1831, 1987],
            [1455, 2297, 1651, 2336],
            [1739, 2290, 1834, 2336],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_center_crop_batch_torch(
    coin_counting_yolo_nas_onnx_static_bs_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0)
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.96446, 0.91637, 0.86476, 0.66761, 0.61128]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.96446, 0.91637, 0.86476, 0.66761, 0.61128]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 4, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 4, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1245, 2053, 1429, 2233],
            [1493, 1871, 1726, 2097],
            [1253, 1695, 1831, 1987],
            [1455, 2297, 1651, 2336],
            [1739, 2290, 1834, 2336],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_with_static_batch_size_and_center_crop_list_torch(
    coin_counting_yolo_nas_onnx_static_bs_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )

    model = YOLONasForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=coin_counting_yolo_nas_onnx_static_bs_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.96446, 0.91637, 0.86476, 0.66761, 0.61128]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.96446, 0.91637, 0.86476, 0.66761, 0.61128]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 4, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 4, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1245, 2053, 1429, 2233],
            [1493, 1871, 1726, 2097],
            [1253, 1695, 1831, 1987],
            [1455, 2297, 1651, 2336],
            [1739, 2290, 1834, 2336],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy,
        atol=2,
    )
