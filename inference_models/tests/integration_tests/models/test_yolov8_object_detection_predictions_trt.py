import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    yolov8_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_coin_counting_trt_package
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
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
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy(
    yolov8_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_coin_counting_trt_package
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
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
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch(
    yolov8_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_coin_counting_trt_package
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
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
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_list(
    yolov8_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_coin_counting_trt_package
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
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
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_batch(
    yolov8_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )

    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov8_coin_counting_trt_package
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
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
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
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9956,
                0.9727,
                0.9653,
                0.9468,
                0.9448,
                0.9390,
                0.9302,
                0.9287,
                0.9155,
                0.9019,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1304, 614, 3024, 1918],
            [1714, 2571, 1884, 2759],
            [2678, 806, 2866, 974],
            [1744, 2294, 1914, 2469],
            [1260, 2058, 1424, 2233],
            [1469, 2302, 1624, 2467],
            [929, 1843, 1091, 1997],
            [1514, 1880, 1718, 2089],
            [1177, 2632, 1374, 2846],
            [1099, 2348, 1260, 2522],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
