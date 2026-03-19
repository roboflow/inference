import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    yolo_nas_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolonas.yolonas_object_detection_trt import (
        YOLONasForObjectDetectionTRT,
    )

    model = YOLONasForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo_nas_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    # warmup
    for _ in range(5):
        _ = model(coins_counting_image_numpy)
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.8926,
                0.8762,
                0.8634,
                0.8558,
                0.8423,
                0.7766,
                0.7712,
                0.7642,
                0.6817,
                0.6397,
                0.4446,
                0.4340,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([2, 1, 0, 0, 0, 3, 0, 3, 2, 2, 0, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1693, 2548, 1910, 2775],
            [1161, 2618, 1389, 2868],
            [1445, 2291, 1641, 2483],
            [913, 1823, 1109, 2017],
            [1080, 2334, 1275, 2537],
            [2664, 763, 2888, 1000],
            [1727, 2285, 1931, 2483],
            [1490, 1862, 1740, 2101],
            [1727, 2283, 1932, 2487],
            [1238, 2041, 1438, 2243],
            [1484, 1864, 1743, 2106],
            [1236, 2040, 1439, 2245],
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
    yolo_nas_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolonas.yolonas_object_detection_trt import (
        YOLONasForObjectDetectionTRT,
    )

    model = YOLONasForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo_nas_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    # warmup
    for _ in range(5):
        _ = model([coins_counting_image_numpy, coins_counting_image_numpy])
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.8926,
                0.8762,
                0.8634,
                0.8558,
                0.8423,
                0.7766,
                0.7712,
                0.7642,
                0.6817,
                0.6397,
                0.4446,
                0.4340,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([2, 1, 0, 0, 0, 3, 0, 3, 2, 2, 0, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1693, 2548, 1910, 2775],
            [1161, 2618, 1389, 2868],
            [1445, 2291, 1641, 2483],
            [913, 1823, 1109, 2017],
            [1080, 2334, 1275, 2537],
            [2664, 763, 2888, 1000],
            [1727, 2285, 1931, 2483],
            [1490, 1862, 1740, 2101],
            [1727, 2283, 1932, 2487],
            [1238, 2041, 1438, 2243],
            [1484, 1864, 1743, 2106],
            [1236, 2040, 1439, 2245],
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
                0.8926,
                0.8762,
                0.8634,
                0.8558,
                0.8423,
                0.7766,
                0.7712,
                0.7642,
                0.6817,
                0.6397,
                0.4446,
                0.4340,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([2, 1, 0, 0, 0, 3, 0, 3, 2, 2, 0, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1693, 2548, 1910, 2775],
            [1161, 2618, 1389, 2868],
            [1445, 2291, 1641, 2483],
            [913, 1823, 1109, 2017],
            [1080, 2334, 1275, 2537],
            [2664, 763, 2888, 1000],
            [1727, 2285, 1931, 2483],
            [1490, 1862, 1740, 2101],
            [1727, 2283, 1932, 2487],
            [1238, 2041, 1438, 2243],
            [1484, 1864, 1743, 2106],
            [1236, 2040, 1439, 2245],
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
    yolo_nas_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolonas.yolonas_object_detection_trt import (
        YOLONasForObjectDetectionTRT,
    )

    model = YOLONasForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo_nas_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    # warmup
    for _ in range(5):
        _ = model(coins_counting_image_torch)
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.8929,
                0.8762,
                0.8625,
                0.8573,
                0.8434,
                0.7718,
                0.7705,
                0.7628,
                0.6723,
                0.6343,
                0.4533,
                0.4388,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([2, 1, 0, 0, 0, 0, 3, 3, 2, 2, 0, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1693, 2548, 1910, 2774],
            [1161, 2618, 1389, 2868],
            [1445, 2291, 1641, 2483],
            [913, 1823, 1110, 2017],
            [1080, 2334, 1275, 2537],
            [1727, 2285, 1931, 2482],
            [2664, 763, 2887, 1001],
            [1491, 1862, 1740, 2101],
            [1727, 2283, 1932, 2487],
            [1238, 2041, 1438, 2243],
            [1485, 1864, 1743, 2106],
            [1236, 2040, 1439, 2245],
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
def test_trt_package_torch_multiple_predictions_in_row(
    yolo_nas_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolonas.yolonas_object_detection_trt import (
        YOLONasForObjectDetectionTRT,
    )

    model = YOLONasForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo_nas_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    # warmup
    for _ in range(5):
        _ = model(coins_counting_image_torch)
    for _ in range(8):
        predictions = model(coins_counting_image_torch)

        # then
        assert torch.allclose(
            predictions[0].confidence.cpu(),
            torch.tensor(
                [
                    0.8929,
                    0.8762,
                    0.8625,
                    0.8573,
                    0.8434,
                    0.7718,
                    0.7705,
                    0.7628,
                    0.6723,
                    0.6343,
                    0.4533,
                    0.4388,
                ]
            ).cpu(),
            atol=0.01,
        )
        assert torch.allclose(
            predictions[0].class_id.cpu(),
            torch.tensor([2, 1, 0, 0, 0, 0, 3, 3, 2, 2, 0, 1], dtype=torch.int32).cpu(),
        )
        expected_xyxy = torch.tensor(
            [
                [1693, 2548, 1910, 2774],
                [1161, 2618, 1389, 2868],
                [1445, 2291, 1641, 2483],
                [913, 1823, 1110, 2017],
                [1080, 2334, 1275, 2537],
                [1727, 2285, 1931, 2482],
                [2664, 763, 2887, 1001],
                [1491, 1862, 1740, 2101],
                [1727, 2283, 1932, 2487],
                [1238, 2041, 1438, 2243],
                [1485, 1864, 1743, 2106],
                [1236, 2040, 1439, 2245],
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
    yolo_nas_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolonas.yolonas_object_detection_trt import (
        YOLONasForObjectDetectionTRT,
    )

    model = YOLONasForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo_nas_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    # warmup
    for _ in range(5):
        _ = model([coins_counting_image_torch, coins_counting_image_torch])
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.8929,
                0.8762,
                0.8625,
                0.8573,
                0.8434,
                0.7718,
                0.7705,
                0.7628,
                0.6723,
                0.6343,
                0.4533,
                0.4388,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([2, 1, 0, 0, 0, 0, 3, 3, 2, 2, 0, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1693, 2548, 1910, 2774],
            [1161, 2618, 1389, 2868],
            [1445, 2291, 1641, 2483],
            [913, 1823, 1110, 2017],
            [1080, 2334, 1275, 2537],
            [1727, 2285, 1931, 2482],
            [2664, 763, 2887, 1001],
            [1491, 1862, 1740, 2101],
            [1727, 2283, 1932, 2487],
            [1238, 2041, 1438, 2243],
            [1485, 1864, 1743, 2106],
            [1236, 2040, 1439, 2245],
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
                0.8929,
                0.8762,
                0.8625,
                0.8573,
                0.8434,
                0.7718,
                0.7705,
                0.7628,
                0.6723,
                0.6343,
                0.4533,
                0.4388,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([2, 1, 0, 0, 0, 0, 3, 3, 2, 2, 0, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1693, 2548, 1910, 2774],
            [1161, 2618, 1389, 2868],
            [1445, 2291, 1641, 2483],
            [913, 1823, 1110, 2017],
            [1080, 2334, 1275, 2537],
            [1727, 2285, 1931, 2482],
            [2664, 763, 2887, 1001],
            [1491, 1862, 1740, 2101],
            [1727, 2283, 1932, 2487],
            [1238, 2041, 1438, 2243],
            [1485, 1864, 1743, 2106],
            [1236, 2040, 1439, 2245],
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
    yolo_nas_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolonas.yolonas_object_detection_trt import (
        YOLONasForObjectDetectionTRT,
    )

    model = YOLONasForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolo_nas_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    # warmup
    for _ in range(5):
        _ = model(torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0))
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0)
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.8929,
                0.8762,
                0.8625,
                0.8573,
                0.8434,
                0.7718,
                0.7705,
                0.7628,
                0.6723,
                0.6343,
                0.4533,
                0.4388,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([2, 1, 0, 0, 0, 0, 3, 3, 2, 2, 0, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1693, 2548, 1910, 2774],
            [1161, 2618, 1389, 2868],
            [1445, 2291, 1641, 2483],
            [913, 1823, 1110, 2017],
            [1080, 2334, 1275, 2537],
            [1727, 2285, 1931, 2482],
            [2664, 763, 2887, 1001],
            [1491, 1862, 1740, 2101],
            [1727, 2283, 1932, 2487],
            [1238, 2041, 1438, 2243],
            [1485, 1864, 1743, 2106],
            [1236, 2040, 1439, 2245],
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
                0.8929,
                0.8762,
                0.8625,
                0.8573,
                0.8434,
                0.7718,
                0.7705,
                0.7628,
                0.6723,
                0.6343,
                0.4533,
                0.4388,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([2, 1, 0, 0, 0, 0, 3, 3, 2, 2, 0, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1693, 2548, 1910, 2774],
            [1161, 2618, 1389, 2868],
            [1445, 2291, 1641, 2483],
            [913, 1823, 1110, 2017],
            [1080, 2334, 1275, 2537],
            [1727, 2285, 1931, 2482],
            [2664, 763, 2887, 1001],
            [1491, 1862, 1740, 2101],
            [1727, 2283, 1932, 2487],
            [1238, 2041, 1438, 2243],
            [1485, 1864, 1743, 2106],
            [1236, 2040, 1439, 2245],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
