import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9809,
                0.9674,
                0.9638,
                0.9622,
                0.9581,
                0.9558,
                0.9555,
                0.9540,
                0.9493,
                0.9489,
                0.4168,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1324, 538, 3070, 1970],
            [1707, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2352, 1254, 2522],
            [1508, 1884, 1720, 2093],
            [2681, 802, 2867, 976],
            [929, 1843, 1091, 2004],
            [929, 1845, 1091, 2004],
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
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9809,
                0.9674,
                0.9638,
                0.9622,
                0.9581,
                0.9558,
                0.9555,
                0.9540,
                0.9493,
                0.9489,
                0.4168,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1324, 538, 3070, 1970],
            [1707, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2352, 1254, 2522],
            [1508, 1884, 1720, 2093],
            [2681, 802, 2867, 976],
            [929, 1843, 1091, 2004],
            [929, 1845, 1091, 2004],
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
                0.9809,
                0.9674,
                0.9638,
                0.9622,
                0.9581,
                0.9558,
                0.9555,
                0.9540,
                0.9493,
                0.9489,
                0.4168,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1324, 538, 3070, 1970],
            [1707, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2352, 1254, 2522],
            [1508, 1884, 1720, 2093],
            [2681, 802, 2867, 976],
            [929, 1843, 1091, 2004],
            [929, 1845, 1091, 2004],
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
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9809,
                0.9674,
                0.9638,
                0.9622,
                0.9581,
                0.9558,
                0.9555,
                0.9540,
                0.9493,
                0.9489,
                0.4168,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1324, 538, 3070, 1970],
            [1707, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2352, 1254, 2522],
            [1508, 1884, 1720, 2093],
            [2681, 802, 2867, 976],
            [929, 1843, 1091, 2004],
            [929, 1845, 1091, 2004],
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
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9809,
                0.9674,
                0.9638,
                0.9622,
                0.9581,
                0.9558,
                0.9555,
                0.9540,
                0.9493,
                0.9489,
                0.4168,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1324, 538, 3070, 1970],
            [1707, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2352, 1254, 2522],
            [1508, 1884, 1720, 2093],
            [2681, 802, 2867, 976],
            [929, 1843, 1091, 2004],
            [929, 1845, 1091, 2004],
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
                0.9809,
                0.9674,
                0.9638,
                0.9622,
                0.9581,
                0.9558,
                0.9555,
                0.9540,
                0.9493,
                0.9489,
                0.4168,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1324, 538, 3070, 1970],
            [1707, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2352, 1254, 2522],
            [1508, 1884, 1720, 2093],
            [2681, 802, 2867, 976],
            [929, 1843, 1091, 2004],
            [929, 1845, 1091, 2004],
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
    rfdetr_coin_counting_trt_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_object_detection_trt import (
        RFDetrForObjectDetectionTRT,
    )

    model = RFDetrForObjectDetectionTRT.from_pretrained(
        model_name_or_path=rfdetr_coin_counting_trt_package,
        engine_host_code_allowed=True,
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
                0.9809,
                0.9674,
                0.9638,
                0.9622,
                0.9581,
                0.9558,
                0.9555,
                0.9540,
                0.9493,
                0.9489,
                0.4168,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1324, 538, 3070, 1970],
            [1707, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2352, 1254, 2522],
            [1508, 1884, 1720, 2093],
            [2681, 802, 2867, 976],
            [929, 1843, 1091, 2004],
            [929, 1845, 1091, 2004],
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
                0.9809,
                0.9674,
                0.9638,
                0.9622,
                0.9581,
                0.9558,
                0.9555,
                0.9540,
                0.9493,
                0.9489,
                0.4168,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1324, 538, 3070, 1970],
            [1707, 2572, 1887, 2760],
            [1172, 2635, 1372, 2850],
            [1744, 2296, 1914, 2472],
            [1464, 2305, 1627, 2475],
            [1255, 2063, 1423, 2233],
            [1091, 2352, 1254, 2522],
            [1508, 1884, 1720, 2093],
            [2681, 802, 2867, 976],
            [929, 1843, 1091, 2004],
            [929, 1845, 1091, 2004],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
