import numpy as np
import pytest
import torch
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.models.yolov8.yolov8_object_detection_torch_script import (
    YOLOv8ForObjectDetectionTorchScript,
)


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_letterbox_batch_numpy(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
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
            [1296, 528, 3024, 1979],
            [1172, 2632, 1376, 2847],
            [1709, 2572, 1885, 2757],
            [1744, 2295, 1914, 2467],
            [1513, 1880, 1717, 2089],
            [1254, 2057, 1424, 2226],
            [1090, 2351, 1260, 2522],
            [1462, 2304, 1623, 2468],
            [2684, 800, 2868, 967],
            [931, 1843, 1090, 1999],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_letterbox_torch(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
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
            [1296, 528, 3024, 1979],
            [1172, 2632, 1376, 2847],
            [1709, 2572, 1885, 2757],
            [1744, 2295, 1914, 2467],
            [1513, 1880, 1717, 2089],
            [1254, 2057, 1424, 2226],
            [1090, 2351, 1260, 2522],
            [1462, 2304, 1623, 2468],
            [2684, 800, 2868, 967],
            [931, 1843, 1090, 1999],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_letterbox_batch_torch(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_package,
        device=DEFAULT_DEVICE,
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
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
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
            [1296, 528, 3024, 1979],
            [1172, 2632, 1376, 2847],
            [1709, 2572, 1885, 2757],
            [1744, 2295, 1914, 2467],
            [1513, 1880, 1717, 2089],
            [1254, 2057, 1424, 2226],
            [1090, 2351, 1260, 2522],
            [1462, 2304, 1623, 2468],
            [2684, 800, 2868, 967],
            [931, 1843, 1090, 1999],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_letterbox_batch_torch_list(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
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
            [1296, 528, 3024, 1979],
            [1172, 2632, 1376, 2847],
            [1709, 2572, 1885, 2757],
            [1744, 2295, 1914, 2467],
            [1513, 1880, 1717, 2089],
            [1254, 2057, 1424, 2226],
            [1090, 2351, 1260, 2522],
            [1462, 2304, 1623, 2468],
            [2684, 800, 2868, 967],
            [931, 1843, 1090, 1999],
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
def test_torch_script_package_with_static_batch_size_and_letterbox__fused_nms_numpy(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
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
            [1296, 528, 3024, 1979],
            [1172, 2632, 1376, 2847],
            [1709, 2572, 1885, 2757],
            [1744, 2295, 1914, 2467],
            [1513, 1880, 1717, 2089],
            [1254, 2057, 1424, 2226],
            [1090, 2351, 1260, 2522],
            [1462, 2304, 1623, 2468],
            [2684, 800, 2868, 967],
            [931, 1843, 1090, 1999],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_letterbox_fused_nms_batch_numpy(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
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
            [1296, 528, 3024, 1979],
            [1172, 2632, 1376, 2847],
            [1709, 2572, 1885, 2757],
            [1744, 2295, 1914, 2467],
            [1513, 1880, 1717, 2089],
            [1254, 2057, 1424, 2226],
            [1090, 2351, 1260, 2522],
            [1462, 2304, 1623, 2468],
            [2684, 800, 2868, 967],
            [931, 1843, 1090, 1999],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_letterbox_fused_nms_torch(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
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
            [1296, 528, 3024, 1979],
            [1172, 2632, 1376, 2847],
            [1709, 2572, 1885, 2757],
            [1744, 2295, 1914, 2467],
            [1513, 1880, 1717, 2089],
            [1254, 2057, 1424, 2226],
            [1090, 2351, 1260, 2522],
            [1462, 2304, 1623, 2468],
            [2684, 800, 2868, 967],
            [931, 1843, 1090, 1999],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_letterbox_fused_nms_batch_torch(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package,
        device=DEFAULT_DEVICE,
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
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
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
            [1296, 528, 3024, 1979],
            [1172, 2632, 1376, 2847],
            [1709, 2572, 1885, 2757],
            [1744, 2295, 1914, 2467],
            [1513, 1880, 1717, 2089],
            [1254, 2057, 1424, 2226],
            [1090, 2351, 1260, 2522],
            [1462, 2304, 1623, 2468],
            [2684, 800, 2868, 967],
            [931, 1843, 1090, 1999],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_letterbox_fused_nms_batch_torch_list(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
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
            [1296, 528, 3024, 1979],
            [1172, 2632, 1376, 2847],
            [1709, 2572, 1885, 2757],
            [1744, 2295, 1914, 2467],
            [1513, 1880, 1717, 2089],
            [1254, 2057, 1424, 2226],
            [1090, 2351, 1260, 2522],
            [1462, 2304, 1623, 2468],
            [2684, 800, 2868, 967],
            [931, 1843, 1090, 1999],
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
def test_torch_script_package_with_static_batch_size_and_letterbox_numpy(
    coin_counting_yolov8n_torch_script_static_bs_letterbox_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9608,
                0.9449,
                0.9339,
                0.9266,
                0.9143,
                0.9090,
                0.8923,
                0.8791,
                0.8508,
                0.8202,
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
            [1296, 528, 3024, 1979],
            [1172, 2632, 1376, 2847],
            [1709, 2572, 1885, 2757],
            [1744, 2295, 1914, 2467],
            [1513, 1880, 1717, 2089],
            [1254, 2057, 1424, 2226],
            [1090, 2351, 1260, 2522],
            [1462, 2304, 1623, 2468],
            [2684, 800, 2868, 967],
            [931, 1843, 1090, 1999],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_numpy(
    coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1710, 2569, 1889, 2759],
            [1509, 1876, 1720, 2098],
            [1253, 2055, 1424, 2226],
            [1467, 2300, 1628, 2467],
            [1095, 2352, 1260, 2519],
            [1172, 2634, 1375, 2846],
            [1746, 2295, 1913, 2469],
            [932, 1842, 1094, 2000],
            [1302, 509, 2732, 1926],
            [2674, 809, 2720, 963],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_torch(
    coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1710, 2569, 1889, 2759],
            [1509, 1876, 1720, 2098],
            [1253, 2055, 1424, 2226],
            [1467, 2300, 1628, 2467],
            [1095, 2352, 1260, 2519],
            [1172, 2634, 1375, 2846],
            [1746, 2295, 1913, 2469],
            [932, 1842, 1094, 2000],
            [1302, 509, 2732, 1926],
            [2674, 809, 2720, 963],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_batch_numpy(
    coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1710, 2569, 1889, 2759],
            [1509, 1876, 1720, 2098],
            [1253, 2055, 1424, 2226],
            [1467, 2300, 1628, 2467],
            [1095, 2352, 1260, 2519],
            [1172, 2634, 1375, 2846],
            [1746, 2295, 1913, 2469],
            [932, 1842, 1094, 2000],
            [1302, 509, 2732, 1926],
            [2674, 809, 2720, 963],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_torch(
    coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1710, 2569, 1889, 2759],
            [1509, 1876, 1720, 2098],
            [1253, 2055, 1424, 2226],
            [1467, 2300, 1628, 2467],
            [1095, 2352, 1260, 2519],
            [1172, 2634, 1375, 2846],
            [1746, 2295, 1913, 2469],
            [932, 1842, 1094, 2000],
            [1302, 509, 2732, 1926],
            [2674, 809, 2720, 963],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_batch_torch(
    coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package,
        device=DEFAULT_DEVICE,
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
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1710, 2569, 1889, 2759],
            [1509, 1876, 1720, 2098],
            [1253, 2055, 1424, 2226],
            [1467, 2300, 1628, 2467],
            [1095, 2352, 1260, 2519],
            [1172, 2634, 1375, 2846],
            [1746, 2295, 1913, 2469],
            [932, 1842, 1094, 2000],
            [1302, 509, 2732, 1926],
            [2674, 809, 2720, 963],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_torch_list(
    coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_dynamic_bs_static_crop_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1710, 2569, 1889, 2759],
            [1509, 1876, 1720, 2098],
            [1253, 2055, 1424, 2226],
            [1467, 2300, 1628, 2467],
            [1095, 2352, 1260, 2519],
            [1172, 2634, 1375, 2846],
            [1746, 2295, 1913, 2469],
            [932, 1842, 1094, 2000],
            [1302, 509, 2732, 1926],
            [2674, 809, 2720, 963],
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


@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_nms_fused_numpy(
    coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1710, 2569, 1889, 2759],
            [1509, 1876, 1720, 2098],
            [1253, 2055, 1424, 2226],
            [1467, 2300, 1628, 2467],
            [1095, 2352, 1260, 2519],
            [1172, 2634, 1375, 2846],
            [1746, 2295, 1913, 2469],
            [932, 1842, 1094, 2000],
            [1302, 509, 2732, 1926],
            [2674, 809, 2720, 963],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_nms_fused_batch_numpy(
    coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1710, 2569, 1889, 2759],
            [1509, 1876, 1720, 2098],
            [1253, 2055, 1424, 2226],
            [1467, 2300, 1628, 2467],
            [1095, 2352, 1260, 2519],
            [1172, 2634, 1375, 2846],
            [1746, 2295, 1913, 2469],
            [932, 1842, 1094, 2000],
            [1302, 509, 2732, 1926],
            [2674, 809, 2720, 963],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_nms_fused_torch(
    coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1710, 2569, 1889, 2759],
            [1509, 1876, 1720, 2098],
            [1253, 2055, 1424, 2226],
            [1467, 2300, 1628, 2467],
            [1095, 2352, 1260, 2519],
            [1172, 2634, 1375, 2846],
            [1746, 2295, 1913, 2469],
            [932, 1842, 1094, 2000],
            [1302, 509, 2732, 1926],
            [2674, 809, 2720, 963],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_nms_fused_batch_torch(
    coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package,
        device=DEFAULT_DEVICE,
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
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1710, 2569, 1889, 2759],
            [1509, 1876, 1720, 2098],
            [1253, 2055, 1424, 2226],
            [1467, 2300, 1628, 2467],
            [1095, 2352, 1260, 2519],
            [1172, 2634, 1375, 2846],
            [1746, 2295, 1913, 2469],
            [932, 1842, 1094, 2000],
            [1302, 509, 2732, 1926],
            [2674, 809, 2720, 963],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_static_crop_stretch_nms_fused_torch_list(
    coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_torch_script_static_bs_static_crop_stretch_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor(
            [
                0.96573,
                0.95899,
                0.95171,
                0.93903,
                0.93697,
                0.93555,
                0.91717,
                0.89604,
                0.54311,
                0.42117,
            ]
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 4, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1710, 2569, 1889, 2759],
            [1509, 1876, 1720, 2098],
            [1253, 2055, 1424, 2226],
            [1467, 2300, 1628, 2467],
            [1095, 2352, 1260, 2519],
            [1172, 2634, 1375, 2846],
            [1746, 2295, 1913, 2469],
            [932, 1842, 1094, 2000],
            [1302, 509, 2732, 1926],
            [2674, 809, 2720, 963],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_numpy(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_torch(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_batch_numpy(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 0, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_torch(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_batch_torch(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0)
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 0, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_torch_list(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 0, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_nms_fused_numpy(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491, 0.32732123]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
            [1365, 2067, 1422, 2153],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_nms_fused_torch(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491, 0.32732123]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
            [1365, 2067, 1422, 2153],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_nms_fused_batch_numpy(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491, 0.32732123]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491, 0.32732123]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 0, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
            [1365, 2067, 1422, 2153],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_nms_fused_torch(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491, 0.32732123]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
            [1365, 2067, 1422, 2153],
        ],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=2,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_nms_fused_batch_torch(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0)
    )

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491, 0.32732123]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491, 0.32732123]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 0, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
            [1365, 2067, 1422, 2153],
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
@pytest.mark.torch_models
def test_torchscript_package_with_static_batch_size_and_center_crop_nms_fused_torch_list(
    coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package: str,
    coins_counting_image_torch: torch.Tensor,
) -> None:
    # given
    model = YOLOv8ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=coin_counting_yolov8n_toch_script_static_bs_center_crop_fused_nms_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491, 0.32732123]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.6618964, 0.4666715, 0.43694144, 0.3340491, 0.32732123]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([1, 0, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([1, 0, 1, 1, 1], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [
            [1755, 2297, 1831, 2336],
            [1191, 1691, 1771, 2332],
            [1379, 2069, 1422, 2159],
            [1651, 1976, 1696, 2015],
            [1365, 2067, 1422, 2153],
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
