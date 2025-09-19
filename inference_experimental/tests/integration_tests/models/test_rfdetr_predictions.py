import numpy as np
import torch
from inference_exp.models.rfdetr.rfdetr_object_detection_onnx import (
    RFDetrForObjectDetectionONNX,
)
from inference_exp.models.rfdetr.rfdetr_object_detection_pytorch import (
    RFDetrForObjectDetectionTorch,
)


def test_torch_package_with_stretch_resize_and_contrast_stretching_numpy(
    coin_counting_rfdetr_nano_torch_cs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = RFDetrForObjectDetectionTorch.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_torch_cs_stretch_package,
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425, 0.5340, 0.5133]
        ),
        atol=0.01
    )
    expected_xyxy = torch.tensor([
        [1704, 2567, 1894, 2756],
        [1741, 2292, 1921, 2468],
        [1460, 2297, 1630, 2468],
        [1500, 1875, 1730, 2096],
        [2673,  792, 2876,  978],
        [1158, 2620, 1383, 2849],
        [1247, 2055, 1430, 2227],
        [1087, 2342, 1264, 2524],
        [919, 1835, 1102, 2009],
        [919, 1835, 1102, 2009],
        [1247, 2055, 1430, 2227]
    ], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy,
        expected_xyxy,
        atol=2,
    )


def test_torch_package_with_stretch_resize_and_contrast_stretching_numpy_batch(
    coin_counting_rfdetr_nano_torch_cs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = RFDetrForObjectDetectionTorch.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_torch_cs_stretch_package,
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425, 0.5340, 0.5133]
        ),
        atol=0.01
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425, 0.5340, 0.5133]
        ),
        atol=0.01
    )
    expected_xyxy = torch.tensor([
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
        [1247, 2055, 1430, 2227]
    ], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy,
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy,
        expected_xyxy,
        atol=2,
    )


def test_torch_package_with_stretch_resize_and_contrast_stretching_torch(
    coin_counting_rfdetr_nano_torch_cs_stretch_package: str,
    coins_counting_image_torch: np.ndarray,
) -> None:
    # given
    model = RFDetrForObjectDetectionTorch.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_torch_cs_stretch_package,
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425, 0.5340, 0.5133]
        ),
        atol=0.01
    )
    expected_xyxy = torch.tensor([
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
        [1247, 2055, 1430, 2227]
    ], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy,
        expected_xyxy,
        atol=2,
    )


def test_torch_package_with_stretch_resize_and_contrast_stretching_torch_batch(
    coin_counting_rfdetr_nano_torch_cs_stretch_package: str,
    coins_counting_image_torch: np.ndarray,
) -> None:
    # given
    model = RFDetrForObjectDetectionTorch.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_torch_cs_stretch_package,
    )

    # when
    predictions = model(torch.stack([coins_counting_image_torch, coins_counting_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425, 0.5340, 0.5133]
        ),
        atol=0.01
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425, 0.5340, 0.5133]
        ),
        atol=0.01
    )
    expected_xyxy = torch.tensor([
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
        [1247, 2055, 1430, 2227]
    ], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy,
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy,
        expected_xyxy,
        atol=2,
    )


def test_torch_package_with_stretch_resize_and_contrast_stretching_torch_list(
    coin_counting_rfdetr_nano_torch_cs_stretch_package: str,
    coins_counting_image_torch: np.ndarray,
) -> None:
    # given
    model = RFDetrForObjectDetectionTorch.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_torch_cs_stretch_package,
    )

    # when
    predictions = model([coins_counting_image_torch, coins_counting_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425, 0.5340, 0.5133]
        ),
        atol=0.01
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425, 0.5340, 0.5133]
        ),
        atol=0.01
    )
    expected_xyxy = torch.tensor([
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
        [1247, 2055, 1430, 2227]
    ], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy,
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy,
        expected_xyxy,
        atol=2,
    )


def test_onnx_package_with_stretch_resize_and_contrast_stretching_numpy(
    coin_counting_rfdetr_nano_onnx_cs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_cs_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"]
    )

    # when
    predictions = model(coins_counting_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425]
        ),
        atol=0.01
    )
    expected_xyxy = torch.tensor([
        [1704, 2567, 1894, 2756],
        [1741, 2292, 1921, 2468],
        [1460, 2297, 1630, 2468],
        [1500, 1875, 1730, 2096],
        [2673, 792, 2876, 978],
        [1158, 2620, 1383, 2849],
        [1247, 2055, 1430, 2227],
        [1087, 2342, 1264, 2524],
        [919, 1835, 1102, 2009],

    ], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy,
        expected_xyxy,
        atol=2,
    )


def test_onnx_package_with_stretch_resize_and_contrast_stretching_batch_numpy(
    coin_counting_rfdetr_nano_onnx_cs_stretch_package: str,
    coins_counting_image_numpy: np.ndarray,
) -> None:
    # given
    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_cs_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"]
    )

    # when
    predictions = model([coins_counting_image_numpy, coins_counting_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425]
        ),
        atol=0.01
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425]
        ),
        atol=0.01
    )
    expected_xyxy = torch.tensor([
        [1704, 2567, 1894, 2756],
        [1741, 2292, 1921, 2468],
        [1460, 2297, 1630, 2468],
        [1500, 1875, 1730, 2096],
        [2673, 792, 2876, 978],
        [1158, 2620, 1383, 2849],
        [1247, 2055, 1430, 2227],
        [1087, 2342, 1264, 2524],
        [919, 1835, 1102, 2009],

    ], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy,
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy,
        expected_xyxy,
        atol=2,
    )


def test_onnx_package_with_stretch_resize_and_contrast_stretching_torch(
    coin_counting_rfdetr_nano_onnx_cs_stretch_package: str,
    coins_counting_image_torch: np.ndarray,
) -> None:
    # given
    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_cs_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"]
    )

    # when
    predictions = model(coins_counting_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425]
        ),
        atol=0.01
    )
    expected_xyxy = torch.tensor([
        [1704, 2567, 1894, 2756],
        [1741, 2292, 1921, 2468],
        [1460, 2297, 1630, 2468],
        [1500, 1875, 1730, 2096],
        [2673, 792, 2876, 978],
        [1158, 2620, 1383, 2849],
        [1247, 2055, 1430, 2227],
        [1087, 2342, 1264, 2524],
        [919, 1835, 1102, 2009],

    ], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy,
        expected_xyxy,
        atol=2,
    )


def test_onnx_package_with_stretch_resize_and_contrast_stretching_torch_batch(
    coin_counting_rfdetr_nano_onnx_cs_stretch_package: str,
    coins_counting_image_torch: np.ndarray,
) -> None:
    # given
    model = RFDetrForObjectDetectionONNX.from_pretrained(
        model_name_or_path=coin_counting_rfdetr_nano_onnx_cs_stretch_package,
        onnx_execution_providers=["CPUExecutionProvider"]
    )

    # when
    predictions = model(torch.stack([coins_counting_image_torch]*2, dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425]
        ),
        atol=0.01
    )
    assert torch.allclose(
        predictions[1].confidence,
        torch.tensor(
            [0.8575, 0.8568, 0.8105, 0.7940, 0.7364, 0.6872, 0.6419, 0.5810, 0.5425]
        ),
        atol=0.01
    )
    expected_xyxy = torch.tensor([
        [1704, 2567, 1894, 2756],
        [1741, 2292, 1921, 2468],
        [1460, 2297, 1630, 2468],
        [1500, 1875, 1730, 2096],
        [2673, 792, 2876, 978],
        [1158, 2620, 1383, 2849],
        [1247, 2055, 1430, 2227],
        [1087, 2342, 1264, 2524],
        [919, 1835, 1102, 2009],

    ], dtype=torch.int32)
    assert torch.allclose(
        predictions[0].xyxy,
        expected_xyxy,
        atol=2,
    )
    assert torch.allclose(
        predictions[1].xyxy,
        expected_xyxy,
        atol=2,
    )