import numpy as np
import pytest
import torch

CONFIDENCE_ATOL = 0.01
XYXY_ATOL = 2


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_static_bs_numpy(
    yolo26n_object_detection_sunflowers_stretch_onnx_static_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(sunflowers_image_numpy)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9143,
                0.9104,
                0.9095,
                0.9090,
                0.9074,
                0.8875,
                0.8867,
                0.8808,
                0.8680,
                0.8654,
                0.8650,
                0.8545,
                0.8507,
                0.8400,
                0.8332,
                0.8285,
                0.8132,
                0.8071,
                0.8066,
                0.7853,
                0.7597,
                0.7349,
                0.6755,
                0.6251,
                0.5847,
                0.5548,
                0.5237,
                0.4825,
                0.4071,
                0.3962,
                0.3794,
                0.3701,
                0.3431,
                0.3362,
            ]
        ),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu().float(),
        torch.tensor(
            [
                [614, 75, 753, 215],
                [1375, 224, 1575, 427],
                [284, 318, 372, 397],
                [8, 381, 92, 465],
                [583, 236, 744, 351],
                [301, 153, 416, 266],
                [1242, 395, 1350, 488],
                [323, 958, 385, 1022],
                [571, 359, 796, 564],
                [100, 335, 192, 431],
                [1298, 260, 1379, 363],
                [918, 282, 1067, 448],
                [1806, 638, 1874, 710],
                [470, 399, 571, 494],
                [68, 1009, 183, 1080],
                [1443, 867, 1502, 941],
                [80, 686, 224, 826],
                [122, 421, 198, 486],
                [1498, 421, 1622, 558],
                [445, 144, 508, 213],
                [1233, 776, 1270, 812],
                [1381, 480, 1437, 543],
                [1478, 808, 1586, 933],
                [1872, 637, 1920, 723],
                [429, 575, 490, 637],
                [1867, 734, 1919, 904],
                [484, 158, 548, 237],
                [1731, 517, 1811, 578],
                [1708, 243, 1838, 375],
                [1612, 370, 1669, 419],
                [194, 339, 278, 413],
                [877, 269, 917, 306],
                [764, 210, 807, 262],
                [1622, 448, 1782, 550],
            ],
            dtype=torch.float32,
        ),
        atol=XYXY_ATOL,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_static_bs_batch_numpy(
    yolo26n_object_detection_sunflowers_stretch_onnx_static_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([sunflowers_image_numpy, sunflowers_image_numpy])

    print(f"predictions[0].confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"predictions[0].class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"predictions[0].xyxy: {predictions[0].xyxy.cpu().tolist()}")
    print(f"predictions[1].confidence: {predictions[1].confidence.cpu().tolist()}")
    print(f"predictions[1].class_id: {predictions[1].class_id.cpu().tolist()}")
    print(f"predictions[1].xyxy: {predictions[1].xyxy.cpu().tolist()}")

    expected_confidence = torch.tensor(
        [
            0.9143,
            0.9104,
            0.9095,
            0.9090,
            0.9074,
            0.8875,
            0.8867,
            0.8808,
            0.8680,
            0.8654,
            0.8650,
            0.8545,
            0.8507,
            0.8400,
            0.8332,
            0.8285,
            0.8132,
            0.8071,
            0.8066,
            0.7853,
            0.7597,
            0.7349,
            0.6755,
            0.6251,
            0.5847,
            0.5548,
            0.5237,
            0.4825,
            0.4071,
            0.3962,
            0.3794,
            0.3701,
            0.3431,
            0.3362,
        ]
    )
    expected_class_id = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=torch.int32,
    )
    expected_xyxy = torch.tensor(
        [
            [614, 75, 753, 215],
            [1375, 224, 1575, 427],
            [284, 318, 372, 397],
            [8, 381, 92, 465],
            [583, 236, 744, 351],
            [301, 153, 416, 266],
            [1242, 395, 1350, 488],
            [323, 958, 385, 1022],
            [571, 359, 796, 564],
            [100, 335, 192, 431],
            [1298, 260, 1379, 363],
            [918, 282, 1067, 448],
            [1806, 638, 1874, 710],
            [470, 399, 571, 494],
            [68, 1009, 183, 1080],
            [1443, 867, 1502, 941],
            [80, 686, 224, 826],
            [122, 421, 198, 486],
            [1498, 421, 1622, 558],
            [445, 144, 508, 213],
            [1233, 776, 1270, 812],
            [1381, 480, 1437, 543],
            [1478, 808, 1586, 933],
            [1872, 637, 1920, 723],
            [429, 575, 490, 637],
            [1867, 734, 1919, 904],
            [484, 158, 548, 237],
            [1731, 517, 1811, 578],
            [1708, 243, 1838, 375],
            [1612, 370, 1669, 419],
            [194, 339, 278, 413],
            [877, 269, 917, 306],
            [764, 210, 807, 262],
            [1622, 448, 1782, 550],
        ],
        dtype=torch.float32,
    )

    assert len(predictions) == 2
    for i in range(2):
        assert torch.allclose(
            predictions[i].confidence.cpu(), expected_confidence, atol=CONFIDENCE_ATOL
        )
        assert torch.allclose(predictions[i].class_id.cpu(), expected_class_id)
        assert torch.allclose(
            predictions[i].xyxy.cpu().float(), expected_xyxy, atol=XYXY_ATOL
        )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_static_bs_torch(
    yolo26n_object_detection_sunflowers_stretch_onnx_static_package: str,
    sunflowers_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(sunflowers_image_torch)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9138,
                0.9129,
                0.9093,
                0.9072,
                0.9066,
                0.8872,
                0.8866,
                0.8802,
                0.8677,
                0.8655,
                0.8649,
                0.8550,
                0.8512,
                0.8389,
                0.8325,
                0.8287,
                0.8076,
                0.8065,
                0.7929,
                0.7866,
                0.7598,
                0.7346,
                0.6740,
                0.6266,
                0.5711,
                0.5586,
                0.5413,
                0.5219,
                0.4788,
                0.3979,
                0.3721,
                0.3587,
                0.3419,
                0.3406,
            ]
        ),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu().float(),
        torch.tensor(
            [
                [614, 75, 753, 215],
                [1375, 224, 1575, 427],
                [284, 318, 372, 397],
                [8, 381, 92, 465],
                [583, 236, 744, 351],
                [301, 153, 416, 266],
                [1242, 395, 1350, 488],
                [323, 958, 385, 1022],
                [570, 359, 796, 565],
                [1298, 260, 1379, 363],
                [100, 335, 192, 431],
                [918, 282, 1067, 448],
                [1806, 638, 1873, 710],
                [470, 399, 571, 494],
                [68, 1009, 183, 1080],
                [1443, 867, 1502, 941],
                [122, 421, 198, 486],
                [1498, 421, 1622, 557],
                [80, 686, 224, 826],
                [445, 144, 508, 213],
                [1233, 776, 1270, 812],
                [1381, 480, 1437, 543],
                [1478, 808, 1586, 933],
                [1872, 637, 1920, 723],
                [429, 575, 490, 637],
                [1867, 733, 1919, 904],
                [1708, 243, 1838, 375],
                [484, 158, 548, 237],
                [1731, 517, 1811, 578],
                [1612, 370, 1669, 419],
                [877, 269, 917, 306],
                [194, 339, 278, 413],
                [1622, 448, 1782, 550],
                [764, 210, 807, 262],
            ],
            dtype=torch.float32,
        ),
        atol=XYXY_ATOL,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_dynamic_bs_numpy(
    yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(sunflowers_image_numpy)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9143,
                0.9104,
                0.9095,
                0.9090,
                0.9074,
                0.8875,
                0.8867,
                0.8808,
                0.8680,
                0.8654,
                0.8650,
                0.8545,
                0.8507,
                0.8400,
                0.8332,
                0.8285,
                0.8132,
                0.8071,
                0.8066,
                0.7853,
                0.7597,
                0.7349,
                0.6755,
                0.6251,
                0.5847,
                0.5548,
                0.5237,
                0.4825,
                0.4071,
                0.3962,
                0.3794,
                0.3701,
                0.3431,
                0.3362,
            ]
        ),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu().float(),
        torch.tensor(
            [
                [614, 75, 753, 215],
                [1375, 224, 1575, 427],
                [284, 318, 372, 397],
                [8, 381, 92, 465],
                [583, 236, 744, 351],
                [301, 153, 416, 266],
                [1242, 395, 1350, 488],
                [323, 958, 385, 1022],
                [571, 359, 796, 564],
                [100, 335, 192, 431],
                [1298, 260, 1379, 363],
                [918, 282, 1067, 448],
                [1806, 638, 1874, 710],
                [470, 399, 571, 494],
                [68, 1009, 183, 1080],
                [1443, 867, 1502, 941],
                [80, 686, 224, 826],
                [122, 421, 198, 486],
                [1498, 421, 1622, 558],
                [445, 144, 508, 213],
                [1233, 776, 1270, 812],
                [1381, 480, 1437, 543],
                [1478, 808, 1586, 933],
                [1872, 637, 1920, 723],
                [429, 575, 490, 637],
                [1867, 734, 1919, 904],
                [484, 158, 548, 237],
                [1731, 517, 1811, 578],
                [1708, 243, 1838, 375],
                [1612, 370, 1669, 419],
                [194, 339, 278, 413],
                [877, 269, 917, 306],
                [764, 210, 807, 262],
                [1622, 448, 1782, 550],
            ],
            dtype=torch.float32,
        ),
        atol=XYXY_ATOL,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_dynamic_bs_batch_numpy(
    yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([sunflowers_image_numpy, sunflowers_image_numpy])

    print(f"predictions[0].confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"predictions[0].class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"predictions[0].xyxy: {predictions[0].xyxy.cpu().tolist()}")
    print(f"predictions[1].confidence: {predictions[1].confidence.cpu().tolist()}")
    print(f"predictions[1].class_id: {predictions[1].class_id.cpu().tolist()}")
    print(f"predictions[1].xyxy: {predictions[1].xyxy.cpu().tolist()}")

    expected_confidence = torch.tensor(
        [
            0.9141,
            0.9103,
            0.9094,
            0.9087,
            0.9075,
            0.8875,
            0.8865,
            0.8807,
            0.8676,
            0.8652,
            0.8648,
            0.8547,
            0.8507,
            0.8397,
            0.8333,
            0.8286,
            0.8174,
            0.8074,
            0.8065,
            0.7853,
            0.7598,
            0.7347,
            0.6756,
            0.6251,
            0.5842,
            0.5553,
            0.5240,
            0.4812,
            0.3960,
            0.3806,
            0.3703,
            0.3662,
            0.3433,
            0.3362,
            0.2541,
        ]
    )
    expected_class_id = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=torch.int32,
    )
    expected_xyxy = torch.tensor(
        [
            [614, 75, 753, 215],
            [1375, 224, 1575, 427],
            [284, 318, 372, 397],
            [8, 381, 92, 465],
            [583, 236, 744, 351],
            [301, 153, 416, 266],
            [1242, 395, 1350, 488],
            [323, 958, 385, 1022],
            [571, 359, 796, 564],
            [100, 335, 192, 431],
            [1298, 260, 1379, 363],
            [918, 282, 1067, 448],
            [1806, 638, 1874, 710],
            [470, 399, 571, 494],
            [68, 1009, 183, 1080],
            [1443, 867, 1502, 941],
            [80, 686, 224, 826],
            [122, 421, 198, 486],
            [1498, 421, 1622, 558],
            [445, 144, 508, 213],
            [1233, 776, 1270, 812],
            [1381, 480, 1437, 543],
            [1478, 808, 1586, 933],
            [1872, 637, 1920, 723],
            [429, 575, 490, 637],
            [1867, 734, 1919, 904],
            [484, 158, 548, 237],
            [1731, 517, 1811, 578],
            [1612, 370, 1669, 419],
            [194, 339, 278, 413],
            [877, 269, 917, 306],
            [1708, 243, 1838, 375],
            [764, 210, 807, 262],
            [1622, 448, 1782, 550],
            [1708, 242, 1838, 375],
        ],
        dtype=torch.float32,
    )

    assert len(predictions) == 2
    for i in range(2):
        assert torch.allclose(
            predictions[i].confidence.cpu(), expected_confidence, atol=CONFIDENCE_ATOL
        )
        assert torch.allclose(predictions[i].class_id.cpu(), expected_class_id)
        assert torch.allclose(
            predictions[i].xyxy.cpu().float(), expected_xyxy, atol=XYXY_ATOL
        )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_dynamic_bs_torch(
    yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package: str,
    sunflowers_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(sunflowers_image_torch)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9138,
                0.9129,
                0.9093,
                0.9072,
                0.9066,
                0.8872,
                0.8866,
                0.8802,
                0.8677,
                0.8655,
                0.8649,
                0.8550,
                0.8512,
                0.8389,
                0.8325,
                0.8287,
                0.8076,
                0.8065,
                0.7929,
                0.7866,
                0.7598,
                0.7346,
                0.6740,
                0.6266,
                0.5711,
                0.5586,
                0.5413,
                0.5219,
                0.4788,
                0.3979,
                0.3721,
                0.3587,
                0.3419,
                0.3406,
            ]
        ),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu().float(),
        torch.tensor(
            [
                [614, 75, 753, 215],
                [1375, 224, 1575, 427],
                [284, 318, 372, 397],
                [8, 381, 92, 465],
                [583, 236, 744, 351],
                [301, 153, 416, 266],
                [1242, 395, 1350, 488],
                [323, 958, 385, 1022],
                [570, 359, 796, 565],
                [1298, 260, 1379, 363],
                [100, 335, 192, 431],
                [918, 282, 1067, 448],
                [1806, 638, 1873, 710],
                [470, 399, 571, 494],
                [68, 1009, 183, 1080],
                [1443, 867, 1502, 941],
                [122, 421, 198, 486],
                [1498, 421, 1622, 557],
                [80, 686, 224, 826],
                [445, 144, 508, 213],
                [1233, 776, 1270, 812],
                [1381, 480, 1437, 543],
                [1478, 808, 1586, 933],
                [1872, 637, 1920, 723],
                [429, 575, 490, 637],
                [1867, 733, 1919, 904],
                [1708, 243, 1838, 375],
                [484, 158, 548, 237],
                [1731, 517, 1811, 578],
                [1612, 370, 1669, 419],
                [877, 269, 917, 306],
                [194, 339, 278, 413],
                [1622, 448, 1782, 550],
                [764, 210, 807, 262],
            ],
            dtype=torch.float32,
        ),
        atol=XYXY_ATOL,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_static_bs_numpy(
    yolo26n_object_detection_sunflowers_letterbox_onnx_static_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(sunflowers_image_numpy)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9024,
                0.8974,
                0.8921,
                0.8903,
                0.8801,
                0.8698,
                0.8588,
                0.8463,
                0.8455,
                0.8399,
                0.8382,
                0.8366,
                0.8351,
                0.8315,
                0.8202,
                0.8043,
                0.8014,
                0.8003,
                0.7885,
                0.7736,
                0.7234,
                0.6309,
                0.5806,
                0.5594,
                0.4611,
                0.4557,
                0.4505,
                0.4410,
                0.4089,
                0.4004,
                0.3922,
                0.3667,
                0.3571,
                0.3170,
                0.2513,
            ]
        ),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu().float(),
        torch.tensor(
            [
                [1374, 223, 1579, 422],
                [1805, 633, 1873, 707],
                [80, 691, 220, 824],
                [299, 146, 419, 267],
                [1706, 236, 1837, 371],
                [327, 954, 380, 1017],
                [282, 315, 375, 394],
                [611, 73, 751, 214],
                [99, 330, 194, 425],
                [1500, 418, 1627, 553],
                [918, 285, 1066, 446],
                [557, 352, 800, 571],
                [1297, 254, 1379, 358],
                [573, 233, 751, 350],
                [1241, 395, 1345, 484],
                [10, 378, 95, 459],
                [443, 142, 509, 212],
                [465, 393, 576, 489],
                [1437, 862, 1505, 940],
                [1232, 771, 1272, 810],
                [120, 418, 199, 479],
                [1868, 734, 1920, 888],
                [483, 160, 547, 234],
                [64, 1003, 191, 1075],
                [1474, 805, 1581, 926],
                [428, 566, 490, 635],
                [1732, 511, 1809, 574],
                [1873, 636, 1920, 726],
                [1610, 368, 1668, 409],
                [877, 266, 915, 304],
                [1622, 437, 1766, 542],
                [761, 277, 834, 333],
                [770, 211, 811, 256],
                [200, 338, 280, 408],
                [766, 210, 807, 254],
            ],
            dtype=torch.float32,
        ),
        atol=XYXY_ATOL,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_static_bs_batch_numpy(
    yolo26n_object_detection_sunflowers_letterbox_onnx_static_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([sunflowers_image_numpy, sunflowers_image_numpy])

    print(f"predictions[0].confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"predictions[0].class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"predictions[0].xyxy: {predictions[0].xyxy.cpu().tolist()}")
    print(f"predictions[1].confidence: {predictions[1].confidence.cpu().tolist()}")
    print(f"predictions[1].class_id: {predictions[1].class_id.cpu().tolist()}")
    print(f"predictions[1].xyxy: {predictions[1].xyxy.cpu().tolist()}")

    expected_confidence = torch.tensor(
        [
            0.9024,
            0.8974,
            0.8921,
            0.8903,
            0.8801,
            0.8698,
            0.8588,
            0.8463,
            0.8455,
            0.8399,
            0.8382,
            0.8366,
            0.8351,
            0.8315,
            0.8202,
            0.8043,
            0.8014,
            0.8003,
            0.7885,
            0.7736,
            0.7234,
            0.6309,
            0.5806,
            0.5594,
            0.4611,
            0.4557,
            0.4505,
            0.4410,
            0.4089,
            0.4004,
            0.3922,
            0.3667,
            0.3571,
            0.3170,
            0.2513,
        ]
    )
    expected_class_id = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=torch.int32,
    )
    expected_xyxy = torch.tensor(
        [
            [1374, 223, 1579, 422],
            [1805, 633, 1873, 707],
            [80, 691, 220, 824],
            [299, 146, 419, 267],
            [1706, 236, 1837, 371],
            [327, 954, 380, 1017],
            [282, 315, 375, 394],
            [611, 73, 751, 214],
            [99, 330, 194, 425],
            [1500, 418, 1627, 553],
            [918, 285, 1066, 446],
            [557, 352, 800, 571],
            [1297, 254, 1379, 358],
            [573, 233, 751, 350],
            [1241, 395, 1345, 484],
            [10, 378, 95, 459],
            [443, 142, 509, 212],
            [465, 393, 576, 489],
            [1437, 862, 1505, 940],
            [1232, 771, 1272, 810],
            [120, 418, 199, 479],
            [1868, 734, 1920, 888],
            [483, 160, 547, 234],
            [64, 1003, 191, 1075],
            [1474, 805, 1581, 926],
            [428, 566, 490, 635],
            [1732, 511, 1809, 574],
            [1873, 636, 1920, 726],
            [1610, 368, 1668, 409],
            [877, 266, 915, 304],
            [1622, 437, 1766, 542],
            [761, 277, 834, 333],
            [770, 211, 811, 256],
            [200, 338, 280, 408],
            [766, 210, 807, 254],
        ],
        dtype=torch.float32,
    )

    assert len(predictions) == 2
    for i in range(2):
        assert torch.allclose(
            predictions[i].confidence.cpu(), expected_confidence, atol=CONFIDENCE_ATOL
        )
        assert torch.allclose(predictions[i].class_id.cpu(), expected_class_id)
        assert torch.allclose(
            predictions[i].xyxy.cpu().float(), expected_xyxy, atol=XYXY_ATOL
        )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_static_bs_torch(
    yolo26n_object_detection_sunflowers_letterbox_onnx_static_package: str,
    sunflowers_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(sunflowers_image_torch)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9027,
                0.8974,
                0.8922,
                0.8907,
                0.8800,
                0.8695,
                0.8588,
                0.8464,
                0.8457,
                0.8399,
                0.8379,
                0.8374,
                0.8352,
                0.8341,
                0.8200,
                0.8040,
                0.8006,
                0.8003,
                0.7891,
                0.7740,
                0.7239,
                0.6127,
                0.5803,
                0.5597,
                0.4576,
                0.4522,
                0.4485,
                0.4405,
                0.4094,
                0.3995,
                0.3903,
                0.3628,
                0.3516,
                0.3212,
                0.2515,
            ]
        ),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu().float(),
        torch.tensor(
            [
                [1374, 223, 1579, 422],
                [1805, 633, 1873, 707],
                [80, 691, 220, 824],
                [299, 146, 419, 266],
                [1706, 236, 1837, 371],
                [327, 954, 380, 1017],
                [282, 315, 375, 394],
                [611, 73, 751, 214],
                [99, 330, 194, 425],
                [1500, 418, 1627, 553],
                [918, 285, 1066, 446],
                [557, 353, 800, 571],
                [1297, 254, 1379, 358],
                [573, 233, 751, 350],
                [1241, 395, 1345, 484],
                [10, 378, 95, 459],
                [465, 393, 576, 489],
                [443, 142, 509, 212],
                [1437, 861, 1505, 940],
                [1232, 771, 1272, 810],
                [120, 418, 199, 479],
                [1868, 734, 1920, 888],
                [483, 160, 547, 234],
                [64, 1003, 191, 1075],
                [428, 566, 490, 635],
                [1733, 511, 1809, 574],
                [1474, 805, 1581, 926],
                [1873, 636, 1920, 726],
                [1610, 368, 1668, 409],
                [877, 266, 915, 304],
                [1622, 437, 1766, 542],
                [761, 277, 834, 333],
                [770, 211, 811, 256],
                [200, 338, 280, 408],
                [766, 210, 807, 254],
            ],
            dtype=torch.float32,
        ),
        atol=XYXY_ATOL,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_dynamic_bs_numpy(
    yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(sunflowers_image_numpy)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9024,
                0.8974,
                0.8921,
                0.8903,
                0.8801,
                0.8698,
                0.8588,
                0.8463,
                0.8455,
                0.8399,
                0.8382,
                0.8366,
                0.8351,
                0.8315,
                0.8202,
                0.8043,
                0.8014,
                0.8003,
                0.7885,
                0.7736,
                0.7234,
                0.6309,
                0.5806,
                0.5594,
                0.4611,
                0.4557,
                0.4505,
                0.4410,
                0.4089,
                0.4004,
                0.3922,
                0.3667,
                0.3571,
                0.3170,
                0.2513,
            ]
        ),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu().float(),
        torch.tensor(
            [
                [1374, 223, 1579, 422],
                [1805, 633, 1873, 707],
                [80, 691, 220, 824],
                [299, 146, 419, 267],
                [1706, 236, 1837, 371],
                [327, 954, 380, 1017],
                [282, 315, 375, 394],
                [611, 73, 751, 214],
                [99, 330, 194, 425],
                [1500, 418, 1627, 553],
                [918, 285, 1066, 446],
                [557, 352, 800, 571],
                [1297, 254, 1379, 358],
                [573, 233, 751, 350],
                [1241, 395, 1345, 484],
                [10, 378, 95, 459],
                [443, 142, 509, 212],
                [465, 393, 576, 489],
                [1437, 862, 1505, 940],
                [1232, 771, 1272, 810],
                [120, 418, 199, 479],
                [1868, 734, 1920, 888],
                [483, 160, 547, 234],
                [64, 1003, 191, 1075],
                [1474, 805, 1581, 926],
                [428, 566, 490, 635],
                [1732, 511, 1809, 574],
                [1873, 636, 1920, 726],
                [1610, 368, 1668, 409],
                [877, 266, 915, 304],
                [1622, 437, 1766, 542],
                [761, 277, 834, 333],
                [770, 211, 811, 256],
                [200, 338, 280, 408],
                [766, 210, 807, 254],
            ],
            dtype=torch.float32,
        ),
        atol=XYXY_ATOL,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_dynamic_bs_batch_numpy(
    yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([sunflowers_image_numpy, sunflowers_image_numpy])

    print(f"predictions[0].confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"predictions[0].class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"predictions[0].xyxy: {predictions[0].xyxy.cpu().tolist()}")
    print(f"predictions[1].confidence: {predictions[1].confidence.cpu().tolist()}")
    print(f"predictions[1].class_id: {predictions[1].class_id.cpu().tolist()}")
    print(f"predictions[1].xyxy: {predictions[1].xyxy.cpu().tolist()}")

    expected_confidence = torch.tensor(
        [
            0.9026,
            0.8976,
            0.8922,
            0.8905,
            0.8802,
            0.8696,
            0.8589,
            0.8465,
            0.8456,
            0.8400,
            0.8384,
            0.8371,
            0.8350,
            0.8325,
            0.8201,
            0.8044,
            0.8015,
            0.8003,
            0.7888,
            0.7737,
            0.7234,
            0.6387,
            0.5806,
            0.5593,
            0.4595,
            0.4559,
            0.4507,
            0.4409,
            0.4088,
            0.4003,
            0.3924,
            0.3661,
            0.3570,
            0.3180,
            0.2505,
        ]
    )
    expected_class_id = torch.tensor(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=torch.int32,
    )
    expected_xyxy = torch.tensor(
        [
            [1374, 223, 1579, 422],
            [1805, 633, 1873, 707],
            [80, 691, 220, 824],
            [299, 147, 419, 267],
            [1706, 236, 1837, 371],
            [327, 954, 380, 1017],
            [282, 315, 375, 394],
            [611, 73, 751, 214],
            [99, 330, 194, 425],
            [1500, 418, 1627, 553],
            [918, 285, 1066, 446],
            [557, 352, 800, 571],
            [1297, 254, 1379, 358],
            [573, 233, 751, 350],
            [1241, 395, 1345, 484],
            [10, 378, 95, 459],
            [443, 142, 509, 212],
            [465, 393, 576, 489],
            [1437, 862, 1505, 940],
            [1232, 771, 1272, 810],
            [120, 418, 199, 479],
            [1868, 734, 1920, 888],
            [483, 160, 547, 234],
            [64, 1003, 191, 1075],
            [1474, 805, 1581, 926],
            [428, 566, 490, 635],
            [1732, 511, 1809, 574],
            [1873, 636, 1920, 726],
            [1610, 368, 1668, 409],
            [877, 266, 915, 304],
            [1622, 437, 1766, 542],
            [761, 277, 834, 333],
            [770, 211, 811, 256],
            [200, 338, 280, 408],
            [766, 210, 807, 254],
        ],
        dtype=torch.float32,
    )

    assert len(predictions) == 2
    for i in range(2):
        assert torch.allclose(
            predictions[i].confidence.cpu(), expected_confidence, atol=CONFIDENCE_ATOL
        )
        assert torch.allclose(predictions[i].class_id.cpu(), expected_class_id)
        assert torch.allclose(
            predictions[i].xyxy.cpu().float(), expected_xyxy, atol=XYXY_ATOL
        )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_dynamic_bs_torch(
    yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package: str,
    sunflowers_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(sunflowers_image_torch)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9027,
                0.8974,
                0.8922,
                0.8907,
                0.8800,
                0.8695,
                0.8588,
                0.8464,
                0.8457,
                0.8399,
                0.8379,
                0.8374,
                0.8352,
                0.8341,
                0.8200,
                0.8040,
                0.8006,
                0.8003,
                0.7891,
                0.7740,
                0.7239,
                0.6127,
                0.5803,
                0.5597,
                0.4576,
                0.4522,
                0.4485,
                0.4405,
                0.4094,
                0.3995,
                0.3903,
                0.3628,
                0.3516,
                0.3212,
                0.2515,
            ]
        ),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu().float(),
        torch.tensor(
            [
                [1374, 223, 1579, 422],
                [1805, 633, 1873, 707],
                [80, 691, 220, 824],
                [299, 146, 419, 266],
                [1706, 236, 1837, 371],
                [327, 954, 380, 1017],
                [282, 315, 375, 394],
                [611, 73, 751, 214],
                [99, 330, 194, 425],
                [1500, 418, 1627, 553],
                [918, 285, 1066, 446],
                [557, 353, 800, 571],
                [1297, 254, 1379, 358],
                [573, 233, 751, 350],
                [1241, 395, 1345, 484],
                [10, 378, 95, 459],
                [465, 393, 576, 489],
                [443, 142, 509, 212],
                [1437, 861, 1505, 940],
                [1232, 771, 1272, 810],
                [120, 418, 199, 479],
                [1868, 734, 1920, 888],
                [483, 160, 547, 234],
                [64, 1003, 191, 1075],
                [428, 566, 490, 635],
                [1733, 511, 1809, 574],
                [1474, 805, 1581, 926],
                [1873, 636, 1920, 726],
                [1610, 368, 1668, 409],
                [877, 266, 915, 304],
                [1622, 437, 1766, 542],
                [761, 277, 834, 333],
                [770, 211, 811, 256],
                [200, 338, 280, 408],
                [766, 210, 807, 254],
            ],
            dtype=torch.float32,
        ),
        atol=XYXY_ATOL,
    )
