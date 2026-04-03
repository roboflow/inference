import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.yolo26.yolo26_object_detection_torch_script import (
    YOLO26ForObjectDetectionTorchScript,
)

CONFIDENCE_ATOL = 0.01
XYXY_ATOL = 2

IS_GPU = str(DEFAULT_DEVICE).startswith("cuda")

GPU_STRETCH_TORCH_CONFIDENCE = [
    0.9142,
    0.9139,
    0.9094,
    0.9070,
    0.9065,
    0.8872,
    0.8866,
    0.8802,
    0.8677,
    0.8652,
    0.8652,
    0.8551,
    0.8511,
    0.8387,
    0.8322,
    0.8286,
    0.8074,
    0.8060,
    0.7985,
    0.7867,
    0.7590,
    0.7341,
    0.6732,
    0.6264,
    0.5695,
    0.5569,
    0.5237,
    0.5214,
    0.4762,
    0.3973,
    0.3710,
    0.3622,
    0.3415,
    0.3410,
]
GPU_STRETCH_TORCH_XYXY = [
    [1375, 224, 1575, 427],
    [614, 75, 753, 215],
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
]

GPU_LETTERBOX_TORCH_CONFIDENCE = [
    0.9027,
    0.8973,
    0.8923,
    0.8909,
    0.8802,
    0.8695,
    0.8588,
    0.8464,
    0.8458,
    0.8400,
    0.8380,
    0.8372,
    0.8352,
    0.8343,
    0.8204,
    0.8042,
    0.8010,
    0.8003,
    0.7896,
    0.7742,
    0.7239,
    0.6118,
    0.5799,
    0.5596,
    0.4578,
    0.4530,
    0.4482,
    0.4400,
    0.4087,
    0.3990,
    0.3905,
    0.3641,
    0.3518,
    0.3196,
    0.2517,
]
GPU_LETTERBOX_TORCH_XYXY = [
    [1374, 223, 1579, 422],
    [1805, 633, 1873, 707],
    [80, 691, 220, 824],
    [299, 147, 419, 266],
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
    [1868, 734, 1920, 887],
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
]

CPU_STRETCH_NUMPY_CONFIDENCE = [
    0.9144,
    0.9107,
    0.9096,
    0.9088,
    0.9072,
    0.8875,
    0.8867,
    0.8807,
    0.8677,
    0.8655,
    0.8649,
    0.8546,
    0.8505,
    0.8397,
    0.8330,
    0.8284,
    0.8122,
    0.8074,
    0.8059,
    0.7853,
    0.7591,
    0.7345,
    0.6749,
    0.6253,
    0.5835,
    0.5537,
    0.5239,
    0.4796,
    0.3954,
    0.3804,
    0.3697,
    0.3693,
    0.3433,
    0.3367,
    0.2512,
]
CPU_STRETCH_NUMPY_XYXY = [
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
    [1806, 638, 1873, 710],
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
    [764, 211, 807, 262],
    [1622, 448, 1782, 550],
    [1708, 242, 1838, 375],
]

CPU_STRETCH_TORCH_CONFIDENCE = [
    0.9165,
    0.9137,
    0.9097,
    0.9079,
    0.9071,
    0.8877,
    0.8865,
    0.8804,
    0.8675,
    0.8659,
    0.8655,
    0.8546,
    0.8509,
    0.8395,
    0.8345,
    0.8288,
    0.8090,
    0.8062,
    0.7857,
    0.7599,
    0.7572,
    0.7323,
    0.6741,
    0.6299,
    0.5878,
    0.5560,
    0.5242,
    0.4819,
    0.4464,
    0.3967,
    0.3718,
    0.3644,
    0.3409,
    0.3387,
]
CPU_STRETCH_TORCH_XYXY = [
    [1375, 224, 1575, 427],
    [614, 74, 753, 215],
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
    [1806, 638, 1873, 710],
    [470, 399, 571, 494],
    [68, 1009, 183, 1080],
    [1443, 867, 1502, 941],
    [122, 421, 198, 486],
    [1498, 421, 1622, 558],
    [445, 144, 508, 213],
    [1233, 776, 1270, 812],
    [80, 686, 224, 826],
    [1381, 480, 1437, 543],
    [1478, 808, 1586, 933],
    [1872, 637, 1920, 723],
    [429, 575, 490, 637],
    [1867, 734, 1919, 904],
    [484, 158, 548, 237],
    [1732, 517, 1811, 578],
    [1708, 243, 1838, 375],
    [1612, 370, 1669, 419],
    [877, 269, 917, 306],
    [194, 339, 278, 413],
    [764, 211, 807, 262],
    [1622, 448, 1782, 550],
]

CPU_LETTERBOX_TORCH_CONFIDENCE = [
    0.9027,
    0.8973,
    0.8921,
    0.8907,
    0.8802,
    0.8697,
    0.8590,
    0.8464,
    0.8455,
    0.8402,
    0.8377,
    0.8372,
    0.8369,
    0.8353,
    0.8203,
    0.8038,
    0.8000,
    0.7999,
    0.7910,
    0.7745,
    0.7243,
    0.6312,
    0.5796,
    0.5601,
    0.4570,
    0.4486,
    0.4433,
    0.4404,
    0.4076,
    0.3988,
    0.3893,
    0.3643,
    0.3583,
    0.3188,
    0.2513,
]
CPU_LETTERBOX_TORCH_XYXY = [
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
    [557, 353, 800, 571],
    [573, 233, 751, 350],
    [1297, 254, 1379, 358],
    [1241, 395, 1345, 484],
    [10, 378, 95, 459],
    [465, 393, 576, 489],
    [443, 142, 509, 212],
    [1437, 861, 1505, 940],
    [1232, 771, 1272, 810],
    [120, 418, 199, 479],
    [1868, 734, 1920, 887],
    [483, 160, 547, 234],
    [64, 1003, 191, 1075],
    [428, 566, 490, 635],
    [1733, 511, 1809, 574],
    [1474, 805, 1581, 925],
    [1873, 636, 1920, 726],
    [1610, 368, 1668, 409],
    [877, 266, 915, 304],
    [1622, 437, 1765, 542],
    [770, 211, 811, 256],
    [761, 277, 834, 333],
    [200, 338, 280, 408],
    [766, 210, 807, 254],
]


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_stretch_numpy(
    yolo26n_object_detection_sunflowers_stretch_torch_script_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        sunflowers_image_numpy, confidence=0.25, key_points_threshold=0.3
    )

    assert len(predictions) == 1

    expected_confidence = torch.tensor(CPU_STRETCH_NUMPY_CONFIDENCE)
    expected_class_id = torch.ones(35, dtype=torch.int32)
    expected_xyxy = torch.tensor(CPU_STRETCH_NUMPY_XYXY, dtype=torch.float32)

    assert torch.allclose(
        predictions[0].confidence.cpu(), expected_confidence, atol=CONFIDENCE_ATOL
    )
    assert torch.allclose(predictions[0].class_id.cpu(), expected_class_id)
    assert torch.allclose(
        predictions[0].xyxy.cpu().float(), expected_xyxy, atol=XYXY_ATOL
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_stretch_batch_numpy(
    yolo26n_object_detection_sunflowers_stretch_torch_script_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        [sunflowers_image_numpy, sunflowers_image_numpy],
        confidence=0.25,
        key_points_threshold=0.3,
    )

    expected_confidence = torch.tensor(CPU_STRETCH_NUMPY_CONFIDENCE)
    expected_class_id = torch.ones(35, dtype=torch.int32)
    expected_xyxy = torch.tensor(CPU_STRETCH_NUMPY_XYXY, dtype=torch.float32)

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
@pytest.mark.torch_models
def test_torchscript_package_stretch_torch(
    yolo26n_object_detection_sunflowers_stretch_torch_script_package: str,
    sunflowers_image_torch: torch.Tensor,
) -> None:
    model = YOLO26ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        sunflowers_image_torch, confidence=0.25, key_points_threshold=0.3
    )

    assert len(predictions) == 1

    if IS_GPU:
        expected_confidence = torch.tensor(GPU_STRETCH_TORCH_CONFIDENCE)
        expected_class_id = torch.ones(34, dtype=torch.int32)
        expected_xyxy = torch.tensor(GPU_STRETCH_TORCH_XYXY, dtype=torch.float32)
    else:
        expected_confidence = torch.tensor(CPU_STRETCH_TORCH_CONFIDENCE)
        expected_class_id = torch.ones(34, dtype=torch.int32)
        expected_xyxy = torch.tensor(CPU_STRETCH_TORCH_XYXY, dtype=torch.float32)

    assert torch.allclose(
        predictions[0].confidence.cpu(), expected_confidence, atol=CONFIDENCE_ATOL
    )
    assert torch.allclose(predictions[0].class_id.cpu(), expected_class_id)
    assert torch.allclose(
        predictions[0].xyxy.cpu().float(), expected_xyxy, atol=XYXY_ATOL
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_package_letterbox_numpy(
    yolo26n_object_detection_sunflowers_letterbox_torch_script_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        sunflowers_image_numpy, confidence=0.25, key_points_threshold=0.3
    )

    assert len(predictions) == 1
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor(
            [
                0.9024,
                0.8974,
                0.8921,
                0.8904,
                0.8801,
                0.8697,
                0.8587,
                0.8462,
                0.8455,
                0.8397,
                0.8380,
                0.8369,
                0.8352,
                0.8330,
                0.8200,
                0.8040,
                0.8015,
                0.8003,
                0.7890,
                0.7740,
                0.7233,
                0.6346,
                0.5803,
                0.5597,
                0.4606,
                0.4560,
                0.4502,
                0.4415,
                0.4093,
                0.3999,
                0.3924,
                0.3667,
                0.3576,
                0.3177,
                0.2502,
            ]
        ),
        atol=CONFIDENCE_ATOL,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.ones(35, dtype=torch.int32),
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
@pytest.mark.torch_models
def test_torchscript_package_letterbox_batch_numpy(
    yolo26n_object_detection_sunflowers_letterbox_torch_script_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        [sunflowers_image_numpy, sunflowers_image_numpy],
        confidence=0.25,
        key_points_threshold=0.3,
    )

    expected_confidence = torch.tensor(
        [
            0.9024,
            0.8974,
            0.8921,
            0.8904,
            0.8801,
            0.8697,
            0.8587,
            0.8462,
            0.8455,
            0.8397,
            0.8380,
            0.8369,
            0.8352,
            0.8330,
            0.8200,
            0.8040,
            0.8015,
            0.8003,
            0.7890,
            0.7740,
            0.7233,
            0.6346,
            0.5803,
            0.5597,
            0.4606,
            0.4560,
            0.4502,
            0.4415,
            0.4093,
            0.3999,
            0.3924,
            0.3667,
            0.3576,
            0.3177,
            0.2502,
        ]
    )
    expected_class_id = torch.ones(35, dtype=torch.int32)
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
@pytest.mark.torch_models
def test_torchscript_package_letterbox_torch(
    yolo26n_object_detection_sunflowers_letterbox_torch_script_package: str,
    sunflowers_image_torch: torch.Tensor,
) -> None:
    model = YOLO26ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        sunflowers_image_torch, confidence=0.25, key_points_threshold=0.3
    )

    assert len(predictions) == 1

    if IS_GPU:
        expected_confidence = torch.tensor(GPU_LETTERBOX_TORCH_CONFIDENCE)
        expected_xyxy = torch.tensor(GPU_LETTERBOX_TORCH_XYXY, dtype=torch.float32)
    else:
        expected_confidence = torch.tensor(CPU_LETTERBOX_TORCH_CONFIDENCE)
        expected_xyxy = torch.tensor(CPU_LETTERBOX_TORCH_XYXY, dtype=torch.float32)

    assert torch.allclose(
        predictions[0].confidence.cpu(), expected_confidence, atol=CONFIDENCE_ATOL
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.ones(35, dtype=torch.int32),
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu().float(), expected_xyxy, atol=XYXY_ATOL
    )
