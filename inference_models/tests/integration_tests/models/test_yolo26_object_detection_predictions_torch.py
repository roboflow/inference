import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.yolo26.yolo26_object_detection_torch_script import (
    YOLO26ForObjectDetectionTorchScript,
)

CONFIDENCE_ATOL = 0.01
XYXY_ATOL = 2


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
    numpy_predictions = model(
        sunflowers_image_torch.permute(1, 2, 0).numpy(),
        input_color_format="rgb",
        confidence=0.25,
        key_points_threshold=0.3,
    )

    assert len(predictions) == 1

    expected_confidence = numpy_predictions[0].confidence.cpu()
    expected_class_id = numpy_predictions[0].class_id.cpu()
    expected_xyxy = numpy_predictions[0].xyxy.cpu().float()

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
    numpy_predictions = model(
        sunflowers_image_torch.permute(1, 2, 0).numpy(),
        input_color_format="rgb",
        confidence=0.25,
        key_points_threshold=0.3,
    )

    assert len(predictions) == 1

    expected_confidence = numpy_predictions[0].confidence.cpu()
    expected_xyxy = numpy_predictions[0].xyxy.cpu().float()

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


@pytest.mark.slow
@pytest.mark.torch_models
def test_torchscript_per_class_confidence_filters_detections(
    yolo26n_object_detection_sunflowers_stretch_torch_script_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.weights_providers.entities import RecommendedParameters

    model = YOLO26ForObjectDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )
    class_names = list(model.class_names)
    model.recommended_parameters = RecommendedParameters(
        confidence=0.25,
        per_class_confidence={class_names[1]: 1.01},
    )
    predictions = model(sunflowers_image_numpy, confidence="best")
    assert predictions[0].class_id.numel() == 0
