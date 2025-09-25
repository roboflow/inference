import numpy as np
import pytest
import torch
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.models.yolov8.yolov8_key_points_detection_torch_script import (
    YOLOv8ForKeyPointsDetectionTorchScript,
)


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_center_crop_package_numpy(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(people_walking_image_numpy)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [741, 192],
                    [0, 0],
                    [0, 0],
                    [698, 178],
                    [749, 180],
                    [661, 251],
                    [770, 255],
                    [633, 372],
                    [801, 365],
                    [636, 467],
                    [809, 448],
                    [680, 436],
                    [748, 438],
                    [692, 578],
                    [735, 585],
                    [715, 716],
                    [714, 726],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [433, 192],
                    [483, 192],
                    [398, 257],
                    [491, 258],
                    [368, 365],
                    [511, 360],
                    [368, 448],
                    [522, 446],
                    [415, 429],
                    [475, 430],
                    [417, 584],
                    [473, 578],
                    [436, 712],
                    [466, 700],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.3142,
                    0.0000,
                    0.0000,
                    0.8722,
                    0.9678,
                    0.9978,
                    0.9983,
                    0.9788,
                    0.9849,
                    0.9391,
                    0.9490,
                    0.9994,
                    0.9995,
                    0.9971,
                    0.9974,
                    0.9213,
                    0.9227,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.5982,
                    0.9159,
                    0.9904,
                    0.9942,
                    0.9521,
                    0.9705,
                    0.8870,
                    0.9113,
                    0.9973,
                    0.9977,
                    0.9870,
                    0.9877,
                    0.7204,
                    0.7233,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_center_crop_package_batch_numpy(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([people_walking_image_numpy, people_walking_image_numpy])

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [741, 192],
                [0, 0],
                [0, 0],
                [698, 178],
                [749, 180],
                [661, 251],
                [770, 255],
                [633, 372],
                [801, 365],
                [636, 467],
                [809, 448],
                [680, 436],
                [748, 438],
                [692, 578],
                [735, 585],
                [715, 716],
                [714, 726],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [433, 192],
                [483, 192],
                [398, 257],
                [491, 258],
                [368, 365],
                [511, 360],
                [368, 448],
                [522, 446],
                [415, 429],
                [475, 430],
                [417, 584],
                [473, 578],
                [436, 712],
                [466, 700],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.3142,
                0.0000,
                0.0000,
                0.8722,
                0.9678,
                0.9978,
                0.9983,
                0.9788,
                0.9849,
                0.9391,
                0.9490,
                0.9994,
                0.9995,
                0.9971,
                0.9974,
                0.9213,
                0.9227,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.5982,
                0.9159,
                0.9904,
                0.9942,
                0.9521,
                0.9705,
                0.8870,
                0.9113,
                0.9973,
                0.9977,
                0.9870,
                0.9877,
                0.7204,
                0.7233,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_center_crop_package_torch(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(people_walking_image_torch)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [741, 192],
                    [0, 0],
                    [0, 0],
                    [698, 178],
                    [749, 180],
                    [661, 251],
                    [770, 255],
                    [633, 372],
                    [801, 365],
                    [636, 467],
                    [809, 448],
                    [680, 436],
                    [748, 438],
                    [692, 578],
                    [735, 585],
                    [715, 716],
                    [714, 726],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [433, 192],
                    [483, 192],
                    [398, 257],
                    [491, 258],
                    [368, 365],
                    [511, 360],
                    [368, 448],
                    [522, 446],
                    [415, 429],
                    [475, 430],
                    [417, 584],
                    [473, 578],
                    [436, 712],
                    [466, 700],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.3142,
                    0.0000,
                    0.0000,
                    0.8722,
                    0.9678,
                    0.9978,
                    0.9983,
                    0.9788,
                    0.9849,
                    0.9391,
                    0.9490,
                    0.9994,
                    0.9995,
                    0.9971,
                    0.9974,
                    0.9213,
                    0.9227,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.5982,
                    0.9159,
                    0.9904,
                    0.9942,
                    0.9521,
                    0.9705,
                    0.8870,
                    0.9113,
                    0.9973,
                    0.9977,
                    0.9870,
                    0.9877,
                    0.7204,
                    0.7233,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_center_crop_package_batch_torch(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        torch.stack([people_walking_image_torch, people_walking_image_torch], dim=0)
    )

    # then
    expected_xy = torch.tensor(
        [
            [
                [741, 192],
                [0, 0],
                [0, 0],
                [698, 178],
                [749, 180],
                [661, 251],
                [770, 255],
                [633, 372],
                [801, 365],
                [636, 467],
                [809, 448],
                [680, 436],
                [748, 438],
                [692, 578],
                [735, 585],
                [715, 716],
                [714, 726],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [433, 192],
                [483, 192],
                [398, 257],
                [491, 258],
                [368, 365],
                [511, 360],
                [368, 448],
                [522, 446],
                [415, 429],
                [475, 430],
                [417, 584],
                [473, 578],
                [436, 712],
                [466, 700],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.3142,
                0.0000,
                0.0000,
                0.8722,
                0.9678,
                0.9978,
                0.9983,
                0.9788,
                0.9849,
                0.9391,
                0.9490,
                0.9994,
                0.9995,
                0.9971,
                0.9974,
                0.9213,
                0.9227,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.5982,
                0.9159,
                0.9904,
                0.9942,
                0.9521,
                0.9705,
                0.8870,
                0.9113,
                0.9973,
                0.9977,
                0.9870,
                0.9877,
                0.7204,
                0.7233,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_center_crop_package_list_torch(
    yolov8n_pose_torchscript_static_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([people_walking_image_torch, people_walking_image_torch])

    # then
    expected_xy = torch.tensor(
        [
            [
                [741, 192],
                [0, 0],
                [0, 0],
                [698, 178],
                [749, 180],
                [661, 251],
                [770, 255],
                [633, 372],
                [801, 365],
                [636, 467],
                [809, 448],
                [680, 436],
                [748, 438],
                [692, 578],
                [735, 585],
                [715, 716],
                [714, 726],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [433, 192],
                [483, 192],
                [398, 257],
                [491, 258],
                [368, 365],
                [511, 360],
                [368, 448],
                [522, 446],
                [415, 429],
                [475, 430],
                [417, 584],
                [473, 578],
                [436, 712],
                [466, 700],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.3142,
                0.0000,
                0.0000,
                0.8722,
                0.9678,
                0.9978,
                0.9983,
                0.9788,
                0.9849,
                0.9391,
                0.9490,
                0.9994,
                0.9995,
                0.9971,
                0.9974,
                0.9213,
                0.9227,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.5982,
                0.9159,
                0.9904,
                0.9942,
                0.9521,
                0.9705,
                0.8870,
                0.9113,
                0.9973,
                0.9977,
                0.9870,
                0.9877,
                0.7204,
                0.7233,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_numpy(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(people_walking_image_numpy)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [697, 190],
                    [747, 188],
                    [672, 258],
                    [765, 255],
                    [638, 374],
                    [793, 356],
                    [640, 450],
                    [791, 407],
                    [689, 441],
                    [747, 440],
                    [700, 576],
                    [735, 578],
                    [716, 686],
                    [722, 692],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [418, 203],
                    [470, 205],
                    [391, 264],
                    [488, 265],
                    [371, 368],
                    [512, 360],
                    [370, 448],
                    [525, 445],
                    [414, 420],
                    [476, 422],
                    [416, 557],
                    [479, 563],
                    [422, 675],
                    [464, 684],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.7607,
                    0.3590,
                    0.9806,
                    0.9198,
                    0.9519,
                    0.7508,
                    0.8595,
                    0.6070,
                    0.9949,
                    0.9904,
                    0.9740,
                    0.9554,
                    0.6294,
                    0.5400,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.4374,
                    0.7757,
                    0.9780,
                    0.9896,
                    0.9354,
                    0.9757,
                    0.8797,
                    0.9364,
                    0.9964,
                    0.9976,
                    0.9830,
                    0.9885,
                    0.6630,
                    0.7122,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_batch_numpy(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([people_walking_image_numpy, people_walking_image_numpy])

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [697, 190],
                [747, 188],
                [672, 258],
                [765, 255],
                [638, 374],
                [793, 356],
                [640, 450],
                [791, 407],
                [689, 441],
                [747, 440],
                [700, 576],
                [735, 578],
                [716, 686],
                [722, 692],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [418, 203],
                [470, 205],
                [391, 264],
                [488, 265],
                [371, 368],
                [512, 360],
                [370, 448],
                [525, 445],
                [414, 420],
                [476, 422],
                [416, 557],
                [479, 563],
                [422, 675],
                [464, 684],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.7607,
                0.3590,
                0.9806,
                0.9198,
                0.9519,
                0.7508,
                0.8595,
                0.6070,
                0.9949,
                0.9904,
                0.9740,
                0.9554,
                0.6294,
                0.5400,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.4374,
                0.7757,
                0.9780,
                0.9896,
                0.9354,
                0.9757,
                0.8797,
                0.9364,
                0.9964,
                0.9976,
                0.9830,
                0.9885,
                0.6630,
                0.7122,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_torch(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(people_walking_image_torch)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [697, 190],
                    [747, 188],
                    [672, 258],
                    [765, 255],
                    [638, 374],
                    [793, 356],
                    [640, 450],
                    [791, 407],
                    [689, 441],
                    [747, 440],
                    [700, 576],
                    [735, 578],
                    [716, 686],
                    [722, 692],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [418, 203],
                    [470, 205],
                    [391, 264],
                    [488, 265],
                    [371, 368],
                    [512, 360],
                    [370, 448],
                    [525, 445],
                    [414, 420],
                    [476, 422],
                    [416, 557],
                    [479, 563],
                    [422, 675],
                    [464, 684],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.7607,
                    0.3590,
                    0.9806,
                    0.9198,
                    0.9519,
                    0.7508,
                    0.8595,
                    0.6070,
                    0.9949,
                    0.9904,
                    0.9740,
                    0.9554,
                    0.6294,
                    0.5400,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.4374,
                    0.7757,
                    0.9780,
                    0.9896,
                    0.9354,
                    0.9757,
                    0.8797,
                    0.9364,
                    0.9964,
                    0.9976,
                    0.9830,
                    0.9885,
                    0.6630,
                    0.7122,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_batch_torch(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        torch.stack([people_walking_image_torch, people_walking_image_torch], dim=0)
    )

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [697, 190],
                [747, 188],
                [672, 258],
                [765, 255],
                [638, 374],
                [793, 356],
                [640, 450],
                [791, 407],
                [689, 441],
                [747, 440],
                [700, 576],
                [735, 578],
                [716, 686],
                [722, 692],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [418, 203],
                [470, 205],
                [391, 264],
                [488, 265],
                [371, 368],
                [512, 360],
                [370, 448],
                [525, 445],
                [414, 420],
                [476, 422],
                [416, 557],
                [479, 563],
                [422, 675],
                [464, 684],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.7607,
                0.3590,
                0.9806,
                0.9198,
                0.9519,
                0.7508,
                0.8595,
                0.6070,
                0.9949,
                0.9904,
                0.9740,
                0.9554,
                0.6294,
                0.5400,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.4374,
                0.7757,
                0.9780,
                0.9896,
                0.9354,
                0.9757,
                0.8797,
                0.9364,
                0.9964,
                0.9976,
                0.9830,
                0.9885,
                0.6630,
                0.7122,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_center_crop_package_list_torch(
    yolov8n_pose_torchscript_static_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([people_walking_image_torch, people_walking_image_torch])

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [697, 190],
                [747, 188],
                [672, 258],
                [765, 255],
                [638, 374],
                [793, 356],
                [640, 450],
                [791, 407],
                [689, 441],
                [747, 440],
                [700, 576],
                [735, 578],
                [716, 686],
                [722, 692],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [418, 203],
                [470, 205],
                [391, 264],
                [488, 265],
                [371, 368],
                [512, 360],
                [370, 448],
                [525, 445],
                [414, 420],
                [476, 422],
                [416, 557],
                [479, 563],
                [422, 675],
                [464, 684],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.7607,
                0.3590,
                0.9806,
                0.9198,
                0.9519,
                0.7508,
                0.8595,
                0.6070,
                0.9949,
                0.9904,
                0.9740,
                0.9554,
                0.6294,
                0.5400,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.4374,
                0.7757,
                0.9780,
                0.9896,
                0.9354,
                0.9757,
                0.8797,
                0.9364,
                0.9964,
                0.9976,
                0.9830,
                0.9885,
                0.6630,
                0.7122,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_numpy(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(people_walking_image_numpy)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [695, 192],
                    [747, 193],
                    [669, 258],
                    [769, 259],
                    [634, 367],
                    [801, 363],
                    [633, 448],
                    [810, 446],
                    [686, 439],
                    [747, 440],
                    [700, 571],
                    [742, 579],
                    [711, 678],
                    [718, 695],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [421, 200],
                    [483, 202],
                    [397, 258],
                    [493, 260],
                    [374, 366],
                    [510, 362],
                    [370, 446],
                    [525, 446],
                    [413, 424],
                    [476, 426],
                    [418, 563],
                    [480, 571],
                    [426, 675],
                    [462, 687],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.6567,
                    0.7295,
                    0.9855,
                    0.9899,
                    0.9535,
                    0.9741,
                    0.8981,
                    0.9329,
                    0.9969,
                    0.9977,
                    0.9785,
                    0.9852,
                    0.6794,
                    0.7282,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.3024,
                    0.5514,
                    0.9627,
                    0.9788,
                    0.9360,
                    0.9680,
                    0.8823,
                    0.9227,
                    0.9957,
                    0.9968,
                    0.9794,
                    0.9845,
                    0.6798,
                    0.7136,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 194, 821, 691], [353, 188, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9204, 0.9172],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_batch_numpy(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([people_walking_image_numpy, people_walking_image_numpy])

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [695, 192],
                [747, 193],
                [669, 258],
                [769, 259],
                [634, 367],
                [801, 363],
                [633, 448],
                [810, 446],
                [686, 439],
                [747, 440],
                [700, 571],
                [742, 579],
                [711, 678],
                [718, 695],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [421, 200],
                [483, 202],
                [397, 258],
                [493, 260],
                [374, 366],
                [510, 362],
                [370, 446],
                [525, 446],
                [413, 424],
                [476, 426],
                [418, 563],
                [480, 571],
                [426, 675],
                [462, 687],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.6567,
                0.7295,
                0.9855,
                0.9899,
                0.9535,
                0.9741,
                0.8981,
                0.9329,
                0.9969,
                0.9977,
                0.9785,
                0.9852,
                0.6794,
                0.7282,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.3024,
                0.5514,
                0.9627,
                0.9788,
                0.9360,
                0.9680,
                0.8823,
                0.9227,
                0.9957,
                0.9968,
                0.9794,
                0.9845,
                0.6798,
                0.7136,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 194, 821, 691], [353, 188, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[620, 194, 821, 691], [353, 188, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9204, 0.9172],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.9204, 0.9172],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_torch(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(people_walking_image_torch)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [695, 192],
                    [747, 193],
                    [669, 258],
                    [769, 259],
                    [634, 367],
                    [801, 363],
                    [633, 448],
                    [810, 446],
                    [686, 439],
                    [747, 440],
                    [700, 571],
                    [742, 579],
                    [711, 678],
                    [718, 695],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [421, 200],
                    [483, 202],
                    [397, 258],
                    [493, 260],
                    [374, 366],
                    [510, 362],
                    [370, 446],
                    [525, 446],
                    [413, 424],
                    [476, 426],
                    [418, 563],
                    [480, 571],
                    [426, 675],
                    [462, 687],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.6567,
                    0.7295,
                    0.9855,
                    0.9899,
                    0.9535,
                    0.9741,
                    0.8981,
                    0.9329,
                    0.9969,
                    0.9977,
                    0.9785,
                    0.9852,
                    0.6794,
                    0.7282,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.3024,
                    0.5514,
                    0.9627,
                    0.9788,
                    0.9360,
                    0.9680,
                    0.8823,
                    0.9227,
                    0.9957,
                    0.9968,
                    0.9794,
                    0.9845,
                    0.6798,
                    0.7136,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 194, 821, 691], [353, 188, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9204, 0.9172],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_batch_torch(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        torch.stack([people_walking_image_torch, people_walking_image_torch], dim=0)
    )

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [695, 192],
                [747, 193],
                [669, 258],
                [769, 259],
                [634, 367],
                [801, 363],
                [633, 448],
                [810, 446],
                [686, 439],
                [747, 440],
                [700, 571],
                [742, 579],
                [711, 678],
                [718, 695],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [421, 200],
                [483, 202],
                [397, 258],
                [493, 260],
                [374, 366],
                [510, 362],
                [370, 446],
                [525, 446],
                [413, 424],
                [476, 426],
                [418, 563],
                [480, 571],
                [426, 675],
                [462, 687],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.6567,
                0.7295,
                0.9855,
                0.9899,
                0.9535,
                0.9741,
                0.8981,
                0.9329,
                0.9969,
                0.9977,
                0.9785,
                0.9852,
                0.6794,
                0.7282,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.3024,
                0.5514,
                0.9627,
                0.9788,
                0.9360,
                0.9680,
                0.8823,
                0.9227,
                0.9957,
                0.9968,
                0.9794,
                0.9845,
                0.6798,
                0.7136,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 194, 821, 691], [353, 188, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[620, 194, 821, 691], [353, 188, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9204, 0.9172],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.9204, 0.9172],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_yolov8n_pose_torchscript_static_static_crop_letterbox_package_list_torch(
    yolov8n_pose_torchscript_static_static_crop_letterbox_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_letterbox_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([people_walking_image_torch, people_walking_image_torch])

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [695, 192],
                [747, 193],
                [669, 258],
                [769, 259],
                [634, 367],
                [801, 363],
                [633, 448],
                [810, 446],
                [686, 439],
                [747, 440],
                [700, 571],
                [742, 579],
                [711, 678],
                [718, 695],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [421, 200],
                [483, 202],
                [397, 258],
                [493, 260],
                [374, 366],
                [510, 362],
                [370, 446],
                [525, 446],
                [413, 424],
                [476, 426],
                [418, 563],
                [480, 571],
                [426, 675],
                [462, 687],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.6567,
                0.7295,
                0.9855,
                0.9899,
                0.9535,
                0.9741,
                0.8981,
                0.9329,
                0.9969,
                0.9977,
                0.9785,
                0.9852,
                0.6794,
                0.7282,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.3024,
                0.5514,
                0.9627,
                0.9788,
                0.9360,
                0.9680,
                0.8823,
                0.9227,
                0.9957,
                0.9968,
                0.9794,
                0.9845,
                0.6798,
                0.7136,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 194, 821, 691], [353, 188, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[620, 194, 821, 691], [353, 188, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9204, 0.9172],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.9204, 0.9172],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_numpy(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(people_walking_image_numpy)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [410, 206],
                    [473, 204],
                    [391, 262],
                    [486, 262],
                    [372, 365],
                    [508, 361],
                    [374, 446],
                    [524, 446],
                    [416, 426],
                    [479, 427],
                    [415, 556],
                    [480, 561],
                    [419, 676],
                    [462, 687],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [692, 197],
                    [750, 200],
                    [670, 251],
                    [767, 257],
                    [619, 342],
                    [791, 359],
                    [633, 402],
                    [817, 443],
                    [679, 424],
                    [740, 427],
                    [684, 566],
                    [738, 570],
                    [700, 667],
                    [720, 677],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.3172,
                    0.5746,
                    0.9575,
                    0.9849,
                    0.9243,
                    0.9809,
                    0.8973,
                    0.9601,
                    0.9969,
                    0.9983,
                    0.9828,
                    0.9904,
                    0.5732,
                    0.6567,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.3832,
                    0.3886,
                    0.8239,
                    0.9429,
                    0.6111,
                    0.8868,
                    0.4590,
                    0.7139,
                    0.9915,
                    0.9954,
                    0.9587,
                    0.9774,
                    0.5449,
                    0.6345,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[354, 196, 541, 699], [621, 196, 824, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.8941, 0.8838],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_batch_numpy(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([people_walking_image_numpy, people_walking_image_numpy])

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [410, 206],
                [473, 204],
                [391, 262],
                [486, 262],
                [372, 365],
                [508, 361],
                [374, 446],
                [524, 446],
                [416, 426],
                [479, 427],
                [415, 556],
                [480, 561],
                [419, 676],
                [462, 687],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [692, 197],
                [750, 200],
                [670, 251],
                [767, 257],
                [619, 342],
                [791, 359],
                [633, 402],
                [817, 443],
                [679, 424],
                [740, 427],
                [684, 566],
                [738, 570],
                [700, 667],
                [720, 677],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.3172,
                0.5746,
                0.9575,
                0.9849,
                0.9243,
                0.9809,
                0.8973,
                0.9601,
                0.9969,
                0.9983,
                0.9828,
                0.9904,
                0.5732,
                0.6567,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.3832,
                0.3886,
                0.8239,
                0.9429,
                0.6111,
                0.8868,
                0.4590,
                0.7139,
                0.9915,
                0.9954,
                0.9587,
                0.9774,
                0.5449,
                0.6345,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[354, 196, 541, 699], [621, 196, 824, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[354, 196, 541, 699], [621, 196, 824, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.8941, 0.8838],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.8941, 0.8838],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_torch(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(people_walking_image_torch)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [410, 206],
                    [473, 204],
                    [391, 262],
                    [486, 262],
                    [372, 365],
                    [508, 361],
                    [374, 446],
                    [524, 446],
                    [416, 426],
                    [479, 427],
                    [415, 556],
                    [480, 561],
                    [419, 676],
                    [462, 687],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [692, 197],
                    [750, 200],
                    [670, 251],
                    [767, 257],
                    [619, 342],
                    [791, 359],
                    [633, 402],
                    [817, 443],
                    [679, 424],
                    [740, 427],
                    [684, 566],
                    [738, 570],
                    [700, 667],
                    [720, 677],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.3172,
                    0.5746,
                    0.9575,
                    0.9849,
                    0.9243,
                    0.9809,
                    0.8973,
                    0.9601,
                    0.9969,
                    0.9983,
                    0.9828,
                    0.9904,
                    0.5732,
                    0.6567,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.3832,
                    0.3886,
                    0.8239,
                    0.9429,
                    0.6111,
                    0.8868,
                    0.4590,
                    0.7139,
                    0.9915,
                    0.9954,
                    0.9587,
                    0.9774,
                    0.5449,
                    0.6345,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[354, 196, 541, 699], [621, 196, 824, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.8941, 0.8838],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_batch_torch(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        torch.stack([people_walking_image_torch, people_walking_image_torch], dim=0)
    )

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [410, 206],
                [473, 204],
                [391, 262],
                [486, 262],
                [372, 365],
                [508, 361],
                [374, 446],
                [524, 446],
                [416, 426],
                [479, 427],
                [415, 556],
                [480, 561],
                [419, 676],
                [462, 687],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [692, 197],
                [750, 200],
                [670, 251],
                [767, 257],
                [619, 342],
                [791, 359],
                [633, 402],
                [817, 443],
                [679, 424],
                [740, 427],
                [684, 566],
                [738, 570],
                [700, 667],
                [720, 677],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.3172,
                0.5746,
                0.9575,
                0.9849,
                0.9243,
                0.9809,
                0.8973,
                0.9601,
                0.9969,
                0.9983,
                0.9828,
                0.9904,
                0.5732,
                0.6567,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.3832,
                0.3886,
                0.8239,
                0.9429,
                0.6111,
                0.8868,
                0.4590,
                0.7139,
                0.9915,
                0.9954,
                0.9587,
                0.9774,
                0.5449,
                0.6345,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[354, 196, 541, 699], [621, 196, 824, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[354, 196, 541, 699], [621, 196, 824, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.8941, 0.8838],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.8941, 0.8838],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_static_crop_stretch_package_list_torch(
    yolov8n_pose_torchscript_static_static_crop_stretch_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_static_crop_stretch_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([people_walking_image_torch, people_walking_image_torch])

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [410, 206],
                [473, 204],
                [391, 262],
                [486, 262],
                [372, 365],
                [508, 361],
                [374, 446],
                [524, 446],
                [416, 426],
                [479, 427],
                [415, 556],
                [480, 561],
                [419, 676],
                [462, 687],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [692, 197],
                [750, 200],
                [670, 251],
                [767, 257],
                [619, 342],
                [791, 359],
                [633, 402],
                [817, 443],
                [679, 424],
                [740, 427],
                [684, 566],
                [738, 570],
                [700, 667],
                [720, 677],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.3172,
                0.5746,
                0.9575,
                0.9849,
                0.9243,
                0.9809,
                0.8973,
                0.9601,
                0.9969,
                0.9983,
                0.9828,
                0.9904,
                0.5732,
                0.6567,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.3832,
                0.3886,
                0.8239,
                0.9429,
                0.6111,
                0.8868,
                0.4590,
                0.7139,
                0.9915,
                0.9954,
                0.9587,
                0.9774,
                0.5449,
                0.6345,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[354, 196, 541, 699], [621, 196, 824, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[354, 196, 541, 699], [621, 196, 824, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.8941, 0.8838],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.8941, 0.8838],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_numpy(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(people_walking_image_numpy)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [741, 192],
                    [0, 0],
                    [0, 0],
                    [698, 178],
                    [749, 180],
                    [661, 251],
                    [770, 255],
                    [633, 372],
                    [801, 365],
                    [636, 467],
                    [809, 448],
                    [680, 436],
                    [748, 438],
                    [692, 578],
                    [735, 585],
                    [715, 716],
                    [714, 726],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [433, 192],
                    [483, 192],
                    [398, 257],
                    [491, 258],
                    [368, 365],
                    [511, 360],
                    [368, 448],
                    [522, 446],
                    [415, 429],
                    [475, 430],
                    [417, 584],
                    [473, 578],
                    [436, 712],
                    [466, 700],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.3142,
                    0.0000,
                    0.0000,
                    0.8722,
                    0.9678,
                    0.9978,
                    0.9983,
                    0.9788,
                    0.9849,
                    0.9391,
                    0.9490,
                    0.9994,
                    0.9995,
                    0.9971,
                    0.9974,
                    0.9213,
                    0.9227,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.5982,
                    0.9159,
                    0.9904,
                    0.9942,
                    0.9521,
                    0.9705,
                    0.8870,
                    0.9113,
                    0.9973,
                    0.9977,
                    0.9870,
                    0.9877,
                    0.7204,
                    0.7233,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_batch_numpy(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model([people_walking_image_numpy, people_walking_image_numpy])

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [741, 192],
                [0, 0],
                [0, 0],
                [698, 178],
                [749, 180],
                [661, 251],
                [770, 255],
                [633, 372],
                [801, 365],
                [636, 467],
                [809, 448],
                [680, 436],
                [748, 438],
                [692, 578],
                [735, 585],
                [715, 716],
                [714, 726],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [433, 192],
                [483, 192],
                [398, 257],
                [491, 258],
                [368, 365],
                [511, 360],
                [368, 448],
                [522, 446],
                [415, 429],
                [475, 430],
                [417, 584],
                [473, 578],
                [436, 712],
                [466, 700],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.3142,
                0.0000,
                0.0000,
                0.8722,
                0.9678,
                0.9978,
                0.9983,
                0.9788,
                0.9849,
                0.9391,
                0.9490,
                0.9994,
                0.9995,
                0.9971,
                0.9974,
                0.9213,
                0.9227,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.5982,
                0.9159,
                0.9904,
                0.9942,
                0.9521,
                0.9705,
                0.8870,
                0.9113,
                0.9973,
                0.9977,
                0.9870,
                0.9877,
                0.7204,
                0.7233,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_torch(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(people_walking_image_torch)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [741, 192],
                    [0, 0],
                    [0, 0],
                    [698, 178],
                    [749, 180],
                    [661, 251],
                    [770, 255],
                    [633, 372],
                    [801, 365],
                    [636, 467],
                    [809, 448],
                    [680, 436],
                    [748, 438],
                    [692, 578],
                    [735, 585],
                    [715, 716],
                    [714, 726],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [433, 192],
                    [483, 192],
                    [398, 257],
                    [491, 258],
                    [368, 365],
                    [511, 360],
                    [368, 448],
                    [522, 446],
                    [415, 429],
                    [475, 430],
                    [417, 584],
                    [473, 578],
                    [436, 712],
                    [466, 700],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.3142,
                    0.0000,
                    0.0000,
                    0.8722,
                    0.9678,
                    0.9978,
                    0.9983,
                    0.9788,
                    0.9849,
                    0.9391,
                    0.9490,
                    0.9994,
                    0.9995,
                    0.9971,
                    0.9974,
                    0.9213,
                    0.9227,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.5982,
                    0.9159,
                    0.9904,
                    0.9942,
                    0.9521,
                    0.9705,
                    0.8870,
                    0.9113,
                    0.9973,
                    0.9977,
                    0.9870,
                    0.9877,
                    0.7204,
                    0.7233,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_batch_torch(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(
        torch.stack([people_walking_image_torch, people_walking_image_torch], dim=0)
    )

    # then
    expected_xy = torch.tensor(
        [
            [
                [741, 192],
                [0, 0],
                [0, 0],
                [698, 178],
                [749, 180],
                [661, 251],
                [770, 255],
                [633, 372],
                [801, 365],
                [636, 467],
                [809, 448],
                [680, 436],
                [748, 438],
                [692, 578],
                [735, 585],
                [715, 716],
                [714, 726],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [433, 192],
                [483, 192],
                [398, 257],
                [491, 258],
                [368, 365],
                [511, 360],
                [368, 448],
                [522, 446],
                [415, 429],
                [475, 430],
                [417, 584],
                [473, 578],
                [436, 712],
                [466, 700],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.3142,
                0.0000,
                0.0000,
                0.8722,
                0.9678,
                0.9978,
                0.9983,
                0.9788,
                0.9849,
                0.9391,
                0.9490,
                0.9994,
                0.9995,
                0.9971,
                0.9974,
                0.9213,
                0.9227,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.5982,
                0.9159,
                0.9904,
                0.9942,
                0.9521,
                0.9705,
                0.8870,
                0.9113,
                0.9973,
                0.9977,
                0.9870,
                0.9877,
                0.7204,
                0.7233,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_nms_fused_center_crop_package_list_torch(
    yolov8n_pose_torchscript_static_nms_fused_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([people_walking_image_torch, people_walking_image_torch])

    # then
    expected_xy = torch.tensor(
        [
            [
                [741, 192],
                [0, 0],
                [0, 0],
                [698, 178],
                [749, 180],
                [661, 251],
                [770, 255],
                [633, 372],
                [801, 365],
                [636, 467],
                [809, 448],
                [680, 436],
                [748, 438],
                [692, 578],
                [735, 585],
                [715, 716],
                [714, 726],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [433, 192],
                [483, 192],
                [398, 257],
                [491, 258],
                [368, 365],
                [511, 360],
                [368, 448],
                [522, 446],
                [415, 429],
                [475, 430],
                [417, 584],
                [473, 578],
                [436, 712],
                [466, 700],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.3142,
                0.0000,
                0.0000,
                0.8722,
                0.9678,
                0.9978,
                0.9983,
                0.9788,
                0.9849,
                0.9391,
                0.9490,
                0.9994,
                0.9995,
                0.9971,
                0.9974,
                0.9213,
                0.9227,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.5982,
                0.9159,
                0.9904,
                0.9942,
                0.9521,
                0.9705,
                0.8870,
                0.9113,
                0.9973,
                0.9977,
                0.9870,
                0.9877,
                0.7204,
                0.7233,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[619, 124, 821, 747], [352, 136, 539, 744]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor(
            [0.9245, 0.9082],
        ).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_numpy(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    # when
    predictions = model(people_walking_image_numpy)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [697, 190],
                    [747, 188],
                    [672, 258],
                    [765, 255],
                    [638, 374],
                    [793, 356],
                    [640, 450],
                    [791, 407],
                    [689, 441],
                    [747, 440],
                    [700, 576],
                    [735, 578],
                    [716, 686],
                    [722, 692],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [418, 203],
                    [470, 205],
                    [391, 264],
                    [488, 265],
                    [371, 368],
                    [512, 360],
                    [370, 448],
                    [525, 445],
                    [414, 420],
                    [476, 422],
                    [416, 557],
                    [479, 563],
                    [422, 675],
                    [464, 684],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.7607,
                    0.3590,
                    0.9806,
                    0.9198,
                    0.9519,
                    0.7508,
                    0.8595,
                    0.6070,
                    0.9949,
                    0.9904,
                    0.9740,
                    0.9554,
                    0.6294,
                    0.5400,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.4374,
                    0.7757,
                    0.9780,
                    0.9896,
                    0.9354,
                    0.9757,
                    0.8797,
                    0.9364,
                    0.9964,
                    0.9976,
                    0.9830,
                    0.9885,
                    0.6630,
                    0.7122,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_batch_numpy(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_numpy: np.ndarray,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([people_walking_image_numpy, people_walking_image_numpy])

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [697, 190],
                [747, 188],
                [672, 258],
                [765, 255],
                [638, 374],
                [793, 356],
                [640, 450],
                [791, 407],
                [689, 441],
                [747, 440],
                [700, 576],
                [735, 578],
                [716, 686],
                [722, 692],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [418, 203],
                [470, 205],
                [391, 264],
                [488, 265],
                [371, 368],
                [512, 360],
                [370, 448],
                [525, 445],
                [414, 420],
                [476, 422],
                [416, 557],
                [479, 563],
                [422, 675],
                [464, 684],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.7607,
                0.3590,
                0.9806,
                0.9198,
                0.9519,
                0.7508,
                0.8595,
                0.6070,
                0.9949,
                0.9904,
                0.9740,
                0.9554,
                0.6294,
                0.5400,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.4374,
                0.7757,
                0.9780,
                0.9896,
                0.9354,
                0.9757,
                0.8797,
                0.9364,
                0.9964,
                0.9976,
                0.9830,
                0.9885,
                0.6630,
                0.7122,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_torch(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(people_walking_image_torch)

    # then
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [697, 190],
                    [747, 188],
                    [672, 258],
                    [765, 255],
                    [638, 374],
                    [793, 356],
                    [640, 450],
                    [791, 407],
                    [689, 441],
                    [747, 440],
                    [700, 576],
                    [735, 578],
                    [716, 686],
                    [722, 692],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [418, 203],
                    [470, 205],
                    [391, 264],
                    [488, 265],
                    [371, 368],
                    [512, 360],
                    [370, 448],
                    [525, 445],
                    [414, 420],
                    [476, 422],
                    [416, 557],
                    [479, 563],
                    [422, 675],
                    [464, 684],
                ],
            ],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor(
            [
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.7607,
                    0.3590,
                    0.9806,
                    0.9198,
                    0.9519,
                    0.7508,
                    0.8595,
                    0.6070,
                    0.9949,
                    0.9904,
                    0.9740,
                    0.9554,
                    0.6294,
                    0.5400,
                ],
                [
                    0.0000,
                    0.0000,
                    0.0000,
                    0.4374,
                    0.7757,
                    0.9780,
                    0.9896,
                    0.9354,
                    0.9757,
                    0.8797,
                    0.9364,
                    0.9964,
                    0.9976,
                    0.9830,
                    0.9885,
                    0.6630,
                    0.7122,
                ],
            ],
        ).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_batch_torch(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(
        torch.stack([people_walking_image_torch, people_walking_image_torch], dim=0)
    )

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [697, 190],
                [747, 188],
                [672, 258],
                [765, 255],
                [638, 374],
                [793, 356],
                [640, 450],
                [791, 407],
                [689, 441],
                [747, 440],
                [700, 576],
                [735, 578],
                [716, 686],
                [722, 692],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [418, 203],
                [470, 205],
                [391, 264],
                [488, 265],
                [371, 368],
                [512, 360],
                [370, 448],
                [525, 445],
                [414, 420],
                [476, 422],
                [416, 557],
                [479, 563],
                [422, 675],
                [464, 684],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.7607,
                0.3590,
                0.9806,
                0.9198,
                0.9519,
                0.7508,
                0.8595,
                0.6070,
                0.9949,
                0.9904,
                0.9740,
                0.9554,
                0.6294,
                0.5400,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.4374,
                0.7757,
                0.9780,
                0.9896,
                0.9354,
                0.9757,
                0.8797,
                0.9364,
                0.9964,
                0.9976,
                0.9830,
                0.9885,
                0.6630,
                0.7122,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package_list_torch(
    yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package: str,
    people_walking_image_torch: torch.Tensor,
) -> None:
    model = YOLOv8ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolov8n_pose_torchscript_static_nms_fused_static_crop_center_crop_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([people_walking_image_torch, people_walking_image_torch])

    # then
    expected_kp_xy = torch.tensor(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [697, 190],
                [747, 188],
                [672, 258],
                [765, 255],
                [638, 374],
                [793, 356],
                [640, 450],
                [791, 407],
                [689, 441],
                [747, 440],
                [700, 576],
                [735, 578],
                [716, 686],
                [722, 692],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [418, 203],
                [470, 205],
                [391, 264],
                [488, 265],
                [371, 368],
                [512, 360],
                [370, 448],
                [525, 445],
                [414, 420],
                [476, 422],
                [416, 557],
                [479, 563],
                [422, 675],
                [464, 684],
            ],
        ],
        dtype=torch.int32,
    ).cpu()
    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    assert torch.allclose(
        predictions[0][1].xy.cpu(),
        expected_kp_xy,
        atol=2,
    )
    expected_kp_confidence = torch.tensor(
        [
            [
                0.0000,
                0.0000,
                0.0000,
                0.7607,
                0.3590,
                0.9806,
                0.9198,
                0.9519,
                0.7508,
                0.8595,
                0.6070,
                0.9949,
                0.9904,
                0.9740,
                0.9554,
                0.6294,
                0.5400,
            ],
            [
                0.0000,
                0.0000,
                0.0000,
                0.4374,
                0.7757,
                0.9780,
                0.9896,
                0.9354,
                0.9757,
                0.8797,
                0.9364,
                0.9964,
                0.9976,
                0.9830,
                0.9885,
                0.6630,
                0.7122,
            ],
        ],
    ).cpu()
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0][1].confidence.cpu(),
        expected_kp_confidence,
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][1].xyxy.cpu(),
        torch.tensor(
            [[620, 190, 793, 691], [351, 191, 538, 695]],
            dtype=torch.int32,
        ).cpu(),
        atol=2,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1][1].confidence.cpu(),
        torch.tensor([0.9362, 0.9064]).cpu(),
        atol=0.01,
    )
