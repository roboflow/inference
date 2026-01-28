import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.yolo26.yolo26_key_points_detection_torch_script import (
    YOLO26ForKeyPointsDetectionTorchScript,
)

COORD_TOLERANCE = 5
CONF_TOLERANCE = 0.02

LETTERBOX_EXPECTED_KP_XY_1 = (
    [[617, 297]]
    + [[0, 0]] * 6
    + [[973, 350], [541, 844], [1008, 486], [944, 568], [879, 669]]
    + [[0, 0]] * 3
    + [[1022, 373]]
    + [[0, 0]] * 2
    + [[1367, 356]]
    + [[0, 0]] * 2
    + [[1000, 492]]
    + [[0, 0]] * 3
    + [[557, 853], [483, 529]]
    + [[0, 0]]
    + [[589, 319], [441, 438], [244, 605]]
    + [[0, 0]] * 2
)

LETTERBOX_EXPECTED_KP_XY_2 = (
    [[614, 293]]
    + [[0, 0]] * 6
    + [[972, 345]]
    + [[0, 0]]
    + [[1009, 484], [943, 565], [875, 669]]
    + [[0, 0]]
    + [[1276, 593]]
    + [[0, 0]]
    + [[1261, 378]]
    + [[0, 0]] * 2
    + [[1368, 350]]
    + [[0, 0]] * 7
    + [[480, 528]]
    + [[0, 0]] * 2
    + [[436, 436], [240, 605]]
    + [[0, 0]] * 2
)

LETTERBOX_EXPECTED_KP_CONF_1 = (
    [0.4831]
    + [0.0] * 6
    + [0.4811, 0.3102, 0.3857, 0.4450, 0.3725]
    + [0.0] * 3
    + [0.3119]
    + [0.0] * 2
    + [0.3165]
    + [0.0] * 2
    + [0.3564]
    + [0.0] * 3
    + [0.3140, 0.3214]
    + [0.0]
    + [0.3110, 0.3245, 0.3673]
    + [0.0] * 2
)

LETTERBOX_EXPECTED_KP_CONF_2 = (
    [0.3901]
    + [0.0] * 6
    + [0.4405]
    + [0.0]
    + [0.4794, 0.4500, 0.3472]
    + [0.0]
    + [0.3144]
    + [0.0]
    + [0.3296]
    + [0.0] * 2
    + [0.3614]
    + [0.0] * 7
    + [0.3407]
    + [0.0] * 2
    + [0.3056, 0.3206]
    + [0.0] * 2
)

STRETCH_EXPECTED_KP_XY = (
    [[0, 0]] * 9
    + [[1011, 492]]
    + [[0, 0]]
    + [[897, 672]]
    + [[0, 0]] * 11
    + [[887, 670]]
    + [[0, 0]] * 9
)

STRETCH_EXPECTED_KP_CONF = (
    [0.0] * 9 + [0.5310] + [0.0] + [0.3717] + [0.0] * 11 + [0.4403] + [0.0] * 9
)


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_letterbox_numpy(
    yolo26n_pose_basketball_letterbox_torch_script_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(basketball_image_numpy)

    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [LETTERBOX_EXPECTED_KP_XY_1, LETTERBOX_EXPECTED_KP_XY_2],
            dtype=torch.int32,
        ),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor([LETTERBOX_EXPECTED_KP_CONF_1, LETTERBOX_EXPECTED_KP_CONF_2]),
        atol=CONF_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[-33, 246, 1733, 1076], [-8, 252, 1798, 1076]], dtype=torch.int32
        ),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9248, 0.5963]),
        atol=CONF_TOLERANCE,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_letterbox_batch_numpy(
    yolo26n_pose_basketball_letterbox_torch_script_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([basketball_image_numpy, basketball_image_numpy])

    expected_kp_xy = torch.tensor(
        [LETTERBOX_EXPECTED_KP_XY_1, LETTERBOX_EXPECTED_KP_XY_2],
        dtype=torch.int32,
    )
    expected_kp_conf = torch.tensor(
        [LETTERBOX_EXPECTED_KP_CONF_1, LETTERBOX_EXPECTED_KP_CONF_2]
    )
    expected_box_xyxy = torch.tensor(
        [[-33, 246, 1733, 1076], [-8, 252, 1798, 1076]], dtype=torch.int32
    )
    expected_box_conf = torch.tensor([0.9248, 0.5963])

    for i in range(2):
        assert torch.allclose(
            predictions[0][i].xy.cpu(), expected_kp_xy, atol=COORD_TOLERANCE
        )
        assert torch.allclose(
            predictions[0][i].confidence.cpu(), expected_kp_conf, atol=CONF_TOLERANCE
        )
        assert torch.allclose(
            predictions[1][i].xyxy.cpu(), expected_box_xyxy, atol=COORD_TOLERANCE
        )
        assert torch.allclose(
            predictions[1][i].confidence.cpu(), expected_box_conf, atol=CONF_TOLERANCE
        )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_letterbox_torch(
    yolo26n_pose_basketball_letterbox_torch_script_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(basketball_image_torch)

    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor(
            [LETTERBOX_EXPECTED_KP_XY_1, LETTERBOX_EXPECTED_KP_XY_2],
            dtype=torch.int32,
        ),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor([LETTERBOX_EXPECTED_KP_CONF_1, LETTERBOX_EXPECTED_KP_CONF_2]),
        atol=CONF_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor(
            [[-33, 246, 1733, 1076], [-8, 252, 1798, 1076]], dtype=torch.int32
        ),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9248, 0.5963]),
        atol=CONF_TOLERANCE,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_stretch_numpy(
    yolo26n_pose_basketball_stretch_torch_script_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(basketball_image_numpy)

    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor([STRETCH_EXPECTED_KP_XY], dtype=torch.int32),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor([STRETCH_EXPECTED_KP_CONF]),
        atol=CONF_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor([[21, 259, 1929, 1078]], dtype=torch.int32),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9262]),
        atol=CONF_TOLERANCE,
    )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_stretch_batch_numpy(
    yolo26n_pose_basketball_stretch_torch_script_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model([basketball_image_numpy, basketball_image_numpy])

    expected_kp_xy = torch.tensor([STRETCH_EXPECTED_KP_XY], dtype=torch.int32)
    expected_kp_conf = torch.tensor([STRETCH_EXPECTED_KP_CONF])
    expected_box_xyxy = torch.tensor([[21, 259, 1929, 1078]], dtype=torch.int32)
    expected_box_conf = torch.tensor([0.9262])

    for i in range(2):
        assert torch.allclose(
            predictions[0][i].xy.cpu(), expected_kp_xy, atol=COORD_TOLERANCE
        )
        assert torch.allclose(
            predictions[0][i].confidence.cpu(), expected_kp_conf, atol=CONF_TOLERANCE
        )
        assert torch.allclose(
            predictions[1][i].xyxy.cpu(), expected_box_xyxy, atol=COORD_TOLERANCE
        )
        assert torch.allclose(
            predictions[1][i].confidence.cpu(), expected_box_conf, atol=CONF_TOLERANCE
        )


@pytest.mark.slow
@pytest.mark.torch_models
def test_yolo26n_pose_torchscript_stretch_torch(
    yolo26n_pose_basketball_stretch_torch_script_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    model = YOLO26ForKeyPointsDetectionTorchScript.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_torch_script_package,
        device=DEFAULT_DEVICE,
    )

    predictions = model(basketball_image_torch)

    assert torch.allclose(
        predictions[0][0].xy.cpu(),
        torch.tensor([STRETCH_EXPECTED_KP_XY], dtype=torch.int32),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[0][0].confidence.cpu(),
        torch.tensor([STRETCH_EXPECTED_KP_CONF]),
        atol=CONF_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].xyxy.cpu(),
        torch.tensor([[21, 259, 1929, 1078]], dtype=torch.int32),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9262]),
        atol=CONF_TOLERANCE,
    )
