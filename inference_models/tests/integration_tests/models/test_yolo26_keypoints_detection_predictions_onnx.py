import numpy as np
import pytest
import torch

COORD_TOLERANCE = 2
CONF_TOLERANCE = 0.01

LETTERBOX_EXPECTED_KP_XY_1 = [
    [617, 297],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [973, 350],
    [541, 844],
    [1008, 486],
    [944, 568],
    [879, 669],
    [0, 0],
    [0, 0],
    [0, 0],
    [1022, 373],
    [0, 0],
    [0, 0],
    [1367, 356],
    [0, 0],
    [0, 0],
    [1000, 492],
    [0, 0],
    [0, 0],
    [0, 0],
    [557, 853],
    [483, 529],
    [0, 0],
    [589, 319],
    [441, 438],
    [244, 605],
    [0, 0],
    [0, 0],
]

LETTERBOX_EXPECTED_KP_XY_2 = [
    [614, 293],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [972, 345],
    [0, 0],
    [1009, 484],
    [943, 565],
    [875, 669],
    [0, 0],
    [1276, 593],
    [0, 0],
    [1261, 378],
    [0, 0],
    [0, 0],
    [1368, 350],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [480, 528],
    [0, 0],
    [0, 0],
    [436, 436],
    [239, 605],
    [0, 0],
    [0, 0],
]

LETTERBOX_EXPECTED_KP_CONF_1 = [
    0.4832,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.4811,
    0.3102,
    0.3857,
    0.4450,
    0.3724,
    0.0,
    0.0,
    0.0,
    0.3119,
    0.0,
    0.0,
    0.3165,
    0.0,
    0.0,
    0.3565,
    0.0,
    0.0,
    0.0,
    0.3140,
    0.3213,
    0.0,
    0.3109,
    0.3244,
    0.3673,
    0.0,
    0.0,
]

LETTERBOX_EXPECTED_KP_CONF_2 = [
    0.3900,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.4405,
    0.0,
    0.4793,
    0.4500,
    0.3472,
    0.0,
    0.3144,
    0.0,
    0.3296,
    0.0,
    0.0,
    0.3614,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.3407,
    0.0,
    0.0,
    0.3056,
    0.3206,
    0.0,
    0.0,
]

STRETCH_EXPECTED_KP_XY = [
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1011, 492],
    [0, 0],
    [897, 672],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [886, 670],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
]

STRETCH_EXPECTED_KP_CONF = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5310,
    0.0,
    0.3716,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.4403,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_letterbox_numpy(
    yolo26n_pose_basketball_letterbox_onnx_static_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
            [[-33, 246, 1732, 1076], [-8, 252, 1798, 1076]], dtype=torch.int32
        ),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9278, 0.4428]),
        atol=CONF_TOLERANCE,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_letterbox_batch_numpy(
    yolo26n_pose_basketball_letterbox_onnx_static_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
        [[-33, 246, 1732, 1076], [-8, 252, 1798, 1076]], dtype=torch.int32
    )
    expected_box_conf = torch.tensor([0.9278, 0.4428])

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
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_letterbox_torch(
    yolo26n_pose_basketball_letterbox_onnx_static_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
            [[-33, 246, 1732, 1076], [-8, 252, 1798, 1076]], dtype=torch.int32
        ),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9278, 0.4428]),
        atol=CONF_TOLERANCE,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_letterbox_numpy(
    yolo26n_pose_basketball_letterbox_onnx_dynamic_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
            [[-33, 246, 1732, 1076], [-8, 252, 1798, 1076]], dtype=torch.int32
        ),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9278, 0.4428]),
        atol=CONF_TOLERANCE,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_letterbox_batch_numpy(
    yolo26n_pose_basketball_letterbox_onnx_dynamic_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
        [[-33, 246, 1732, 1076], [-8, 252, 1798, 1076]], dtype=torch.int32
    )
    expected_box_conf = torch.tensor([0.9278, 0.4428])

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
            predictions[1][i].confidence.cpu(), expected_box_conf, atol=0.10
        )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_letterbox_torch(
    yolo26n_pose_basketball_letterbox_onnx_dynamic_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
            [[-33, 246, 1732, 1076], [-8, 252, 1798, 1076]], dtype=torch.int32
        ),
        atol=COORD_TOLERANCE,
    )
    assert torch.allclose(
        predictions[1][0].confidence.cpu(),
        torch.tensor([0.9278, 0.4428]),
        atol=CONF_TOLERANCE,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_stretch_numpy(
    yolo26n_pose_basketball_stretch_onnx_static_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
        torch.tensor([0.9263]),
        atol=CONF_TOLERANCE,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_stretch_batch_numpy(
    yolo26n_pose_basketball_stretch_onnx_static_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([basketball_image_numpy, basketball_image_numpy])

    expected_kp_xy = torch.tensor([STRETCH_EXPECTED_KP_XY], dtype=torch.int32)
    expected_kp_conf = torch.tensor([STRETCH_EXPECTED_KP_CONF])
    expected_box_xyxy = torch.tensor([[21, 259, 1929, 1078]], dtype=torch.int32)
    expected_box_conf = torch.tensor([0.9263])

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
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_static_stretch_torch(
    yolo26n_pose_basketball_stretch_onnx_static_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
        torch.tensor([0.9263]),
        atol=CONF_TOLERANCE,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_stretch_numpy(
    yolo26n_pose_basketball_stretch_onnx_dynamic_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
        torch.tensor([0.9263]),
        atol=CONF_TOLERANCE,
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_stretch_batch_numpy(
    yolo26n_pose_basketball_stretch_onnx_dynamic_package: str,
    basketball_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model([basketball_image_numpy, basketball_image_numpy])

    expected_kp_xy = torch.tensor([STRETCH_EXPECTED_KP_XY], dtype=torch.int32)
    expected_kp_conf = torch.tensor([STRETCH_EXPECTED_KP_CONF])
    expected_box_xyxy = torch.tensor([[21, 259, 1929, 1078]], dtype=torch.int32)
    expected_box_conf = torch.tensor([0.9263])

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
@pytest.mark.onnx_extras
def test_yolo26n_pose_onnx_dynamic_stretch_torch(
    yolo26n_pose_basketball_stretch_onnx_dynamic_package: str,
    basketball_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_key_points_detection_onnx import (
        YOLO26ForKeyPointsDetectionOnnx,
    )

    model = YOLO26ForKeyPointsDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_pose_basketball_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
        torch.tensor([0.9263]),
        atol=CONF_TOLERANCE,
    )
