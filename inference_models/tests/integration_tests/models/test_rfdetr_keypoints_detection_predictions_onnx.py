import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_rfdetr_keypoints_onnx_glue_sticks_numpy(
    rfdetr_kp_preview_onnx_glue_sticks_package: str,
    glue_sticks_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_key_points_detection_onnx import (
        RFDetrForKeyPointsONNX,
    )

    model = RFDetrForKeyPointsONNX.from_pretrained(
        model_name_or_path=rfdetr_kp_preview_onnx_glue_sticks_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    key_points_list, detections_list = model(glue_sticks_image_numpy, confidence=0.5)

    # then
    assert len(key_points_list) == 1
    key_points = key_points_list[0]
    assert torch.allclose(
        key_points.xy.cpu().to(torch.int32),
        torch.tensor(
            [
                [[860, 166], [671, 420]],
                [[1278, 657], [895, 532]],
            ],
            dtype=torch.int32,
        ),
        atol=2,
    )
    assert torch.allclose(
        key_points.confidence.cpu(),
        torch.tensor(
            [
                [0.9899, 0.9807],
                [0.9873, 0.9876],
            ]
        ),
        atol=0.01,
    )
    assert key_points.class_id.cpu().tolist() == [0, 0]

    # Covariance: pixel-space per-keypoint (N, K, 2, 2), matching xy's (N, K, *) layout.
    assert key_points.covariance is not None
    assert key_points.covariance.shape == key_points.xy.shape[:2] + (2, 2)
    covariance = key_points.covariance.cpu()
    # All keypoints here are above threshold, so every covariance is finite.
    assert torch.isfinite(covariance).all()
    # Each covariance is symmetric with positive variances on the diagonal.
    assert torch.allclose(covariance[..., 0, 1], covariance[..., 1, 0])
    assert (covariance[..., 0, 0] > 0).all()
    assert (covariance[..., 1, 1] > 0).all()
    # to_supervision() exposes the covariance under data["covariance"] for the
    # Supervision ellipse annotators.
    sv_key_points = key_points.to_supervision()
    assert "covariance" in sv_key_points.data
    assert sv_key_points.data["covariance"].shape == tuple(key_points.covariance.shape)

    assert detections_list is not None
    detections = detections_list[0]
    assert torch.allclose(
        detections.xyxy.cpu().to(torch.int32),
        torch.tensor(
            [
                [625, 124, 909, 441],
                [875, 497, 1286, 680],
            ],
            dtype=torch.int32,
        ),
        atol=2,
    )
    assert torch.allclose(
        detections.confidence.cpu(),
        torch.tensor([0.6288, 0.6260]),
        atol=0.01,
    )
    assert detections.class_id.cpu().tolist() == [0, 0]


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_rfdetr_keypoints_onnx_glue_sticks_batch_numpy(
    rfdetr_kp_preview_onnx_glue_sticks_package: str,
    glue_sticks_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_key_points_detection_onnx import (
        RFDetrForKeyPointsONNX,
    )

    model = RFDetrForKeyPointsONNX.from_pretrained(
        model_name_or_path=rfdetr_kp_preview_onnx_glue_sticks_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    key_points_list, detections_list = model(
        [glue_sticks_image_numpy, glue_sticks_image_numpy], confidence=0.5
    )

    # then
    assert len(key_points_list) == 2
    assert detections_list is not None
    assert len(detections_list) == 2

    expected_kp_xy = torch.tensor(
        [
            [[860, 166], [671, 420]],
            [[1278, 657], [895, 532]],
        ],
        dtype=torch.int32,
    )
    expected_kp_confidence = torch.tensor(
        [
            [0.9899, 0.9807],
            [0.9873, 0.9876],
        ]
    )
    expected_xyxy = torch.tensor(
        [
            [625, 124, 909, 441],
            [875, 497, 1286, 680],
        ],
        dtype=torch.int32,
    )
    for key_points in key_points_list:
        assert torch.allclose(
            key_points.xy.cpu().to(torch.int32), expected_kp_xy, atol=2
        )
        assert torch.allclose(
            key_points.confidence.cpu(), expected_kp_confidence, atol=0.01
        )
        assert key_points.class_id.cpu().tolist() == [0, 0]
    for detections in detections_list:
        assert torch.allclose(
            detections.xyxy.cpu().to(torch.int32), expected_xyxy, atol=2
        )
        assert torch.allclose(
            detections.confidence.cpu(), torch.tensor([0.6288, 0.6260]), atol=0.01
        )
        assert detections.class_id.cpu().tolist() == [0, 0]


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_rfdetr_keypoints_onnx_glue_sticks_torch(
    rfdetr_kp_preview_onnx_glue_sticks_package: str,
    glue_sticks_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.rfdetr.rfdetr_key_points_detection_onnx import (
        RFDetrForKeyPointsONNX,
    )

    model = RFDetrForKeyPointsONNX.from_pretrained(
        model_name_or_path=rfdetr_kp_preview_onnx_glue_sticks_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # when
    key_points_list, detections_list = model(glue_sticks_image_torch, confidence=0.5)

    # then
    assert len(key_points_list) == 1
    key_points = key_points_list[0]
    assert torch.allclose(
        key_points.xy.cpu().to(torch.int32),
        torch.tensor(
            [
                [[857, 170], [673, 417]],
                [[1285, 656], [906, 536]],
            ],
            dtype=torch.int32,
        ),
        atol=2,
    )
    assert torch.allclose(
        key_points.confidence.cpu(),
        torch.tensor(
            [
                [0.9896, 0.9812],
                [0.9868, 0.9867],
            ]
        ),
        atol=0.01,
    )
    assert key_points.class_id.cpu().tolist() == [0, 0]

    assert detections_list is not None
    detections = detections_list[0]
    assert torch.allclose(
        detections.xyxy.cpu().to(torch.int32),
        torch.tensor(
            [
                [625, 124, 908, 442],
                [876, 499, 1285, 680],
            ],
            dtype=torch.int32,
        ),
        atol=2,
    )
    assert torch.allclose(
        detections.confidence.cpu(),
        torch.tensor([0.6338, 0.6074]),
        atol=0.01,
    )
    assert detections.class_id.cpu().tolist() == [0, 0]