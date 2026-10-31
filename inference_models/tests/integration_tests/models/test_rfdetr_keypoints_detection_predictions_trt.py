import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_rfdetr_keypoints_trt_numpy(
    rfdetr_kp_preview_trt_package: str,
    glue_sticks_image_numpy: np.ndarray,
) -> None:
    # given
    if not torch.cuda.is_available():
        pytest.skip("RF-DETR keypoint TRT requires CUDA")

    from inference_models.models.rfdetr.rfdetr_key_points_detection_trt import (
        RFDetrForKeyPointsTRT,
    )

    model = RFDetrForKeyPointsTRT.from_pretrained(
        model_name_or_path=rfdetr_kp_preview_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    key_points_list, detections_list = model(glue_sticks_image_numpy, confidence=0.5)

    # then
    assert len(key_points_list) == 1
    key_points = key_points_list[0]
    assert key_points.xy.ndim == 3
    assert key_points.xy.shape[-1] == 2
    assert key_points.class_id.ndim == 1
    assert key_points.confidence.shape == key_points.xy.shape[:2]
    assert key_points.covariance is not None
    assert key_points.covariance.shape == key_points.xy.shape[:2] + (2, 2)
    assert detections_list is not None
    assert len(detections_list) == 1
    assert detections_list[0].xyxy.shape[0] == key_points.xy.shape[0]
    if key_points.detection_confidence is not None:
        assert torch.allclose(
            key_points.detection_confidence.cpu(),
            detections_list[0].confidence.cpu(),
        )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_rfdetr_keypoints_trt_batch_numpy(
    rfdetr_kp_preview_trt_package: str,
    glue_sticks_image_numpy: np.ndarray,
) -> None:
    # given
    if not torch.cuda.is_available():
        pytest.skip("RF-DETR keypoint TRT requires CUDA")

    from inference_models.models.rfdetr.rfdetr_key_points_detection_trt import (
        RFDetrForKeyPointsTRT,
    )

    model = RFDetrForKeyPointsTRT.from_pretrained(
        model_name_or_path=rfdetr_kp_preview_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    key_points_list, detections_list = model(
        [glue_sticks_image_numpy, glue_sticks_image_numpy], confidence=0.5
    )

    # then
    assert len(key_points_list) == 2
    assert detections_list is not None
    assert len(detections_list) == 2
    for key_points in key_points_list:
        assert key_points.xy.ndim == 3
        assert key_points.xy.shape[-1] == 2
        assert key_points.class_id.ndim == 1
    for detections in detections_list:
        assert detections.xyxy.ndim == 2
        assert detections.xyxy.shape[-1] == 4


@pytest.mark.slow
@pytest.mark.trt_extras
def test_rfdetr_keypoints_trt_torch(
    rfdetr_kp_preview_trt_package: str,
    glue_sticks_image_torch: torch.Tensor,
) -> None:
    # given
    if not torch.cuda.is_available():
        pytest.skip("RF-DETR keypoint TRT requires CUDA")

    from inference_models.models.rfdetr.rfdetr_key_points_detection_trt import (
        RFDetrForKeyPointsTRT,
    )

    model = RFDetrForKeyPointsTRT.from_pretrained(
        model_name_or_path=rfdetr_kp_preview_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    key_points_list, detections_list = model(glue_sticks_image_torch, confidence=0.5)

    # then
    assert len(key_points_list) == 1
    key_points = key_points_list[0]
    assert key_points.xy.ndim == 3
    assert detections_list is not None
    assert len(detections_list) == 1
