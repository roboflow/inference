import numpy as np
import pytest
import torch
from inference_exp.configuration import DEFAULT_DEVICE


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_pipeline_predictions_for_numpy_image(
    mediapipe_face_detector_package: str, l2cs_package: str, man_image_numpy: np.ndarray
) -> None:
    # given
    from inference_exp.model_pipelines.face_and_gaze_detection.mediapipe_l2cs import (
        FaceAndGazeDetectionMPAndL2CS,
    )

    pipeline = FaceAndGazeDetectionMPAndL2CS.from_pretrained(
        face_detector=mediapipe_face_detector_package,
        gaze_detector=l2cs_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = pipeline(man_image_numpy)

    # then
    assert len(results) == 3
    assert np.allclose(
        results[0][0].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )
    assert np.allclose(
        results[1][0].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(results[2][0].yaw.cpu().numpy(), np.array([-0.4165]), atol=1e-3)
    assert np.allclose(
        results[2][0].pitch.cpu().numpy(), np.array([-0.0278]), atol=1e-3
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_pipeline_predictions_for_numpy_image_list(
    mediapipe_face_detector_package: str, l2cs_package: str, man_image_numpy: np.ndarray
) -> None:
    # given
    from inference_exp.model_pipelines.face_and_gaze_detection.mediapipe_l2cs import (
        FaceAndGazeDetectionMPAndL2CS,
    )

    pipeline = FaceAndGazeDetectionMPAndL2CS.from_pretrained(
        face_detector=mediapipe_face_detector_package,
        gaze_detector=l2cs_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = pipeline([man_image_numpy, man_image_numpy])

    # then
    assert len(results) == 3
    assert np.allclose(
        results[0][0].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )
    assert np.allclose(
        results[0][1].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )
    assert np.allclose(
        results[1][0].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(
        results[1][1].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(results[2][0].yaw.cpu().numpy(), np.array([-0.4165]), atol=1e-3)
    assert np.allclose(results[2][1].yaw.cpu().numpy(), np.array([-0.4165]), atol=1e-3)
    assert np.allclose(
        results[2][0].pitch.cpu().numpy(), np.array([-0.0278]), atol=1e-3
    )
    assert np.allclose(
        results[2][1].pitch.cpu().numpy(), np.array([-0.0278]), atol=1e-3
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_pipeline_predictions_for_torch_image(
    mediapipe_face_detector_package: str,
    l2cs_package: str,
    man_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.model_pipelines.face_and_gaze_detection.mediapipe_l2cs import (
        FaceAndGazeDetectionMPAndL2CS,
    )

    pipeline = FaceAndGazeDetectionMPAndL2CS.from_pretrained(
        face_detector=mediapipe_face_detector_package,
        gaze_detector=l2cs_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = pipeline(man_image_torch)

    # then
    assert len(results) == 3
    assert np.allclose(
        results[0][0].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )
    assert np.allclose(
        results[1][0].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(results[2][0].yaw.cpu().numpy(), np.array([-0.4165]), atol=1e-3)
    assert np.allclose(
        results[2][0].pitch.cpu().numpy(), np.array([-0.0278]), atol=1e-3
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_pipeline_predictions_for_torch_batch(
    mediapipe_face_detector_package: str,
    l2cs_package: str,
    man_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.model_pipelines.face_and_gaze_detection.mediapipe_l2cs import (
        FaceAndGazeDetectionMPAndL2CS,
    )

    pipeline = FaceAndGazeDetectionMPAndL2CS.from_pretrained(
        face_detector=mediapipe_face_detector_package,
        gaze_detector=l2cs_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = pipeline(torch.stack([man_image_torch, man_image_torch], dim=0))

    # then
    assert len(results) == 3
    assert np.allclose(
        results[0][0].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )
    assert np.allclose(
        results[0][1].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )
    assert np.allclose(
        results[1][0].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(
        results[1][1].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(results[2][0].yaw.cpu().numpy(), np.array([-0.4165]), atol=1e-3)
    assert np.allclose(results[2][1].yaw.cpu().numpy(), np.array([-0.4165]), atol=1e-3)
    assert np.allclose(
        results[2][0].pitch.cpu().numpy(), np.array([-0.0278]), atol=1e-3
    )
    assert np.allclose(
        results[2][1].pitch.cpu().numpy(), np.array([-0.0278]), atol=1e-3
    )


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_pipeline_predictions_for_torch_list(
    mediapipe_face_detector_package: str,
    l2cs_package: str,
    man_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_exp.model_pipelines.face_and_gaze_detection.mediapipe_l2cs import (
        FaceAndGazeDetectionMPAndL2CS,
    )

    pipeline = FaceAndGazeDetectionMPAndL2CS.from_pretrained(
        face_detector=mediapipe_face_detector_package,
        gaze_detector=l2cs_package,
        device=DEFAULT_DEVICE,
    )

    # when
    results = pipeline([man_image_torch, man_image_torch])

    # then
    assert len(results) == 3
    assert np.allclose(
        results[0][0].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )
    assert np.allclose(
        results[0][1].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )
    assert np.allclose(
        results[1][0].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(
        results[1][1].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(results[2][0].yaw.cpu().numpy(), np.array([-0.4165]), atol=1e-3)
    assert np.allclose(results[2][1].yaw.cpu().numpy(), np.array([-0.4165]), atol=1e-3)
    assert np.allclose(
        results[2][0].pitch.cpu().numpy(), np.array([-0.0278]), atol=1e-3
    )
    assert np.allclose(
        results[2][1].pitch.cpu().numpy(), np.array([-0.0278]), atol=1e-3
    )
