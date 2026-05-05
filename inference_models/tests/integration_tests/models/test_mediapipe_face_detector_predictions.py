import numpy as np
import pytest
import torch


@pytest.mark.torch_models
def test_face_detector_predictions_for_numpy_image(
    mediapipe_face_detector_package: str,
    man_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.mediapipe_face_detection.face_detection import (
        MediaPipeFaceDetector,
    )

    model = MediaPipeFaceDetector.from_pretrained(mediapipe_face_detector_package)

    # when
    results = model(man_image_numpy, confidence=0.25)

    # then
    assert len(results) == 2
    assert np.allclose(
        results[1][0].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(
        results[0][0].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )


@pytest.mark.torch_models
def test_face_detector_predictions_for_numpy_image_list(
    mediapipe_face_detector_package: str,
    man_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.mediapipe_face_detection.face_detection import (
        MediaPipeFaceDetector,
    )

    model = MediaPipeFaceDetector.from_pretrained(mediapipe_face_detector_package)

    # when
    results = model([man_image_numpy, man_image_numpy], confidence=0.25)

    # then
    assert len(results) == 2
    assert np.allclose(
        results[1][0].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(
        results[0][0].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )
    assert np.allclose(
        results[1][1].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(
        results[0][1].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )


@pytest.mark.torch_models
def test_face_detector_predictions_for_torch_image(
    mediapipe_face_detector_package: str, man_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.mediapipe_face_detection.face_detection import (
        MediaPipeFaceDetector,
    )

    model = MediaPipeFaceDetector.from_pretrained(mediapipe_face_detector_package)

    # when
    results = model(man_image_torch, confidence=0.25)

    # then
    assert len(results) == 2
    assert np.allclose(
        results[1][0].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(
        results[0][0].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )


@pytest.mark.torch_models
def test_face_detector_predictions_for_torch_batch(
    mediapipe_face_detector_package: str, man_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.mediapipe_face_detection.face_detection import (
        MediaPipeFaceDetector,
    )

    model = MediaPipeFaceDetector.from_pretrained(mediapipe_face_detector_package)

    # when
    results = model(
        torch.stack([man_image_torch, man_image_torch], dim=0), confidence=0.25
    )

    # then
    assert len(results) == 2
    assert np.allclose(
        results[1][0].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(
        results[0][0].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )
    assert np.allclose(
        results[1][1].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(
        results[0][1].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )


@pytest.mark.torch_models
def test_face_detector_predictions_for_torch_list(
    mediapipe_face_detector_package: str, man_image_torch: torch.Tensor
) -> None:
    # given
    from inference_models.models.mediapipe_face_detection.face_detection import (
        MediaPipeFaceDetector,
    )

    model = MediaPipeFaceDetector.from_pretrained(mediapipe_face_detector_package)

    # when
    results = model([man_image_torch, man_image_torch], confidence=0.25)

    # then
    assert len(results) == 2
    assert np.allclose(
        results[1][0].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(
        results[0][0].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )
    assert np.allclose(
        results[1][1].xyxy.cpu().numpy(), np.array([[197, 133, 420, 356]]), atol=3
    )
    assert np.allclose(
        results[0][1].xy.cpu().numpy(),
        np.array(
            [[[256, 182], [356, 177], [305, 221], [307, 280], [205, 224], [412, 216]]]
        ),
        atol=3,
    )


@pytest.mark.torch_models
def test_per_class_confidence_filters_faces(
    mediapipe_face_detector_package: str,
    man_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.mediapipe_face_detection.face_detection import (
        MediaPipeFaceDetector,
    )
    from inference_models.weights_providers.entities import RecommendedParameters

    model = MediaPipeFaceDetector.from_pretrained(mediapipe_face_detector_package)
    baseline = model(man_image_numpy, confidence=0.25)
    baseline_count = baseline[1][0].xyxy.shape[0]
    assert baseline_count > 0

    model.recommended_parameters = RecommendedParameters(
        confidence=0.25,
        per_class_confidence={name: 1.01 for name in model.class_names},
    )
    filtered = model(man_image_numpy, confidence="best")
    assert filtered[1][0].xyxy.shape[0] < baseline_count
