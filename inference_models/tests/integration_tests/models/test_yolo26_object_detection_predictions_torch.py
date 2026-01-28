import numpy as np
import pytest
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.yolo26.yolo26_object_detection_torch_script import (
    YOLO26ForObjectDetectionTorchScript,
)


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

    predictions = model(sunflowers_image_numpy)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert predictions[0].confidence is not None
    assert predictions[0].class_id is not None
    assert predictions[0].xyxy is not None


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

    predictions = model([sunflowers_image_numpy, sunflowers_image_numpy])

    print(f"predictions[0].confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"predictions[0].class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"predictions[0].xyxy: {predictions[0].xyxy.cpu().tolist()}")
    print(f"predictions[1].confidence: {predictions[1].confidence.cpu().tolist()}")
    print(f"predictions[1].class_id: {predictions[1].class_id.cpu().tolist()}")
    print(f"predictions[1].xyxy: {predictions[1].xyxy.cpu().tolist()}")

    assert len(predictions) == 2
    assert predictions[0].confidence is not None
    assert predictions[1].confidence is not None


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

    predictions = model(sunflowers_image_torch)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert predictions[0].confidence is not None
    assert predictions[0].class_id is not None
    assert predictions[0].xyxy is not None


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

    predictions = model(sunflowers_image_numpy)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert predictions[0].confidence is not None
    assert predictions[0].class_id is not None
    assert predictions[0].xyxy is not None


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

    predictions = model([sunflowers_image_numpy, sunflowers_image_numpy])

    print(f"predictions[0].confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"predictions[0].class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"predictions[0].xyxy: {predictions[0].xyxy.cpu().tolist()}")
    print(f"predictions[1].confidence: {predictions[1].confidence.cpu().tolist()}")
    print(f"predictions[1].class_id: {predictions[1].class_id.cpu().tolist()}")
    print(f"predictions[1].xyxy: {predictions[1].xyxy.cpu().tolist()}")

    assert len(predictions) == 2
    assert predictions[0].confidence is not None
    assert predictions[1].confidence is not None


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

    predictions = model(sunflowers_image_torch)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert predictions[0].confidence is not None
    assert predictions[0].class_id is not None
    assert predictions[0].xyxy is not None
