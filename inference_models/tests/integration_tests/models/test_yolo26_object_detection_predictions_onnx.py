import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.onnx_extras
def test_onnx_package_stretch_static_bs_numpy(
    yolo26n_object_detection_sunflowers_stretch_onnx_static_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_onnx_package_stretch_static_bs_batch_numpy(
    yolo26n_object_detection_sunflowers_stretch_onnx_static_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_onnx_package_stretch_static_bs_torch(
    yolo26n_object_detection_sunflowers_stretch_onnx_static_package: str,
    sunflowers_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_onnx_package_stretch_dynamic_bs_numpy(
    yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_onnx_package_stretch_dynamic_bs_batch_numpy(
    yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_onnx_package_stretch_dynamic_bs_torch(
    yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package: str,
    sunflowers_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_stretch_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_static_bs_numpy(
    yolo26n_object_detection_sunflowers_letterbox_onnx_static_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_static_bs_batch_numpy(
    yolo26n_object_detection_sunflowers_letterbox_onnx_static_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_static_bs_torch(
    yolo26n_object_detection_sunflowers_letterbox_onnx_static_package: str,
    sunflowers_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_static_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_dynamic_bs_numpy(
    yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_dynamic_bs_batch_numpy(
    yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package: str,
    sunflowers_image_numpy: np.ndarray,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
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
@pytest.mark.onnx_extras
def test_onnx_package_letterbox_dynamic_bs_torch(
    yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package: str,
    sunflowers_image_torch: torch.Tensor,
) -> None:
    from inference_models.models.yolo26.yolo26_object_detection_onnx import (
        YOLO26ForObjectDetectionOnnx,
    )

    model = YOLO26ForObjectDetectionOnnx.from_pretrained(
        model_name_or_path=yolo26n_object_detection_sunflowers_letterbox_onnx_dynamic_package,
        onnx_execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = model(sunflowers_image_torch)

    print(f"confidence: {predictions[0].confidence.cpu().tolist()}")
    print(f"class_id: {predictions[0].class_id.cpu().tolist()}")
    print(f"xyxy: {predictions[0].xyxy.cpu().tolist()}")

    assert len(predictions) == 1
    assert predictions[0].confidence is not None
    assert predictions[0].class_id is not None
    assert predictions[0].xyxy is not None
