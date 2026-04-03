import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_numpy(
    yolov10_object_detection_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov10.yolov10_object_detection_trt import (
        YOLOv10ForObjectDetectionTRT,
    )

    model = YOLOv10ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov10_object_detection_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(dog_image_numpy)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.5039]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([16], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[71, 253, 646, 970]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_batch_numpy(
    yolov10_object_detection_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    # given
    from inference_models.models.yolov10.yolov10_object_detection_trt import (
        YOLOv10ForObjectDetectionTRT,
    )

    model = YOLOv10ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov10_object_detection_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([dog_image_numpy, dog_image_numpy])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.5039]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([16], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[71, 253, 646, 970]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.5039]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([16], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[71, 253, 646, 970]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch(
    yolov10_object_detection_trt_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov10.yolov10_object_detection_trt import (
        YOLOv10ForObjectDetectionTRT,
    )

    model = YOLOv10ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov10_object_detection_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(dog_image_torch)

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.5039]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([16], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[71, 253, 646, 970]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_list(
    yolov10_object_detection_trt_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov10.yolov10_object_detection_trt import (
        YOLOv10ForObjectDetectionTRT,
    )

    model = YOLOv10ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov10_object_detection_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model([dog_image_torch, dog_image_torch])

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.5039]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([16], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[71, 253, 646, 970]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.5039]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([16], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[71, 253, 646, 970]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_batch(
    yolov10_object_detection_trt_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov10.yolov10_object_detection_trt import (
        YOLOv10ForObjectDetectionTRT,
    )

    model = YOLOv10ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov10_object_detection_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    predictions = model(torch.stack([dog_image_torch, dog_image_torch], dim=0))

    # then
    assert torch.allclose(
        predictions[0].confidence.cpu(),
        torch.tensor([0.5039]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[0].class_id.cpu(),
        torch.tensor([16], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[71, 253, 646, 970]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[0].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )
    assert torch.allclose(
        predictions[1].confidence.cpu(),
        torch.tensor([0.5039]).cpu(),
        atol=0.01,
    )
    assert torch.allclose(
        predictions[1].class_id.cpu(),
        torch.tensor([16], dtype=torch.int32).cpu(),
    )
    expected_xyxy = torch.tensor(
        [[71, 253, 646, 970]],
        dtype=torch.int32,
    )
    assert torch.allclose(
        predictions[1].xyxy.cpu(),
        expected_xyxy.cpu(),
        atol=5,
    )


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_package_torch_multiple_predictions_in_row(
    yolov10_object_detection_trt_package: str,
    dog_image_torch: torch.Tensor,
) -> None:
    # given
    from inference_models.models.yolov10.yolov10_object_detection_trt import (
        YOLOv10ForObjectDetectionTRT,
    )

    model = YOLOv10ForObjectDetectionTRT.from_pretrained(
        model_name_or_path=yolov10_object_detection_trt_package,
        engine_host_code_allowed=True,
    )

    # when
    for _ in range(8):
        predictions = model(dog_image_torch)

        # then
        assert torch.allclose(
            predictions[0].confidence.cpu(),
            torch.tensor([0.5039]).cpu(),
            atol=0.01,
        )
        assert torch.allclose(
            predictions[0].class_id.cpu(),
            torch.tensor([16], dtype=torch.int32).cpu(),
        )
        expected_xyxy = torch.tensor(
            [[71, 253, 646, 970]],
            dtype=torch.int32,
        )
        assert torch.allclose(
            predictions[0].xyxy.cpu(),
            expected_xyxy.cpu(),
            atol=5,
        )
