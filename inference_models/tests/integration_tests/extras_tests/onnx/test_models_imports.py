import pytest


@pytest.mark.onnx_extras
def test_import_of_onnx_utils() -> None:
    import inference_models.models.common.onnx


@pytest.mark.onnx_extras
def test_import_clip_onnx() -> None:
    from inference_models.models.clip.clip_onnx import ClipOnnx


@pytest.mark.onnx_extras
def test_import_l2cs_onnx() -> None:
    from inference_models.models.l2cs.l2cs_onnx import L2CSNetOnnx


@pytest.mark.onnx_extras
def test_import_resnet_onnx() -> None:
    from inference_models.models.resnet.resnet_classification_onnx import (
        ResNetForClassificationOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolonas_object_detection_onnx() -> None:
    from inference_models.models.yolonas.yolonas_object_detection_onnx import (
        YOLONasForObjectDetectionOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolov5_instance_segmentation_onnx() -> None:
    from inference_models.models.yolov5.yolov5_instance_segmentation_onnx import (
        YOLOv5ForInstanceSegmentationOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolov5_object_detection_onnx() -> None:
    from inference_models.models.yolov5.yolov5_object_detection_onnx import (
        YOLOv5ForObjectDetectionOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolov7_instance_segmentation_onnx() -> None:
    from inference_models.models.yolov7.yolov7_instance_segmentation_onnx import (
        YOLOv7ForInstanceSegmentationOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolov8_object_detection_onnx() -> None:
    from inference_models.models.yolov8.yolov8_object_detection_onnx import (
        YOLOv8ForObjectDetectionOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolov8_instance_segmentation_onnx() -> None:
    from inference_models.models.yolov8.yolov8_instance_segmentation_onnx import (
        YOLOv8ForInstanceSegmentationOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolov8_keypoints_detection_onnx() -> None:
    from inference_models.models.yolov8.yolov8_key_points_detection_onnx import (
        YOLOv8ForKeyPointsDetectionOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolov9_object_detection_onnx() -> None:
    from inference_models.models.yolov9.yolov9_onnx import YOLOv9ForObjectDetectionOnnx


@pytest.mark.onnx_extras
def test_import_yolov10_object_detection_onnx() -> None:
    from inference_models.models.yolov10.yolov10_object_detection_onnx import (
        YOLOv10ForObjectDetectionOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolov11_object_detection_onnx() -> None:
    from inference_models.models.yolov11.yolov11_onnx import (
        YOLOv11ForObjectDetectionOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolov11_instance_segmentation_onnx() -> None:
    from inference_models.models.yolov11.yolov11_onnx import (
        YOLOv11ForInstanceSegmentationOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolov11_key_points_detection_onnx() -> None:
    from inference_models.models.yolov11.yolov11_onnx import (
        YOLOv11ForForKeyPointsDetectionOnnx,
    )


@pytest.mark.onnx_extras
def test_import_yolov12_object_detection_onnx() -> None:
    from inference_models.models.yolov12.yolov12_onnx import (
        YOLOv12ForObjectDetectionOnnx,
    )
