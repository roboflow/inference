import pytest


@pytest.mark.trt_extras
def test_import_of_trt_utils() -> None:
    import inference_exp.models.common.trt


@pytest.mark.trt_extras
def test_import_yolonas_object_detection_trt() -> None:
    from inference_exp.models.yolonas.yolonas_object_detection_trt import (
        YOLONasForObjectDetectionTRT,
    )


@pytest.mark.trt_extras
def test_import_yolov5_instance_segmentation_trt() -> None:
    from inference_exp.models.yolov5.yolov5_instance_segmentation_trt import (
        YOLOv5ForInstanceSegmentationTRT,
    )


@pytest.mark.trt_extras
def test_import_yolov5_object_detection_trt() -> None:
    from inference_exp.models.yolov5.yolov5_object_detection_trt import (
        YOLOv5ForObjectDetectionTRT,
    )


@pytest.mark.trt_extras
def test_import_yolov7_instance_segmentation_trt() -> None:
    from inference_exp.models.yolov7.yolov7_instance_segmentation_trt import (
        YOLOv7ForInstanceSegmentationTRT,
    )


@pytest.mark.trt_extras
def test_import_yolov8_object_detection_trt() -> None:
    from inference_exp.models.yolov8.yolov8_object_detection_trt import (
        YOLOv8ForObjectDetectionTRT,
    )


@pytest.mark.trt_extras
def test_import_yolov8_instance_segmentation_trt() -> None:
    from inference_exp.models.yolov8.yolov8_instance_segmentation_trt import (
        YOLOv8ForInstanceSegmentationTRT,
    )


@pytest.mark.trt_extras
def test_import_yolov8_keypoints_detection_trt() -> None:
    from inference_exp.models.yolov8.yolov8_key_points_detection_trt import (
        YOLOv8ForKeyPointsDetectionTRT,
    )


@pytest.mark.trt_extras
def test_import_yolov9_object_detection_trt() -> None:
    from inference_exp.models.yolov9.yolov9_trt import YOLOv9ForObjectDetectionTRT


@pytest.mark.trt_extras
def test_import_yolov10_object_detection_trt() -> None:
    from inference_exp.models.yolov10.yolov10_object_detection_trt import (
        YOLOv10ForObjectDetectionTRT,
    )


@pytest.mark.trt_extras
def test_import_yolov11_object_detection_trt() -> None:
    from inference_exp.models.yolov11.yolov11_trt import YOLOv11ForObjectDetectionTRT


@pytest.mark.trt_extras
def test_import_yolov11_instance_segmentation_trt() -> None:
    from inference_exp.models.yolov11.yolov11_trt import (
        YOLOv11ForInstanceSegmentationTRT,
    )


@pytest.mark.trt_extras
def test_import_yolov11_key_points_detection_trt() -> None:
    from inference_exp.models.yolov11.yolov11_trt import (
        YOLOv11ForForKeyPointsDetectionTRT,
    )


@pytest.mark.trt_extras
def test_import_yolov12_object_detection_trt() -> None:
    from inference_exp.models.yolov12.yolov12_trt import YOLOv12ForObjectDetectionTRT
