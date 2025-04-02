import numpy as np

import supervision as sv

from inference import get_model


def test_yolov8n_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8n-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov8s_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8s-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov8m_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8m-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov8l_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8l-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov8x_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8x-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolo_nas_s_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolo-nas-s-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolo_nas_m_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolo-nas-m-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolo_nas_l_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolo-nas-l-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov10n_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov10n-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov10s_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov10s-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov10m_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov10m-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov10b_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov10b-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov10l_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov10l-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov10x_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov10x-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov11n_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov11n-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov11s_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov11s-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov11m_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov11m-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov11l_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov11l-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov11x_object_detection_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov11x-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov8n_instance_segmentation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8n-seg-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result.mask) > 0, "At least one prediction is expected"


def test_yolov8s_instance_segmentation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8s-seg-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result.mask) > 0, "At least one prediction is expected"


def test_yolov8m_instance_segmentation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8m-seg-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result.mask) > 0, "At least one prediction is expected"


def test_yolov8l_instance_segmentation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8l-seg-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result.mask) > 0, "At least one prediction is expected"


def test_yolov8x_instance_segmentation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8x-seg-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result.mask) > 0, "At least one prediction is expected"


def test_yolov11n_instance_segmentation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov11n-seg-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result.mask) > 0, "At least one prediction is expected"



def test_yolov11s_instance_segmentation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov11s-seg-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result.mask) > 0, "At least one prediction is expected"


def test_yolov11m_instance_segmentation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov11m-seg-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result.mask) > 0, "At least one prediction is expected"


def test_yolov11l_instance_segmentation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov11l-seg-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result.mask) > 0, "At least one prediction is expected"


def test_yolov11x_instance_segmentation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov11x-seg-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.Detections.from_inference(result_raw)

    # then
    assert len(result.mask) > 0, "At least one prediction is expected"


def test_yolov8n_pose_estimation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8n-pose-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.KeyPoints.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov8s_pose_estimation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8s-pose-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.KeyPoints.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov8m_pose_estimation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8m-pose-640")

    # when
    result_raw = model.infer(reference_image)[0]
    result = sv.KeyPoints.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov8l_pose_estimation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8l-pose-640")

    # when
    result_raw = model.infer(reference_image, confidence=0.3)[0]
    result = sv.KeyPoints.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"


def test_yolov8x_pose_estimation_inference(reference_image: np.ndarray) -> None:
    # given
    model = get_model("yolov8x-pose-640")

    # when
    result_raw = model.infer(reference_image, confidence=0.3)[0]
    result = sv.KeyPoints.from_inference(result_raw)

    # then
    assert len(result) > 0, "At least one prediction is expected"
