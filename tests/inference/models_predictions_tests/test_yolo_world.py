import numpy as np
import supervision as sv

from inference.models import YOLOWorld

PIXEL_TOLERANCE = 0.05
CONFIDENCE_TOLERANCE = 5e-3


def test_yolo_world_v1_s_against_single_image(person_image: np.ndarray) -> None:
    # given
    model = YOLOWorld(model_id="yolo_world/s")
    model.set_classes(["person", "bag", "shoe"])
    expected_detections = sv.Detections(
        xyxy=np.array(
            [
                [273.49, 160.23, 358.42, 378.65],
                [318.68, 365.2, 331.2, 378.42],
                [299.9, 353.42, 316.59, 366.49],
                [339.87, 271.46, 370.65, 331.83],
                [276.42, 201.16, 298.01, 279.31],
            ]
        ),
        confidence=np.array([0.95061, 0.51519, 0.50877, 0.41699, 0.03059]),
        class_id=np.array([0, 2, 2, 1, 1]),
    )

    # when
    results = model.infer(person_image, confidence=0.03).dict(
        by_alias=True, exclude_none=True
    )
    detection_results = sv.Detections.from_inference(results)

    # then
    assert len(detection_results) == len(
        expected_detections
    ), "Expected the same number of boxes"
    assert np.allclose(
        detection_results.xyxy, expected_detections.xyxy, atol=PIXEL_TOLERANCE
    ), "Boxes coordinates detection differ"
    assert np.allclose(
        detection_results.confidence,
        expected_detections.confidence,
        atol=CONFIDENCE_TOLERANCE,
    ), "Confidences differ"
    assert np.allclose(
        detection_results.class_id, expected_detections.class_id
    ), "Classes id differ"


def test_yolo_world_v1_s_against_single_image_with_only_one_detected_box(
    person_image: np.ndarray,
) -> None:
    # given
    model = YOLOWorld(model_id="yolo_world/s")
    model.set_classes(["person"])
    expected_detections = sv.Detections(
        xyxy=np.array(
            [
                [273.18, 160.18, 358.61, 378.85],
            ]
        ),
        confidence=np.array([0.9503]),
        class_id=np.array([0]),
    )

    # when
    results = model.infer(person_image, confidence=0.03).dict(
        by_alias=True, exclude_none=True
    )
    detection_results = sv.Detections.from_inference(results)

    # then
    assert len(detection_results) == len(
        expected_detections
    ), "Expected the same number of boxes"
    assert np.allclose(
        detection_results.xyxy, expected_detections.xyxy, atol=PIXEL_TOLERANCE
    ), "Boxes coordinates detection differ"
    assert np.allclose(
        detection_results.confidence,
        expected_detections.confidence,
        atol=CONFIDENCE_TOLERANCE,
    ), "Confidences differ"
    assert np.allclose(
        detection_results.class_id, expected_detections.class_id
    ), "Classes id differ"


def test_yolo_world_v1_m_against_single_image(person_image: np.ndarray) -> None:
    # given
    model = YOLOWorld(model_id="yolo_world/m")
    model.set_classes(["person", "bag", "shoe"])
    expected_detections = sv.Detections(
        xyxy=np.array(
            [
                [274.99, 160.44, 357.9, 377.55],
                [341.12, 271.7, 370.4, 332.35],
                [318.83, 365.58, 331.21, 378.4],
                [299.89, 354.25, 317.38, 365.97],
            ]
        ),
        confidence=np.array([0.94808, 0.57941, 0.24767, 0.13392]),
        class_id=np.array([0, 1, 2, 2]),
    )

    # when
    results = model.infer(person_image, confidence=0.03).dict(
        by_alias=True, exclude_none=True
    )
    detection_results = sv.Detections.from_inference(results)

    # then
    assert len(detection_results) == len(
        expected_detections
    ), "Expected the same number of boxes"
    assert np.allclose(
        detection_results.xyxy,
        expected_detections.xyxy,
        atol=PIXEL_TOLERANCE,
    ), "Boxes coordinates detection differ"
    assert np.allclose(
        detection_results.confidence,
        expected_detections.confidence,
        atol=CONFIDENCE_TOLERANCE,
    ), "Confidences differ"
    assert np.allclose(
        detection_results.class_id, expected_detections.class_id
    ), "Classes id differ"


def test_yolo_world_v1_l_against_single_image(person_image: np.ndarray) -> None:
    # given
    model = YOLOWorld(model_id="yolo_world/l")
    model.set_classes(["person", "bag", "shoe"])
    expected_detections = sv.Detections(
        xyxy=np.array(
            [
                [275.29, 160.82, 358.69, 378.55],
                [341.38, 281.8, 370.6, 332.59],
                [318.92, 365.75, 330.91, 378.6],
                [301, 353.85, 316.4, 367.33],
            ]
        ),
        confidence=np.array([0.94448, 0.36421, 0.16429, 0.12968]),
        class_id=np.array([0, 1, 2, 2]),
    )

    # when
    results = model.infer(person_image, confidence=0.03).dict(
        by_alias=True, exclude_none=True
    )
    detection_results = sv.Detections.from_inference(results)

    # then
    assert len(detection_results) == len(
        expected_detections
    ), "Expected the same number of boxes"
    assert np.allclose(
        detection_results.xyxy, expected_detections.xyxy, atol=PIXEL_TOLERANCE
    ), "Boxes coordinates detection differ"
    assert np.allclose(
        detection_results.confidence,
        expected_detections.confidence,
        atol=CONFIDENCE_TOLERANCE,
    ), "Confidences differ"
    assert np.allclose(
        detection_results.class_id, expected_detections.class_id
    ), "Classes id differ"


def test_yolo_world_v1_x_against_single_image(person_image: np.ndarray) -> None:
    # given
    model = YOLOWorld(model_id="yolo_world/x")
    model.set_classes(["person", "bag", "shoe"])
    expected_detections = sv.Detections(
        xyxy=np.array(
            [
                [274.19, 160.17, 358.73, 378.62],
                [341.56, 275.65, 370.53, 332.55],
                [319.11, 365.47, 331.18, 378.43],
                [300.87, 354.24, 317.05, 365.83],
            ]
        ),
        confidence=np.array([0.951, 0.63286, 0.22856, 0.18254]),
        class_id=np.array([0, 1, 2, 2]),
    )

    # when
    results = model.infer(person_image, confidence=0.03).dict(
        by_alias=True, exclude_none=True
    )
    detection_results = sv.Detections.from_inference(results)

    # then
    assert len(detection_results) == len(
        expected_detections
    ), "Expected the same number of boxes"
    assert np.allclose(
        detection_results.xyxy, expected_detections.xyxy, atol=PIXEL_TOLERANCE
    ), "Boxes coordinates detection differ"
    assert np.allclose(
        detection_results.confidence,
        expected_detections.confidence,
        atol=CONFIDENCE_TOLERANCE,
    ), "Confidences differ"
    assert np.allclose(
        detection_results.class_id, expected_detections.class_id
    ), "Classes id differ"


def test_yolo_world_v2_s_against_single_image(person_image: np.ndarray) -> None:
    # given
    model = YOLOWorld(model_id="yolo_world/v2-s")
    model.set_classes(["person", "bag", "shoe"])
    expected_detections = sv.Detections(
        xyxy=np.array(
            [
                [273.8, 160.44, 358.38, 378.8],
                [340.7, 279.38, 370.77, 333.23],
                [319.09, 365.05, 331.67, 378.05],
                [299.7, 353.88, 318.19, 368.31],
            ]
        ),
        confidence=np.array([0.92775, 0.46766, 0.22858, 0.19605]),
        class_id=np.array([0, 1, 2, 2]),
    )

    # when
    results = model.infer(person_image, confidence=0.03).dict(
        by_alias=True, exclude_none=True
    )
    detection_results = sv.Detections.from_inference(results)

    # then
    assert len(detection_results) == len(
        expected_detections
    ), "Expected the same number of boxes"
    assert np.allclose(
        detection_results.xyxy, expected_detections.xyxy, atol=PIXEL_TOLERANCE
    ), "Boxes coordinates detection differ"
    assert np.allclose(
        detection_results.confidence,
        expected_detections.confidence,
        atol=CONFIDENCE_TOLERANCE,
    ), "Confidences differ"
    assert np.allclose(
        detection_results.class_id, expected_detections.class_id
    ), "Classes id differ"


def test_yolo_world_v2_m_against_single_image(person_image: np.ndarray) -> None:
    # given
    model = YOLOWorld(model_id="yolo_world/v2-m")
    model.set_classes(["person", "bag", "shoe"])
    expected_detections = sv.Detections(
        xyxy=np.array(
            [
                [274.42, 160.3, 358.54, 378.19],
                [340.04, 273.48, 370.63, 332.29],
                [318.64, 365.07, 331.66, 378.3],
                [299.92, 353.52, 316.1, 365.73],
            ]
        ),
        confidence=np.array([0.92252, 0.31467, 0.27939, 0.25805]),
        class_id=np.array([0, 1, 2, 2]),
    )

    # when
    results = model.infer(person_image, confidence=0.03).dict(
        by_alias=True, exclude_none=True
    )
    detection_results = sv.Detections.from_inference(results)
    # then
    assert len(detection_results) == len(
        expected_detections
    ), "Expected the same number of boxes"
    assert np.allclose(
        detection_results.xyxy, expected_detections.xyxy, atol=PIXEL_TOLERANCE
    ), "Boxes coordinates detection differ"
    assert np.allclose(
        detection_results.confidence,
        expected_detections.confidence,
        atol=CONFIDENCE_TOLERANCE,
    ), "Confidences differ"
    assert np.allclose(
        detection_results.class_id, expected_detections.class_id
    ), "Classes id differ"


def test_yolo_world_v2_l_against_single_image(person_image: np.ndarray) -> None:
    # given
    model = YOLOWorld(model_id="yolo_world/v2-l")
    model.set_classes(["person", "bag", "shoe"])
    expected_detections = sv.Detections(
        xyxy=np.array(
            [
                [274.18, 160.19, 358.83, 378.69],
                [340.96, 281.83, 370.56, 332.45],
                [318.8, 365.77, 331.11, 378.45],
                [300.43, 354.22, 316.5, 368.4],
            ]
        ),
        confidence=np.array([0.95149, 0.19823, 0.12165, 0.096857]),
        class_id=np.array([0, 1, 2, 2]),
    )

    # when
    results = model.infer(person_image, confidence=0.03).dict(
        by_alias=True, exclude_none=True
    )
    detection_results = sv.Detections.from_inference(results)

    # then
    assert len(detection_results) == len(
        expected_detections
    ), "Expected the same number of boxes"
    assert np.allclose(
        detection_results.xyxy, expected_detections.xyxy, atol=PIXEL_TOLERANCE
    ), "Boxes coordinates detection differ"
    assert np.allclose(
        detection_results.confidence,
        expected_detections.confidence,
        atol=CONFIDENCE_TOLERANCE,
    ), "Confidences differ"
    assert np.allclose(
        detection_results.class_id, expected_detections.class_id
    ), "Classes id differ"


def test_yolo_world_v2_x_against_single_image(person_image: np.ndarray) -> None:
    # given
    model = YOLOWorld(model_id="yolo_world/v2-x")
    model.set_classes(["person", "bag", "shoe"])
    expected_detections = sv.Detections(
        xyxy=np.array(
            [
                [274.59, 160.12, 359.33, 379.37],
                [341.31, 277.65, 370.6, 332.26],
                [318.86, 365.56, 331.56, 378.71],
                [300.91, 353.9, 316.95, 365.79],
            ]
        ),
        confidence=np.array([0.96406, 0.14249, 0.079248, 0.072569]),
        class_id=np.array([0, 1, 2, 2]),
    )

    # when
    results = model.infer(person_image, confidence=0.03).dict(
        by_alias=True, exclude_none=True
    )
    detection_results = sv.Detections.from_inference(results)

    # then
    assert len(detection_results) == len(
        expected_detections
    ), "Expected the same number of boxes"
    assert np.allclose(
        detection_results.xyxy, expected_detections.xyxy, atol=PIXEL_TOLERANCE
    ), "Boxes coordinates detection differ"
    assert np.allclose(
        detection_results.confidence,
        expected_detections.confidence,
        atol=CONFIDENCE_TOLERANCE,
    ), "Confidences differ"
    assert np.allclose(
        detection_results.class_id, expected_detections.class_id
    ), "Classes id differ"
