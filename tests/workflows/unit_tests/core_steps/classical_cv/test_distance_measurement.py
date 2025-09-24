import math

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.classical_cv.distance_measurement.v1 import (
    OUTPUT_KEY_CENTIMETER,
    OUTPUT_KEY_PIXEL,
    DistanceMeasurementBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import Batch


def test_distance_measurement_block_pixel_ratio_vertical():
    # given
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [3409.4, 387.72, 3652.7, 1875.3],
                [622.75, 1791, 3321.7, 2832.8],
                [1592.4, 263.17, 2677.2, 985.28],
            ]
        ),
        confidence=np.array([0.83615, 0.8006, 0.62719]),
        mask=None,
        class_id=np.array([2, 0, 1]),
        data={
            "class_name": np.array(
                ["fork", "computer keyboard", "mouse"], dtype="<U17"
            ),
        },
    )
    calibration_type = "pixel to centimeter"
    object_1_class_name = "mouse"
    object_2_class_name = "computer keyboard"
    pixel_ratio = 86.9
    reference_axis = "vertical"
    reference_object_class_name = None
    reference_height = None
    reference_width = None

    # when
    block = DistanceMeasurementBlockV1()
    result = block.run(
        predictions=predictions,
        calibration_method=calibration_type,
        object_1_class_name=object_1_class_name,
        object_2_class_name=object_2_class_name,
        pixel_ratio=pixel_ratio,
        reference_axis=reference_axis,
        reference_object_class_name=reference_object_class_name,
        reference_height=reference_height,
        reference_width=reference_width,
    )

    # then
    expected_result = {OUTPUT_KEY_CENTIMETER: 9.275028768, OUTPUT_KEY_PIXEL: 806}
    tolerance = 1e-5  # Define a tolerance level

    # Compare the float values within the specified tolerance
    assert math.isclose(
        result[OUTPUT_KEY_CENTIMETER],
        expected_result[OUTPUT_KEY_CENTIMETER],
        rel_tol=tolerance,
    ), f"Expected {expected_result[OUTPUT_KEY_CENTIMETER]} cm, but got {result[OUTPUT_KEY_CENTIMETER]} cm"

    # Compare the integer values directly
    assert (
        result[OUTPUT_KEY_PIXEL] == expected_result[OUTPUT_KEY_PIXEL]
    ), f"Expected {expected_result[OUTPUT_KEY_PIXEL]} pixels, but got {result[OUTPUT_KEY_PIXEL]} pixels"


def test_distance_measurement_block_pixel_ratio_horizontal():
    # given
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [1761.8, 483.12, 2971.4, 3573.8],
                [137.25, 1731.7, 983.1, 2993.6],
                [841, 1356, 1074, 1587],
            ]
        ),
        confidence=np.array([0.96508, 0.96155, 0.9497]),
        mask=None,
        class_id=np.array([0, 1, 0]),
        tracker_id=None,
        data={
            "class_name": np.array(
                ["computer keyboard", "mouse", "coin"], dtype="<U17"
            ),
        },
    )
    calibration_type = "pixel to centimeter"
    object_1_class_name = "mouse"
    object_2_class_name = "computer keyboard"
    pixel_ratio = 98
    reference_axis = "horizontal"
    reference_object_class_name = None
    reference_height = None
    reference_width = None

    # when
    block = DistanceMeasurementBlockV1()
    result = block.run(
        predictions=predictions,
        calibration_method=calibration_type,
        object_1_class_name=object_1_class_name,
        object_2_class_name=object_2_class_name,
        pixel_ratio=pixel_ratio,
        reference_axis=reference_axis,
        reference_object_class_name=reference_object_class_name,
        reference_height=reference_height,
        reference_width=reference_width,
    )

    # then
    expected_result = {"distance_cm": 7.948979591836735, "distance_pixel": 779}
    tolerance = 1e-3  # Define a tolerance level

    # Compare the float value within the specified tolerance
    assert math.isclose(
        result["distance_cm"], expected_result["distance_cm"], rel_tol=tolerance
    ), f"Expected {expected_result['distance_cm']} cm, but got {result['distance_cm']} cm"

    # Compare the integer value directly
    assert (
        result["distance_pixel"] == expected_result["distance_pixel"]
    ), f"Expected {expected_result['distance_pixel']} pixels, but got {result['distance_pixel']} pixels"


def test_distance_measurement_block_reference_object_vertical():
    # given
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [494.09, 1750.4, 3594.6, 2961.6],
                [1085.8, 135.34, 2350.1, 982.24],
                [2487, 836, 2723, 1067],
            ]
        ),
        confidence=np.array([0.97317, 0.90317, 0.92105]),
        mask=None,
        class_id=np.array([0, 1, 0]),
        data={
            "scaling_relative_to_root_parent": np.array([1, 1, 1]),
            "detection_id": np.array(
                [
                    "556ebe7b-3394-4174-93be-6e0112bf6cc9",
                    "4d2a0956-bf0b-4a0a-aa5a-175333cbee90",
                    "53d6c43f-6057-44e8-9901-3db1b0f42776",
                ],
                dtype="<U36",
            ),
            "class_name": np.array(
                ["computer keyboard", "mouse", "coin"], dtype="<U17"
            ),
        },
    )
    calibration_type = "reference object"
    object_1_class_name = "computer keyboard"
    object_2_class_name = "mouse"
    pixel_ratio = None
    reference_axis = "vertical"
    reference_object_class_name = "coin"
    reference_height = 2.426
    reference_width = 2.426

    # when
    block = DistanceMeasurementBlockV1()
    result = block.run(
        predictions=predictions,
        calibration_method=calibration_type,
        object_1_class_name=object_1_class_name,
        object_2_class_name=object_2_class_name,
        pixel_ratio=pixel_ratio,
        reference_axis=reference_axis,
        reference_object_class_name=reference_object_class_name,
        reference_height=reference_height,
        reference_width=reference_width,
    )

    # then
    expected_result = {"distance_cm": 7.979306209850107, "distance_pixel": 768}
    tolerance = 1e-3  # Define a tolerance level

    # Compare the float value within the specified tolerance
    assert math.isclose(
        result["distance_cm"], expected_result["distance_cm"], rel_tol=tolerance
    ), f"Expected {expected_result['distance_cm']} cm, but got {result['distance_cm']} cm"

    # Compare the integer value directly
    assert (
        result["distance_pixel"] == expected_result["distance_pixel"]
    ), f"Expected {expected_result['distance_pixel']} pixels, but got {result['distance_pixel']} pixels"


def test_distance_measurement_block_reference_object_horizontal():
    # given
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [1761.8, 483.12, 2971.4, 3573.8],
                [137.25, 1731.7, 983.1, 2993.6],
                [841, 1356, 1074, 1587],
            ]
        ),
        confidence=np.array([0.96508, 0.96155, 0.9497]),
        mask=None,
        class_id=np.array([0, 1, 0]),
        tracker_id=None,
        data={
            "class_name": np.array(
                ["computer keyboard", "mouse", "coin"], dtype="<U17"
            ),
        },
    )
    calibration_type = "reference object"
    object_1_class_name = "mouse"
    object_2_class_name = "computer keyboard"
    pixel_ratio = None
    reference_axis = "horizontal"
    reference_object_class_name = "coin"
    reference_height = 2.426
    reference_width = 2.426

    # when
    block = DistanceMeasurementBlockV1()
    result = block.run(
        predictions=predictions,
        calibration_method=calibration_type,
        object_1_class_name=object_1_class_name,
        object_2_class_name=object_2_class_name,
        pixel_ratio=pixel_ratio,
        reference_axis=reference_axis,
        reference_object_class_name=reference_object_class_name,
        reference_height=reference_height,
        reference_width=reference_width,
    )

    # then
    expected_result = {"distance_cm": 8.145922413793105, "distance_pixel": 779}
    tolerance = 1e-3  # Define a tolerance level

    # Compare the float value within the specified tolerance
    assert math.isclose(
        result["distance_cm"], expected_result["distance_cm"], rel_tol=tolerance
    ), f"Expected {expected_result['distance_cm']} cm, but got {result['distance_cm']} cm"

    # Compare the integer value directly
    assert (
        result["distance_pixel"] == expected_result["distance_pixel"]
    ), f"Expected {expected_result['distance_pixel']} pixels, but got {result['distance_pixel']} pixels"


def test_distance_measurement_block_reference_object_with_empty_reference_object():
    # given
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [1761.8, 483.12, 2971.4, 3573.8],
                [137.25, 1731.7, 983.1, 2993.6],
                [841, 1356, 1074, 1587],
            ]
        ),
        confidence=np.array([0.96508, 0.96155, 0.9497]),
        mask=None,
        class_id=np.array([0, 1, 0]),
        tracker_id=None,
        data={
            "class_name": np.array(
                ["computer keyboard", "mouse", "coin"], dtype="<U17"
            ),
        },
    )
    calibration_type = "reference object"
    object_1_class_name = "mouse"
    object_2_class_name = "computer keyboard"
    pixel_ratio = None
    reference_axis = "horizontal"
    reference_object_class_name = None
    reference_height = None
    reference_width = None

    # when
    block = DistanceMeasurementBlockV1()

    # when
    block = DistanceMeasurementBlockV1()
    with pytest.raises(
        expected_exception=ValueError,
        match="Reference class 'None' not found in predictions.",
    ):
        result = block.run(
            predictions=predictions,
            calibration_method=calibration_type,
            object_1_class_name=object_1_class_name,
            object_2_class_name=object_2_class_name,
            pixel_ratio=pixel_ratio,
            reference_axis=reference_axis,
            reference_object_class_name=reference_object_class_name,
            reference_height=reference_height,
            reference_width=reference_width,
        )


def test_distance_measurement_block_pixel_ratio_with_empty_pixel_ratio():
    # given
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [1761.8, 483.12, 2971.4, 3573.8],
                [137.25, 1731.7, 983.1, 2993.6],
                [841, 1356, 1074, 1587],
            ]
        ),
        confidence=np.array([0.96508, 0.96155, 0.9497]),
        mask=None,
        class_id=np.array([0, 1, 0]),
        tracker_id=None,
        data={
            "class_name": np.array(
                ["computer keyboard", "mouse", "coin"], dtype="<U17"
            ),
        },
    )
    calibration_type = "pixel to centimeter"
    object_1_class_name = "mouse"
    object_2_class_name = "computer keyboard"
    pixel_ratio = None
    reference_axis = "horizontal"
    reference_object_class_name = None
    reference_height = None
    reference_width = None

    # when
    block = DistanceMeasurementBlockV1()

    # when
    block = DistanceMeasurementBlockV1()
    with pytest.raises(
        expected_exception=ValueError,
        match="Pixel-to-centimeter ratio must be provided.",
    ):
        block.run(
            predictions=predictions,
            calibration_method=calibration_type,
            object_1_class_name=object_1_class_name,
            object_2_class_name=object_2_class_name,
            pixel_ratio=pixel_ratio,
            reference_axis=reference_axis,
            reference_object_class_name=reference_object_class_name,
            reference_height=reference_height,
            reference_width=reference_width,
        )


def test_distance_measurement_block_pixel_ratio_with_negative_pixel_ratio():
    # given
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [1761.8, 483.12, 2971.4, 3573.8],
                [137.25, 1731.7, 983.1, 2993.6],
                [841, 1356, 1074, 1587],
            ]
        ),
        confidence=np.array([0.96508, 0.96155, 0.9497]),
        mask=None,
        class_id=np.array([0, 1, 0]),
        tracker_id=None,
        data={
            "class_name": np.array(
                ["computer keyboard", "mouse", "coin"], dtype="<U17"
            ),
        },
    )
    calibration_type = "pixel to centimeter"
    object_1_class_name = "mouse"
    object_2_class_name = "computer keyboard"
    pixel_ratio = -4.20
    reference_axis = "horizontal"
    reference_object_class_name = None
    reference_height = None
    reference_width = None

    # when
    block = DistanceMeasurementBlockV1()
    with pytest.raises(
        expected_exception=ValueError,
        match="Pixel-to-centimeter ratio must be greater than zero.",
    ):
        block.run(
            predictions=predictions,
            calibration_method=calibration_type,
            object_1_class_name=object_1_class_name,
            object_2_class_name=object_2_class_name,
            pixel_ratio=pixel_ratio,
            reference_axis=reference_axis,
            reference_object_class_name=reference_object_class_name,
            reference_height=reference_height,
            reference_width=reference_width,
        )


def test_distance_measurement_block_with_inexistent_object_1_class_name():
    # given
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [3409.4, 387.72, 3652.7, 1875.3],
                [622.75, 1791, 3321.7, 2832.8],
                [1592.4, 263.17, 2677.2, 985.28],
            ]
        ),
        confidence=np.array([0.83615, 0.8006, 0.62719]),
        mask=None,
        class_id=np.array([2, 0, 1]),
        data={
            "class_name": np.array(
                ["fork", "computer keyboard", "mouse"], dtype="<U17"
            ),
        },
    )
    calibration_type = "pixel to centimeter"
    object_1_class_name = "rat"
    object_2_class_name = "computer keyboard"
    pixel_ratio = 86.9
    reference_axis = "vertical"
    reference_object_class_name = None
    reference_height = None
    reference_width = None

    # when
    block = DistanceMeasurementBlockV1()
    with pytest.raises(
        expected_exception=ValueError,
        match="Reference class 'rat' or 'computer keyboard' not found in predictions.",
    ):
        block.run(
            predictions=predictions,
            calibration_method=calibration_type,
            object_1_class_name=object_1_class_name,
            object_2_class_name=object_2_class_name,
            pixel_ratio=pixel_ratio,
            reference_axis=reference_axis,
            reference_object_class_name=reference_object_class_name,
            reference_height=reference_height,
            reference_width=reference_width,
        )


def test_distance_measurement_block_with_horizontal_overlapping_target_objects():
    predictions = sv.Detections(
        xyxy=np.array([[2294.2, 1951.2, 3577, 2813], [352.11, 311.64, 3463, 1516.6]]),
        mask=None,
        confidence=np.array([0.98209, 0.96483]),
        class_id=np.array([1, 0]),
        tracker_id=None,
        data={"class_name": np.array(["mouse", "computer keyboard"], dtype="<U17")},
    )
    calibration_type = "pixel to centimeter"
    object_1_class_name = "mouse"
    object_2_class_name = "computer keyboard"
    pixel_ratio = 86.9
    reference_axis = "horizontal"
    reference_object_class_name = None
    reference_height = None
    reference_width = None

    # when
    block = DistanceMeasurementBlockV1()
    result = block.run(
        predictions=predictions,
        calibration_method=calibration_type,
        object_1_class_name=object_1_class_name,
        object_2_class_name=object_2_class_name,
        pixel_ratio=pixel_ratio,
        reference_axis=reference_axis,
        reference_object_class_name=reference_object_class_name,
        reference_height=reference_height,
        reference_width=reference_width,
    )

    # then
    expected_result = {"distance_cm": 0, "distance_pixel": 0}
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


def test_distance_measurement_block_with_vertical_overlapping_target_objects():

    predictions = sv.Detections(
        xyxy=np.array(
            [
                [3249.1, 1337.4, 4031.7, 2464.5],
                [315.94, 638.84, 3007.9, 1693.2],
            ]
        ),
        mask=None,
        confidence=np.array([0.98184, 0.97315]),
        class_id=np.array([1, 0]),
        data={"class_name": np.array(["mouse", "computer keyboard"], dtype="<U17")},
    )
    calibration_type = "pixel to centimeter"
    object_1_class_name = "mouse"
    object_2_class_name = "computer keyboard"
    pixel_ratio = 86.9
    reference_axis = "vertical"
    reference_object_class_name = None
    reference_height = None
    reference_width = None

    # when
    block = DistanceMeasurementBlockV1()
    result = block.run(
        predictions=predictions,
        calibration_method=calibration_type,
        object_1_class_name=object_1_class_name,
        object_2_class_name=object_2_class_name,
        pixel_ratio=pixel_ratio,
        reference_axis=reference_axis,
        reference_object_class_name=reference_object_class_name,
        reference_height=reference_height,
        reference_width=reference_width,
    )

    # then
    expected_result = {OUTPUT_KEY_CENTIMETER: 0, OUTPUT_KEY_PIXEL: 0}
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
