from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.transformations.perspective_correction import (
    pick_largest_perspective_polygons,
    sort_polygon_vertices_clockwise,
    roll_polygon_vertices_to_start_from_leftmost_bottom,
    extend_perspective_polygon,
    generate_transformation_matrix,
)


PERSPECTIVE_POLYGON_LIST = [[1, 1], [10, 1], [10, 10], [1, 10]]
LARGE_PERSPECTIVE_POLYGON_LIST = [[1, 1], [100, 1], [100, 100], [1, 100]]
PERSPECTIVE_POLYGON_NP_ARRAY = np.array(PERSPECTIVE_POLYGON_LIST)
LARGE_PERSPECTIVE_POLYGON_NP_ARRAY = np.array(LARGE_PERSPECTIVE_POLYGON_LIST)


@pytest.mark.parametrize(
    "perspective_polygons_batch, expected_exception",
    [
        (1, pytest.raises(ValueError, match="Unexpected type of input")),
        ("cat", pytest.raises(ValueError, match="Unexpected type of input")),
        (np.array([]), pytest.raises(ValueError, match="Unexpected type of input")),
        (np.array([1, 2, 3, 4]), pytest.raises(ValueError, match="Unexpected type of input")),
        (PERSPECTIVE_POLYGON_NP_ARRAY, pytest.raises(ValueError, match="Unexpected type of input")),
        ([], pytest.raises(ValueError, match="Unexpected empty batch")),
        ([[], []], pytest.raises(ValueError, match="Unexpected empty batch element")),
        ([np.array([]), np.array([])], pytest.raises(ValueError, match="Unexpected empty batch element")),
        ([np.array([[1, 2], [3, 4], [5, 6]])], pytest.raises(ValueError, match="Unexpected shape of batch element")),
        ([[PERSPECTIVE_POLYGON_NP_ARRAY], []], pytest.raises(ValueError, match="Unexpected empty batch element")),
        ([PERSPECTIVE_POLYGON_NP_ARRAY, np.array([])], pytest.raises(ValueError, match="Unexpected empty batch element")),
        ([[PERSPECTIVE_POLYGON_LIST], []], pytest.raises(ValueError, match="Unexpected empty batch element")),
        ([PERSPECTIVE_POLYGON_LIST, []], pytest.raises(ValueError, match="Unexpected empty batch element")),
        ([1, 2, 3, 4], pytest.raises(ValueError, match="Unexpected type of batch element")),
        (PERSPECTIVE_POLYGON_LIST, pytest.raises(ValueError, match="No batch element consists of 4 vertices")),
        ([[1, 2], [3, 4], [5, 6]], pytest.raises(ValueError, match="No batch element consists of 4 vertices")),
        ([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], pytest.raises(ValueError, match="No batch element consists of 4 vertices")),
        ([[], []], pytest.raises(ValueError, match="Unexpected empty batch element")),
        ([PERSPECTIVE_POLYGON_NP_ARRAY], does_not_raise()),
        ([[PERSPECTIVE_POLYGON_NP_ARRAY]], does_not_raise()),
        ([[PERSPECTIVE_POLYGON_NP_ARRAY, []]], does_not_raise()),
        ([PERSPECTIVE_POLYGON_LIST], does_not_raise()),
        ([[PERSPECTIVE_POLYGON_LIST]], does_not_raise()),
        ([[PERSPECTIVE_POLYGON_LIST, []]], does_not_raise()),
    ]
)
def test_pick_largest_perspective_polygons_exceptions(perspective_polygons_batch: Any, expected_exception):
    with expected_exception:
        pick_largest_perspective_polygons(perspective_polygons_batch=perspective_polygons_batch)


@pytest.mark.parametrize(
    "perspective_polygons_batch, expected_largest_polygons",
    [
        ([PERSPECTIVE_POLYGON_LIST], [PERSPECTIVE_POLYGON_NP_ARRAY]),
        ([PERSPECTIVE_POLYGON_LIST, PERSPECTIVE_POLYGON_LIST], [PERSPECTIVE_POLYGON_NP_ARRAY, PERSPECTIVE_POLYGON_NP_ARRAY]),
        ([PERSPECTIVE_POLYGON_LIST, LARGE_PERSPECTIVE_POLYGON_LIST], [PERSPECTIVE_POLYGON_NP_ARRAY, LARGE_PERSPECTIVE_POLYGON_NP_ARRAY]),
        ([[PERSPECTIVE_POLYGON_LIST, PERSPECTIVE_POLYGON_LIST]], [PERSPECTIVE_POLYGON_NP_ARRAY]),
        ([[PERSPECTIVE_POLYGON_LIST, LARGE_PERSPECTIVE_POLYGON_LIST]], [LARGE_PERSPECTIVE_POLYGON_NP_ARRAY]),
        (
            [
                [PERSPECTIVE_POLYGON_LIST, LARGE_PERSPECTIVE_POLYGON_LIST],
                [PERSPECTIVE_POLYGON_LIST, LARGE_PERSPECTIVE_POLYGON_LIST]
            ],
            [
                LARGE_PERSPECTIVE_POLYGON_NP_ARRAY,
                LARGE_PERSPECTIVE_POLYGON_NP_ARRAY
            ]
        ),
    ]
)
def test_pick_largest_perspective_polygons(perspective_polygons_batch, expected_largest_polygons):
    largest_polygons = pick_largest_perspective_polygons(perspective_polygons_batch=perspective_polygons_batch)
    assert isinstance(largest_polygons, list)
    assert len(largest_polygons) == len(expected_largest_polygons)
    assert all(np.array_equal(l, e) for l, e in zip(largest_polygons, expected_largest_polygons))


COUNTER_CLOCKWISE_POLYGON = np.array([[1, 1], [1, 10], [10, 10], [10, 1]])
CLOCKWISE_POLYGON = np.array([[1, 10], [1, 1], [10, 1], [10, 10]])


@pytest.mark.parametrize(
    "polygon, expected_clockwise_polygon",
    [
        (COUNTER_CLOCKWISE_POLYGON, CLOCKWISE_POLYGON),
        (np.roll(COUNTER_CLOCKWISE_POLYGON, 2), CLOCKWISE_POLYGON),
        (np.roll(COUNTER_CLOCKWISE_POLYGON, 4), CLOCKWISE_POLYGON),
        (np.roll(COUNTER_CLOCKWISE_POLYGON, 6), CLOCKWISE_POLYGON),
    ]
)
def test_sort_polygon_vertices_clockwise(polygon: np.ndarray, expected_clockwise_polygon: np.array):
    clockwise_polygon = sort_polygon_vertices_clockwise(polygon=polygon)
    assert np.array_equal(clockwise_polygon, expected_clockwise_polygon)


POLYGON = np.array([[1, 1], [10, 1], [10, 10], [1, 10]])
ROLLED_POLYGON = np.array([[1, 10], [1, 1], [10, 1], [10, 10]])


@pytest.mark.parametrize(
    "polygon, expected_rolled_polygon",
    [
        (POLYGON, ROLLED_POLYGON),
        (np.roll(POLYGON, 2), ROLLED_POLYGON),
        (np.roll(POLYGON, 4), ROLLED_POLYGON),
        (np.roll(POLYGON, 6), ROLLED_POLYGON),
    ]
)
def test_roll_polygon_vertices_to_start_from_leftmost_bottom(polygon: np.ndarray, expected_rolled_polygon: np.array):
    rolled_polygon = roll_polygon_vertices_to_start_from_leftmost_bottom(polygon=polygon)
    assert np.array_equal(rolled_polygon, expected_rolled_polygon)


ORIG_POLYGON = np.array([[100, 110], [100, 100], [110, 100], [110, 110]])
DETECTIONS_WITHIN_POLYGON = sv.Detections(xyxy=np.array([[105, 105, 106, 106]]))
DETECTIONS_ON_THE_CORNERS = sv.Detections(xyxy=np.array([[99, 99, 101, 100], [109, 99, 111, 100], [99, 109, 101, 110], [109, 109, 111, 110]]))
DETECTIONS_EXTENDING_FROM_THE_LEFT = sv.Detections(xyxy=np.array([[90, 105, 92, 106]]))
DETECTIONS_EXTENDING_FROM_THE_RIGHT = sv.Detections(xyxy=np.array([[120, 105, 122, 106]]))
DETECTIONS_EXTENDING_FROM_THE_TOP = sv.Detections(xyxy=np.array([[105, 90, 106, 92]]))
DETECTIONS_EXTENDING_FROM_THE_BOTTOM = sv.Detections(xyxy=np.array([[105, 120, 106, 122]]))


@pytest.mark.parametrize(
    "polygon, detections, bbox_position, expected_extended_polygon",
    [
        (ORIG_POLYGON, sv.Detections.empty(), "", ORIG_POLYGON),
        (ORIG_POLYGON, sv.Detections.empty(), sv.Position.BOTTOM_CENTER, ORIG_POLYGON),
        (ORIG_POLYGON, DETECTIONS_WITHIN_POLYGON, sv.Position.BOTTOM_CENTER, ORIG_POLYGON),
        (ORIG_POLYGON, DETECTIONS_ON_THE_CORNERS, sv.Position.BOTTOM_CENTER, ORIG_POLYGON),
        (
            ORIG_POLYGON,
            sv.Detections.merge([DETECTIONS_WITHIN_POLYGON, DETECTIONS_EXTENDING_FROM_THE_LEFT]),
            sv.Position.BOTTOM_CENTER,
            np.array([[91, 110], [91, 100], [110, 100], [110, 110]])
        ),
        (
            ORIG_POLYGON,
            sv.Detections.merge([DETECTIONS_WITHIN_POLYGON, DETECTIONS_EXTENDING_FROM_THE_RIGHT]),
            sv.Position.BOTTOM_CENTER,
            np.array([[100, 110], [100, 100], [121, 100], [121, 110]])
        ),
        (
            ORIG_POLYGON,
            sv.Detections.merge([DETECTIONS_WITHIN_POLYGON, DETECTIONS_EXTENDING_FROM_THE_TOP]),
            sv.Position.BOTTOM_CENTER,
            np.array([[100, 110], [100, 92], [110, 92], [110, 110]])
        ),
        (
            ORIG_POLYGON,
            sv.Detections.merge([DETECTIONS_WITHIN_POLYGON, DETECTIONS_EXTENDING_FROM_THE_BOTTOM]),
            sv.Position.BOTTOM_CENTER,
            np.array([[100, 122], [100, 100], [110, 100], [110, 122]])
        ),
        (
            ORIG_POLYGON,
            sv.Detections.merge([
                DETECTIONS_WITHIN_POLYGON,
                DETECTIONS_EXTENDING_FROM_THE_LEFT,
                DETECTIONS_EXTENDING_FROM_THE_RIGHT,
                DETECTIONS_EXTENDING_FROM_THE_TOP,
                DETECTIONS_EXTENDING_FROM_THE_BOTTOM
            ]),
            sv.Position.BOTTOM_CENTER,
            np.array([[91, 122], [91, 92], [121, 92], [121, 122]])
        ),
    ]
)
def test_extend_perspective_polygon(polygon: np.ndarray, detections: sv.Detections, bbox_position: sv.Position, expected_extended_polygon: np.array):
    extended_polygon = extend_perspective_polygon(
        polygon=polygon,
        detections=detections,
        bbox_position=bbox_position,
    )
    assert np.array_equal(extended_polygon, expected_extended_polygon)


@pytest.mark.parametrize(
    "src_polygon, detections, bbox_position, transformed_rect_width, transformed_rect_height, expected_transformation_matrix",
    [
        (
            ORIG_POLYGON,
            sv.Detections.merge([
                DETECTIONS_WITHIN_POLYGON,
                DETECTIONS_ON_THE_CORNERS,
                DETECTIONS_EXTENDING_FROM_THE_LEFT,
                DETECTIONS_EXTENDING_FROM_THE_RIGHT,
                DETECTIONS_EXTENDING_FROM_THE_TOP,
                DETECTIONS_EXTENDING_FROM_THE_BOTTOM
            ]),
            sv.Position.BOTTOM_CENTER,
            1000,
            1000,
            np.array([[       33.3, -4.7325e-16,     -3030.3],
                      [          0,        33.3,     -3063.6],
                      [ 7.5867e-18,  1.8967e-18,           1]])
        ),
    ]
)
def test_generate_transformation_matrix(src_polygon, detections, bbox_position, transformed_rect_width, transformed_rect_height, expected_transformation_matrix):
    transformation_matrix = generate_transformation_matrix(
        src_polygon=src_polygon,
        detections=detections,
        transformed_rect_width=transformed_rect_width,
        transformed_rect_height=transformed_rect_height,
        detections_anchor=bbox_position,
    )

    assert np.allclose(transformation_matrix, expected_transformation_matrix)


    