import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.transformations.polygon_simplification import (
    calculate_simplified_polygon,
)


@pytest.mark.parametrize(
    "polygon, required_number_of_vertices, expected_simplified_polygon",
    [
        (
            np.array([[10, 1], [10, 10], [20, 10], [20, 1]]),
            4,
            np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
        ),
        (
            np.roll(np.array([[10, 1], [10, 10], [20, 10], [20, 1]]), shift=2),
            4,
            np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
        ),
        (
            np.array([[10, 10], [10, 1], [20, 1], [20, 10]]),
            4,
            np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
        ),
        (
            np.roll(np.array([[10, 10], [10, 1], [20, 1], [20, 10]]), shift=2),
            4,
            np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
        ),
        (
            np.array([[10, 10], [10, 5], [10, 1], [15, 1], [20, 1], [20, 5], [20, 10]]),
            4,
            np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
        ),
        (
            np.array([[10, 10], [5, 5], [10, 1], [15, 1], [20, 1], [25, 5], [20, 10]]),
            4,
            np.array([[5, 5], [10, 10], [25, 5], [20, 1]])
        ),
        (  # drop valley
            np.roll(np.array([[10, 10], [10, 1], [15, 1], [15, 9], [16, 1], [20, 1], [20, 10]]), shift=2),
            4,
            np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
        ),
        (  # drop two valleys
            np.roll(np.array([[10, 10], [10, 1], [15, 1], [15, 9], [16, 1], [20, 1], [20, 10], [18, 10], [17, 2], [16, 10]]), shift=2),
            4,
            np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
        ),
    ]
)
def test_calculate_simplified_polygon(polygon, required_number_of_vertices, expected_simplified_polygon):
    mask = sv.polygon_to_mask(
        polygon=polygon,
        resolution_wh=(np.max(polygon, axis=0) + 10)
    )
    simplified_polygon = calculate_simplified_polygon(
        mask=mask,
        required_number_of_vertices=required_number_of_vertices,
    )
    assert np.allclose(simplified_polygon, expected_simplified_polygon)
