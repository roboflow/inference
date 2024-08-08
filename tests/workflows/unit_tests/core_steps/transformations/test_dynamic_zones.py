import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.transformations.dynamic_zones.v1 import (
    calculate_simplified_polygon,
)


def test_dynamic_zones_no_simplification_required():
    # given
    polygon = np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
    mask = sv.polygon_to_mask(
        polygon=polygon, resolution_wh=(np.max(polygon, axis=0) + 10)
    )

    # when
    simplified_polygon = calculate_simplified_polygon(
        mask=mask,
        required_number_of_vertices=len(polygon),
    )

    # then
    assert np.allclose(
        simplified_polygon, np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
    ), "Polygon should not be modified if it already contains required number of vertices"


def test_dynamic_zones_resulting_in_convex_polygon():
    # given
    polygon = np.array(
        [
            [10, 10],
            [10, 1],
            [15, 1],
            [15, 9],
            [16, 1],
            [20, 1],
            [20, 10],
            [18, 10],
            [17, 2],
            [16, 10],
        ]
    )
    mask = sv.polygon_to_mask(
        polygon=polygon, resolution_wh=(np.max(polygon, axis=0) + 10)
    )

    # when
    simplified_polygon = calculate_simplified_polygon(
        mask=mask,
        required_number_of_vertices=4,
    )

    # then
    assert np.allclose(
        simplified_polygon, np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
    ), (
        "Valleys ([15, 1], [15, 9], [16, 1]) on the edge between [10, 1] and [20, 1] "
        "and ([18, 10], [17, 2], [16, 10]) on the edge between [20, 10] and [10, 10] "
        "should be dropped and shape of the polygon should remain unchanged"
    )


def test_dynamic_zones_drop_intermediate_points():
    # given
    polygon = np.array(
        np.array([[10, 10], [10, 5], [10, 1], [15, 1], [20, 1], [20, 5], [20, 10]])
    )
    mask = sv.polygon_to_mask(
        polygon=polygon, resolution_wh=(np.max(polygon, axis=0) + 10)
    )

    # when
    simplified_polygon = calculate_simplified_polygon(
        mask=mask,
        required_number_of_vertices=4,
    )

    # then
    assert np.allclose(
        simplified_polygon, np.array([[10, 1], [10, 10], [20, 10], [20, 1]])
    ), (
        "Intermediate points [10, 5] (between [10, 10] and [10, 1]), "
        "[15, 1] (between [10, 1] and [20, 1]) and [20, 5] (between [20, 1] and [20, 10]) "
        "should be dropped and shape of polygon should remain unchanged."
    )
