import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.transformations.bounding_rect.v1 import (
    calculate_minimum_bounding_rectangle,
)


def test_calculate_minimum_bounding_rectangle():
    # given
    polygon = np.array(
        [
            [10, 10],
            [10, 1],
            [20, 1],
            [20, 10],
            [15, 5]
        ]
    )
    mask = sv.polygon_to_mask(
        polygon=polygon, resolution_wh=(np.max(polygon, axis=0) + 10)
    )

    # when
    box, width, height, angle = calculate_minimum_bounding_rectangle(mask=mask)

    # then
    expected_box = np.array([[10, 1], [20, 1], [20, 10], [10, 10]])
    assert np.allclose(box, expected_box), (
        f"Expected bounding box to be {expected_box}, but got {box}"
    )
    assert np.isclose(width, 9), f"Expected width to be 9, but got {width}"
    assert np.isclose(height, 10), f"Expected height to be 10, but got {height}"
    assert angle == 90 or angle == -90, f"Expected angle to be 90 or -90, but got {angle}"
