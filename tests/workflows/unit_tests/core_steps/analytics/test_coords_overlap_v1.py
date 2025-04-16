import pytest

from inference.core.workflows.core_steps.analytics.overlap.v1 import (
    OverlapBlockV1,
)

def test_coords_overlap():
    assert not OverlapBlockV1.coords_overlap(
        [0, 0, 20, 20], [15, 15, 35, 35], "Center Overlap"
    )
    assert not OverlapBlockV1.coords_overlap(
        [10, 10, 20, 20], [30, 30, 40, 40], "Any Overlap"
    )
    assert OverlapBlockV1.coords_overlap(
        [20, 20, 30, 30], [15, 15, 35, 35], "Center Overlap"
    )
    assert OverlapBlockV1.coords_overlap(
        [0, 0, 20, 20], [15, 15, 35, 35], "Any Overlap"
    )