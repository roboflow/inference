"""Zone geometry shared by ``roboflow_core/line_counter@v2`` and ``roboflow_core/time_in_zone@v3``.

Expected values are derived by hand from latest V1
``core_steps/analytics/_zone_geometry.py``; the task's randomized parity
probe compares the predicates with V1 directly.
"""

import math

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.analytics import zones

DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "mps",
        id="mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="needs MPS"
        ),
    ),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
]
# Horizontal line drawn left to right at y = 50: above is +1 (in).
LINE = ((0.0, 50.0), (100.0, 50.0))
SQUARE = [[10, 10], [10, 30], [30, 30], [30, 10]]


def test_parse_line_accepts_lists_tuples_and_arrays():
    expected = ((1.0, 2.0), (3.0, 4.5))

    assert zones.parse_line([[1, 2], [3, 4.5]]) == expected
    assert zones.parse_line(((1, 2), (3, 4.5))) == expected
    assert zones.parse_line(np.array([[1, 2], [3, 4.5]])) == expected
    assert zones.parse_line([[np.int64(1), np.float32(2)], [3, 4.5]]) == expected


@pytest.mark.parametrize(
    "value, message",
    [
        ([[0, 0]], "exactly 2 points"),
        ([[0, 0], [1, 1], [2, 2]], "exactly 2 points"),
        ([], "exactly 2 points"),
        ("[[0, 0], [1, 1]]", "must be a list"),
        (None, "must be a list"),
        ([[0, 0], [1]], r"line_segment\[1\] must be a point"),
        ([[0, 0], [1, 2, 3]], "must be a point"),
        ([[0, 0], ["1", 2]], "must be numbers"),
        ([[0, 0], [True, 2]], "must be numbers"),
        ([[0, 0], [np.bool_(True), 2]], "must be numbers"),
        ([[0, 0], [math.nan, 2]], "must be finite"),
        ([[0, 0], [math.inf, 2]], "must be finite"),
        ([[5, 5], [5.0, 5.0]], "two different points"),
    ],
)
def test_parse_line_rejects_bad_values(value, message):
    with pytest.raises(ValueError, match=message):
        zones.parse_line(value)


def test_parse_line_error_names_the_parameter():
    with pytest.raises(ValueError, match=r"^line\[0\] must be a point"):
        zones.parse_line([[0], [1, 1]], name="line")


def test_parse_polygons_reads_one_polygon_a_list_of_polygons_or_none():
    one = zones.parse_polygons(SQUARE)
    many = zones.parse_polygons([SQUARE, [[0, 0], [5, 0], [0, 5]]])

    assert one == (tuple((float(x), float(y)) for x, y in SQUARE),)
    assert len(many) == 2 and many[0] == one[0]
    assert many[1] == ((0.0, 0.0), (5.0, 0.0), (0.0, 5.0))
    assert zones.parse_polygons([]) == ()
    assert zones.parse_polygons(np.array(SQUARE)) == one
    assert zones.parse_polygons([np.array(SQUARE)]) == one


@pytest.mark.parametrize(
    "value, message",
    [
        ([[0, 0], [1, 1]], r"zone must be a polygon of at least 3"),
        ([SQUARE, [[0, 0], [1, 1]]], r"zone\[1\] must be a polygon of at least 3"),
        ([SQUARE, []], r"zone\[1\] must be a polygon of at least 3"),
        ([[0, 0], [1, 1], [2, "x"]], r"zone\[2\] coordinates must be numbers"),
        ([[0, 0], [1, 1], [2, math.nan]], "must be finite"),
        ([[0, 0], [1, False], [2, 2]], "must be numbers"),
        ([[[0, 0], [1, 1], [2]]], r"zone\[0\]\[2\] must be a point"),
        ("zone", "must be a list"),
        (5, "must be a list"),
    ],
)
def test_parse_polygons_rejects_bad_values(value, message):
    with pytest.raises(ValueError, match=message):
        zones.parse_polygons(value)


def test_parse_anchor_accepts_every_box_anchor_and_rejects_others():
    for anchor in zones.ANCHORS:
        assert zones.parse_anchor(anchor) == anchor

    for bad in ("CENTER_OF_MASS", "center", None, 3):
        with pytest.raises(ValueError, match="triggering_anchor must be one of"):
            zones.parse_anchor(bad)


@pytest.mark.parametrize("device", DEVICES)
def test_anchor_points_follow_v1_positions_and_keep_device_and_dtype(device):
    xyxy = torch.tensor([[10.0, 20.0, 30.0, 60.0]], device=device)

    points = zones.anchor_points(xyxy, zones.ANCHORS)

    assert points.shape == (len(zones.ANCHORS), 1, 2)
    assert points.device.type == device and points.dtype == torch.float32
    by_name = dict(zip(zones.ANCHORS, points[:, 0].cpu().tolist()))
    assert by_name == {
        "CENTER": [20.0, 40.0],
        "CENTER_LEFT": [10.0, 40.0],
        "CENTER_RIGHT": [30.0, 40.0],
        "TOP_CENTER": [20.0, 20.0],
        "TOP_LEFT": [10.0, 20.0],
        "TOP_RIGHT": [30.0, 20.0],
        "BOTTOM_LEFT": [10.0, 60.0],
        "BOTTOM_CENTER": [20.0, 60.0],
        "BOTTOM_RIGHT": [30.0, 60.0],
    }


def test_anchor_points_accept_noncontiguous_float64_boxes():
    wide = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], dtype=torch.float64)
    xyxy = wide[:, ::2]

    points = zones.anchor_points(xyxy, ("CENTER",))

    assert not xyxy.is_contiguous()
    assert points.dtype == torch.float64
    assert points.tolist() == [[[2.0, 4.0]]]


def test_anchor_points_of_no_rows_have_no_rows():
    points = zones.anchor_points(torch.zeros((0, 4)), ("CENTER", "TOP_LEFT"))

    assert points.shape == (2, 0, 2)


@pytest.mark.parametrize("device", DEVICES)
def test_line_sides_classify_left_right_straddling_and_beside(device):
    # Rows: above, below, straddling, on the line, beside the segment, end.
    xyxy = torch.tensor(
        [
            [10.0, 10.0, 20.0, 20.0],
            [10.0, 60.0, 20.0, 70.0],
            [10.0, 40.0, 20.0, 60.0],
            [10.0, 50.0, 20.0, 50.0],
            [110.0, 10.0, 120.0, 20.0],
            [90.0, 10.0, 100.0, 20.0],
        ],
        device=device,
    )
    points = zones.anchor_points(xyxy, zones.DEFAULT_LINE_ANCHORS)

    sides = zones.line_sides(points, LINE)

    assert sides.device.type == device and sides.dtype == torch.int8
    # On the line counts as right (cross >= 0), as V1; segment ends included.
    assert sides.cpu().tolist() == [1, -1, 0, -1, 0, 1]


def test_line_sides_follow_the_drawn_direction():
    above = zones.anchor_points(torch.tensor([[40.0, 10.0, 60.0, 20.0]]), ("CENTER",))
    reversed_line = (LINE[1], LINE[0])

    assert zones.line_sides(above, LINE).tolist() == [1]
    assert zones.line_sides(above, reversed_line).tolist() == [-1]


def test_line_sides_of_a_diagonal_line_use_the_exact_segment_extent():
    line = ((0.0, 0.0), (10.0, 10.0))
    points = torch.tensor([[[0.0, 5.0], [10.0, 0.0], [-1.0, 0.0], [10.0, 11.0]]])

    # The line runs down-right on screen; its left (+1) is the upper right.
    assert zones.line_sides(points, line).tolist() == [-1, 1, 0, 0]


def test_polygon_mask_matches_cv2_raster_with_boundary_inside():
    mask = zones.polygon_mask(zones.parse_polygons(SQUARE)[0])

    assert mask.dtype == torch.bool and mask.device.type == "cpu"
    assert mask.shape == (32, 32)
    assert mask[10, 10] and mask[30, 30] and mask[20, 20]
    assert not mask[9, 20] and not mask[31, 20] and not mask[20, 31]
    assert int(mask.sum()) == 21 * 21


def test_polygon_mask_truncates_fractional_vertices():
    polygon = ((10.9, 10.9), (10.9, 30.9), (30.9, 30.9), (30.9, 10.9))

    mask = zones.polygon_mask(polygon)

    assert torch.equal(mask, zones.polygon_mask(zones.parse_polygons(SQUARE)[0]))


def test_points_in_mask_uses_ceil_pixels_and_requires_every_anchor():
    mask = zones.polygon_mask(zones.parse_polygons(SQUARE)[0])
    points = torch.tensor(
        [
            [[29.5, 20.0], [30.5, 20.0], [9.5, 20.0], [-3.0, 20.0], [20.0, 900.0]],
            [[20.0, 20.0], [20.0, 20.0], [20.0, 20.0], [20.0, 20.0], [20.0, 20.0]],
        ],
        dtype=torch.float64,
    )

    inside = zones.points_in_mask(points, mask)

    # ceil(29.5) = 30 inside; ceil(30.5) = 31 outside; ceil(9.5) = 10 inside.
    assert inside.tolist() == [True, False, True, False, False]
    assert zones.points_in_mask(points[1:], mask).tolist() == [True] * 5


@pytest.mark.parametrize("device", DEVICES)
def test_points_in_mask_stays_on_the_points_device(device):
    mask = zones.polygon_mask(zones.parse_polygons(SQUARE)[0]).to(device)
    points = torch.tensor([[[20.0, 20.0], [50.0, 50.0]]], device=device)

    inside = zones.points_in_mask(points, mask)

    assert inside.device.type == device
    assert inside.cpu().tolist() == [True, False]
