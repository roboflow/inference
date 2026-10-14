"""Zone drawings: polygon zones, line counters and reference paths.

The oracle is the latest V1 block run on the same pixels (``support.run_v1``).
Each V1 test gets a fresh V1 instance, so V1's mask caches cannot leak between
cases. V1 test modules are not imported.
"""

from typing import Any, Dict

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.visualizations.line_zone.v1 import (
    LineCounterZoneVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.polygon_zone.v1 import (
    PolygonZoneVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.reference_path.v1 import (
    ReferencePathVisualizationBlockV1,
)
from roboflow_workflows.v2.blocks.analytics.zones import parse_line, parse_polygons
from roboflow_workflows.v2.blocks.visualization.annotations.line_counter_visualization.v1 import LineCounterVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.polygon_zone_visualization.v1 import PolygonZoneVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations.reference_path_visualization.v1 import ReferencePathVisualizationBlock
from roboflow_workflows.v2.blocks.visualization.annotations._shared.zones import polygon_zones
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from tests.v2.unit.blocks.visualization.annotations.support import (
    image_of,
    pixels_of,
    run_v1,
    scene,
)

HEIGHT, WIDTH = 96, 128
TRIANGLE = [[10, 10], [90, 20], [40, 80]]
SQUARE = [[60, 30], [120, 30], [120, 90], [60, 90]]
LINE_DEFAULTS = {
    "color": "#5bb573",
    "thickness": 2,
    "text_thickness": 1,
    "text_scale": 1.0,
    "count_in": 0,
    "count_out": 0,
    "opacity": 0.3,
}


def run_v2(block_class: type, image: ImageData, **parameters: Any) -> ImageData:
    """Validate literal parameters as the compiler does, then ``run``."""
    params = spec_of(block_class).validate_params(
        {**parameters, "image": "$inputs.image"}
    )
    result = block_class().run(**{**dict(params), "image": image})

    return result["image"]


def v1_line(pixels: np.ndarray, **parameters: Any) -> np.ndarray:
    output = run_v1(
        LineCounterZoneVisualizationBlockV1, pixels, **{**LINE_DEFAULTS, **parameters}
    )

    return output


@pytest.mark.parametrize(
    "zone, color, opacity",
    [
        (TRIANGLE, "#5bb573", 0.3),
        ([TRIANGLE, SQUARE], "RED", 0.5),
        ([[-20, -10], [70, 5], [30, 140]], "rgb(10, 200, 250)", 1.0),
        ([[10.9, 10.2], [90.7, 20.5], [40.1, 80.99]], "WHITE", 0.7),
        (SQUARE, "bgr(0, 0, 255)", 0.0),
    ],
    ids=["triangle", "two-overlapping", "beyond-frame", "fractional", "transparent"],
)
def test_polygon_zone_equals_v1(zone: Any, color: str, opacity: float) -> None:
    # given
    pixels = scene(HEIGHT, WIDTH)

    # when
    output = run_v2(
        PolygonZoneVisualizationBlock,
        image_of(pixels),
        zone=zone,
        color=color,
        opacity=opacity,
    )

    # then
    expected = run_v1(
        PolygonZoneVisualizationBlockV1, pixels, zone=zone, color=color, opacity=opacity
    )
    assert np.array_equal(pixels_of(output), expected)


def test_polygon_zone_brightens_inside_and_saturates_like_v1() -> None:
    # given: a bright image, so pixel + opacity * color exceeds 255
    pixels = np.full((HEIGHT, WIDTH, 3), 230, dtype=np.uint8)

    # when
    output = pixels_of(
        run_v2(
            PolygonZoneVisualizationBlock, image_of(pixels), zone=SQUARE, opacity=0.5
        )
    )

    # then: 230 + 0.5 * (91, 181, 115) = (275.5, 320.5, 287.5) -> 255 inside
    assert output[60, 90].tolist() == [255, 255, 255]
    assert output[5, 5].tolist() == [230, 230, 230]


def test_empty_zone_returns_an_equal_new_image() -> None:
    # given
    image = image_of(scene(HEIGHT, WIDTH))

    # when
    output = run_v2(PolygonZoneVisualizationBlock, image, zone=[])

    # then
    assert torch.equal(output.tensor_image, image.tensor_image)
    assert output.tensor_image.data_ptr() != image.tensor_image.data_ptr()
    assert output.image_id == image.image_id


@pytest.mark.parametrize(
    "parameters",
    [
        {"zone": [[0, 50], [127, 50]]},
        {"zone": [[5, 90], [120, 8]], "thickness": 5, "count_in": 12, "count_out": 7},
        {"zone": [[64, 2], [64, 94]], "thickness": 1, "text_scale": 0.4},
        {"zone": [[-30, 40], [200, 60]], "text_thickness": 2, "opacity": 1.0},
        {"zone": [[3, 3], [40, 40]], "color": "YELLOW", "text_scale": 0.6},
    ],
    ids=[
        "horizontal",
        "diagonal-thick",
        "vertical-small-text",
        "beyond-frame",
        "corner",
    ],
)
def test_line_counter_equals_v1(parameters: Dict[str, Any]) -> None:
    # given
    pixels = scene(HEIGHT, WIDTH, seed=1)

    # when
    output = run_v2(LineCounterVisualizationBlock, image_of(pixels), **parameters)

    # then
    assert np.array_equal(pixels_of(output), v1_line(pixels, **parameters))


def test_line_counter_draws_the_thickness_of_each_call() -> None:
    # given: one block instance; V1 cached the line mask without its thickness
    pixels = scene(HEIGHT, WIDTH)
    block = LineCounterVisualizationBlock()
    params = dict(
        spec_of(LineCounterVisualizationBlock).validate_params(
            {"image": "$inputs.image", "zone": [[0, 50], [127, 50]]}
        )
    )

    # when
    thin = block.run(**{**params, "image": image_of(pixels), "thickness": 1})
    thick = block.run(**{**params, "image": image_of(pixels), "thickness": 9})

    # then
    zone = [[0, 50], [127, 50]]
    assert np.array_equal(
        pixels_of(thin["image"]), v1_line(pixels, zone=zone, thickness=1)
    )
    assert np.array_equal(
        pixels_of(thick["image"]), v1_line(pixels, zone=zone, thickness=9)
    )


def test_line_counter_writes_counts_on_white_at_the_first_point() -> None:
    # given
    pixels = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)

    # when
    output = pixels_of(
        run_v2(
            LineCounterVisualizationBlock,
            image_of(pixels),
            zone=[[64, 20], [64, 90]],
            count_in=3,
            count_out=1,
            text_scale=0.4,
            opacity=0.0,
        )
    )

    # then: a white box with black text centred on (64, 20); no visible line
    (width, height), _ = cv2.getTextSize(
        "in: 3, out: 1", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
    )
    left, top = 64 - width // 2, 20 - height // 2
    box = output[top : top + height + 1, left : left + width + 1]
    assert (box == 255).all(axis=2).mean() > 0.5
    assert (box < 128).all(axis=2).any()
    assert output[top - 1, 64].tolist() == [40, 40, 40]
    assert output[60, 64].tolist() == [40, 40, 40]


@pytest.mark.parametrize(
    "parameters",
    [
        {"reference_path": [[10, 90], [50, 40], [120, 40]]},
        {"reference_path": [[10, 90, 7], [50, 40, 8]], "thickness": 4},
        {"reference_path": [[60.8, 30.2]], "thickness": 3, "color": "RED"},
        {"reference_path": [[-40, 10], [64, 48], [180, 120]], "thickness": 1},
        {"reference_path": np.array([[5, 5], [100, 80], [5, 80]]), "color": "#FF00FF"},
    ],
    ids=["open-path", "extra-coordinates", "single-point", "beyond-frame", "array"],
)
def test_reference_path_equals_v1(parameters: Dict[str, Any]) -> None:
    # given
    pixels = scene(HEIGHT, WIDTH, seed=2)
    arguments = {"color": "#5bb573", "thickness": 2, **parameters}

    # when
    output = run_v2(ReferencePathVisualizationBlock, image_of(pixels), **arguments)

    # then
    expected = run_v1(ReferencePathVisualizationBlockV1, pixels, **arguments)
    assert np.array_equal(pixels_of(output), expected)


def test_reference_path_is_open() -> None:
    # given: a U shape; a closed polyline would join its two top ends
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    path = [[20, 10], [20, 80], [100, 80], [100, 10]]

    # when
    output = pixels_of(
        run_v2(ReferencePathVisualizationBlock, image_of(pixels), reference_path=path)
    )

    # then
    assert output[80, 60].any()
    assert not output[10, 60].any()


def test_empty_reference_path_returns_an_equal_image() -> None:
    # given: V1 raised IndexError for an empty path
    image = image_of(scene(HEIGHT, WIDTH))

    # when
    output = run_v2(ReferencePathVisualizationBlock, image, reference_path=[])

    # then
    assert torch.equal(output.tensor_image, image.tensor_image)


@pytest.mark.parametrize(
    "block_class, parameters",
    [
        (PolygonZoneVisualizationBlock, {"zone": [TRIANGLE, SQUARE], "opacity": 0.6}),
        (LineCounterVisualizationBlock, {"zone": [[0, 50], [127, 60]], "count_in": 4}),
        (ReferencePathVisualizationBlock, {"reference_path": TRIANGLE, "thickness": 3}),
    ],
)
def test_input_is_unchanged_and_a_strided_input_draws_the_same(
    block_class: type, parameters: Dict[str, Any]
) -> None:
    # given: the same pixels as a contiguous image and as a strided view
    pixels = scene(HEIGHT, WIDTH, seed=3)
    contiguous = image_of(pixels)
    wide = torch.zeros((3, HEIGHT, WIDTH * 2), dtype=torch.uint8)
    wide[:, :, ::2] = torch.from_numpy(pixels).permute(2, 0, 1)
    strided = ImageData.from_tensor(wide[:, :, ::2], image_id="strided")
    before = strided.tensor_image.clone()

    # when
    from_contiguous = run_v2(block_class, contiguous, **parameters)
    from_strided = run_v2(block_class, strided, **parameters)

    # then
    assert torch.equal(strided.tensor_image, before)
    assert torch.equal(from_strided.tensor_image, from_contiguous.tensor_image)
    assert from_strided.tensor_image.is_contiguous()
    assert from_strided.image_id == "strided"


@pytest.mark.parametrize(
    "value, expected",
    [
        (TRIANGLE, 1),
        ([TRIANGLE, SQUARE], 2),
        ((tuple(point) for point in TRIANGLE), None),
        ([], 0),
    ],
    ids=["one-polygon", "two-polygons", "generator", "empty"],
)
def test_polygon_zones_reads_one_or_many_polygons(value: Any, expected: Any) -> None:
    # when / then
    if expected is None:
        with pytest.raises(ValueError, match="zone must be a list; got generator"):
            polygon_zones(value)
    else:
        polygons = polygon_zones(value)
        assert len(polygons) == expected
        assert all(p.dtype == np.int32 and p.shape[1] == 2 for p in polygons)


@pytest.mark.parametrize(
    "block_class, name, value, message",
    [
        (PolygonZoneVisualizationBlock, "zone", [[0, 0], [5, 5]], "zone must be a polygon of at least 3 [x, y] points"),
        (PolygonZoneVisualizationBlock, "zone", [TRIANGLE, []], "zone[1] must be a polygon of at least 3 [x, y] points"),
        (PolygonZoneVisualizationBlock, "zone", [[0, 0, 1], [5, 5, 1], [9, 0, 1]], "zone[0][0] must be a point [x, y]"),
        (PolygonZoneVisualizationBlock, "zone", [[0, True], [5, 5], [9, 0]], "zone[0] coordinates must be numbers"),
        (PolygonZoneVisualizationBlock, "zone", [[0, float("nan")], [5, 5], [9, 0]], "zone[0] coordinates must be finite"),
        (PolygonZoneVisualizationBlock, "zone", [[0, 3e9], [5, 5], [9, 0]], "zone must be finite int32 pixels"),
        (PolygonZoneVisualizationBlock, "opacity", 1.5, "opacity must be a number in [0.0, 1.0]"),
        (PolygonZoneVisualizationBlock, "color", "#zz0000", "color: invalid color"),
        (LineCounterVisualizationBlock, "zone", TRIANGLE, "zone must be exactly 2 points"),
        (LineCounterVisualizationBlock, "zone", [[5, 5], [5.0, 5]], "zone needs two different points"),
        (LineCounterVisualizationBlock, "thickness", 0, "thickness must be an integer >= 1"),
        (LineCounterVisualizationBlock, "text_thickness", 0, "text_thickness must be an integer >= 1"),
        (LineCounterVisualizationBlock, "text_scale", 0, "text_scale must be a number > 0.0"),
        (LineCounterVisualizationBlock, "count_in", -1, "count_in must be an integer >= 0"),
        (ReferencePathVisualizationBlock, "reference_path", [[1]], "reference_path[0] must be a point [x, y, ...]"),
        (ReferencePathVisualizationBlock, "thickness", 0, "thickness must be an integer >= 1"),
    ],
)  # fmt: skip
def test_a_selected_value_fails_with_the_literal_error_text(
    block_class: type, name: str, value: Any, message: str
) -> None:
    # given
    geometry = {
        PolygonZoneVisualizationBlock: {"zone": TRIANGLE},
        LineCounterVisualizationBlock: {"zone": [[0, 5], [9, 5]]},
        ReferencePathVisualizationBlock: {"reference_path": TRIANGLE},
    }[block_class]
    refs = {"image": "$inputs.image", **geometry}
    spec = spec_of(block_class)

    # when: the literal at compile time, then the same value selected at run
    with pytest.raises(ParamsValidationError) as compile_error:
        spec.validate_params({**refs, name: value})
    params = dict(spec.validate_params({**refs, name: f"$inputs.{name}"}))
    with pytest.raises(ValueError) as run_error:
        block_class().run(**{**params, name: value, "image": image_of(scene())})

    # then
    assert message in str(compile_error.value)
    assert message in str(run_error.value)


@pytest.mark.parametrize(
    "block_class, parse, value",
    [
        (PolygonZoneVisualizationBlock, parse_polygons, TRIANGLE),
        (PolygonZoneVisualizationBlock, parse_polygons, [TRIANGLE, SQUARE]),
        (PolygonZoneVisualizationBlock, parse_polygons, np.array(TRIANGLE)),
        (PolygonZoneVisualizationBlock, parse_polygons, []),
        (PolygonZoneVisualizationBlock, parse_polygons, [[0, 0], [5, 5]]),
        (PolygonZoneVisualizationBlock, parse_polygons, [[[0, 0, 1], [5, 5, 1], [9, 0, 1]]]),
        (PolygonZoneVisualizationBlock, parse_polygons, "0,0 5,5 9,0"),
        (LineCounterVisualizationBlock, parse_line, [[0, 5], [9, 5]]),
        (LineCounterVisualizationBlock, parse_line, [[0, 5], [0, 5]]),
        (LineCounterVisualizationBlock, parse_line, [[0, 5, 1], [9, 5, 1]]),
        (LineCounterVisualizationBlock, parse_line, [[0, 5]]),
    ],
)  # fmt: skip
def test_a_drawing_accepts_a_geometry_exactly_when_the_zone_blocks_do(
    block_class: type, parse: Any, value: Any
) -> None:
    # given
    params = dict(
        spec_of(block_class).validate_params(
            {"image": "$inputs.image", "zone": "$inputs.zone"}
        )
    )

    # when
    try:
        parse(value, name="zone")
        expected_error = None
    except ValueError as error:
        expected_error = str(error)
    try:
        block_class().run(**{**params, "zone": value, "image": image_of(scene())})
        drawing_error = None
    except ValueError as error:
        drawing_error = str(error)

    # then
    assert drawing_error == expected_error


def test_a_selected_non_list_geometry_names_the_expected_shape() -> None:
    # given: literals of another type are rejected by the schema at compile
    spec = spec_of(ReferencePathVisualizationBlock)
    params = dict(
        spec.validate_params(
            {"image": "$inputs.image", "reference_path": "$inputs.path"}
        )
    )

    # when
    with pytest.raises(ValueError) as error:
        ReferencePathVisualizationBlock().run(
            **{**params, "reference_path": "10,20", "image": image_of(scene())}
        )

    # then
    assert "reference_path must be a list of points, got '10,20'" in str(error.value)


@pytest.mark.parametrize(
    "block_class, geometry",
    [
        (PolygonZoneVisualizationBlock, {"zone": TRIANGLE}),
        (LineCounterVisualizationBlock, {"zone": [[0, 5], [9, 5]]}),
        (ReferencePathVisualizationBlock, {"reference_path": TRIANGLE}),
    ],
)
def test_v1_copy_image_is_rejected_with_the_chaining_remedy(
    block_class: type, geometry: Dict[str, Any]
) -> None:
    # when
    with pytest.raises(ParamsValidationError) as error:
        spec_of(block_class).validate_params(
            {"image": "$inputs.image", "copy_image": False, **geometry}
        )

    # then
    assert "image: $steps.<previous>.image" in str(error.value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "block_class, parameters",
    [
        (PolygonZoneVisualizationBlock, {"zone": [TRIANGLE, SQUARE], "opacity": 0.45}),
        (LineCounterVisualizationBlock, {"zone": [[5, 90], [120, 8]], "count_out": 2}),
        (ReferencePathVisualizationBlock, {"reference_path": SQUARE, "thickness": 3}),
    ],
)
def test_cuda_image_draws_on_its_device_like_cpu(
    block_class: type, parameters: Dict[str, Any]
) -> None:
    # given
    host = image_of(scene(HEIGHT, WIDTH, seed=4))
    device = host.copy(device="cuda")

    # when
    on_cpu = run_v2(block_class, host, **parameters)
    on_cuda = run_v2(block_class, device, **parameters)

    # then
    assert on_cuda.device.type == "cuda"
    assert torch.equal(on_cuda.tensor_image.cpu(), on_cpu.tensor_image)
