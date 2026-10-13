"""Zone drawings in compiled V2 workflows.

The geometry is a workflow input, written once and read by every drawing
(and, later, by the counters). Counts arrive per image, as a counter's
outputs would::

    image ─▶ polygon zone ─▶ line counter ─▶ reference path ─▶ output
             $inputs.zone    $inputs.line     $inputs.path
                             $inputs.count_in / count_out (one per image)

Pixel parity with V1 is the unit suite's job. These tests check what a
workflow builder relies on: the blocks chain, images of different sizes keep
their rows, selected values draw like literals and fail with the same text,
gates skip images and inputs stay unchanged.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.visualization.annotations import zones
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.plan import CompileOptions

MODES = ("run", "phases")
GREY = 100
ZONE = [[8, 8], [56, 8], [56, 40], [8, 40]]
LINE = [[0, 52], [80, 52]]
PATH = [[70, 4], [70, 30], [90, 30]]
ZONE_COLOR = (91, 181, 115)
GEOMETRY_INPUTS = [
    {"type": "WorkflowParameter", "name": "zone"},
    {"type": "WorkflowParameter", "name": "line"},
    {"type": "WorkflowParameter", "name": "path"},
]
COUNT_INPUTS = [
    {"type": "WorkflowBatchInput", "name": "count_in"},
    {"type": "WorkflowBatchInput", "name": "count_out"},
]


def chain(zone: Any, line: Any, path: Any, *, opacity: Any = 0.3) -> List[dict]:
    steps = [
        {
            "type": "v2/polygon_zone_visualization@v1",
            "name": "zone",
            "image": "$inputs.image",
            "zone": zone,
            "opacity": opacity,
        },
        {
            "type": "v2/line_counter_visualization@v1",
            "name": "counter",
            "image": "$steps.zone.image",
            "zone": line,
            "count_in": "$inputs.count_in",
            "count_out": "$inputs.count_out",
            "text_scale": 0.3,
        },
        {
            "type": "v2/reference_path_visualization@v1",
            "name": "path",
            "image": "$steps.counter.image",
            "reference_path": path,
            "color": "RED",
        },
    ]

    return steps


def compile_plan(
    steps: List[dict],
    *,
    inputs: Sequence[dict] = (),
    mode: str = "run",
):
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}, *inputs],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": "drawn", "selector": "$steps.path.image"}
        ],
    }
    plan = compile_workflow(
        definition,
        catalogue=create_catalogue().with_blocks(list(zones.BLOCKS)),
        options=CompileOptions(block_execution=mode),
    )

    return plan


def run_rows(
    steps: List[dict],
    images: Sequence[ImageData],
    *,
    inputs: Sequence[dict] = (),
    values: Optional[Dict[str, Any]] = None,
    mode: str = "run",
) -> List[Dict[str, Any]]:
    plan = compile_plan(steps, inputs=[*COUNT_INPUTS, *inputs], mode=mode)
    counts = {"count_in": [0] * len(images), "count_out": [0] * len(images)}
    rows = (
        plan.create_session({})
        .run({"image": list(images), **counts, **(values or {})})
        .rows()
    )

    return rows


def grey_image(height: int, width: int, *, image_id: str) -> ImageData:
    pixels = np.full((height, width, 3), GREY, dtype=np.uint8)
    image = ImageData.from_numpy_rgb(pixels, image_id=image_id)

    return image


def pixels_of(image: ImageData) -> np.ndarray:
    pixels = image.tensor_image.permute(1, 2, 0).cpu().numpy()

    return pixels


@pytest.mark.parametrize("mode", MODES)
def test_the_three_drawings_chain_on_images_of_different_sizes(mode: str) -> None:
    # given
    images = [
        grey_image(64, 96, image_id="small"),
        grey_image(120, 160, image_id="large"),
    ]
    before = [pixels_of(image).copy() for image in images]

    # when
    rows = run_rows(
        chain("$inputs.zone", "$inputs.line", "$inputs.path"),
        images,
        inputs=GEOMETRY_INPUTS,
        values={
            "zone": ZONE,
            "line": LINE,
            "path": PATH,
            "count_in": [3, 4],
            "count_out": [1, 2],
        },
        mode=mode,
    )

    # then: each row keeps its image's identity and size
    for row, image, original in zip(rows, images, before):
        drawn = pixels_of(row["drawn"])
        assert row["drawn"].image_id == image.image_id
        assert drawn.shape == original.shape
        assert np.array_equal(pixels_of(image), original)
        # the zone brightens by 0.3 * color; the path is solid red
        tinted = np.round(GREY + 0.3 * np.array(ZONE_COLOR, dtype=np.float32))
        assert drawn[24, 30].tolist() == tinted.tolist()
        assert drawn[20, 70].tolist() == [255, 0, 0]
        # the counts sit on white at the line's first point; far pixels stay
        assert (drawn[48:56, 0:6] == 255).all(axis=2).any()
        assert drawn[-1, -1].tolist() == [GREY] * 3


def test_counts_of_each_image_change_only_its_label() -> None:
    # given: two equal frames counted differently
    images = [grey_image(64, 96, image_id=f"frame-{i}") for i in range(2)]

    # when
    rows = run_rows(
        chain(ZONE, LINE, PATH),
        images,
        values={"count_in": [0, 12], "count_out": [0, 5]},
    )

    # then
    first, second = (pixels_of(row["drawn"]) for row in rows)
    rows_changed = np.nonzero((first != second).any(axis=(1, 2)))[0]
    assert rows_changed.size > 0
    assert rows_changed.min() >= 52 - 8 and rows_changed.max() <= 52 + 8


def test_selected_geometry_and_style_draw_like_the_same_literals() -> None:
    # given
    images = [grey_image(64, 96, image_id="a"), grey_image(80, 120, image_id="b")]
    inputs = [*GEOMETRY_INPUTS, {"type": "WorkflowParameter", "name": "opacity"}]
    selected = chain(
        "$inputs.zone", "$inputs.line", "$inputs.path", opacity="$inputs.opacity"
    )
    plan = compile_plan(selected, inputs=[*COUNT_INPUTS, *inputs])
    session = plan.create_session({})
    counts = {"count_in": [2, 2], "count_out": [1, 1]}

    results = []
    for zone, opacity in ((ZONE, 0.3), ([ZONE, [[60, 40], [90, 60], [60, 60]]], 0.8)):
        # when
        by_selector = session.run(
            {
                "image": images,
                "zone": zone,
                "line": LINE,
                "path": PATH,
                "opacity": opacity,
                **counts,
            }
        ).rows()
        by_literal = run_rows(
            chain(zone, LINE, PATH, opacity=opacity), images, values=counts
        )

        # then
        for selector_row, literal_row in zip(by_selector, by_literal):
            assert np.array_equal(
                pixels_of(selector_row["drawn"]), pixels_of(literal_row["drawn"])
            )
        results.append(pixels_of(by_selector[0]["drawn"]))

    # different selected values give different pictures from one plan
    assert not np.array_equal(results[0], results[1])


def test_a_selected_line_fails_with_the_literal_compile_error_text() -> None:
    # given
    three_points = [[0, 1], [2, 3], [4, 5]]

    # when
    with pytest.raises(Exception) as compile_error:
        compile_plan(chain(ZONE, three_points, PATH), inputs=COUNT_INPUTS)
    plan = compile_plan(
        chain(ZONE, "$inputs.line", PATH), inputs=[*COUNT_INPUTS, GEOMETRY_INPUTS[1]]
    )
    with pytest.raises(StepExecutionError) as run_error:
        plan.create_session({}).run(
            {
                "image": [grey_image(32, 32, image_id="x")],
                "line": three_points,
                "count_in": [0],
                "count_out": [0],
            }
        )

    # then
    expected = "zone must be exactly 2 points [[x1, y1], [x2, y2]]; got 3 items"
    assert expected in str(compile_error.value)
    assert expected in str(run_error.value)


@pytest.mark.parametrize("mode", MODES)
def test_a_gated_image_is_not_drawn_and_the_others_are(mode: str) -> None:
    # given
    steps = [
        {
            "type": "v2/continue_if",
            "name": "keep",
            "condition": "$inputs.keep",
            "next_steps": ["$steps.zone"],
        },
        *chain(ZONE, LINE, PATH),
    ]
    images = [grey_image(64, 96, image_id=f"frame-{i}") for i in range(3)]

    # when
    rows = run_rows(
        steps,
        images,
        inputs=[{"type": "WorkflowBatchInput", "name": "keep"}],
        values={"keep": [True, False, True]},
        mode=mode,
    )

    # then
    assert rows[1]["drawn"] is None
    assert np.array_equal(pixels_of(rows[0]["drawn"]), pixels_of(rows[2]["drawn"]))
    assert not np.array_equal(pixels_of(rows[0]["drawn"]), pixels_of(images[0]))
