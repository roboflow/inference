"""The six shape blocks in compiled V2 workflows fed by a real model block.

Fake models (``annotations/scene.py``) stand in for loaded weights; the
instance segmentation block and the shape blocks are the real V2 blocks::

    instance seg ─▶ halo ─▶ circle ─▶ ellipse ─▶ corner ─▶ dot ─▶ triangle

Pixel equality with V1 is the unit suites' job. These tests check what a
workflow builder relies on, in both block execution modes: the chain equals
the blocks called one by one, images of a ragged batch do not affect each
other, selected values behave like literals, inputs stay unchanged, and box
readers never materialize a compact mask view.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.visualization.annotations.circle import (
    CircleVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.corner import (
    CornerVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.dot import (
    DotVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.ellipse import (
    EllipseVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.halo import (
    HaloVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.triangle import (
    TriangleVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.integration.blocks.visualization.annotations.scene import (
    ALLEY_SIZE,
    SceneLoader,
    alley_image,
    annotation_catalogue,
    changed,
    pixels_of,
    scene_image,
    street_image,
)

MODES = ("run", "phases")
SHAPE_BLOCKS = (
    HaloVisualizationBlock,
    CircleVisualizationBlock,
    EllipseVisualizationBlock,
    CornerVisualizationBlock,
    DotVisualizationBlock,
    TriangleVisualizationBlock,
)
SEG = {
    "type": "v2/instance_segmentation@v1",
    "name": "seg",
    "image": "$inputs.image",
    "model_id": "shapes-seg",
}
COMPACT = "test/synthetic_compact_segmentation@v1"
# Chain order and each block's non-default style.
CHAIN = (
    ("halo", HaloVisualizationBlock, {"kernel_size": 15}),
    ("circle", CircleVisualizationBlock, {"thickness": 1}),
    ("ellipse", EllipseVisualizationBlock, {"start_angle": 0, "end_angle": 360}),
    ("corner", CornerVisualizationBlock, {"thickness": 2, "corner_length": 6}),
    ("dot", DotVisualizationBlock, {"position": "CENTER_OF_MASS", "radius": 3}),
    ("triangle", TriangleVisualizationBlock, {"base": 8, "height": 6}),
)


def shape_catalogue():
    catalogue = annotation_catalogue().with_blocks(list(SHAPE_BLOCKS))

    return catalogue


def compile_plan(
    steps: List[Dict[str, Any]],
    outputs: Dict[str, str],
    *,
    inputs: Sequence[Dict[str, Any]] = (),
    mode: str = "run",
):
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}, *inputs],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    plan = compile_workflow(
        definition,
        catalogue=shape_catalogue(),
        options=CompileOptions(block_execution=mode),
    )

    return plan


def run_rows(
    steps: List[Dict[str, Any]],
    outputs: Dict[str, str],
    images: Sequence[ImageData],
    *,
    inputs: Sequence[Dict[str, Any]] = (),
    values: Optional[Dict[str, Any]] = None,
    mode: str = "run",
) -> List[Dict[str, Any]]:
    plan = compile_plan(steps, outputs, inputs=inputs, mode=mode)
    session = plan.create_session({"models": ModelCache(SceneLoader())})
    rows = session.run({"image": list(images), **(values or {})}).rows()

    return rows


def chain_steps(predictions: str = "$steps.seg.segmentation") -> List[dict]:
    steps, image = [SEG], "$inputs.image"
    for name, block, style in CHAIN:
        steps.append(
            {
                "type": block.type,
                "name": name,
                "image": image,
                "predictions": predictions,
                **style,
            }
        )
        image = f"$steps.{name}.image"

    return steps


CHAIN_OUTPUTS = {
    **{name: f"$steps.{name}.image" for name, _, _ in CHAIN},
    "predictions": "$steps.seg.segmentation",
}


def draw_one_by_one(image: ImageData, predictions: Any) -> Dict[str, ImageData]:
    # The same chain, each block validated and called directly.
    drawn = {}
    for name, block, style in CHAIN:
        params = spec_of(block).validate_params(
            {"image": "$inputs.image", "predictions": "$inputs.p", **style}
        )
        image = block().run(
            **{**dict(params), "image": image, "predictions": predictions}
        )["image"]
        drawn[name] = image

    return drawn


@pytest.mark.parametrize("mode", MODES)
def test_the_chain_equals_the_blocks_called_one_by_one_over_a_ragged_batch(
    mode: str,
) -> None:
    images = [street_image(), alley_image()]

    rows = run_rows(chain_steps(), CHAIN_OUTPUTS, images, mode=mode)

    for image, row in zip(images, rows):
        expected = draw_one_by_one(image, row["predictions"])
        for name, _, _ in CHAIN:
            assert np.array_equal(pixels_of(row[name]), pixels_of(expected[name])), name
            before = image if name == "halo" else row[_previous(name)]
            assert changed(before, row[name]).any(), f"{name} must draw"
        assert row["triangle"].image_id == image.image_id


def _previous(name: str) -> str:
    names = [step_name for step_name, _, _ in CHAIN]

    return names[names.index(name) - 1]


def test_an_image_of_the_batch_draws_as_it_does_alone() -> None:
    street, alley = street_image(), alley_image()

    together = run_rows(chain_steps(), CHAIN_OUTPUTS, [street, alley])
    alone = run_rows(chain_steps(), CHAIN_OUTPUTS, [alley])

    assert np.array_equal(
        pixels_of(together[1]["triangle"]), pixels_of(alone[0]["triangle"])
    )


@pytest.mark.parametrize("mode", MODES)
def test_an_image_without_objects_passes_every_block_unchanged(mode: str) -> None:
    empty = scene_image("empty", ALLEY_SIZE, (), road=True)

    rows = run_rows(chain_steps(), CHAIN_OUTPUTS, [empty, street_image()], mode=mode)

    assert len(rows[0]["predictions"]) == 0
    for name, _, _ in CHAIN:
        assert not changed(empty, rows[0][name]).any(), name
        assert rows[0][name].tensor_image.data_ptr() != empty.tensor_image.data_ptr()


@pytest.mark.parametrize("mode", MODES)
def test_inputs_and_predictions_are_not_changed(mode: str) -> None:
    images = [street_image(), alley_image()]
    before = [image.tensor_image.clone() for image in images]
    steps = chain_steps("$steps.seg.predictions")
    outputs = {"triangle": "$steps.triangle.image", "seg": "$steps.seg.predictions"}

    rows = run_rows(steps, outputs, images, mode=mode)

    model_only = run_rows([SEG], {"seg": "$steps.seg.predictions"}, images)
    for image, pixels in zip(images, before):
        assert torch.equal(image.tensor_image, pixels)
    for image, row, reference in zip(images, rows, model_only):
        assert changed(image, row["triangle"]).any()
        assert torch.equal(row["seg"].mask, reference["seg"].mask)
        assert torch.equal(row["seg"].xyxy, reference["seg"].xyxy)


def compact_chain(*, with_halo: bool, dot_position: str) -> List[dict]:
    """The chain on the synthetic compact view; optionally without halo."""
    steps = chain_steps()
    steps[0] = {**SEG, "type": COMPACT}
    if not with_halo:
        steps = [step for step in steps if step["name"] != "halo"]
        steps[1]["image"] = "$inputs.image"
    for step in steps:
        if step["name"] == "dot":
            step["position"] = dot_position

    return steps


def test_box_readers_never_materialize_a_compact_view() -> None:
    outputs = {"view": "$steps.seg.segmentation", "out": "$steps.triangle.image"}

    rows = run_rows(
        compact_chain(with_halo=False, dot_position="CENTER"),
        outputs,
        [street_image()],
    )

    assert rows[0]["view"].materialization_count == 0
    assert changed(street_image(), rows[0]["out"]).any()


def test_halo_and_centre_of_mass_share_one_accurate_materialization() -> None:
    outputs = {"view": "$steps.seg.segmentation", "out": "$steps.triangle.image"}

    rows = run_rows(
        compact_chain(with_halo=True, dot_position="CENTER_OF_MASS"),
        outputs,
        [street_image()],
    )

    assert rows[0]["view"].materialization_count == 1


def test_coarse_centre_of_mass_and_halo_never_take_the_accurate_path() -> None:
    steps = chain_steps()
    steps[0] = {**SEG, "type": COMPACT}
    for step in steps[1:]:
        if step["name"] in ("halo", "dot"):
            step["mask_quality"] = "coarse"

    rows = run_rows(
        steps,
        {"view": "$steps.seg.segmentation", "out": "$steps.triangle.image"},
        [street_image()],
    )

    assert rows[0]["view"].materialization_count == 0
    assert changed(street_image(), rows[0]["out"]).any()


SELECTED = {
    "halo": ("kernel_size", 15, 31),
    "circle": ("thickness", 1, -1),
    "ellipse": ("end_angle", 360, 180),
    "corner": ("corner_length", 6, -4),
    "dot": ("radius", 3, 6),
    "triangle": ("position", "TOP_CENTER", "BOTTOM_LEFT"),
}


def selected_steps(values: Dict[str, Any]) -> List[dict]:
    steps = chain_steps()
    for step in steps[1:]:
        field = SELECTED[step["name"]][0]
        step[field] = values[step["name"]]

    return steps


def test_selected_values_draw_like_the_same_literals() -> None:
    images = [street_image(), alley_image()]
    selectors = {name: f"$inputs.{name}" for name in SELECTED}
    inputs = [{"type": "WorkflowParameter", "name": name} for name in SELECTED]
    plan = compile_plan(selected_steps(selectors), CHAIN_OUTPUTS, inputs=inputs)
    session = plan.create_session({"models": ModelCache(SceneLoader())})

    for choice in (1, 2):
        values = {name: entry[choice] for name, entry in SELECTED.items()}
        by_selector = session.run({"image": images, **values}).rows()
        by_literal = run_rows(selected_steps(values), CHAIN_OUTPUTS, images)
        for selector_row, literal_row in zip(by_selector, by_literal):
            for name in SELECTED:
                assert np.array_equal(
                    pixels_of(selector_row[name]), pixels_of(literal_row[name])
                ), (name, choice)


def test_a_selected_value_fails_with_the_literal_compile_error_text() -> None:
    literal = selected_steps({**{n: e[1] for n, e in SELECTED.items()}, "halo": 0})
    with pytest.raises(Exception) as compile_error:
        compile_plan(literal, {"out": "$steps.triangle.image"})
    selected = selected_steps(
        {**{n: e[1] for n, e in SELECTED.items()}, "halo": "$inputs.kernel"}
    )
    plan = compile_plan(
        selected,
        {"out": "$steps.triangle.image"},
        inputs=[{"type": "WorkflowParameter", "name": "kernel"}],
    )
    session = plan.create_session({"models": ModelCache(SceneLoader())})

    with pytest.raises(StepExecutionError) as run_error:
        session.run({"image": [street_image()], "kernel": 0})

    expected = "kernel_size must be an integer >= 1, got 0"
    assert expected in str(compile_error.value)
    assert expected in str(run_error.value)


def test_halo_refuses_boxes_without_masks_at_compile_time() -> None:
    steps = [
        {**SEG, "type": "v2/object_detection@v1", "name": "detect"},
        {
            "type": HaloVisualizationBlock.type,
            "name": "halo",
            "image": "$inputs.image",
            "predictions": "$steps.detect.predictions",
        },
    ]

    with pytest.raises(Exception) as error:
        compile_plan(steps, {"halo": "$steps.halo.image"})

    assert "object_detection_prediction" in str(error.value)


@pytest.mark.parametrize("block", SHAPE_BLOCKS, ids=lambda block: block.type)
def test_copy_image_is_rejected_with_the_chaining_remedy(block: type) -> None:
    step = {
        "type": block.type,
        "name": "shape",
        "image": "$inputs.image",
        "predictions": "$steps.seg.segmentation",
        "copy_image": True,
    }

    with pytest.raises(Exception) as error:
        compile_plan([SEG, step], {"out": "$steps.shape.image"})

    assert "copy_image" in str(error.value) and "$steps." in str(error.value)
