"""Model comparison, rich labels and text display in compiled V2 workflows.

The fake ``shapes-seg`` model (``scene.py``) stands in for loaded weights;
the model, comparison, label and text blocks are the real V2 blocks. Two
steps of the same model with different confidence thresholds play two
models that disagree::

    seg_strict (confidence 0.75) ─┐
                                  ├─▶ compare ─▶ rich labels of B ─▶ count text
    seg_loose  (confidence 0.35) ─┘

On the street the loose model alone finds the yellow ring (0.7); both find
the red ellipse (0.9) and the blue triangle (0.8). Pixel exactness against
V1 is the unit suites' job; these tests check the workflow outcomes.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.visualization.annotations.model_comparison import (
    ModelComparisonVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rich_label import (
    RichLabelVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text_display import (
    TextDisplayBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.integration.blocks.visualization.annotations.scene import (
    ALLEY,
    ALLEY_SIZE,
    STREET,
    STREET_SIZE,
    SceneLoader,
    alley_image,
    annotation_catalogue,
    changed,
    pixels_of,
    shape_mask,
    street_image,
)

MODES = ("run", "phases")
OPACITY = 0.5
GREEN, RED, BLACK = (0, 255, 0), (255, 0, 0), (0, 0, 0)
STRICT = {
    "type": "v2/instance_segmentation@v1",
    "name": "seg_strict",
    "image": "$inputs.image",
    "model_id": "shapes-seg",
    "confidence": 0.75,
}
LOOSE = {**STRICT, "name": "seg_loose", "confidence": 0.35}
COMPARE = {
    "type": "v2/model_comparison_visualization@v1",
    "name": "compare",
    "image": "$inputs.image",
    "predictions_a": "$steps.seg_strict.predictions",
    "predictions_b": "$steps.seg_loose.predictions",
    "opacity": OPACITY,
}
LABELS = {
    "type": "v2/rich_label_visualization@v1",
    "name": "labels",
    "image": "$steps.compare.image",
    "predictions": "$steps.seg_loose.predictions",
    "text": "Class and Confidence",
    "font_size": 9,
    "text_padding": 2,
}
COUNTS = {
    "type": "v2/text_display@v1",
    "name": "counts",
    "image": "$steps.labels.image",
    "text": "A {{ $parameters.a }} B {{ $parameters.b }}",
    "text_parameters": {
        "a": "$steps.seg_strict.predictions",
        "b": "$steps.seg_loose.predictions",
    },
    "text_parameters_operations": {
        "a": [{"type": "SequenceLength"}],
        "b": [{"type": "SequenceLength"}],
    },
    "font_scale": 0.3,
    "font_thickness": 1,
    "padding": 2,
    "offset_x": 0,
    "offset_y": 0,
}
OUTPUTS = {
    "compare": "$steps.compare.image",
    "labels": "$steps.labels.image",
    "counts": "$steps.counts.image",
}


def catalogue():
    extended = annotation_catalogue().with_blocks(
        [
            ModelComparisonVisualizationBlock,
            RichLabelVisualizationBlock,
            TextDisplayBlock,
        ]
    )

    return extended


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
        definition, catalogue=catalogue(), options=CompileOptions(block_execution=mode)
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


def tinted(pixels: np.ndarray, color) -> np.ndarray:
    blended = np.round(pixels * (1 - OPACITY) + np.array(color) * OPACITY)

    return blended.astype(np.uint8)


def direct_text(image: ImageData, text: str) -> np.ndarray:
    # The text block run outside the workflow with a literal sentence.
    literal = {
        name: value
        for name, value in COUNTS.items()
        if name not in ("type", "name", "image", "text_parameters_operations")
    }
    params = spec_of(TextDisplayBlock).validate_params(
        {**literal, "image": "$inputs.image", "text": text, "text_parameters": {}}
    )
    result = TextDisplayBlock().run(**{**dict(params), "image": image})
    pixels = pixels_of(result["image"])

    return pixels


@pytest.mark.parametrize("mode", MODES)
def test_disagreement_is_colored_per_image_of_a_ragged_batch(mode: str) -> None:
    # Given: a street (strict finds 2, loose finds 3) and an alley (1 and 1)
    street, alley = street_image(), alley_image()

    rows = run_rows(
        [STRICT, LOOSE, COMPARE],
        {"compare": "$steps.compare.image"},
        [street, alley],
        mode=mode,
    )

    # Then: on the street only the ring is red; agreed shapes are untouched
    before = pixels_of(street)
    after = pixels_of(rows[0]["compare"])
    red_ellipse, blue_triangle, yellow_ring = (
        shape_mask(s, STREET_SIZE) for s in STREET
    )
    agreed = red_ellipse | blue_triangle
    np.testing.assert_array_equal(after[agreed], before[agreed])
    np.testing.assert_array_equal(after[yellow_ring], tinted(before[yellow_ring], RED))
    neither = ~(agreed | yellow_ring)
    np.testing.assert_array_equal(after[neither], tinted(before[neither], BLACK))
    assert not (after == tinted(before, GREEN)).all(axis=2).any()

    # And: on the alley both found the triangle, so nothing is red or green
    alley_before = pixels_of(alley)
    alley_after = pixels_of(rows[1]["compare"])
    triangle = shape_mask(ALLEY[0], ALLEY_SIZE)
    np.testing.assert_array_equal(alley_after[triangle], alley_before[triangle])
    np.testing.assert_array_equal(
        alley_after[~triangle], tinted(alley_before[~triangle], BLACK)
    )


def test_swapping_models_swaps_the_colors() -> None:
    swapped = {
        **COMPARE,
        "predictions_a": "$steps.seg_loose.predictions",
        "predictions_b": "$steps.seg_strict.predictions",
    }
    street = street_image()

    rows = run_rows(
        [STRICT, LOOSE, swapped], {"compare": "$steps.compare.image"}, [street]
    )

    ring = shape_mask(STREET[2], STREET_SIZE)
    before = pixels_of(street)
    np.testing.assert_array_equal(
        pixels_of(rows[0]["compare"])[ring], tinted(before[ring], GREEN)
    )


@pytest.mark.parametrize("mode", MODES)
def test_chain_labels_loose_rows_and_writes_both_counts(mode: str) -> None:
    street, alley = street_image(), alley_image()

    rows = run_rows(
        [STRICT, LOOSE, COMPARE, LABELS, COUNTS], OUTPUTS, [street, alley], mode=mode
    )

    for row, sentence in zip(rows, ("A 2 B 3", "A 1 B 1")):
        # Rich labels change the comparison image near each loose row.
        labelled = changed(row["compare"], row["labels"])
        assert labelled.any()
        # The counts are the selected predictions' lengths.
        np.testing.assert_array_equal(
            pixels_of(row["counts"]), direct_text(row["labels"], sentence)
        )
        # Inputs are never changed by the chain.
    np.testing.assert_array_equal(pixels_of(street), pixels_of(street_image()))
    np.testing.assert_array_equal(pixels_of(alley), pixels_of(alley_image()))


def test_three_labels_on_the_street_one_on_the_alley() -> None:
    # Each label background is a solid row color; count distinct label areas
    # by the rows the loose model returned: the label image changes once per row.
    street, alley = street_image(), alley_image()
    single_row_steps = [STRICT, LOOSE, COMPARE, {**LABELS, "text": "Index"}]

    rows = run_rows(
        single_row_steps,
        {"compare": "$steps.compare.image", "labels": "$steps.labels.image"},
        [street, alley],
    )

    street_changed = changed(rows[0]["compare"], rows[0]["labels"])
    alley_changed = changed(rows[1]["compare"], rows[1]["labels"])
    # TOP_LEFT labels sit above each box's top-left corner.
    for shape in STREET:
        x0, y0, _, _ = shape.xyxy
        assert street_changed[max(y0 - 3, 0), x0 + 1]
    x0, y0, _, _ = ALLEY[0].xyxy
    assert alley_changed[max(y0 - 3, 0), x0 + 1]
    assert not alley_changed[:, 60:].any()


def test_selected_style_values_behave_like_literals() -> None:
    inputs = [
        {"type": "WorkflowParameter", "name": "font"},
        {"type": "WorkflowParameter", "name": "color"},
    ]
    selected = [
        STRICT,
        LOOSE,
        {**COMPARE, "color_b": "$inputs.color"},
        {**LABELS, "font_family": "$inputs.font"},
    ]
    literal = [
        STRICT,
        LOOSE,
        {**COMPARE, "color_b": "#FF00FF"},
        {**LABELS, "font_family": "Noto Sans"},
    ]
    street = street_image()
    outputs = {"labels": "$steps.labels.image"}

    by_selector = run_rows(
        selected,
        outputs,
        [street],
        inputs=inputs,
        values={"font": "noto_sans", "color": "#FF00FF"},
    )
    by_literal = run_rows(literal, outputs, [street])

    np.testing.assert_array_equal(
        pixels_of(by_selector[0]["labels"]), pixels_of(by_literal[0]["labels"])
    )


def test_a_selected_value_fails_with_the_literal_compile_error_text() -> None:
    with pytest.raises(Exception) as compile_error:
        compile_plan(
            [STRICT, LOOSE, {**COMPARE, "opacity": 1.5}],
            {"compare": "$steps.compare.image"},
        )
    plan = compile_plan(
        [STRICT, LOOSE, {**COMPARE, "opacity": "$inputs.opacity"}],
        {"compare": "$steps.compare.image"},
        inputs=[{"type": "WorkflowParameter", "name": "opacity"}],
    )
    session = plan.create_session({"models": ModelCache(SceneLoader())})

    with pytest.raises(StepExecutionError) as run_error:
        session.run({"image": [street_image()], "opacity": 1.5})

    expected = "opacity must be a number in [0.0, 1.0], got 1.5"
    assert expected in str(compile_error.value)
    assert expected in str(run_error.value)


def test_operations_for_an_unknown_parameter_fail_compilation() -> None:
    broken = {
        **COUNTS,
        "text_parameters_operations": {"c": [{"type": "SequenceLength"}]},
    }

    with pytest.raises(Exception) as error:
        compile_plan([STRICT, LOOSE, COMPARE, LABELS, broken], OUTPUTS)

    assert "names `c`, which is not a text_parameters key" in str(error.value)


def test_unknown_selected_font_fails_the_step_with_the_font_list() -> None:
    plan = compile_plan(
        [STRICT, LOOSE, COMPARE, {**LABELS, "font_family": "$inputs.font"}],
        {"labels": "$steps.labels.image"},
        inputs=[{"type": "WorkflowParameter", "name": "font"}],
    )
    session = plan.create_session({"models": ModelCache(SceneLoader())})

    with pytest.raises(StepExecutionError) as error:
        session.run({"image": [street_image()], "font": "Comic Sans"})

    assert "Unknown font family: 'Comic Sans'" in str(error.value)
    assert "Noto Sans" in str(error.value)
