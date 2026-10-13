"""``v2/text_display@v1`` against the latest V1 block and its template rules.

The V1 oracle (``support.run_v1``) gets exactly the validated V2 values, so
every placement, alignment, background and clipping case must be pixel-equal.
"""

from typing import Any, Dict

import numpy as np
import pytest
from roboflow_workflows.core_steps.visualizations.text_display.v1 import (
    TextDisplayVisualizationBlockV1,
)
from roboflow_workflows.v2.blocks.query import QueryDefinitionError
from roboflow_workflows.v2.blocks.visualization.annotations.text_display import (
    TextDisplayBlock,
)
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from tests.v2.unit.blocks.visualization.annotations.support import (
    OBJECTS,
    detections,
    image_of,
    pixels_of,
    run_v1,
    scene,
)

HEIGHT, WIDTH = 72, 120


def validated(**parameters: Any) -> Dict[str, Any]:
    params = spec_of(TextDisplayBlock).validate_params(
        {"image": "$inputs.image", **parameters}
    )

    return dict(params)


def display(pixels: np.ndarray, **parameters: Any) -> np.ndarray:
    arguments = {**validated(**parameters), "image": image_of(pixels)}
    result = TextDisplayBlock().run(**arguments)
    drawn = pixels_of(result["image"])

    return drawn


def v1_display(pixels: np.ndarray, **parameters: Any) -> np.ndarray:
    arguments = {
        name: value
        for name, value in validated(**parameters).items()
        if name != "image"
    }
    expected = run_v1(TextDisplayVisualizationBlockV1, pixels, **arguments)

    return expected


V1_CASES = {
    "defaults": dict(text="Hello"),
    "center anchor, multiline, center align": dict(
        text="Cars\nin lane 3\n\nend",
        anchor="center",
        text_align="center",
        font_scale=0.5,
    ),
    "bottom right, right align, rounded": dict(
        text="one\nthree",
        anchor="bottom_right",
        text_align="right",
        border_radius=6,
        font_scale=0.4,
        offset_x=-4,
        offset_y=-4,
    ),
    "translucent rounded background": dict(
        text="50%", background_opacity=0.5, border_radius=8, background_color="#3366CC"
    ),
    "transparent background": dict(text="no box", background_color="Transparent"),
    "zero opacity draws text only": dict(text="text only", background_opacity=0.0),
    "absolute, partly outside": dict(
        text="edge", position_mode="absolute", position_x=110, position_y=-30
    ),
    "box larger than the image": dict(
        text="far too wide a line", font_scale=1.4, padding=4
    ),
    "negative offset clamps to the edge": dict(
        text="clamp", offset_x=-50, offset_y=-50
    ),
    "thick text, no padding": dict(
        text="W", font_thickness=4, padding=0, font_scale=1.2
    ),
    "top center anchor": dict(text="Top", anchor="top_center"),
}


@pytest.mark.parametrize("case", list(V1_CASES))
def test_pixels_equal_v1(case: str) -> None:
    pixels = scene(HEIGHT, WIDTH)

    drawn = display(pixels, **V1_CASES[case])

    np.testing.assert_array_equal(drawn, v1_display(pixels, **V1_CASES[case]))
    assert (drawn != pixels).any()


def test_anchor_ignores_case_when_selected() -> None:
    # V1 lower-cased the anchor; a selected "BOTTOM_LEFT" keeps working.
    pixels = scene(HEIGHT, WIDTH)
    params = validated(text="low", anchor="$inputs.anchor")

    result = TextDisplayBlock().run(
        **{**params, "image": image_of(pixels), "anchor": "BOTTOM_LEFT"}
    )

    expected = display(pixels, text="low", anchor="bottom_left")
    np.testing.assert_array_equal(pixels_of(result["image"]), expected)


def test_template_fills_parameters_after_their_operations() -> None:
    # Given: selected predictions counted by SequenceLength, and a plain value
    pixels = scene(HEIGHT, WIDTH)
    image = image_of(pixels)
    params = validated(
        text="Cars: {{ $parameters.cars }} at {{$parameters.zone}} ({{ $parameters.cars }})",
        text_parameters={"cars": "$steps.model.predictions", "zone": "$inputs.zone"},
        text_parameters_operations={"cars": [{"type": "SequenceLength"}]},
        font_scale=0.3,
    )
    params["text_parameters"] = {"cars": detections(image, OBJECTS), "zone": "gate"}

    # When
    result = TextDisplayBlock().run(**{**params, "image": image})

    # Then: the same pixels as the finished sentence
    expected = display(pixels, text="Cars: 3 at gate (3)", font_scale=0.3)
    np.testing.assert_array_equal(pixels_of(result["image"]), expected)


def test_template_matches_v1_for_unknown_placeholders_and_none() -> None:
    pixels = scene(HEIGHT, WIDTH)
    parameters = dict(
        text="{{ $parameters.missing }} / {{ $parameters.value }}",
        text_parameters={"value": None},
        font_scale=0.4,
    )

    drawn = display(pixels, **parameters)

    np.testing.assert_array_equal(drawn, v1_display(pixels, **parameters))


def test_input_image_is_unchanged() -> None:
    pixels = scene(HEIGHT, WIDTH)
    image = image_of(pixels, image_id="kept")

    result = TextDisplayBlock().run(**{**validated(text="x"), "image": image})

    np.testing.assert_array_equal(pixels_of(image), pixels)
    assert result["image"].image_id == "kept"


def test_operations_naming_no_parameter_fail_compilation() -> None:
    with pytest.raises(
        ParamsValidationError, match="names `count`, which is not a text_parameters key"
    ):
        validated(
            text="{{ $parameters.cars }}",
            text_parameters={"cars": "$steps.model.predictions"},
            text_parameters_operations={"count": [{"type": "SequenceLength"}]},
        )


def test_invalid_operation_fails_compilation() -> None:
    with pytest.raises((ParamsValidationError, QueryDefinitionError)):
        validated(
            text="{{ $parameters.cars }}",
            text_parameters={"cars": 1},
            text_parameters_operations={"cars": [{"type": "NoSuchOperation"}]},
        )


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("text_color", "nope", "text_color: invalid color"),
        ("background_color", "nope", "background_color: invalid color"),
        (
            "background_opacity",
            2,
            r"background_opacity must be a number in \[0.0, 1.0\]",
        ),
        ("font_scale", 0.05, r"font_scale must be a number in \[0.1, 10.0\]"),
        ("font_thickness", 11, "font_thickness must be at most 10"),
        ("padding", -1, "padding must be an integer >= 0"),
        ("border_radius", -2, "border_radius must be an integer >= 0"),
        # A literal fails StrictInt itself; the wording differs.
        ("position_x", 1.5, "position_x.* integer"),
    ],
)
def test_literal_and_selected_values_fail_with_the_same_message(
    field: str, value: Any, message: str
) -> None:
    with pytest.raises(ParamsValidationError, match=message):
        validated(text="x", **{field: value})

    params = validated(text="x", **{field: "$inputs.value"})
    with pytest.raises(ValueError, match=message):
        TextDisplayBlock().run(**{**params, "image": image_of(scene()), field: value})


@pytest.mark.parametrize(
    "field, value, message",
    [
        (
            "text_align",
            "middle",
            r"text_align must be one of \['left', 'center', 'right'\]",
        ),
        (
            "position_mode",
            "pixels",
            r"position_mode must be one of \['absolute', 'relative'\]",
        ),
        ("anchor", "middle", "anchor must be one of"),
    ],
)
def test_selected_choices_are_checked(field: str, value: str, message: str) -> None:
    # V1 drew an unknown text_align as left and position_mode as relative.
    params = validated(text="x", **{field: "$inputs.value"})

    with pytest.raises(ValueError, match=message):
        TextDisplayBlock().run(**{**params, "image": image_of(scene()), field: value})


def test_copy_image_is_rejected() -> None:
    with pytest.raises(ParamsValidationError, match="copy_image is not a V2 parameter"):
        validated(text="x", copy_image=True)
