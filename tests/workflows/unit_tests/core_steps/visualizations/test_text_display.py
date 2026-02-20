import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    SequenceLength,
)
from inference.core.workflows.core_steps.visualizations.text_display.v1 import (
    BlockManifest,
    TextDisplayVisualizationBlockV1,
    format_text_with_parameters,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def test_text_display_validation_when_valid_manifest_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/text_display@v1",
        "name": "text_display1",
        "image": "$inputs.image",
        "text": "Hello World",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.type == "roboflow_core/text_display@v1"
    assert result.name == "text_display1"
    assert result.image == "$inputs.image"
    assert result.text == "Hello World"


def test_text_display_validation_when_full_manifest_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/text_display@v1",
        "name": "text_display1",
        "image": "$inputs.image",
        "text": "Count: {{ $parameters.count }}",
        "text_parameters": {"count": "$steps.model.predictions"},
        "text_parameters_operations": {"count": [{"type": "SequenceLength"}]},
        "text_color": "WHITE",
        "background_color": "BLACK",
        "background_opacity": 0.8,
        "font_scale": 1.5,
        "font_thickness": 2,
        "padding": 15,
        "text_align": "center",
        "border_radius": 5,
        "position_mode": "relative",
        "anchor": "top_left",
        "offset_x": 10,
        "offset_y": 10,
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.text == "Count: {{ $parameters.count }}"
    assert result.text_parameters == {"count": "$steps.model.predictions"}
    assert result.text_color == "WHITE"
    assert result.background_opacity == 0.8
    assert result.font_scale == 1.5
    assert result.text_align == "center"
    assert result.position_mode == "relative"
    assert result.anchor == "top_left"


def test_text_display_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/text_display@v1",
        "name": "text_display1",
        "image": "invalid",
        "text": "Hello World",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_text_display_visualization_block_basic() -> None:
    # given
    block = TextDisplayVisualizationBlockV1()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((500, 500, 3), dtype=np.uint8),
        ),
        text="Hello World",
        text_parameters={},
        text_parameters_operations={},
        text_color="WHITE",
        background_color="BLACK",
        background_opacity=1.0,
        font_scale=1.0,
        font_thickness=2,
        padding=10,
        text_align="left",
        border_radius=0,
        position_mode="relative",
        position_x=0,
        position_y=0,
        anchor="top_left",
        offset_x=10,
        offset_y=10,
        copy_image=True,
    )

    # then
    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")
    # dimensions of output match input
    assert output.get("image").numpy_image.shape == (500, 500, 3)
    # check if the image is modified (text was drawn)
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((500, 500, 3), dtype=np.uint8)
    )


def test_text_display_visualization_block_with_absolute_positioning() -> None:
    # given
    block = TextDisplayVisualizationBlockV1()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((500, 500, 3), dtype=np.uint8),
        ),
        text="Positioned Text",
        text_parameters={},
        text_parameters_operations={},
        text_color="WHITE",
        background_color="BLACK",
        background_opacity=1.0,
        font_scale=1.0,
        font_thickness=2,
        padding=10,
        text_align="left",
        border_radius=0,
        position_mode="absolute",
        position_x=100,
        position_y=100,
        anchor="top_left",
        offset_x=0,
        offset_y=0,
        copy_image=True,
    )

    # then
    assert output is not None
    assert "image" in output
    assert output.get("image").numpy_image.shape == (500, 500, 3)


def test_text_display_visualization_block_with_transparent_background() -> None:
    # given
    block = TextDisplayVisualizationBlockV1()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((500, 500, 3), dtype=np.uint8),
        ),
        text="No Background",
        text_parameters={},
        text_parameters_operations={},
        text_color="WHITE",
        background_color="transparent",
        background_opacity=1.0,
        font_scale=1.0,
        font_thickness=2,
        padding=10,
        text_align="left",
        border_radius=0,
        position_mode="relative",
        position_x=0,
        position_y=0,
        anchor="center",
        offset_x=0,
        offset_y=0,
        copy_image=True,
    )

    # then
    assert output is not None
    assert "image" in output


def test_text_display_visualization_block_with_multiline_text() -> None:
    # given
    block = TextDisplayVisualizationBlockV1()

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((500, 500, 3), dtype=np.uint8),
        ),
        text="Line 1\nLine 2\nLine 3",
        text_parameters={},
        text_parameters_operations={},
        text_color="WHITE",
        background_color="BLACK",
        background_opacity=1.0,
        font_scale=1.0,
        font_thickness=2,
        padding=10,
        text_align="center",
        border_radius=5,
        position_mode="relative",
        position_x=0,
        position_y=0,
        anchor="center",
        offset_x=0,
        offset_y=0,
        copy_image=True,
    )

    # then
    assert output is not None
    assert "image" in output


def test_text_display_visualization_block_copy_image_false() -> None:
    # given
    block = TextDisplayVisualizationBlockV1()
    original_image = np.zeros((500, 500, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=original_image,
        ),
        text="Hello",
        text_parameters={},
        text_parameters_operations={},
        text_color="WHITE",
        background_color="BLACK",
        background_opacity=1.0,
        font_scale=1.0,
        font_thickness=2,
        padding=10,
        text_align="left",
        border_radius=0,
        position_mode="relative",
        position_x=0,
        position_y=0,
        anchor="top_left",
        offset_x=10,
        offset_y=10,
        copy_image=False,
    )

    # then
    assert output is not None
    # When copy_image=False, the original image should be modified in place
    assert not np.array_equal(original_image, np.zeros((500, 500, 3), dtype=np.uint8))
    # Output should share memory with the original input array
    assert np.shares_memory(output["image"].numpy_image, original_image)


def test_text_display_visualization_block_copy_image_true() -> None:
    # given
    block = TextDisplayVisualizationBlockV1()
    original_image = np.zeros((500, 500, 3), dtype=np.uint8)

    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=original_image,
        ),
        text="Hello",
        text_parameters={},
        text_parameters_operations={},
        text_color="WHITE",
        background_color="BLACK",
        background_opacity=1.0,
        font_scale=1.0,
        font_thickness=2,
        padding=10,
        text_align="left",
        border_radius=0,
        position_mode="relative",
        position_x=0,
        position_y=0,
        anchor="top_left",
        offset_x=10,
        offset_y=10,
        copy_image=True,
    )

    # then
    assert output is not None
    # When copy_image=True, the original image should NOT be modified
    assert np.array_equal(original_image, np.zeros((500, 500, 3), dtype=np.uint8))
    # Output should NOT share memory with the original input array
    assert not np.shares_memory(output["image"].numpy_image, original_image)


def test_format_text_with_parameters_simple() -> None:
    # given
    text = "Count: {{ $parameters.count }}"
    text_parameters = {"count": 5}
    text_parameters_operations = {}

    # when
    result = format_text_with_parameters(
        text, text_parameters, text_parameters_operations
    )

    # then
    assert result == "Count: 5"


def test_format_text_with_parameters_multiple() -> None:
    # given
    text = "Found {{ $parameters.count }} objects of class {{ $parameters.class_name }}"
    text_parameters = {"count": 3, "class_name": "person"}
    text_parameters_operations = {}

    # when
    result = format_text_with_parameters(
        text, text_parameters, text_parameters_operations
    )

    # then
    assert result == "Found 3 objects of class person"


def test_format_text_with_parameters_missing_parameter() -> None:
    # given
    text = "Value: {{ $parameters.missing }}"
    text_parameters = {}
    text_parameters_operations = {}

    # when
    result = format_text_with_parameters(
        text, text_parameters, text_parameters_operations
    )

    # then
    # Missing parameters should be left as-is
    assert result == "Value: {{ $parameters.missing }}"


def test_format_text_with_parameters_with_operations() -> None:
    # given
    text = "Count: {{ $parameters.items }}"
    text_parameters = {"items": [1, 2, 3, 4, 5]}
    text_parameters_operations = {"items": [SequenceLength(type="SequenceLength")]}

    # when
    result = format_text_with_parameters(
        text, text_parameters, text_parameters_operations
    )

    # then
    assert result == "Count: 5"


def test_format_text_with_no_parameters() -> None:
    # given
    text = "Static text with no parameters"
    text_parameters = {}
    text_parameters_operations = {}

    # when
    result = format_text_with_parameters(
        text, text_parameters, text_parameters_operations
    )

    # then
    assert result == "Static text with no parameters"


@pytest.mark.parametrize(
    "anchor",
    [
        "center",
        "top_left",
        "top_center",
        "top_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
        "center_left",
        "center_right",
    ],
)
def test_text_display_all_anchor_positions(anchor: str) -> None:
    # given
    block = TextDisplayVisualizationBlockV1()

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((500, 500, 3), dtype=np.uint8),
        ),
        text="Test",
        text_parameters={},
        text_parameters_operations={},
        text_color="WHITE",
        background_color="BLACK",
        background_opacity=1.0,
        font_scale=1.0,
        font_thickness=2,
        padding=10,
        text_align="left",
        border_radius=0,
        position_mode="relative",
        position_x=0,
        position_y=0,
        anchor=anchor,
        offset_x=0,
        offset_y=0,
        copy_image=True,
    )

    # then
    assert output is not None
    assert "image" in output


@pytest.mark.parametrize("text_align", ["left", "center", "right"])
def test_text_display_all_alignments(text_align: str) -> None:
    # given
    block = TextDisplayVisualizationBlockV1()

    # when
    output = block.run(
        image=WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((500, 500, 3), dtype=np.uint8),
        ),
        text="Line 1\nLonger Line 2",
        text_parameters={},
        text_parameters_operations={},
        text_color="WHITE",
        background_color="BLACK",
        background_opacity=1.0,
        font_scale=1.0,
        font_thickness=2,
        padding=10,
        text_align=text_align,
        border_radius=0,
        position_mode="relative",
        position_x=0,
        position_y=0,
        anchor="center",
        offset_x=0,
        offset_y=0,
        copy_image=True,
    )

    # then
    assert output is not None
    assert "image" in output
