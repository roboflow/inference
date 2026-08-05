import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.visualizations.common.fonts import (
    FONTS_REGISTRY,
)
from inference.core.workflows.core_steps.visualizations.rich_label.v1 import (
    RichLabelManifest,
    RichLabelVisualizationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def _build_image(height: int = 400, width: int = 400) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def _build_predictions(class_names=("person", "car")) -> sv.Detections:
    boxes = [[10 + 120 * i, 10, 100 + 120 * i, 100] for i in range(len(class_names))]

    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float64),
        class_id=np.arange(len(class_names)),
        confidence=np.full(len(class_names), 0.9),
        data={"class_name": np.array(class_names)},
    )


def _run_block(block: RichLabelVisualizationBlockV1, **overrides):
    kwargs = {
        "image": _build_image(),
        "predictions": _build_predictions(),
        "copy_image": True,
        "color_palette": "DEFAULT",
        "palette_size": 10,
        "custom_colors": None,
        "color_axis": "CLASS",
        "text": "Class",
        "text_position": "TOP_LEFT",
        "text_color": "WHITE",
        "font_family": "geist_mono",
        "text_size_mode": "Manual",
        "font_size": 14,
        "text_padding": 10,
        "border_radius": 0,
        "max_line_length": None,
    }
    kwargs.update(overrides)

    return block.run(**kwargs)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_rich_label_validation_when_valid_manifest_is_given(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/rich_label_visualization@v1",
        "name": "rich_label1",
        "predictions": "$steps.od_model.predictions",
        images_field_alias: "$inputs.image",
        "text": "Class",
        "text_position": "TOP_LEFT",
        "text_color": "WHITE",
        "font_family": "geist_mono",
        "font_size": 14,
        "text_padding": 10,
        "border_radius": 0,
    }

    # when
    result = RichLabelManifest.model_validate(data)

    # then
    assert result == RichLabelManifest(
        type="roboflow_core/rich_label_visualization@v1",
        name="rich_label1",
        images="$inputs.image",
        predictions="$steps.od_model.predictions",
        text="Class",
        text_position="TOP_LEFT",
        text_color="WHITE",
        font_family="Geist Mono",
        text_size_mode="Manual",
        font_size=14,
        text_padding=10,
        border_radius=0,
    )


def test_rich_label_validation_applies_geist_mono_as_default_font() -> None:
    # given
    data = {
        "type": "roboflow_core/rich_label_visualization@v1",
        "name": "rich_label1",
        "predictions": "$steps.od_model.predictions",
        "image": "$inputs.image",
    }

    # when
    result = RichLabelManifest.model_validate(data)

    # then
    assert result.font_family == "Geist Mono"


def test_rich_label_validation_accepts_legacy_snake_case_font_identifier() -> None:
    data = {
        "type": "roboflow_core/rich_label_visualization@v1",
        "name": "rich_label1",
        "predictions": "$steps.od_model.predictions",
        "image": "$inputs.image",
        "font_family": "geist_mono",
    }

    result = RichLabelManifest.model_validate(data)

    assert result.font_family == "Geist Mono"


def test_rich_label_validation_when_invalid_image_is_given() -> None:
    # given
    data = {
        "type": "roboflow_core/rich_label_visualization@v1",
        "name": "rich_label1",
        "images": "invalid",
        "predictions": "$steps.od_model.predictions",
    }

    # when
    with pytest.raises(ValidationError):
        _ = RichLabelManifest.model_validate(data)


@pytest.mark.parametrize("font_family", ["comic_sans", "/usr/share/fonts/evil.ttf"])
def test_rich_label_validation_when_unregistered_font_family_is_given(
    font_family: str,
) -> None:
    # given - unknown identifiers and filesystem paths must both be rejected
    data = {
        "type": "roboflow_core/rich_label_visualization@v1",
        "name": "rich_label1",
        "images": "$inputs.image",
        "predictions": "$steps.od_model.predictions",
        "font_family": font_family,
    }

    # when
    with pytest.raises(ValidationError):
        _ = RichLabelManifest.model_validate(data)


def test_manifest_font_family_enum_matches_fonts_registry() -> None:
    schema = RichLabelManifest.model_json_schema()
    enum_values = schema["properties"]["font_family"]["anyOf"][0]["enum"]

    assert set(enum_values) == {
        metadata.display_name for metadata in FONTS_REGISTRY.values()
    }


def test_rich_label_block_is_registered_in_loader() -> None:
    # given
    from inference.core.workflows.core_steps.loader import load_blocks

    # then
    assert RichLabelVisualizationBlockV1 in load_blocks()


def test_rich_label_visualization_block(bundled_fonts) -> None:
    # given
    block = RichLabelVisualizationBlockV1()

    # when
    output = _run_block(block)

    # then
    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")
    assert output.get("image").numpy_image.shape == (400, 400, 3)
    assert output.get("image").numpy_image.dtype == np.uint8
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((400, 400, 3), dtype=np.uint8)
    ), "Image should be modified by label rendering"


def test_rich_label_visualization_block_raises_on_unknown_font_delivered_at_runtime() -> (
    None
):
    # given - an unknown font id may reach run() through an input selector
    block = RichLabelVisualizationBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        _ = _run_block(block, font_family="comic_sans")

    # then
    assert "comic_sans" in str(error.value)
    assert "Geist Mono" in str(error.value)


def test_rich_label_visualization_block_raises_on_font_path_delivered_at_runtime() -> (
    None
):
    # given
    block = RichLabelVisualizationBlockV1()

    # when
    with pytest.raises(ValueError):
        _ = _run_block(block, font_family="/etc/passwd")


def test_rich_label_visualization_block_with_empty_detections() -> None:
    # given
    block = RichLabelVisualizationBlockV1()

    # when
    output = _run_block(block, predictions=sv.Detections.empty())

    # then
    assert output.get("image").numpy_image.shape == (400, 400, 3)
    assert np.array_equal(
        output.get("image").numpy_image, np.zeros((400, 400, 3), dtype=np.uint8)
    ), "Image should be unmodified when there are no detections"


def test_rich_label_visualization_block_with_non_ascii_labels(bundled_fonts) -> None:
    # given - Cyrillic and Greek characters, covered by the noto_sans build
    block = RichLabelVisualizationBlockV1()
    predictions = _build_predictions(class_names=("человек", "αυτοκίνητο"))

    # when
    output = _run_block(block, predictions=predictions, font_family="noto_sans")

    # then
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((400, 400, 3), dtype=np.uint8)
    ), "Non-ASCII labels should render visible text"


def test_rich_label_visualization_block_reuses_cached_annotator(bundled_fonts) -> None:
    # given
    block = RichLabelVisualizationBlockV1()

    # when
    _ = _run_block(block)
    _ = _run_block(block)

    # then
    assert len(block.annotatorCache) == 1, (
        "Same configuration should reuse a single annotator (and its "
        "loaded font) instead of reloading per frame"
    )

    # when
    _ = _run_block(block, font_size=24)
    _ = _run_block(block, font_family="inter", font_size=24)

    # then
    assert (
        len(block.annotatorCache) == 3
    ), "Different font configurations should produce distinct cache entries"


def test_rich_label_visualization_block_with_max_line_length(bundled_fonts) -> None:
    # given
    block = RichLabelVisualizationBlockV1()
    predictions = _build_predictions(
        class_names=("a very long label that should wrap over multiple lines",)
    )

    # when
    output = _run_block(block, predictions=predictions, max_line_length=10)

    # then
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((400, 400, 3), dtype=np.uint8)
    )
