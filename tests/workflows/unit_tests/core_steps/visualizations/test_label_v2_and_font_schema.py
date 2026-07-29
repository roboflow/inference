import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.visualizations.common.fonts.registry import (
    FONTS_REGISTRY,
)
from inference.core.workflows.core_steps.visualizations.label.v2 import (
    LabelManifestV2,
    LabelVisualizationBlockV2,
)
from inference.core.workflows.core_steps.visualizations.rich_label.v1 import (
    RichLabelManifest,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def test_rich_label_font_schema_is_inline_enum_of_display_names() -> None:
    schema = RichLabelManifest.model_json_schema()
    font_family = schema["properties"]["font_family"]
    font_branch = font_family["anyOf"][0]
    text_branch = schema["properties"]["text"]["anyOf"][0]

    # human-readable display names, one per registry entry
    assert set(font_branch["enum"]) == {
        metadata.display_name for metadata in FONTS_REGISTRY.values()
    }
    assert "geist_mono" not in font_branch["enum"]
    assert font_family["default"] == "Geist Mono"
    # inline enum with the same shape as the `text` field - the Workflow
    # Builder renders a dropdown only for inline enums (not $refs)
    assert set(text_branch.keys()) == set(font_branch.keys()) == {"enum", "type"}


def test_rich_label_manifest_exposes_font_title() -> None:
    schema = RichLabelManifest.model_json_schema()
    font_family = schema["properties"]["font_family"]

    assert font_family["title"] == "Font"
    assert schema["properties"]["font_size"]["default"] == 14
    assert schema["properties"]["text_size_mode"]["default"] == "Manual"


def test_label_v2_manifest_defaults_to_manual_text_size_mode() -> None:
    schema = LabelManifestV2.model_json_schema()

    assert schema["properties"]["text_size_mode"]["default"] == "Manual"
    assert schema["properties"]["text_scale"]["title"] == "Scale"


def test_label_v2_block_is_registered_in_loader() -> None:
    from inference.core.workflows.core_steps.loader import load_blocks

    assert LabelVisualizationBlockV2 in load_blocks()


def test_label_v2_automatic_mode_scales_text_for_smaller_images(bundled_fonts) -> None:
    block = LabelVisualizationBlockV2()
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((540, 960, 3), dtype=np.uint8),
    )
    predictions = sv.Detections(
        xyxy=np.array([[10, 10, 100, 100]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
        data={"class_name": np.array(["person"])},
    )

    manual_output = block.run(
        image=image,
        predictions=predictions,
        copy_image=True,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=None,
        color_axis="CLASS",
        text="Class",
        text_position="TOP_LEFT",
        text_color="WHITE",
        text_size_mode="Manual",
        text_scale=1.0,
        text_thickness=1,
        text_padding=10,
        border_radius=0,
    )
    automatic_output = block.run(
        image=image,
        predictions=predictions,
        copy_image=True,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=None,
        color_axis="CLASS",
        text="Class",
        text_position="TOP_LEFT",
        text_color="WHITE",
        text_size_mode="Automatic",
        text_scale=1.0,
        text_thickness=1,
        text_padding=10,
        border_radius=0,
    )

    assert not np.array_equal(
        manual_output["image"].numpy_image, automatic_output["image"].numpy_image
    )
