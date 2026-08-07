import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.visualizations.common.fonts.registry import (
    FONTS_REGISTRY,
)
from inference.core.workflows.core_steps.visualizations.label.v2 import (
    LabelManifestV2,
    LabelVisualizationBlockV2,
)
from inference.core.workflows.core_steps.visualizations.label.v2_tensor import (
    LabelVisualizationBlockV2 as LabelVisualizationBlockV2Tensor,
)
from inference.core.workflows.core_steps.visualizations.rich_label.v1 import (
    RichLabelManifest,
)
from inference.core.workflows.execution_engine.constants import CLASS_NAMES_KEY
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections as NativeDetections

# The loader binds `_tensor` siblings under the same names when
# ENABLE_TENSOR_DATA_REPRESENTATION is set - hence the flag-opposed
# _NUMPY_ONLY / _TENSOR_ONLY split below.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="loader binds the tensor-native sibling under "
    "ENABLE_TENSOR_DATA_REPRESENTATION — see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
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


@_NUMPY_ONLY
def test_label_v2_block_is_registered_in_loader() -> None:
    from inference.core.workflows.core_steps.loader import load_blocks

    assert LabelVisualizationBlockV2 in load_blocks()


@_TENSOR_ONLY
def test_label_v2_tensor_block_is_registered_in_loader() -> None:
    from inference.core.workflows.core_steps.loader import load_blocks

    blocks = load_blocks()
    assert LabelVisualizationBlockV2Tensor in blocks
    assert LabelVisualizationBlockV2 not in blocks


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


def _native_predictions() -> NativeDetections:
    return NativeDetections(
        xyxy=torch.tensor([[10, 10, 100, 100]], dtype=torch.float32),
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([0.9], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "person"}},
    )


def _run_label_v2_tensor_block(block, image, predictions, **overrides):
    kwargs = {
        "image": image,
        "predictions": predictions,
        "copy_image": True,
        "color_palette": "DEFAULT",
        "palette_size": 10,
        "custom_colors": None,
        "color_axis": "CLASS",
        "text": "Class",
        "text_position": "TOP_LEFT",
        "text_color": "WHITE",
        "text_size_mode": "Manual",
        "text_scale": 1.0,
        "text_thickness": 1,
        "text_padding": 10,
        "border_radius": 0,
    }
    kwargs.update(overrides)
    return block.run(**kwargs)


@_TENSOR_ONLY
def test_label_v2_automatic_mode_scales_text_for_smaller_images_tensor_native(
    bundled_fonts,
) -> None:
    block = LabelVisualizationBlockV2Tensor()
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((540, 960, 3), dtype=np.uint8),
    )

    manual_output = _run_label_v2_tensor_block(
        block, image, _native_predictions(), text_size_mode="Manual"
    )
    automatic_output = _run_label_v2_tensor_block(
        block, image, _native_predictions(), text_size_mode="Automatic"
    )

    assert not np.array_equal(
        manual_output["image"].numpy_image, automatic_output["image"].numpy_image
    )


@_TENSOR_ONLY
def test_label_v2_tensor_native_output_matches_numpy_block(bundled_fonts) -> None:
    # given - the same frame and semantically-identical predictions in both
    # representations; the tensor sibling must reproduce the numpy block's
    # rendering byte-for-byte (it reuses the numpy drawing internals).
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((540, 960, 3), dtype=np.uint8),
    )
    sv_predictions = sv.Detections(
        xyxy=np.array([[10, 10, 100, 100]], dtype=np.float64),
        class_id=np.array([0]),
        confidence=np.array([0.9]),
        data={"class_name": np.array(["person"])},
    )

    # when
    numpy_output = LabelVisualizationBlockV2().run(
        image=image,
        predictions=sv_predictions,
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
    tensor_output = _run_label_v2_tensor_block(
        LabelVisualizationBlockV2Tensor(),
        image,
        _native_predictions(),
        text_size_mode="Automatic",
    )

    # then
    assert np.array_equal(
        numpy_output["image"].numpy_image, tensor_output["image"].numpy_image
    )


@_TENSOR_ONLY
def test_label_v2_tensor_native_empty_predictions_passthrough_is_device_resident() -> (
    None
):
    # given - a tensor-source image and an empty prediction: the sibling must
    # pass the tensor representation through without materialising numpy
    block = LabelVisualizationBlockV2Tensor()
    tensor = torch.zeros((3, 240, 320), dtype=torch.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        tensor_image=tensor,
    )
    empty_predictions = NativeDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
    )

    # when
    copied_output = _run_label_v2_tensor_block(
        block, image, empty_predictions, copy_image=True
    )
    shared_output = _run_label_v2_tensor_block(
        block, image, empty_predictions, copy_image=False
    )

    # then
    assert image._numpy_image is None, "passthrough must not materialise numpy"
    for output in [copied_output, shared_output]:
        assert output["image"].is_tensor_materialised() is True
        assert output["image"]._numpy_image is None
        assert torch.equal(output["image"].tensor_image, image.tensor_image)
    # copy semantics: independent storage when copy_image=True, shared otherwise
    assert (
        copied_output["image"].tensor_image.data_ptr()
        != image.tensor_image.data_ptr()
    )
    assert (
        shared_output["image"].tensor_image.data_ptr()
        == image.tensor_image.data_ptr()
    )
