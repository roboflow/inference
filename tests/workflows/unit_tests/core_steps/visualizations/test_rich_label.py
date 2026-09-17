import numpy as np
import pytest
import supervision as sv
import torch
from pydantic import ValidationError

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.visualizations.common.fonts import (
    FONTS_REGISTRY,
)
from inference.core.workflows.core_steps.visualizations.rich_label.v1 import (
    RichLabelManifest,
    RichLabelVisualizationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.rich_label.v1_tensor import (
    RichLabelVisualizationBlockV1 as RichLabelVisualizationBlockV1Tensor,
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


@_NUMPY_ONLY
def test_rich_label_block_is_registered_in_loader() -> None:
    # given
    from inference.core.workflows.core_steps.loader import load_blocks

    # then
    assert RichLabelVisualizationBlockV1 in load_blocks()


@_TENSOR_ONLY
def test_rich_label_tensor_block_is_registered_in_loader() -> None:
    # given
    from inference.core.workflows.core_steps.loader import load_blocks

    # when
    blocks = load_blocks()

    # then
    assert RichLabelVisualizationBlockV1Tensor in blocks
    assert RichLabelVisualizationBlockV1 not in blocks


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


def _build_native_predictions(class_names=("person", "car")) -> NativeDetections:
    boxes = [[10 + 120 * i, 10, 100 + 120 * i, 100] for i in range(len(class_names))]

    return NativeDetections(
        xyxy=torch.tensor(boxes, dtype=torch.float32),
        class_id=torch.arange(len(class_names), dtype=torch.long),
        confidence=torch.full((len(class_names),), 0.9, dtype=torch.float32),
        image_metadata={
            CLASS_NAMES_KEY: {i: name for i, name in enumerate(class_names)}
        },
    )


def _run_tensor_block(block: RichLabelVisualizationBlockV1Tensor, **overrides):
    kwargs = {
        "image": _build_image(),
        "predictions": _build_native_predictions(),
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


@_TENSOR_ONLY
def test_rich_label_visualization_block_tensor_native(bundled_fonts) -> None:
    # given
    block = RichLabelVisualizationBlockV1Tensor()

    # when
    output = _run_tensor_block(block)

    # then
    assert output is not None
    assert "image" in output
    assert hasattr(output.get("image"), "numpy_image")
    assert output.get("image").numpy_image.shape == (400, 400, 3)
    assert output.get("image").numpy_image.dtype == np.uint8
    assert not np.array_equal(
        output.get("image").numpy_image, np.zeros((400, 400, 3), dtype=np.uint8)
    ), "Image should be modified by label rendering"


@_TENSOR_ONLY
def test_rich_label_tensor_native_output_matches_numpy_block(bundled_fonts) -> None:
    # given - the same frame and semantically-identical predictions in both
    # representations; the tensor sibling must reproduce the numpy block's
    # rendering byte-for-byte (it reuses the numpy drawing internals).
    numpy_block = RichLabelVisualizationBlockV1()
    tensor_block = RichLabelVisualizationBlockV1Tensor()

    # when
    numpy_output = _run_block(numpy_block, text_size_mode="Automatic")
    tensor_output = _run_tensor_block(tensor_block, text_size_mode="Automatic")

    # then
    assert np.array_equal(
        numpy_output["image"].numpy_image, tensor_output["image"].numpy_image
    )


@_TENSOR_ONLY
def test_rich_label_visualization_block_tensor_native_raises_on_unknown_font() -> None:
    # given - an unknown font id may reach run() through an input selector
    block = RichLabelVisualizationBlockV1Tensor()

    # when
    with pytest.raises(ValueError) as error:
        _ = _run_tensor_block(block, font_family="comic_sans")

    # then
    assert "comic_sans" in str(error.value)


@_TENSOR_ONLY
def test_rich_label_visualization_block_with_empty_detections_tensor_native() -> None:
    # given
    block = RichLabelVisualizationBlockV1Tensor()
    empty_predictions = NativeDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
    )

    # when
    output = _run_tensor_block(block, predictions=empty_predictions)

    # then
    assert output.get("image").numpy_image.shape == (400, 400, 3)
    assert np.array_equal(
        output.get("image").numpy_image, np.zeros((400, 400, 3), dtype=np.uint8)
    ), "Image should be unmodified when there are no detections"


@_TENSOR_ONLY
def test_rich_label_tensor_native_empty_predictions_passthrough_is_device_resident() -> (
    None
):
    # given - a tensor-source image and an empty prediction: the sibling must
    # pass the tensor representation through without materialising numpy
    block = RichLabelVisualizationBlockV1Tensor()
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        tensor_image=torch.zeros((3, 240, 320), dtype=torch.uint8),
    )
    empty_predictions = NativeDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.long),
        confidence=torch.zeros((0,), dtype=torch.float32),
    )

    # when
    copied_output = _run_tensor_block(
        block, image=image, predictions=empty_predictions, copy_image=True
    )
    shared_output = _run_tensor_block(
        block, image=image, predictions=empty_predictions, copy_image=False
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
