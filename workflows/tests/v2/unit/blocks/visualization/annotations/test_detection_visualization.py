"""Detection visualization: boxes then labels, by reference to the two blocks.

The oracle of each implementation is the box implementation of the same
quality chained into the label implementation of the same quality, run with
the same parameters. Equality with that chain covers label texts, row order,
palette, anchors, frame mapping and global occlusion (all boxes, then all
labels).
"""

import copy
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pytest
import torch
from pydantic import Field
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BLOCKS,
    BoundingBoxVisualizationBlock,
    DetectionVisualizationBlock,
    LabelVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.bounding_box_visualization.v1 import (
    AccurateBoundingBoxes,
    FastBoundingBoxes,
)
from roboflow_workflows.v2.blocks.visualization.annotations.detection_visualization.v1 import (
    AccurateDetections,
    FastDetections,
)
from roboflow_workflows.v2.blocks.visualization.annotations.label_visualization.v2 import (
    AccurateLabels,
    FastLabels,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import RowRef
from roboflow_workflows.v2.engine import (
    Block,
    BlockParams,
    Catalogue,
    CompileOptions,
    Output,
    Ref,
    compile_workflow,
)
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from tests.v2.unit.blocks.visualization.annotations.support import (
    OBJECTS,
    detections,
    image_of,
    instance_detections,
    scene,
)

SMALL_TEXT = {
    "text_size_mode": "Manual",
    "text_scale": 0.4,
    "text_thickness": 1,
    "text_padding": 3,
}
PALETTE_FIELDS = ("color_palette", "palette_size", "custom_colors", "color_axis")
BOX_FIELDS = ("thickness", "roundness")
CHAINS = {
    AccurateDetections: (AccurateBoundingBoxes, AccurateLabels),
    FastDetections: (FastBoundingBoxes, FastLabels),
}
IMPLEMENTATIONS = [AccurateDetections, FastDetections]
CROP_XYXY = (10, 8, 50, 40)


# -- helpers ------------------------------------------------------------------


def _run_with(
    implementation: Any,
    block_class: type,
    *,
    image: ImageData,
    predictions: Any,
    selected: Dict[str, Any] = None,
    **literals: Any,
) -> ImageData:
    # Validate like the engine (selectors for "selected"), then run with values.
    selected = selected or {}
    params = spec_of(block_class).validate_params(
        {
            **literals,
            **{name: f"$inputs.{name}" for name in selected},
            "image": "$inputs.image",
            "predictions": "$inputs.predictions",
        }
    )
    if isinstance(implementation, type):
        implementation = implementation()
    arguments = {**dict(params), **selected, "image": image, "predictions": predictions}
    result = implementation.run(**arguments)

    return result["image"]


def _chain(
    implementation: type, *, image: ImageData, predictions: Any, **parameters: Any
) -> ImageData:
    boxes, labels = CHAINS[implementation]
    box_values = {
        name: value
        for name, value in parameters.items()
        if name in PALETTE_FIELDS + BOX_FIELDS
    }
    label_values = {
        name: value for name, value in parameters.items() if name not in BOX_FIELDS
    }
    boxed = _run_with(
        boxes,
        BoundingBoxVisualizationBlock,
        image=image,
        predictions=predictions,
        **box_values,
    )
    labelled = _run_with(
        labels,
        LabelVisualizationBlock,
        image=boxed,
        predictions=predictions,
        **label_values,
    )

    return labelled


def _assert_matches_chain(
    implementation: type, *, image: ImageData, predictions: Any, **parameters: Any
) -> ImageData:
    drawn = _run_with(
        implementation,
        DetectionVisualizationBlock,
        image=image,
        predictions=predictions,
        **parameters,
    )
    expected = _chain(
        implementation, image=image, predictions=predictions, **parameters
    )
    assert torch.equal(drawn.tensor_image, expected.tensor_image)

    return drawn


def _tensors_of(predictions: Any) -> list:
    names = ("xyxy", "class_id", "confidence", "mask")
    tensors = [
        getattr(predictions, name)
        for name in names
        if isinstance(getattr(predictions, name, None), torch.Tensor)
    ]

    return tensors


# -- contract -----------------------------------------------------------------


def test_block_is_registered_once_with_accurate_first() -> None:
    assert BLOCKS.count(DetectionVisualizationBlock) == 1
    assert (
        DetectionVisualizationBlock.type == "roboflow_core/detection_visualization@v1"
    )
    assert DetectionVisualizationBlock.implementations == (
        AccurateDetections,
        FastDetections,
    )
    names = [item.name for item in spec_of(DetectionVisualizationBlock).implementations]
    assert names == ["accurate", "fast"]
    assert list(DetectionVisualizationBlock.outputs) == ["image"]
    assert DetectionVisualizationBlock.outputs["image"].source == "image"


def test_params_are_the_label_params_plus_box_width_and_roundness() -> None:
    params = spec_of(DetectionVisualizationBlock).validate_params(
        {"image": "$inputs.image", "predictions": "$inputs.predictions"}
    )
    label_defaults = spec_of(LabelVisualizationBlock).validate_params(
        {"image": "$inputs.image", "predictions": "$inputs.predictions"}
    )

    fields = spec_of(DetectionVisualizationBlock).params_model.model_fields
    label_fields = spec_of(LabelVisualizationBlock).params_model.model_fields
    assert set(fields) == set(label_fields) | {"thickness", "roundness"}
    assert params.thickness == 2
    assert params.roundness == 0.0
    for name in label_fields:
        assert getattr(params, name) == getattr(label_defaults, name)


@pytest.mark.parametrize("copy_image", [True, False])
def test_copy_image_is_rejected_with_a_remedy(copy_image: bool) -> None:
    with pytest.raises(ParamsValidationError) as caught:
        spec_of(DetectionVisualizationBlock).validate_params(
            {
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                "copy_image": copy_image,
            }
        )

    message = str(caught.value) + str(caught.value.__cause__)
    assert "copy_image" in message and "delete the field" in message


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("thickness", 0, "thickness must be an integer >= 1"),
        ("roundness", 1.5, "roundness must be a number in [0.0, 1.0]"),
        ("roundness", -0.1, "roundness must be a number in [0.0, 1.0]"),
        ("text_scale", 0, "text_scale must be a number > 0.0"),
        ("text_padding", -1, "text_padding must be an integer >= 0"),
        ("border_radius", -1, "border_radius must be an integer >= 0"),
        ("palette_size", 0, "palette_size must be at least 1, got 0"),
    ],
)
def test_invalid_literals_fail_at_compile(name: str, value: Any, message: str) -> None:
    # Box checks are new; label and palette checks are the inherited validators.
    with pytest.raises(ParamsValidationError) as caught:
        spec_of(DetectionVisualizationBlock).validate_params(
            {
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
                name: value,
            }
        )

    assert message in str(caught.value) + str(caught.value.__cause__)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("thickness", 0, "thickness must be an integer >= 1"),
        ("thickness", 2.5, "thickness must be an integer >= 1"),
        ("thickness", True, "thickness must be an integer >= 1"),
        ("roundness", 1.5, "roundness must be a number in [0.0, 1.0]"),
        ("roundness", "round", "roundness must be a number in [0.0, 1.0]"),
        ("text_padding", -1, "text_padding must be an integer >= 0"),
        ("text_position", "NOWHERE", "text_position must be one of"),
    ],
)
def test_invalid_selected_values_fail_before_drawing(
    implementation: type, name: str, value: Any, message: str
) -> None:
    image = image_of(scene())
    predictions = detections(image)
    original = image.tensor_image.clone()

    with pytest.raises(ValueError, match="must be") as caught:
        _run_with(
            implementation,
            DetectionVisualizationBlock,
            image=image,
            predictions=predictions,
            selected={name: value},
        )

    assert message in str(caught.value)
    assert torch.equal(image.tensor_image, original)


# -- selection ----------------------------------------------------------------


class FixedDetections(Block):
    type = "test/detection_visualization_rows@v1"
    outputs = {"predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image for synthetic detections.")

    def run(self, *, image):
        result = {"predictions": detections(image)}

        return result


def _definition(execution: Any = None) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "images", "kind": ["image"]}],
        "steps": [
            {"type": FixedDetections.type, "name": "detect", "image": "$inputs.images"},
            {
                "type": DetectionVisualizationBlock.type,
                "name": "draw",
                "image": "$inputs.images",
                "predictions": "$steps.detect.predictions",
                **SMALL_TEXT,
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "drawn", "selector": "$steps.draw.image"}
        ],
    }
    if execution is not None:
        definition["execution"] = execution

    return definition


@pytest.mark.parametrize(
    ("execution", "deployment", "expected"),
    [
        (None, None, "accurate"),
        (None, "accurate", "accurate"),
        (None, "fast", "fast"),
        ({"quality": "fast"}, None, "fast"),
        ({"quality": "accurate"}, "fast", "accurate"),
    ],
)
def test_quality_selects_the_implementation_and_the_workflow_runs(
    execution: Any, deployment: Any, expected: str
) -> None:
    catalogue = Catalogue([FixedDetections, DetectionVisualizationBlock])
    plan = compile_workflow(
        _definition(execution),
        catalogue=catalogue,
        options=CompileOptions(quality=deployment),
    )
    assert plan.step(("draw",)).implementation.name == expected
    image = image_of(scene(), image_id="source")
    original = image.tensor_image.clone()

    with plan.create_session() as session:
        (row,) = session.run({"images": [image]}).rows()

    implementation = {"accurate": AccurateDetections, "fast": FastDetections}[expected]
    reference = _chain(
        implementation, image=image, predictions=detections(image), **SMALL_TEXT
    )
    assert torch.equal(image.tensor_image, original)
    assert row["drawn"].image_id == image.image_id
    assert torch.equal(row["drawn"].tensor_image, reference.tensor_image)


# -- pixels by reference ------------------------------------------------------


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "objects", [(), OBJECTS[:1], OBJECTS], ids=["empty", "single", "overlapping"]
)
@pytest.mark.parametrize(
    "style",
    [
        {},
        {"thickness": 4, "roundness": 0.5},
        {"thickness": 1, "color_axis": "INDEX", "color_palette": "ROBOFLOW"},
        {
            "color_palette": "CUSTOM",
            "custom_colors": ["#f10d1d", "#07e529", "#1325e9"],
            "text_color": "#000000",
            "border_radius": 2,
        },
    ],
)
def test_output_equals_the_box_then_label_chain(
    implementation: type, objects: tuple, style: Dict[str, Any]
) -> None:
    image = image_of(scene())

    drawn = _assert_matches_chain(
        implementation,
        image=image,
        predictions=detections(image, objects),
        **SMALL_TEXT,
        **style,
    )

    if objects:
        assert not torch.equal(drawn.tensor_image, image.tensor_image)
    else:
        assert torch.equal(drawn.tensor_image, image.tensor_image)
        assert drawn.tensor_image.data_ptr() != image.tensor_image.data_ptr()


def test_all_boxes_are_drawn_before_any_label() -> None:
    # Row 0's label at its top-left corner sits inside row 1's box outline
    # band only if drawn first; per-object interleaving would let box 1 cover it.
    image = image_of(np.zeros((64, 80, 3), dtype=np.uint8))
    objects = ((10.0, 30.0, 70.0, 60.0, 0, 0.9), (5.0, 10.0, 75.0, 25.0, 1, 0.8))
    style = dict(
        SMALL_TEXT,
        thickness=6,
        color_palette="CUSTOM",
        custom_colors=["#ff0000", "#0000ff"],
    )
    predictions = detections(image, objects)

    drawn = _run_with(
        AccurateDetections,
        DetectionVisualizationBlock,
        image=image,
        predictions=predictions,
        **style,
    )
    labels_only = _run_with(
        AccurateLabels,
        LabelVisualizationBlock,
        image=image,
        predictions=detections(image, objects[:1]),
        **{key: value for key, value in style.items() if key != "thickness"},
    )
    boxes_only = _run_with(
        AccurateBoundingBoxes,
        BoundingBoxVisualizationBlock,
        image=image,
        predictions=predictions,
        **{key: value for key, value in style.items() if key not in SMALL_TEXT},
    )
    label_pixels = labels_only.tensor_image.any(dim=0)

    assert label_pixels.any()
    # The case is meaningful: a later row's box crosses row 0's label.
    blue = torch.tensor([0, 0, 255], dtype=torch.uint8)[:, None]
    under_label = boxes_only.tensor_image[:, label_pixels]
    assert (under_label == blue).all(dim=0).any()
    assert torch.equal(
        drawn.tensor_image[:, label_pixels], labels_only.tensor_image[:, label_pixels]
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    "text",
    [
        "Class",
        "Confidence",
        "Class and Confidence",
        "Index",
        "Dimensions",
        "Tracker Id",
        "Time In Zone",
        "plate",
    ],
)
def test_label_texts_and_metadata_keys_follow_the_label_block(
    implementation: type, text: str
) -> None:
    image = image_of(scene())
    rows = [{"plate": "AB 123"}, {"plate": "XY 9"}, {}]

    _assert_matches_chain(
        implementation,
        image=image,
        predictions=detections(image, rows=rows),
        selected={"text": text},
        **SMALL_TEXT,
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize(
    ("text", "position"), [("Area", "TOP_LEFT"), ("Class", "CENTER_OF_MASS")]
)
def test_instance_masks_serve_area_and_centre_of_mass(
    implementation: type, text: str, position: str
) -> None:
    image = image_of(scene())

    _assert_matches_chain(
        implementation,
        image=image,
        predictions=instance_detections(image),
        text=text,
        text_position=position,
        **SMALL_TEXT,
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_label_block_errors_are_kept(implementation: type) -> None:
    image = image_of(scene())

    with pytest.raises(ValueError, match="CENTER_OF_MASS"):
        _run_with(
            implementation,
            DetectionVisualizationBlock,
            image=image,
            predictions=detections(image),
            text_position="CENTER_OF_MASS",
        )
    with pytest.raises(ValueError, match="neither one of"):
        _run_with(
            implementation,
            DetectionVisualizationBlock,
            image=image,
            predictions=detections(image),
            selected={"text": "unknown_key"},
        )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_selected_values_draw_like_the_same_literals(implementation: type) -> None:
    image = image_of(scene())
    predictions = detections(image)
    values = {"thickness": 3, "roundness": 0.5, "text": "Class and Confidence"}

    selected = _run_with(
        implementation,
        DetectionVisualizationBlock,
        image=image,
        predictions=predictions,
        selected=values,
        **SMALL_TEXT,
    )
    literal = _run_with(
        implementation,
        DetectionVisualizationBlock,
        image=image,
        predictions=predictions,
        **values,
        **SMALL_TEXT,
    )

    assert torch.equal(selected.tensor_image, literal.tensor_image)


# -- frames -------------------------------------------------------------------


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_crop_predictions_are_drawn_where_they_lie_in_the_parent(
    implementation: type,
) -> None:
    image = image_of(scene())
    crop = image.crop(CROP_XYXY, image_id="crop")
    local = [(2.5, 3.25, 20.75, 18.5, 0, 0.9), (15.0, 10.5, 38.25, 31.0, 2, 0.4)]
    x0, y0 = CROP_XYXY[:2]
    shifted = [
        (x1 + x0, y1 + y0, x2 + x0, y2 + y0, *rest) for x1, y1, x2, y2, *rest in local
    ]

    drawn = _assert_matches_chain(
        implementation,
        image=image,
        predictions=detections(crop, local),
        **SMALL_TEXT,
    )
    in_parent = _run_with(
        implementation,
        DetectionVisualizationBlock,
        image=image,
        predictions=detections(image, shifted),
        **SMALL_TEXT,
    )

    assert torch.equal(drawn.tensor_image, in_parent.tensor_image)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_parent_predictions_drawn_on_a_crop_are_clipped_to_it(
    implementation: type,
) -> None:
    image = image_of(scene())
    crop = image.crop(CROP_XYXY, image_id="crop")

    drawn = _assert_matches_chain(
        implementation,
        image=crop,
        predictions=detections(image),
        **SMALL_TEXT,
    )

    assert drawn.size_hw == crop.size_hw
    assert not torch.equal(drawn.tensor_image, crop.tensor_image)


# -- ownership ----------------------------------------------------------------


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("carrier", ["boxes", "masks"])
def test_input_image_and_predictions_are_not_changed(
    implementation: type, carrier: str
) -> None:
    image = image_of(scene())
    if carrier == "boxes":
        predictions = detections(image, rows=[{"plate": "A"}, {}, {"plate": "C"}])
    else:
        predictions = instance_detections(image)
    image_before = image.tensor_image.clone()
    tensors_before = [tensor.clone() for tensor in _tensors_of(predictions)]
    image_metadata_before = copy.deepcopy(predictions.image_metadata)
    rows_before = copy.deepcopy(predictions.bboxes_metadata)

    drawn = _run_with(
        implementation,
        DetectionVisualizationBlock,
        image=image,
        predictions=predictions,
        selected={"text": "Area" if carrier == "masks" else "plate"},
        **SMALL_TEXT,
    )

    assert torch.equal(image.tensor_image, image_before)
    for before, after in zip(tensors_before, _tensors_of(predictions)):
        assert torch.equal(before, after)
    assert predictions.image_metadata == image_metadata_before
    assert predictions.bboxes_metadata == rows_before
    assert not torch.equal(drawn.tensor_image, image_before)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_output_keeps_identity_frames_video_metadata_and_parent_row(
    implementation: type,
) -> None:
    video = VideoMetadata(
        video_identifier="camera-1",
        frame_number=42,
        frame_timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
        fps=25.0,
    )
    frame = ImageData.from_numpy_rgb(
        scene(64, 80), image_id="video-frame", video_metadata=video
    )
    image = frame.crop((8, 4, 72, 52), image_id="crop").with_parent_row(
        RowRef("detections", 3)
    )

    drawn = _run_with(
        implementation,
        DetectionVisualizationBlock,
        image=image,
        predictions=detections(image),
        **SMALL_TEXT,
    )

    assert drawn.image_id == image.image_id
    assert drawn.parent == image.parent and drawn.root == image.root
    assert drawn.video_metadata == image.video_metadata
    assert drawn.parent_row == RowRef("detections", 3)
    assert drawn.composite_sources is None
    assert (
        drawn.tensor_image.dtype == torch.uint8 and drawn.tensor_image.is_contiguous()
    )
    assert drawn.tensor_image.data_ptr() != image.tensor_image.data_ptr()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_one_instance_draws_images_with_different_row_counts(
    implementation: type,
) -> None:
    # The instance is reused across invocations; no state carries over.
    instance = implementation()
    first = image_of(scene(seed=1), image_id="first")
    second = image_of(scene(seed=2), image_id="second")

    for image, objects in ((first, OBJECTS), (second, OBJECTS[1:2]), (first, ())):
        drawn = _run_with(
            instance,
            DetectionVisualizationBlock,
            image=image,
            predictions=detections(image, objects),
            **SMALL_TEXT,
        )
        expected = _chain(
            implementation,
            image=image,
            predictions=detections(image, objects),
            **SMALL_TEXT,
        )
        assert torch.equal(drawn.tensor_image, expected.tensor_image)
