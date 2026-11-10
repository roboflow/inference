"""The ten annotation blocks in compiled V2 workflows fed by real model blocks.

Fake models (``scene.py``) stand in for loaded weights; the model blocks,
crop, relabel, gate and annotation blocks are the real V2 blocks::

    instance seg ─▶ mask ─▶ bounding box ─▶ label          one segmentation view
    semantic seg ─▶ semantic overlay                       native class grid
    instance seg ─▶ dynamic crop ─▶ instance seg ─▶ box on the parent image
    instance seg ─▶ dynamic crop ─▶ classifier ─▶ classes replacement ─▶ label

Pixel exactness of each painter is the unit suites' job. These tests check
what a workflow builder relies on: blocks compose, ragged batches keep their
rows, selected values behave like literals, gates skip images, inputs stay
unchanged, frames map between crop and parent, and ``mask_quality`` follows
decision 027.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.visualization.annotations import (
    MaskVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.instance_segmentation import InstanceDetections
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
    scene_image,
    shape_mask,
    street_image,
)

MODES = ("run", "phases")
# V1's default text scale 1.0 is too large for these small scenes.
LABEL_SCALE = 0.35
SEG = {
    "type": "roboflow_core/roboflow_instance_segmentation_model@v4",
    "name": "seg",
    "image": "$inputs.image",
    "model_id": "shapes-seg",
}
SEMANTIC = {
    "type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
    "name": "semantic",
    "image": "$inputs.image",
    "model_id": "shapes-semantic",
}
CROP = {
    "type": "roboflow_core/dynamic_crop@v1",
    "name": "crop",
    "image": "$inputs.image",
    "predictions": "$steps.seg.predictions",
}


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
        catalogue=annotation_catalogue(),
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
    loader: Optional[SceneLoader] = None,
) -> List[Dict[str, Any]]:
    plan = compile_plan(steps, outputs, inputs=inputs, mode=mode)
    session = plan.create_session({"models": ModelCache(loader or SceneLoader())})
    rows = session.run({"image": list(images), **(values or {})}).rows()

    return rows


def draw(block_type: str, name: str, predictions: str, **params: Any) -> dict:
    step = {
        "type": block_type,
        "name": name,
        "image": params.pop("image", "$inputs.image"),
        "predictions": predictions,
        **params,
    }

    return step


def union_of_masks(shapes, size_hw) -> np.ndarray:
    union = np.zeros(size_hw, dtype=bool)
    for shape in shapes:
        union |= shape_mask(shape, size_hw)

    return union


def box_band(shapes, size_hw, *, margin: int) -> np.ndarray:
    """Pixels within ``margin`` of a box outline: where a box stroke may land."""
    height, width = size_hw
    band = np.zeros(size_hw, dtype=bool)
    for shape in shapes:
        x0, y0, x1, y1 = shape.xyxy
        outer = np.zeros(size_hw, dtype=bool)
        outer[max(y0 - margin, 0) : y1 + margin, max(x0 - margin, 0) : x1 + margin] = (
            True
        )
        inner = np.zeros(size_hw, dtype=bool)
        inner[y0 + margin : max(y1 - margin, 0), x0 + margin : max(x1 - margin, 0)] = (
            True
        )
        band |= outer & ~inner

    return band


def bounding_rect(region: np.ndarray):
    ys, xs = np.nonzero(region)

    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# --- model composition -------------------------------------------------------


CHAIN = [
    SEG,
    draw("roboflow_core/mask_visualization@v1", "mask", "$steps.seg.segmentation"),
    draw(
        "roboflow_core/bounding_box_visualization@v1",
        "box",
        "$steps.seg.segmentation",
        image="$steps.mask.image",
    ),
    draw(
        "roboflow_core/label_visualization@v2",
        "label",
        "$steps.seg.segmentation",
        image="$steps.box.image",
        text_scale=LABEL_SCALE,
    ),
]
CHAIN_OUTPUTS = {
    "mask": "$steps.mask.image",
    "box": "$steps.box.image",
    "label": "$steps.label.image",
}


@pytest.mark.parametrize("mode", MODES)
def test_instance_masks_boxes_and_labels_chain_over_a_ragged_batch(mode: str) -> None:
    # given: three objects on the street, one on the smaller alley
    images = [street_image(), alley_image()]
    loader = SceneLoader()

    # when
    rows = run_rows(CHAIN, CHAIN_OUTPUTS, images, mode=mode, loader=loader)

    # then: one model load and one batch for both images
    assert loader.loaded == [ModelRequest("shapes-seg")]
    for image, row, shapes, size in zip(
        images, rows, (STREET, ALLEY), (STREET_SIZE, ALLEY_SIZE)
    ):
        for name in CHAIN_OUTPUTS:
            assert row[name].image_id == image.image_id
            assert tuple(row[name].tensor_image.shape) == (3, *size)
        # the overlay tints exactly the instance masks, not their boxes
        assert np.array_equal(changed(image, row["mask"]), union_of_masks(shapes, size))
        # box strokes land on each row's box outline; each label patch sits
        # on its row's top-left corner (V1 default: TOP_LEFT, padding 10)
        strokes = changed(row["mask"], row["box"])
        assert strokes.any()
        assert not (strokes & ~box_band(shapes, size, margin=2)).any()
        labels = changed(row["box"], row["label"])
        top_left = np.zeros(size, dtype=bool)
        for shape in shapes:
            x0, y0 = shape.xyxy[:2]
            corner = (slice(max(y0 - 32, 0), y0 + 2), slice(max(x0 - 2, 0), x0 + 48))
            assert labels[corner].any()
            top_left[corner] = True
        assert not (labels & ~top_left).any()


def test_rows_of_one_image_never_reach_another_image() -> None:
    # given: the alley alone, and the alley batched after the street
    alone = run_rows(CHAIN, CHAIN_OUTPUTS, [alley_image()])
    batched = run_rows(CHAIN, CHAIN_OUTPUTS, [street_image(), alley_image()])

    # then
    assert np.array_equal(pixels_of(alone[0]["label"]), pixels_of(batched[1]["label"]))


def test_all_ten_blocks_draw_an_image_without_objects_as_an_unchanged_copy() -> None:
    # given: a scene with no painted objects and no road
    blank = scene_image("blank", ALLEY_SIZE, (), road=False)
    steps, outputs = gallery_steps()

    # when
    (row,) = run_rows(steps, outputs, [blank])

    # then: every block returns a new image; only the classification label
    # always draws (the classifier still names a top class)
    for name in outputs:
        assert row[name] is not None
        assert row[name].image_id == "blank"
        assert row[name].tensor_image.data_ptr() != blank.tensor_image.data_ptr()
        if name != "classification_label":
            assert np.array_equal(pixels_of(row[name]), pixels_of(blank)), name
    assert changed(blank, row["classification_label"]).any()


def gallery_steps():
    """One step per annotation block, each drawing on the input image."""
    steps = [
        SEG,
        SEMANTIC,
        {
            "type": "roboflow_core/roboflow_keypoint_detection_model@v3",
            "name": "keypoints",
            "image": "$inputs.image",
            "model_id": "shapes-keypoints",
        },
        {
            "type": "roboflow_core/roboflow_classification_model@v3",
            "name": "scene_class",
            "image": "$inputs.image",
            "model_id": "colors",
        },
        draw(
            "roboflow_core/bounding_box_visualization@v1",
            "bounding_box",
            "$steps.seg.segmentation",
        ),
        draw("roboflow_core/label_visualization@v2", "label", "$steps.seg.segmentation"),
        draw("roboflow_core/polygon_visualization@v2", "polygon", "$steps.seg.segmentation"),
        draw("roboflow_core/mask_visualization@v1", "mask", "$steps.seg.segmentation"),
        # V1 has no default skeleton for two keypoints; name the one edge
        draw(
            "roboflow_core/keypoint_visualization@v1",
            "keypoint",
            "$steps.keypoints.predictions",
            edges=[[1, 2]],
        ),
        draw(
            "roboflow_core/classification_label_visualization@v1",
            "classification_label",
            "$steps.scene_class.predictions",
        ),
        draw("roboflow_core/crop_visualization@v1", "crop", "$steps.seg.segmentation"),
        draw("roboflow_core/blur_visualization@v1", "blur", "$steps.seg.segmentation"),
        draw("roboflow_core/pixelate_visualization@v1", "pixelate", "$steps.seg.segmentation"),
        draw(
            "roboflow_core/semantic_segmentation_visualization@v1",
            "semantic_overlay",
            "$steps.semantic.predictions",
        ),
    ]
    names = [step["name"] for step in steps[4:]]
    outputs = {name: f"$steps.{name}.image" for name in names}

    return steps, outputs


@pytest.mark.parametrize("mode", MODES)
def test_all_ten_blocks_leave_inputs_and_predictions_unchanged(mode: str) -> None:
    # given
    images = [street_image(), alley_image()]
    before = [image.tensor_image.clone() for image in images]
    steps, outputs = gallery_steps()
    outputs = {**outputs, "seg": "$steps.seg.predictions"}

    # when
    rows = run_rows(steps, outputs, images, mode=mode)
    masks_before = [row["seg"].mask.clone() for row in rows]
    boxes_before = [row["seg"].xyxy.clone() for row in rows]

    # then: inputs keep their pixels; every block changed its own copy
    for image, pixels, row in zip(images, before, rows):
        assert torch.equal(image.tensor_image, pixels)
        for name in outputs:
            if name != "seg":
                assert changed(image, row[name]).any(), name
    for row, masks, boxes in zip(rows, masks_before, boxes_before):
        assert torch.equal(row["seg"].mask, masks)
        assert torch.equal(row["seg"].xyxy, boxes)


def test_native_semantic_model_feeds_the_semantic_overlay() -> None:
    # given: background, road and car classes from the semantic model
    image = street_image()
    steps = [
        SEMANTIC,
        draw(
            "roboflow_core/semantic_segmentation_visualization@v1",
            "overlay",
            "$steps.semantic.predictions",
            opacity=0.5,
        ),
    ]
    outputs = {"overlay": "$steps.overlay.image", "grid": "$steps.semantic.predictions"}

    # when
    (row,) = run_rows(steps, outputs, [image])

    # then: background stays; road and car look like the mask overlay of one
    # mask per class with the same palette and opacity
    labels = row["grid"].segmentation_map.cpu().numpy()
    assert set(np.unique(labels).tolist()) == {0, 1, 2}
    background = labels == 0
    overlay = pixels_of(row["overlay"])
    assert np.array_equal(overlay[background], pixels_of(image)[background])
    per_class = InstanceDetections(
        xyxy=torch.tensor([[0.0, 0.0, 112.0, 72.0]] * 2),
        class_id=torch.tensor([1, 2]),
        confidence=torch.ones(2),
        mask=torch.from_numpy(np.stack([labels == 1, labels == 2])),
    )
    reference = MaskVisualizationBlock().run(
        image=image,
        predictions=per_class,
        opacity=0.5,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=[],
        color_axis="CLASS",
        mask_quality="accurate",
    )["image"]
    assert np.array_equal(overlay, pixels_of(reference))


def test_semantic_overlay_lists_the_model_classes_for_an_unknown_skip_class() -> None:
    steps = [
        SEMANTIC,
        draw(
            "roboflow_core/semantic_segmentation_visualization@v1",
            "overlay",
            "$steps.semantic.predictions",
            skip_classes=["truck"],
        ),
    ]

    with pytest.raises(StepExecutionError) as error:
        run_rows(steps, {"overlay": "$steps.overlay.image"}, [street_image()])

    message = str(error.value)
    assert "truck" in message
    for name in ("background", "road", "car"):
        assert name in message


def test_semantic_grid_of_a_resized_image_is_not_drawn_on_the_original() -> None:
    steps = [
        {
            "type": "roboflow_core/resize@v1",
            "name": "small",
            "image": "$inputs.image",
            "width": 56,
            "height": 36,
        },
        {**SEMANTIC, "image": "$steps.small.image"},
        draw(
            "roboflow_core/semantic_segmentation_visualization@v1",
            "overlay",
            "$steps.semantic.predictions",
        ),
    ]

    with pytest.raises(StepExecutionError) as error:
        run_rows(steps, {"overlay": "$steps.overlay.image"}, [street_image()])

    assert "on the image the model saw" in str(error.value)


# --- mask_quality (decision 027) ---------------------------------------------


def quality_steps(producer: str, quality: str, *, field: str = "segmentation"):
    seg = {**SEG, "type": producer}
    steps = [
        seg,
        draw(
            "roboflow_core/mask_visualization@v1",
            "mask",
            f"$steps.seg.{field}",
            mask_quality=quality,
        ),
        draw(
            "roboflow_core/polygon_visualization@v2",
            "polygon",
            f"$steps.seg.{field}",
            mask_quality=quality,
            image="$steps.mask.image",
        ),
    ]
    outputs = {
        "mask": "$steps.mask.image",
        "polygon": "$steps.polygon.image",
        "view": "$steps.seg.segmentation",
    }

    return steps, outputs


def test_accurate_and_coarse_draw_the_same_pixels_for_the_dense_model() -> None:
    # given: the built-in instance segmentation block supplies dense masks
    images = [street_image(), alley_image()]
    pixels = {}

    # when
    for quality in ("accurate", "coarse"):
        steps, outputs = quality_steps("roboflow_core/roboflow_instance_segmentation_model@v4", quality)
        rows = run_rows(steps, outputs, images)
        pixels[quality] = [pixels_of(row["polygon"]) for row in rows]

    # then
    for accurate, coarse in zip(pixels["accurate"], pixels["coarse"]):
        assert np.array_equal(accurate, coarse)


COMPACT = "test/synthetic_compact_segmentation@v1"


def test_accurate_on_a_compact_view_equals_drawing_its_full_resolution_masks() -> None:
    # given: the synthetic grid view, and the same rows as dense predictions
    images = [street_image(), alley_image()]
    view_steps, outputs = quality_steps(COMPACT, "accurate")
    dense_steps, _ = quality_steps(COMPACT, "accurate", field="predictions")

    # when
    on_view = run_rows(view_steps, outputs, images)
    on_dense = run_rows(dense_steps, outputs, images)

    # then: both mask readers shared one accurate materialization of the view
    for view_row, dense_row in zip(on_view, on_dense):
        assert np.array_equal(
            pixels_of(view_row["polygon"]), pixels_of(dense_row["polygon"])
        )
        assert view_row["view"].materialization_count == 1


def test_coarse_on_a_compact_view_differs_only_along_mask_edges() -> None:
    # given
    images = [street_image(), alley_image()]
    rows = {}
    for quality in ("accurate", "coarse"):
        steps, outputs = quality_steps(COMPACT, quality)
        rows[quality] = run_rows(steps, outputs, images)

    for image, accurate, coarse, shapes, size in zip(
        images,
        rows["accurate"],
        rows["coarse"],
        (STREET, ALLEY),
        (STREET_SIZE, ALLEY_SIZE),
    ):
        # then: coarse never took the accurate full-resolution path
        assert coarse["view"].materialization_count == 0
        # whole grid cells make blocky edges: pixels differ, but only near an
        # edge of a painted shape (within one stride-4 cell)
        difference = changed(accurate["mask"], coarse["mask"])
        assert difference.any()
        near_edge = np.zeros(size, dtype=bool)
        for shape in shapes:
            mask = torch.from_numpy(shape_mask(shape, size)).float()[None, None]
            grown = torch.nn.functional.max_pool2d(mask, 9, stride=1, padding=4)
            shrunk = -torch.nn.functional.max_pool2d(-mask, 9, stride=1, padding=4)
            near_edge |= (grown - shrunk)[0, 0].numpy().astype(bool)
        assert not (difference & ~near_edge).any()
        # far from edges both qualities tint the same pixels
        assert np.array_equal(
            changed(image, accurate["mask"]) & ~near_edge,
            changed(image, coarse["mask"]) & ~near_edge,
        )


def test_box_and_label_readers_never_materialize_a_compact_view() -> None:
    steps = [
        {**SEG, "type": COMPACT},
        draw("roboflow_core/bounding_box_visualization@v1", "box", "$steps.seg.segmentation"),
        draw(
            "roboflow_core/label_visualization@v2",
            "label",
            "$steps.seg.segmentation",
            image="$steps.box.image",
            text_scale=LABEL_SCALE,
        ),
    ]
    outputs = {"label": "$steps.label.image", "view": "$steps.seg.segmentation"}

    images = [street_image(), alley_image()]

    rows = run_rows(steps, outputs, images)

    for image, row in zip(images, rows):
        assert row["view"].materialization_count == 0
        assert changed(image, row["label"]).any()


# --- frames: crops and parents -----------------------------------------------


def test_detections_found_in_crops_are_drawn_at_their_place_on_the_parent() -> None:
    # given: crops of each object, segmented again by the same model
    steps = [
        SEG,
        CROP,
        {**SEG, "name": "inner", "image": "$steps.crop.crops"},
        draw(
            "roboflow_core/bounding_box_visualization@v1", "on_parent", "$steps.inner.segmentation"
        ),
        draw("roboflow_core/bounding_box_visualization@v1", "direct", "$steps.seg.segmentation"),
    ]
    outputs = {"on_parent": "$steps.on_parent.image", "direct": "$steps.direct.image"}
    image = street_image()

    # when
    (row,) = run_rows(steps, outputs, [image])

    # then: one parent drawing per crop; each holds its row's box at the
    # parent position, and together they equal drawing the parent rows
    drawings = row["on_parent"]
    assert len(drawings) == len(STREET)
    direct = changed(image, row["direct"])
    union = np.zeros(STREET_SIZE, dtype=bool)
    for drawing, shape in zip(drawings, STREET):
        assert drawing.image_id == image.image_id
        strokes = changed(image, drawing)
        x0, y0, x1, y1 = bounding_rect(strokes)
        sx0, sy0, sx1, sy1 = shape.xyxy
        assert abs(x0 - sx0) <= 2 and abs(y0 - sy0) <= 2
        assert abs(x1 - sx1) <= 2 and abs(y1 - sy1) <= 2
        union |= strokes
    assert np.array_equal(union, direct)


def test_parent_detections_drawn_on_a_crop_are_clipped_to_it() -> None:
    # given: all parent rows drawn on each crop; other objects lie outside
    steps = [
        SEG,
        CROP,
        draw(
            "roboflow_core/bounding_box_visualization@v1",
            "on_crop",
            "$steps.seg.segmentation",
            image="$steps.crop.crops",
        ),
        draw("roboflow_core/bounding_box_visualization@v1", "direct", "$steps.seg.segmentation"),
    ]
    outputs = {
        "on_crop": "$steps.on_crop.image",
        "crops": "$steps.crop.crops",
        "direct": "$steps.direct.image",
    }

    # when
    (row,) = run_rows(steps, outputs, [street_image()])

    # then: each crop shows the parent drawing's pixels of its own region
    direct = pixels_of(row["direct"])
    for drawing, crop, shape in zip(row["on_crop"], row["crops"], STREET):
        x0, y0, x1, y1 = shape.xyxy
        assert drawing.image_id == crop.image_id
        assert changed(crop, drawing).any()
        assert np.array_equal(pixels_of(drawing), direct[y0:y1, x0:x1])


# --- row-specific relabelling ------------------------------------------------


def test_each_relabelled_row_is_drawn_with_its_own_name() -> None:
    # given: to the segmenter both cars are class 1 "car"; a color classifier
    # reads each row's crop and the replacement renames that row only
    steps = [
        SEG,
        CROP,
        {
            "type": "roboflow_core/roboflow_classification_model@v3",
            "name": "colors",
            "image": "$steps.crop.crops",
            "model_id": "colors",
        },
        {
            "type": "roboflow_core/detections_classes_replacement@v1",
            "name": "relabel",
            "object_detection_predictions": "$steps.seg.predictions",
            "classification_predictions": "$steps.colors.predictions",
            "crops": "$steps.crop.crops",
        },
        draw(
            "roboflow_core/label_visualization@v2",
            "model_names",
            "$steps.seg.predictions",
            text_scale=LABEL_SCALE,
        ),
        draw(
            "roboflow_core/label_visualization@v2",
            "row_names",
            "$steps.relabel.predictions",
            text_scale=LABEL_SCALE,
        ),
    ]
    outputs = {
        "seg": "$steps.seg.predictions",
        "relabel": "$steps.relabel.predictions",
        "model_names": "$steps.model_names.image",
        "row_names": "$steps.row_names.image",
    }
    image = street_image()

    # when
    (row,) = run_rows(steps, outputs, [image])

    # then: each row carries its own crop's name; every label patch changed
    assert row["seg"].class_id.tolist() == [0, 1, 1]
    relabel = row["relabel"]
    assert [r["class"] for r in relabel.bboxes_metadata] == ["red", "blue", "yellow"]
    difference = changed(row["model_names"], row["row_names"])
    for shape in STREET:
        x0, y0 = shape.xyxy[:2]
        assert difference[max(y0 - 32, 0) : y0 + 2, max(x0 - 2, 0) : x0 + 48].any()


# --- selected style and gates ------------------------------------------------


STYLE_INPUTS = [
    {"type": "WorkflowParameter", "name": "opacity"},
    {"type": "WorkflowParameter", "name": "thickness"},
    {"type": "WorkflowParameter", "name": "palette"},
]


def styled_steps(opacity: Any, thickness: Any, palette: Any):
    steps = [
        SEG,
        draw(
            "roboflow_core/mask_visualization@v1",
            "mask",
            "$steps.seg.segmentation",
            opacity=opacity,
            color_palette=palette,
        ),
        draw(
            "roboflow_core/bounding_box_visualization@v1",
            "box",
            "$steps.seg.segmentation",
            image="$steps.mask.image",
            thickness=thickness,
            color_palette=palette,
        ),
    ]

    return steps


def test_selected_style_values_draw_like_the_same_literals() -> None:
    images = [street_image(), alley_image()]
    outputs = {"box": "$steps.box.image"}
    selected = styled_steps("$inputs.opacity", "$inputs.thickness", "$inputs.palette")
    plan = compile_plan(selected, outputs, inputs=STYLE_INPUTS)
    session = plan.create_session({"models": ModelCache(SceneLoader())})

    results = {}
    for values in ((0.2, 1, "DEFAULT"), (0.9, 4, "ROBOFLOW")):
        opacity, thickness, palette = values
        by_selector = session.run(
            {
                "image": images,
                "opacity": opacity,
                "thickness": thickness,
                "palette": palette,
            }
        ).rows()
        by_literal = run_rows(styled_steps(*values), outputs, images)
        for selector_row, literal_row in zip(by_selector, by_literal):
            assert np.array_equal(
                pixels_of(selector_row["box"]), pixels_of(literal_row["box"])
            )
        results[values] = [pixels_of(row["box"]) for row in by_selector]

    # different selected values give different pictures from one plan
    first, second = results.values()
    assert not np.array_equal(first[0], second[0])


def test_a_selected_value_fails_with_the_literal_compile_error_text() -> None:
    with pytest.raises(Exception) as compile_error:
        compile_plan(styled_steps(1.5, 2, "DEFAULT"), {"box": "$steps.box.image"})
    plan = compile_plan(
        styled_steps("$inputs.opacity", 2, "DEFAULT"),
        {"box": "$steps.box.image"},
        inputs=STYLE_INPUTS[:1],
    )
    session = plan.create_session({"models": ModelCache(SceneLoader())})

    with pytest.raises(StepExecutionError) as run_error:
        session.run({"image": [street_image()], "opacity": 1.5})

    expected = "opacity must be a number in [0.0, 1.0], got 1.5"
    assert expected in str(compile_error.value)
    assert expected in str(run_error.value)


@pytest.mark.parametrize("mode", MODES)
def test_a_gated_image_is_not_drawn_and_the_others_are(mode: str) -> None:
    steps = [
        SEG,
        {
            "type": "roboflow_core/continue_if_boolean@v1",
            "name": "keep",
            "condition": "$inputs.keep",
            "next_steps": ["$steps.box"],
        },
        draw("roboflow_core/bounding_box_visualization@v1", "box", "$steps.seg.segmentation"),
    ]
    images = [street_image(), alley_image(), street_image()]

    rows = run_rows(
        steps,
        {"box": "$steps.box.image"},
        images,
        inputs=[{"type": "WorkflowBatchInput", "name": "keep"}],
        values={"keep": [True, False, True]},
        mode=mode,
    )

    assert rows[1]["box"] is None
    for index in (0, 2):
        assert changed(images[index], rows[index]["box"]).any()
    assert np.array_equal(pixels_of(rows[0]["box"]), pixels_of(rows[2]["box"]))


# --- compile-time contract ---------------------------------------------------


def test_copy_image_is_rejected_with_the_chaining_remedy() -> None:
    steps = [SEG, draw("roboflow_core/mask_visualization@v1", "mask", "$steps.seg.segmentation")]
    steps[1]["copy_image"] = False

    with pytest.raises(Exception) as error:
        compile_plan(steps, {"mask": "$steps.mask.image"})

    message = str(error.value)
    assert "copy_image" in message
    assert "delete" in message
    assert "$steps." in message


def test_boxes_without_masks_cannot_feed_a_mask_reader() -> None:
    steps = [
        {**SEG, "type": "roboflow_core/roboflow_object_detection_model@v3", "name": "detect"},
        draw("roboflow_core/mask_visualization@v1", "mask", "$steps.detect.predictions"),
    ]

    with pytest.raises(Exception) as error:
        compile_plan(steps, {"mask": "$steps.mask.image"})

    assert "object_detection_prediction" in str(error.value)
