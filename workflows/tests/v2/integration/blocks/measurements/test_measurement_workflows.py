"""Measurement blocks composed through the real V2 compiler and session.

A stand-in segmenter reads painted objects: channel 0 is 255 inside an
object, channel 1 holds its number, channel 2 its class id. The calibrated
scene (120 x 160 px, 10 px per cm)::

    1 card     class 0   x 10..60,   y 10..40    5 x 3 cm reference
    2 package  class 1   x 80..150,  y 10..50    20 px right of the card
    3 ring     class 2   x 20..60,   y 60..100   hole x 30..50, y 70..90

Expected values below are worked out from these rectangles.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.image import DynamicCropBlock, ImagePreprocessingBlock
from roboflow_workflows.v2.blocks.measurements import BLOCKS as MEASUREMENT_BLOCKS
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, LIST_OF_VALUES_KIND
from roboflow_workflows.v2.data.masks import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.predictions import INSTANCE_SEGMENTATION_PREDICTION_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.instance_segmentation import InstanceDetections

CLASS_NAMES = {0: "card", 1: "package", 2: "ring"}
SIZE_HW = (120, 160)
CARD = (10, 10, 60, 40, 1, 0)
PACKAGE = (80, 10, 150, 50, 2, 1)
RING = (20, 60, 60, 100, 3, 2)
RING_HOLE = (30, 70, 50, 90)
RING_AREA = 40 * 40 - 20 * 20
# Masks measure between pixel centres: the card is 49 x 29, the package
# 69 x 39 in mask units.
PACKAGE_SIZE_CM = {
    "width": 69 * 5 / 49,
    "height": 39 * 3 / 29,
    "longer": 69 * 5 / 49,
    "shorter": 39 * 3 / 29,
}


def painted_scene(objects=(CARD, PACKAGE, RING)) -> ImageData:
    pixels = np.zeros((*SIZE_HW, 3), dtype=np.uint8)
    for x0, y0, x1, y1, number, class_id in objects:
        pixels[y0:y1, x0:x1] = (255, number, class_id)
    if RING in objects:
        x0, y0, x1, y1 = RING_HOLE
        pixels[y0:y1, x0:x1] = 0
    image = ImageData.from_numpy_rgb(pixels)

    return image


class PaintedSegmenterBlock(Block):
    """Segment painted objects, optionally only some classes."""

    type = "test/painted_segmenter@v1"
    outputs = {
        "predictions": Output(INSTANCE_SEGMENTATION_PREDICTION_KIND, source="image"),
        "segmentation": Output(INSTANCE_SEGMENTATION_VIEW_KIND, source="image"),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to segment.")
        classes: Optional[List[int]] = Field(
            default=None, description="Class ids to keep; all when None."
        )
        mask_value: Optional[int] = Field(
            default=None,
            description="Foreground value of uint8 masks; bool masks when None.",
        )

    def run(
        self,
        *,
        image: ImageData,
        classes: Optional[List[int]],
        mask_value: Optional[int],
    ) -> Dict[str, Any]:
        pixels = image.tensor_image
        inside = pixels[0] == 255
        numbers = torch.unique(pixels[1][inside]).tolist()
        masks, boxes, class_ids = [], [], []
        for number in numbers:
            mask = inside & (pixels[1] == number)
            ys, xs = torch.nonzero(mask, as_tuple=True)
            class_id = int(pixels[2][ys[0], xs[0]])
            if classes is not None and class_id not in classes:
                continue
            masks.append(mask)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            class_ids.append(class_id)
        height, width = image.size_hw
        mask = (
            torch.stack(masks)
            if masks
            else torch.zeros((0, height, width), dtype=torch.bool)
        )
        if mask_value is not None:
            mask = mask.to(dtype=torch.uint8) * mask_value
        predictions = InstanceDetections(
            xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            class_id=torch.tensor(class_ids, dtype=torch.int64),
            confidence=torch.full((len(class_ids),), 0.9),
            mask=mask,
            image_metadata={
                **FrameRef.from_image(image).prediction_metadata(),
                "class_names": dict(CLASS_NAMES),
            },
            bboxes_metadata=[{"object": n} for n, c in zip(numbers, class_ids)],
        )
        result = {
            "predictions": predictions,
            "segmentation": SegmentationView.from_dense(predictions),
        }

        return result


CATALOGUE = Catalogue(
    [
        *MEASUREMENT_BLOCKS,
        PaintedSegmenterBlock,
        DynamicCropBlock,
        ImagePreprocessingBlock,
    ],
    kinds=[FLOAT_KIND, LIST_OF_VALUES_KIND],
)
SCENE_INPUTS = {
    "axis": "horizontal",
    "card_width": 5.0,
    "card_height": 3.0,
    "card_size": [5.0, 3.0],
    "pixels_per_cm": 10.0,
    "color": "(255, 3, 2)",
    "tolerance": 0,
}


def scene_steps() -> List[Dict[str, Any]]:
    steps = [
        {
            "type": "test/painted_segmenter@v1",
            "name": "detect",
            "image": "$inputs.image",
        },
        {
            "type": "test/painted_segmenter@v1",
            "name": "reference",
            "image": "$inputs.image",
            "classes": [0],
        },
        {
            "type": "test/painted_segmenter@v1",
            "name": "objects",
            "image": "$inputs.image",
            "classes": [1],
        },
        {
            "type": "roboflow_core/distance_measurement@v1",
            "name": "distance",
            "predictions": "$steps.detect.predictions",
            "object_1_class_name": "card",
            "object_2_class_name": "package",
            "reference_axis": "$inputs.axis",
            "calibration_method": "reference object",
            "reference_object_class_name": "card",
            "reference_width": "$inputs.card_width",
            "reference_height": "$inputs.card_height",
        },
        {
            "type": "roboflow_core/size_measurement@v1",
            "name": "size",
            "object_predictions": "$steps.objects.predictions",
            "reference_predictions": "$steps.reference.predictions",
            "reference_dimensions": "$inputs.card_size",
        },
        {
            "type": "roboflow_core/mask_area_measurement@v1",
            "name": "area",
            "predictions": "$steps.detect.segmentation",
            "pixels_per_unit": "$inputs.pixels_per_cm",
        },
        {
            "type": "roboflow_core/pixel_color_count@v1",
            "name": "color",
            "image": "$inputs.image",
            "target_color": "$inputs.color",
            "tolerance": "$inputs.tolerance",
        },
    ]

    return steps


SCENE_OUTPUTS = {
    "distance_cm": "$steps.distance.distance_cm",
    "distance_pixel": "$steps.distance.distance_pixel",
    "dimensions": "$steps.size.dimensions",
    "areas": "$steps.area.predictions",
    "ring_pixels": "$steps.color.matching_pixels_count",
}


def definition(steps, outputs, *, parameters=tuple(SCENE_INPUTS)) -> Dict[str, Any]:
    workflow = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            *({"type": "WorkflowParameter", "name": name} for name in parameters),
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }

    return workflow


def area_rows(prediction) -> List[tuple]:
    rows = (
        prediction.detections.bboxes_metadata
        if isinstance(prediction, SegmentationView)
        else prediction.bboxes_metadata
    )
    areas = [(row["object"], row["area_px"], row["area_converted"]) for row in rows]

    return areas


def assert_calibrated_scene(row) -> None:
    assert (row["distance_pixel"], row["distance_cm"]) == (20, 2.0)
    assert row["dimensions"] == [pytest.approx(PACKAGE_SIZE_CM)]
    assert area_rows(row["areas"]) == [
        (1, 1500.0, 15.0),
        (2, 2800.0, 28.0),
        (3, float(RING_AREA), RING_AREA / 100),
    ]
    assert isinstance(row["areas"], SegmentationView)
    assert row["ring_pixels"] == RING_AREA


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("pipeline", [False, True], ids=["session", "pipeline"])
def test_calibrated_scene_with_selector_configuration(mode, pipeline) -> None:
    plan = compile_workflow(
        definition(scene_steps(), SCENE_OUTPUTS),
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode),
    )
    session = plan.create_session()
    inputs = {"image": painted_scene(), **SCENE_INPUTS}
    try:
        if pipeline:
            with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipe:
                futures = [pipe.submit(inputs) for _ in range(2)]
            results = [future.result() for future in futures]
        else:
            results = [session.run(inputs)]
    finally:
        session.close()

    for result in results:
        assert_calibrated_scene(result.rows()[0])


def test_configuration_selectors_change_units_and_axis() -> None:
    plan = compile_workflow(
        definition(scene_steps(), SCENE_OUTPUTS), catalogue=CATALOGUE
    )
    inputs = {
        "image": painted_scene(),
        **SCENE_INPUTS,
        "axis": "vertical",  # card and package share rows: no vertical gap
        "card_size": "50,30",  # millimetres
        "pixels_per_cm": 1.0,
        "color": "#FF0300",
        "tolerance": 3,  # (255, 0..6, 0..3): card, package and ring pixels
    }

    row = plan.create_session().run(inputs).rows()[0]

    assert (row["distance_pixel"], row["distance_cm"]) == (0, 0.0)
    assert row["dimensions"][0]["width"] == pytest.approx(10 * 69 * 5 / 49)
    assert [area for _, _, area in area_rows(row["areas"])] == [1500.0, 2800.0, 1200.0]
    assert row["ring_pixels"] == 1500 + 2800 + RING_AREA


def test_invalid_selector_value_fails_the_step() -> None:
    plan = compile_workflow(
        definition(scene_steps(), SCENE_OUTPUTS), catalogue=CATALOGUE
    )
    inputs = {"image": painted_scene(), **SCENE_INPUTS, "pixels_per_cm": 0.0}

    with pytest.raises(StepExecutionError, match="pixels_per_unit must be greater"):
        plan.create_session().run(inputs)


def test_uint8_masks_of_value_255_measure_like_boolean_masks() -> None:
    # Models may emit 0/255 masks; areas count pixels, not mask values.
    steps = scene_steps()
    for segmenter in steps[:3]:
        segmenter["mask_value"] = 255
    dense_area = {
        "type": "roboflow_core/mask_area_measurement@v1",
        "name": "dense_area",
        "predictions": "$steps.detect.predictions",
    }
    outputs = {**SCENE_OUTPUTS, "dense_areas": "$steps.dense_area.predictions"}
    plan = compile_workflow(
        definition([*steps, dense_area], outputs), catalogue=CATALOGUE
    )

    inputs = {"image": painted_scene(), **SCENE_INPUTS}

    row = plan.create_session().run(inputs).rows()[0]

    assert row["dense_areas"].mask.dtype == torch.uint8
    assert_calibrated_scene(row)
    assert [area for _, area, _ in area_rows(row["dense_areas"])] == [
        1500.0,
        2800.0,
        float(RING_AREA),
    ]


@pytest.mark.parametrize(
    "polygon",
    [[[0, 0, 1], [10, 0, 1], [10, 10, 1]], [0, 0, 10, 0, 10, 10]],
    ids=["three-values", "flat"],
)
def test_malformed_selector_polygon_fails_naming_the_parameter(polygon) -> None:
    steps = scene_steps()
    size = next(step for step in steps if step["name"] == "size")
    size["reference_predictions"] = "$inputs.polygon"
    plan = compile_workflow(
        definition(steps, SCENE_OUTPUTS, parameters=(*SCENE_INPUTS, "polygon")),
        catalogue=CATALOGUE,
    )
    inputs = {"image": painted_scene(), **SCENE_INPUTS, "polygon": polygon}

    with pytest.raises(StepExecutionError, match="reference_predictions as a polygon"):
        plan.create_session().run(inputs)


def test_batch_images_are_measured_independently() -> None:
    plan = compile_workflow(
        definition(scene_steps(), SCENE_OUTPUTS), catalogue=CATALOGUE
    )
    moved = (100, 10, 150, 50, 2, 1)  # 40 px right of the card
    images = [painted_scene(), painted_scene((CARD, moved, RING))]

    rows = plan.create_session().run({"image": images, **SCENE_INPUTS}).rows()

    assert [row["distance_pixel"] for row in rows] == [20, 40]
    assert [row["distance_cm"] for row in rows] == [2.0, 4.0]
    assert area_rows(rows[1]["areas"])[1] == (2, 2000.0, 20.0)
    assert rows[0]["areas"].detections.image_metadata["class_names"] == CLASS_NAMES


def test_empty_scene_has_no_areas_no_sizes_and_a_failing_distance() -> None:
    outputs = {
        key: SCENE_OUTPUTS[key] for key in ("dimensions", "areas", "ring_pixels")
    }
    plan = compile_workflow(definition(scene_steps(), outputs), catalogue=CATALOGUE)

    row = (
        plan.create_session()
        .run({"image": painted_scene(()), **SCENE_INPUTS})
        .rows()[0]
    )

    assert row["dimensions"] is None
    assert len(row["areas"]) == 0
    assert row["ring_pixels"] == 0

    distance = compile_workflow(
        definition(scene_steps(), {"cm": "$steps.distance.distance_cm"}),
        catalogue=CATALOGUE,
    )
    with pytest.raises(StepExecutionError, match="'card' or 'package' not found"):
        distance.create_session().run({"image": painted_scene(()), **SCENE_INPUTS})


def test_undemanded_measurements_are_pruned() -> None:
    plan = compile_workflow(
        definition(scene_steps(), {"areas": "$steps.area.predictions"}),
        catalogue=CATALOGUE,
    )

    step_names = {step.path[-1] for step in plan.steps}

    assert {"distance", "size", "color"}.isdisjoint(step_names)
    assert "area" in step_names


CROP_STEPS = [
    {"type": "test/painted_segmenter@v1", "name": "detect", "image": "$inputs.image"},
    {
        "type": "roboflow_core/dynamic_crop@v1",
        "name": "crop",
        "image": "$inputs.image",
        "predictions": "$steps.detect.predictions",
    },
    {
        "type": "test/painted_segmenter@v1",
        "name": "inner",
        "image": "$steps.crop.crops",
    },
    {
        "type": "roboflow_core/mask_area_measurement@v1",
        "name": "area",
        "predictions": "$steps.inner.predictions",
        "pixels_per_unit": 10.0,
    },
    {
        "type": "roboflow_core/pixel_color_count@v1",
        "name": "color",
        "image": "$steps.crop.crops",
        "target_color": "#FF0302",
        "tolerance": 0,
    },
]


def test_areas_on_crops_equal_areas_in_the_full_image() -> None:
    plan = compile_workflow(
        definition(
            CROP_STEPS,
            {
                "areas": "$steps.area.predictions",
                "ring": "$steps.color.matching_pixels_count",
            },
            parameters=(),
        ),
        catalogue=CATALOGUE,
    )

    result = plan.create_session().run({"image": painted_scene()})

    crops = result.outputs.data["areas"][0]
    areas = [area_rows(crop) for crop in crops]
    assert areas == [
        [(1, 1500.0, 15.0)],
        [(2, 2800.0, 28.0)],
        [(3, float(RING_AREA), RING_AREA / 100)],
    ]
    root_offsets = [
        FrameRef.from_record(crop.image_metadata["workflows_v2_frame"]).root.offset_xy
        for crop in crops
    ]
    assert root_offsets == [(10.0, 10.0), (80.0, 10.0), (20.0, 60.0)]
    assert list(result.outputs.data["ring"][0]) == [0, 0, RING_AREA]


RESIZED_STEPS = [
    {
        "type": "roboflow_core/image_preprocessing@v1",
        "name": "half",
        "image": "$inputs.image",
        "task_type": "resize",
        "width": 80,
        "height": 60,
    },
    {
        "type": "test/painted_segmenter@v1",
        "name": "small",
        "image": "$steps.half.image",
    },
    {
        "type": "test/painted_segmenter@v1",
        "name": "reference",
        "image": "$inputs.image",
        "classes": [0],
    },
    {
        "type": "roboflow_core/mask_area_measurement@v1",
        "name": "area",
        "predictions": "$steps.small.predictions",
    },
    {
        "type": "roboflow_core/size_measurement@v1",
        "name": "size",
        "object_predictions": "$steps.small.predictions",
        "reference_predictions": "$steps.reference.predictions",
        "reference_dimensions": [5.0, 3.0],
    },
]


def test_resized_image_is_measured_in_its_own_pixels() -> None:
    plan = compile_workflow(
        definition(RESIZED_STEPS, {"areas": "$steps.area.predictions"}, parameters=()),
        catalogue=CATALOGUE,
    )

    row = plan.create_session().run({"image": painted_scene()}).rows()[0]

    # Half size each way: nearest-neighbour keeps whole painted blocks.
    assert [area for _, area, _ in area_rows(row["areas"])][:2] == [375.0, 700.0]


def test_sizes_refuse_to_mix_pixel_scales() -> None:
    plan = compile_workflow(
        definition(
            RESIZED_STEPS, {"dimensions": "$steps.size.dimensions"}, parameters=()
        ),
        catalogue=CATALOGUE,
    )

    with pytest.raises(StepExecutionError, match="different pixel sizes"):
        plan.create_session().run({"image": painted_scene()})


def test_tensor_inputs_are_not_modified_by_the_workflow() -> None:
    plan = compile_workflow(
        definition(scene_steps(), SCENE_OUTPUTS), catalogue=CATALOGUE
    )
    image = painted_scene()
    pixels_before = image.tensor_image.clone()

    row = plan.create_session().run({"image": image, **SCENE_INPUTS}).rows()[0]

    assert torch.equal(image.tensor_image, pixels_before)
    assert isinstance(row["areas"].detections.xyxy, torch.Tensor)
