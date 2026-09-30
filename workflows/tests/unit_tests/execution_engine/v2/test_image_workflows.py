"""Compiled workflows over the native V2 image blocks.

These run the real compiler, session and executor with ``create_catalogue()``;
nothing is called directly. Image payloads are ``ImageData``; inputs are given
as tensors, or as RGB NumPy arrays through the boundary conversion.
"""

import json
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.blocks.image_data import (
    FrameMapping,
    ImageData,
)
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.errors import ParamsValidationError
from roboflow_workflows.execution_engine.v2.plan import RunResult

REGIONS = [[40, 40, 100, 100], [120, 10, 180, 70]]
IMAGES_INPUT = {"type": "WorkflowBatchInput", "name": "images", "kind": ["image"]}


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _uniform(height: int, width: int, value: int) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _data(result: RunResult, name: str) -> Any:
    (entry,) = result.selections[name].values()

    return result.outputs.data[entry]


def _axis_kinds(result: RunResult, name: str) -> List[str]:
    (entry,) = result.selections[name].values()

    return [axis.kind for axis in result.outputs.layout[entry].axes]


def test_crop_invert_mosaic_keeps_ragged_groups_and_blank_canvas() -> None:
    # given
    definition = {
        "version": "2.0",
        "inputs": [IMAGES_INPUT],
        "steps": [
            {
                "type": "v2/crop",
                "name": "crop",
                "image": "$inputs.images",
                "regions": REGIONS,
            },
            {"type": "v2/invert", "name": "invert", "image": "$steps.crop.crops"},
            {
                "type": "v2/mosaic",
                "name": "mosaic",
                "images": "$steps.invert.image",
                "tile_size": 16,
                "background": 9,
            },
        ],
        "outputs": [
            _output("crops", "$steps.crop.crops"),
            _output("summary", "$steps.crop.summary"),
            _output("mosaic", "$steps.mosaic.image"),
            _output("count", "$steps.mosaic.count"),
        ],
    }
    images = [_uniform(120, 200, 10), _uniform(32, 32, 20), _uniform(110, 110, 30)]
    plan = compile_workflow(definition, catalogue=create_catalogue())

    # when
    result = plan.create_session().run({"images": images})

    # then
    crops = _data(result, "crops")
    assert [len(group) for group in crops] == [2, 0, 1]
    assert [group.indices for group in crops] == [((0, 0), (0, 1)), (), ((2, 0),)]
    assert crops[1].parent_index == (1,)
    assert _axis_kinds(result, "crops") == ["sample", "dynamic_nesting"]
    assert _axis_kinds(result, "summary") == ["sample"]
    assert _axis_kinds(result, "mosaic") == ["sample"]
    assert list(_data(result, "count")) == [2, 0, 1]
    blank = _data(result, "mosaic")[1]
    assert tuple(blank.tensor_image.shape) == (3, 16, 16)
    assert int(blank.tensor_image.min()) == int(blank.tensor_image.max()) == 9
    assert blank.is_composite and blank.composite_sources == ()


def test_brightness_gate_filters_crops_without_renumbering() -> None:
    # given
    bright_then_dark = _uniform(120, 200, 0)
    bright_then_dark[40:100, 40:100] = 255
    dark = _uniform(110, 110, 0)
    definition = {
        "version": "2.0",
        "inputs": [IMAGES_INPUT],
        "steps": [
            {
                "type": "v2/crop",
                "name": "crop",
                "image": "$inputs.images",
                "regions": REGIONS,
            },
            {
                "type": "v2/has_brightness",
                "name": "keep",
                "image": "$steps.crop.crops",
                "minimum": 100,
            },
            {
                "type": "v2/continue_if",
                "name": "gate",
                "condition": "$steps.keep.keep",
                "next_steps": ["$steps.invert"],
            },
            {"type": "v2/invert", "name": "invert", "image": "$steps.crop.crops"},
        ],
        "outputs": [
            _output("keep", "$steps.keep.keep"),
            _output("inverted", "$steps.invert.image"),
        ],
    }
    plan = compile_workflow(definition, catalogue=create_catalogue())

    # when
    result = plan.create_session().run({"images": [bright_then_dark, dark]})

    # then
    keep = _data(result, "keep")
    assert [list(group) for group in keep] == [[True, False], [False]]
    inverted = _data(result, "inverted")
    assert isinstance(inverted, Batch)
    assert [group.indices for group in inverted] == [((0, 0),), ()]
    assert inverted[1].parent_index == (1,)
    assert int(inverted[0][0].tensor_image.max()) == 0
    (entry,) = result.selections["inverted"].values()
    assert result.filtered_paths[entry] == ((0, 1), (1, 0))
    rows = result.rows()
    assert [crop is None for crop in rows[0]["inverted"]] == [False, True]
    assert rows[1]["inverted"] == [None]


def test_nested_crop_resize_crop_composes_provenance_per_source() -> None:
    # given: the same rectangles on two differently sized inputs, and the
    # second crop and resize inside a nested workflow.
    child = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "crops", "kind": ["image"]}],
        "steps": [
            {
                "type": "v2/resize",
                "name": "resize",
                "image": "$inputs.crops",
                "width": 20,
                "height": 10,
                "interpolation": "nearest",
            },
            {
                "type": "v2/crop",
                "name": "crop",
                "image": "$steps.resize.image",
                "regions": [[5, 3, 15, 8]],
            },
        ],
        "outputs": [_output("details", "$steps.crop.crops")],
    }
    definition = {
        "version": "2.0",
        "inputs": [IMAGES_INPUT],
        "steps": [
            {
                "type": "v2/crop",
                "name": "crop",
                "image": "$inputs.images",
                "regions": [[30, 70, 70, 90], [0, 0, 400, 400]],
            },
            {
                "type": "inner_workflow",
                "name": "child",
                "workflow_definition": child,
                "parameter_bindings": {"crops": "$steps.crop.crops"},
            },
        ],
        "outputs": [
            _output("crops", "$steps.crop.crops"),
            _output("details", "$steps.child.details"),
        ],
    }
    first = ImageData.from_tensor(
        torch.zeros((3, 100, 120), dtype=torch.uint8), image_id="camera-a"
    )
    second = torch.zeros((3, 95, 80), dtype=torch.uint8)
    plan = compile_workflow(definition, catalogue=create_catalogue())

    # when
    rows = plan.create_session().run({"images": [first, second]}).rows()

    # then
    first_rows, second_rows = rows
    crops = first_rows["crops"] + second_rows["crops"]
    details = [
        detail for row in rows for per_crop in row["details"] for detail in per_crop
    ]
    assert len(crops) == len(details) == 4
    assert len({image.image_id for image in crops + details}) == 8
    second_id = second_rows["crops"][0].root.frame_id
    assert second_id not in ("camera-a", "") and second_id.startswith("input-")
    assert [detail.root.frame_id for detail in details] == ["camera-a"] * 2 + [
        second_id
    ] * 2
    for detail in details:
        assert detail.parent.frame_size_hw == (10, 20)
        assert detail.parent.offset_xy == (5.0, 3.0)
    # 40x20 crop at (30, 70) resized to 20x10 (scale 2): local (1, 1) of the
    # detail is resized (6, 4), crop (12, 8), root (42, 78).
    assert details[0].root == FrameMapping(
        "camera-a", (100, 120), scale_xy=(2.0, 2.0), offset_xy=(40.0, 76.0)
    )
    assert details[0].root.map_xy(1, 1) == (42.0, 78.0)
    # The clipped full-image crop of the 95x80 input: scale (80/20, 95/10).
    assert details[3].root == FrameMapping(
        second_id, (95, 80), scale_xy=(4.0, 9.5), offset_xy=(20.0, 28.5)
    )


def test_serialized_rows_carry_provenance_and_feed_back_as_inputs() -> None:
    # given
    definition = {
        "version": "2.0",
        "inputs": [IMAGES_INPUT],
        "steps": [
            {
                "type": "v2/resize",
                "name": "resize",
                "image": "$inputs.images",
                "width": 30,
                "height": 10,
            },
            {
                "type": "v2/crop",
                "name": "crop",
                "image": "$steps.resize.image",
                "regions": [[4, 2, 24, 8]],
            },
        ],
        "outputs": [_output("crops", "$steps.crop.crops")],
    }
    session = compile_workflow(
        definition, catalogue=create_catalogue()
    ).create_session()
    pixels = torch.randint(0, 256, (3, 40, 60), dtype=torch.uint8)

    # when
    native = session.run({"images": [pixels]}).rows()
    wire = json.loads(
        json.dumps(session.run({"images": [pixels]}).rows(serialize=True))
    )
    echo = compile_workflow(
        {
            "version": "2.0",
            "inputs": [IMAGES_INPUT],
            "steps": [],
            "outputs": [_output("images", "$inputs.images")],
        },
        catalogue=create_catalogue(),
    ).create_session()
    restored = echo.run({"images": wire[0]["crops"]}).rows()[0]["images"]

    # then
    (serialized,) = wire[0]["crops"]
    (crop,) = native[0]["crops"]
    assert serialized["root"]["scale_xy"] == [2.0, 4.0]
    assert serialized["root"]["offset_xy"] == [8.0, 8.0]
    assert restored.root == FrameMapping(
        serialized["root"]["frame_id"],
        (40, 60),
        scale_xy=(2.0, 4.0),
        offset_xy=(8.0, 8.0),
    )
    assert restored.parent.frame_id == serialized["parent"]["frame_id"]
    assert restored.size_hw == crop.size_hw


def test_bad_regions_literal_fails_compilation_naming_step_and_field() -> None:
    # given
    definition = {
        "version": "2.0",
        "inputs": [IMAGES_INPUT],
        "steps": [
            {
                "type": "v2/crop",
                "name": "crop",
                "image": "$inputs.images",
                "regions": [[10, 0, 0, 10]],
            }
        ],
        "outputs": [_output("crops", "$steps.crop.crops")],
    }

    # when
    with pytest.raises(ParamsValidationError) as error:
        compile_workflow(definition, catalogue=create_catalogue())

    # then
    assert error.value.step_path == ("crop",)
    assert error.value.field_path[0] == "regions"
    assert "$steps.crop" in str(error.value)
