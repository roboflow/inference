"""Compiled workflows over the native V2 image blocks.

These run the real compiler, session and executor with ``create_catalogue()``;
nothing is called directly. They restore the compiled-workflow checks that
used to live in ``test_blocks.py``.
"""

from typing import Any, Dict, List

import numpy as np
import pytest
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
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
    assert blank.shape == (16, 16, 3)
    assert int(blank.min()) == int(blank.max()) == 9


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
    assert int(inverted[0][0].max()) == 0
    (entry,) = result.selections["inverted"].values()
    assert result.filtered_paths[entry] == ((0, 1), (1, 0))
    rows = result.rows()
    assert [crop is None for crop in rows[0]["inverted"]] == [False, True]
    assert rows[1]["inverted"] == [None]


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
