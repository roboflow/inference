"""Native V2 image blocks through the real compiler and executor.

Re-homes the former compiled-workflow checks of ``test_blocks.py``
(image-blocks request R2): crop → invert → mosaic, a brightness gate and a
rejected ``regions`` literal.
"""

import numpy as np
import pytest
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import WorkflowCompileError

REGIONS = [[40, 40, 100, 100], [120, 10, 180, 70]]


def images():
    """Three images: both regions fit, none fits, only the first fits."""
    first = np.full((120, 200, 3), 200, dtype=np.uint8)
    first[10:70, 120:180] = 5
    second = np.full((32, 32, 3), 50, dtype=np.uint8)
    third = np.full((110, 110, 3), 150, dtype=np.uint8)

    return [first, second, third]


def definition(*, gated=False, regions=REGIONS):
    steps = [
        {
            "type": "v2/crop",
            "name": "crop",
            "image": "$inputs.images",
            "regions": regions,
        },
        {"type": "v2/invert", "name": "invert", "image": "$steps.crop.crops"},
        {
            "type": "v2/mosaic",
            "name": "mosaic",
            "images": "$steps.invert.image",
            "tile_size": 16,
            "background": 9,
        },
    ]
    if gated:
        steps[1:1] = [
            {
                "type": "v2/has_brightness",
                "name": "bright",
                "image": "$steps.crop.crops",
                "minimum": 100,
            },
            {
                "type": "v2/continue_if",
                "name": "gate",
                "condition": "$steps.bright.keep",
                "next_steps": ["$steps.invert"],
            },
        ]
    workflow = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "images"}],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in (
                ("crops", "$steps.crop.crops"),
                ("summary", "$steps.crop.summary"),
                ("inverted", "$steps.invert.image"),
                ("mosaic", "$steps.mosaic.image"),
                ("count", "$steps.mosaic.count"),
            )
        ],
    }

    return workflow


def test_crop_invert_mosaic_keeps_ragged_indices_and_empty_groups() -> None:
    plan = compile_workflow(definition(), catalogue=create_catalogue())

    result = plan.create_session().run({"images": images()})

    crops = result.outputs.data["crops"]
    assert [group.indices for group in crops] == [((0, 0), (0, 1)), (), ((2, 0),)]
    assert crops[1].parent_index == (1,)
    assert [axis.kind for axis in result.outputs.layout["crops"].axes] == [
        "sample",
        "dynamic_nesting",
    ]
    assert len(result.outputs.layout["summary"].axes) == 1
    assert len(result.outputs.layout["mosaic"].axes) == 1
    rows = result.rows()
    assert [row["summary"]["crop_count"] for row in rows] == [2, 0, 1]
    assert [row["count"] for row in rows] == [2, 0, 1]
    blank = rows[1]["mosaic"]
    assert blank.shape == (16, 16, 3) and (blank == 9).all()


def test_brightness_gate_filters_a_crop_without_renumbering() -> None:
    plan = compile_workflow(definition(gated=True), catalogue=create_catalogue())

    result = plan.create_session().run({"images": images()})

    inverted = result.outputs.data["inverted"]
    assert inverted[0].indices == ((0, 0),)
    assert result.filtered_paths["inverted"] == ((0, 1),)
    assert [row["count"] for row in result.rows()] == [1, 0, 1]


def test_bad_regions_literal_fails_compilation_naming_step_and_field() -> None:
    with pytest.raises(WorkflowCompileError, match="regions") as caught:
        compile_workflow(
            definition(regions=[[100, 0, 10, 10]]), catalogue=create_catalogue()
        )

    assert caught.value.step_path == ("crop",)
