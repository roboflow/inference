"""The SAM prompt classes live in workflows; the server re-exports them.

One class object per name (`Box`, `Point`, `Sam2Prompt`, `Sam2PromptSet`,
`Sam3Prompt`). That is what keeps `isinstance(raw_point, Point)` in the two
SAM 3 interactive blocks true for a `Point` built through either import path
(round-4 defect 2), lets the server request classes accept prompt sets the
blocks build, and keeps the `ValidationError` a bad prompt raises the same
object either way. The tables below are frozen from the classes as they stood
in `requests/sam2.py` / `requests/sam3.py` before the move, so a later edit to
the moved classes is caught here.
"""

import ast
from pathlib import Path
from typing import List

import pytest
from pydantic import ValidationError

from inference.core.entities.requests import sam2 as server_sam2
from inference.core.entities.requests import sam3 as server_sam3
from inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive import (
    v1 as interactive_v1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive import (
    v1_tensor as interactive_v1_tensor,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything_common import (
    prompts as local,
)

# tests/workflows/unit_tests/core_steps/models/foundation/<file> -> parents[6] is the repo root
PROMPTS_MODULE = (
    Path(__file__).resolve().parents[6]
    / "inference/core/workflows/core_steps/models/foundation/segment_anything_common/prompts.py"
)
IMAGE = {"type": "base64", "value": "aGVsbG8="}
BOX = dict(x=5.0, y=6.0, width=4.0, height=8.0)
POINTS = [dict(x=1.0, y=2.0, positive=True), dict(x=3.0, y=4.0, positive=False)]
PAIRS = [
    (local.Box, server_sam2.Box),
    (local.Point, server_sam2.Point),
    (local.Sam2Prompt, server_sam2.Sam2Prompt),
    (local.Sam2PromptSet, server_sam2.Sam2PromptSet),
    (local.Sam3Prompt, server_sam3.Sam3Prompt),
]


@pytest.mark.parametrize(
    "local_cls,server_cls", PAIRS, ids=[pair[1].__name__ for pair in PAIRS]
)
def test_server_module_re_exports_the_workflows_class(local_cls, server_cls) -> None:
    assert local_cls is server_cls


def test_request_classes_resolve_their_prompt_fields_to_the_workflows_classes() -> None:
    assert (
        server_sam2.Sam2SegmentationRequest.model_fields["prompts"].annotation
        is local.Sam2PromptSet
    )
    assert (
        server_sam3.Sam3SegmentationRequest.model_fields["prompts"].annotation
        == List[local.Sam3Prompt]
    )


def test_prompts_module_imports_nothing_from_the_server() -> None:
    assert PROMPTS_MODULE.is_file(), PROMPTS_MODULE
    tree = ast.parse(PROMPTS_MODULE.read_text(encoding="utf-8"))
    imported = sorted(
        (node.module if isinstance(node, ast.ImportFrom) else alias.name)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
        if (node.module if isinstance(node, ast.ImportFrom) else alias.name)
    )
    assert imported == ["pydantic", "typing"], imported


def test_server_request_classes_accept_block_built_prompt_sets() -> None:
    prompt_set = local.Sam2PromptSet(
        prompts=[local.Sam2Prompt(points=[local.Point(x=1.0, y=2.0, positive=True)])]
    )
    request = server_sam2.Sam2SegmentationRequest(
        image=IMAGE, api_key="k", prompts=prompt_set
    )
    assert request.prompts is prompt_set  # `_coerce_prompts`'s isinstance branch
    request = server_sam3.Sam3SegmentationRequest(
        image=IMAGE, api_key="k", prompts=[local.Sam3Prompt(type="text", text="cat")]
    )
    assert type(request.prompts[0]) is local.Sam3Prompt


def test_sam2_segmentation_request_coerces_nested_dict_prompts_to_workflows_classes() -> (
    None
):
    """`_coerce_prompts` accepts `{"prompts": [...]}` and hands it to
    `Sam2PromptSet(**value)`; pydantic then builds every nested prompt/box/
    point from the same (workflows-owned) classes, not copies."""
    request = server_sam2.Sam2SegmentationRequest(
        image=IMAGE,
        api_key="k",
        prompts={"prompts": [{"box": BOX}, {"points": POINTS}]},
    )
    assert type(request.prompts) is local.Sam2PromptSet
    assert type(request.prompts.prompts[0]) is local.Sam2Prompt
    assert type(request.prompts.prompts[0].box) is local.Box
    assert type(request.prompts.prompts[1].points[0]) is local.Point


def test_sam2_segmentation_request_rejects_invalid_nested_dict_prompts() -> None:
    with pytest.raises(ValidationError) as error:
        server_sam2.Sam2SegmentationRequest(
            image=IMAGE,
            api_key="k",
            prompts={
                "prompts": [{"box": {"x": "a", "y": 6.0, "width": 4.0, "height": 8.0}}]
            },
        )
    assert [(e["loc"], e["type"]) for e in error.value.errors()] == [
        (("prompts", "prompts", 0, "box", "x"), "float_parsing")
    ]


def test_sam3_segmentation_request_coerces_nested_dict_prompts_to_workflows_classes() -> (
    None
):
    request = server_sam3.Sam3SegmentationRequest(
        image=IMAGE, api_key="k", prompts=[{"type": "text", "text": "cat"}]
    )
    assert type(request.prompts[0]) is local.Sam3Prompt


def test_sam3_segmentation_request_rejects_invalid_nested_dict_prompts() -> None:
    with pytest.raises(ValidationError) as error:
        server_sam3.Sam3SegmentationRequest(
            image=IMAGE, api_key="k", prompts=[{"type": "text", "text": 123}]
        )
    assert [(e["loc"], e["type"]) for e in error.value.errors()] == [
        (("prompts", 0, "text"), "string_type")
    ]


def test_schema_descriptions_survive_the_move() -> None:
    """The exact `description=`/docstring text in `prompts.py` must still show
    up in `model_json_schema()` after the move (the HTTP `/model/schema`
    surface for SAM2/SAM3 prompts is generated from these classes)."""
    sam2_prompt_set_schema = local.Sam2PromptSet.model_json_schema()
    assert sam2_prompt_set_schema["properties"]["prompts"]["description"] == (
        "An optional list of prompts for masks to predict. Each prompt can "
        "include a bounding box and / or a set of postive or negative points"
    )

    sam3_prompt_schema = local.Sam3Prompt.model_json_schema()
    assert (
        "Unified prompt that can contain text and/or geometry."
        in sam3_prompt_schema["description"]
    )
    assert (
        "Absolute pixel coordinates are used for boxes. Labels accept 0/1 or booleans."
        in sam3_prompt_schema["description"]
    )
    assert sam3_prompt_schema["properties"]["type"]["description"] == (
        "Optional hint: 'text' or 'visual'. 'visual' requires at least one box."
    )
    assert sam3_prompt_schema["properties"]["text"]["description"] == (
        "Concept to segment as a short noun phrase (e.g. 'person'). All "
        "matching instances are returned. Can be combined with exemplar "
        "boxes in the same prompt."
    )
    assert sam3_prompt_schema["properties"]["output_prob_thresh"]["description"] == (
        "Score threshold for this prompt's outputs. Overrides request-level "
        "threshold if set."
    )
    assert sam3_prompt_schema["properties"]["boxes"]["description"] == (
        "Exemplar boxes in absolute pixels, as XYWH entries "
        "({x, y, width, height}, top-left anchored) or XYXY entries "
        "({x0, y0, x1, y1}). Each box marks an example object; the model "
        "segments every instance matching the exemplars (and text, if "
        "provided), not just the boxed objects. Requires box_labels."
    )
    assert sam3_prompt_schema["properties"]["box_labels"]["description"] == (
        "Per-box exemplar labels, one per entry in boxes: 1/true marks a "
        "positive exemplar (segment objects like this), 0/false marks a "
        "negative exemplar (exclude objects like this). Required when boxes "
        "is set."
    )


# --- the accepted-input matrix of `_as_sam2_points`, both interactive variants ---

VARIANTS = [interactive_v1._as_sam2_points, interactive_v1_tensor._as_sam2_points]
ACCEPTED = [
    ("dict", {"x": 1, "y": 2, "positive": False}, (1.0, 2.0, False)),
    ("dict without positive", {"x": 3, "y": 4}, (3.0, 4.0, True)),
    ("(x, y) list", [5, 6], (5.0, 6.0, True)),
    ("(x, y, positive) tuple", (7, 8, False), (7.0, 8.0, False)),
    (
        "server Point",
        server_sam2.Point(x=9.0, y=10.0, positive=True),
        (9.0, 10.0, True),
    ),
    ("local Point", local.Point(x=11.0, y=12.0, positive=False), (11.0, 12.0, False)),
]
REJECTED = [
    ("string", "1,2", "Each point prompt must be an object or a sequence"),
    ("dict missing y", {"x": 1}, "must define `x` and `y`"),
    ("one-element list", [1], "sequence with two or three values"),
    ("bool coordinates", {"x": True, "y": 1}, "Point coordinates must be numbers"),
]


@pytest.mark.parametrize("as_sam2_points", VARIANTS, ids=["v1", "v1_tensor"])
@pytest.mark.parametrize("case", ACCEPTED, ids=[case[0] for case in ACCEPTED])
def test_as_sam2_points_accepts_every_input_shape_the_original_accepted(
    as_sam2_points, case
) -> None:
    _, raw, expected = case
    [point] = as_sam2_points([raw])
    assert point.to_hashable() == expected
    assert type(point) is local.Point
    if isinstance(raw, local.Point):
        assert point is raw  # an accepted Point object passes through untouched


@pytest.mark.parametrize("as_sam2_points", VARIANTS, ids=["v1", "v1_tensor"])
@pytest.mark.parametrize("case", REJECTED, ids=[case[0] for case in REJECTED])
def test_as_sam2_points_rejects_what_the_original_rejected(
    as_sam2_points, case
) -> None:
    _, raw, message = case
    with pytest.raises(ValueError, match=message):
        as_sam2_points([raw])


# --- frozen behaviour of the moved classes ---


def _prompt_set() -> local.Sam2PromptSet:
    return local.Sam2PromptSet(
        prompts=[
            local.Sam2Prompt(box=local.Box(**BOX)),
            local.Sam2Prompt(points=[local.Point(**p) for p in POINTS]),
        ]
    )


def test_to_sam2_inputs_is_frozen() -> None:
    assert _prompt_set().to_sam2_inputs() == {
        "point_coords": [[], [[1.0, 2.0], [3.0, 4.0]]],
        "point_labels": [[], [1, 0]],
        "box": [[3.0, 2.0, 7.0, 10.0]],
    }
    empty = {"point_coords": None, "point_labels": None, "box": None}
    assert local.Sam2PromptSet().to_sam2_inputs() == empty
    assert local.Sam2PromptSet(prompts=[]).to_sam2_inputs() == empty


def test_remote_payload_shape_is_frozen() -> None:
    expected = {
        "prompts": [
            {"box": {"x": 5.0, "y": 6.0, "width": 4.0, "height": 8.0}},
            {
                "points": [
                    {"x": 1.0, "y": 2.0, "positive": True},
                    {"x": 3.0, "y": 4.0, "positive": False},
                ]
            },
        ]
    }
    assert _prompt_set().dict(exclude_none=True) == expected
    assert _prompt_set().model_dump(exclude_none=True) == expected


def test_num_points_and_hashable_are_frozen() -> None:
    assert _prompt_set().num_points() == 2
    assert local.Point(**POINTS[0]).to_hashable() == (1.0, 2.0, True)


def _outcome(cls, kwargs):
    try:
        return ("ok", cls(**kwargs).model_dump(exclude_none=True))
    except ValidationError as error:
        return ("error", [(e["loc"], e["type"]) for e in error.errors()])


SAM3_TABLE = [
    (
        dict(type="text", text="cat", output_prob_thresh=0.4),
        ("ok", {"type": "text", "text": "cat", "output_prob_thresh": 0.4}),
    ),
    (
        dict(
            type="visual",
            boxes=[{"x": 1, "y": 1, "width": 2, "height": 2}],
            box_labels=[1],
        ),
        (
            "ok",
            {
                "type": "visual",
                "boxes": [{"x": 1.0, "y": 1.0, "width": 2.0, "height": 2.0}],
                "box_labels": [1],
            },
        ),
    ),
    (
        dict(type="text", text="cat", boxes=[{"x0": 0, "y0": 0, "x1": 1, "y1": 1}]),
        (
            "ok",
            {
                "type": "text",
                "text": "cat",
                "boxes": [{"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}],
            },
        ),
    ),
    (dict(type="text", text=123), ("error", [(("text",), "string_type")])),
    (
        dict(type="visual", boxes=[{}]),
        (
            "error",
            [
                (("boxes", 0, "Box", "x"), "missing"),
                (("boxes", 0, "Box", "y"), "missing"),
                (("boxes", 0, "Box", "width"), "missing"),
                (("boxes", 0, "Box", "height"), "missing"),
                (("boxes", 0, "BoxXYXY", "x0"), "missing"),
                (("boxes", 0, "BoxXYXY", "y0"), "missing"),
                (("boxes", 0, "BoxXYXY", "x1"), "missing"),
                (("boxes", 0, "BoxXYXY", "y1"), "missing"),
            ],
        ),
    ),
    (
        dict(type="text", text="cat", output_prob_thresh=1.5),
        ("error", [(("output_prob_thresh",), "value_error")]),
    ),
    (dict(type="visual", text=None), ("error", [(("boxes",), "value_error")])),
    (
        dict(
            type="text",
            text="cat",
            boxes=[{"x": 1, "y": 1, "width": 2, "height": 2}],
            box_labels=[1, 0],
        ),
        ("error", [(("box_labels",), "value_error")]),
    ),
    (
        dict(type="text", text="cat", box_labels=[1]),
        ("error", [(("box_labels",), "value_error")]),
    ),
]


@pytest.mark.parametrize(
    "kwargs,expected", SAM3_TABLE, ids=[str(i) for i in range(len(SAM3_TABLE))]
)
def test_sam3_prompt_validation_is_frozen(kwargs, expected) -> None:
    assert _outcome(local.Sam3Prompt, kwargs) == expected


def test_validation_errors_still_name_the_class() -> None:
    """error_handlers.py:148-152 exposes `str(inner_error)` to HTTP clients."""
    with pytest.raises(ValidationError) as error:
        local.Sam3Prompt(type="text", text=123)
    assert str(error.value).startswith("1 validation error for Sam3Prompt\ntext\n")
    with pytest.raises(ValidationError) as error:
        local.Sam2Prompt(box=dict(x="a", y=2, width=3, height=4))
    assert str(error.value).startswith("1 validation error for Sam2Prompt\nbox.x\n")
    assert [(e["loc"], e["type"]) for e in error.value.errors()] == [
        (("box", "x"), "float_parsing")
    ]
