"""Invocation layouts, binding modes, axis identity, groups, casts and batching."""

import pytest
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    LineageError,
    ParamsValidationError,
)
from roboflow_workflows.execution_engine.v2.plan import Constant, InputPort

from tests.unit_tests.execution_engine.v2.compilation.fixture_blocks import (
    CATALOGUE,
    axes_of,
    batch_input,
    origin_of,
    parameter,
    step,
    workflow,
)

EXPAND = step("expand", "expand", value="$inputs.values")


def _compile(steps, *, inputs=None, outputs=None):
    plan = compile_workflow(
        workflow(steps, outputs, inputs=inputs), catalogue=CATALOGUE
    )

    return plan


def test_item_bindings_broadcast_parents_and_scalars_to_the_deepest_level() -> None:
    plan = _compile(
        [
            EXPAND,
            step(
                "scale",
                "per_child",
                value="$steps.expand.child",
                factor="$inputs.values",
            ),
            step(
                "scale",
                "scalar_factor",
                value="$steps.expand.child",
                factor="$inputs.f",
            ),
        ],
        inputs=[batch_input("values", kind=["float"]), parameter("f", default=3)],
    )

    per_child = plan.step(("per_child",))
    assert axes_of(per_child.invocation_layout) == ["inputs", "expand:child"]
    assert [binding.mode for binding in per_child.bindings] == ["element", "ancestor"]
    assert [binding.mode for binding in plan.step(("scalar_factor",)).bindings] == [
        "element",
        "constant",
    ]


def test_outputs_of_one_step_share_its_expand_axis_but_keep_own_layouts() -> None:
    plan = _compile([EXPAND, step("stable_expand", "tiles", value="$inputs.values")])

    outputs = plan.step(("expand",)).outputs
    assert outputs["child"].layout == outputs["label"].layout
    assert axes_of(outputs["child"].layout) == ["inputs", "expand:child"]
    assert axes_of(outputs["count"].layout) == ["inputs"]
    assert outputs["child"].layout.axes[-1].kind == "dynamic_nesting"
    tile_axis = plan.step(("tiles",)).outputs["tile"].layout.axes[-1]
    assert (tile_axis.id, tile_axis.kind) == ("tiles:tiles", "static_nesting")
    assert plan.axis_origin("expand:child").step == ("expand",)


def test_equal_depth_data_of_unrelated_lineages_never_aligns() -> None:
    two_expansions = [
        EXPAND,
        step("expand", "other", value="$inputs.values"),
        step("scale", "join", value="$steps.expand.child", factor="$steps.other.child"),
    ]
    two_nested_inputs = [step("scale", "join", value="$inputs.a", factor="$inputs.b")]

    with pytest.raises(LineageError) as expansions:
        _compile(two_expansions)
    with pytest.raises(LineageError) as inputs:
        _compile(
            two_nested_inputs,
            inputs=[
                batch_input("a", depth=2, kind=["float"]),
                batch_input("b", depth=2, kind=["float"]),
            ],
        )

    assert "['inputs', 'expand:child']" in str(expansions.value)
    assert "['inputs', 'other:child']" in str(expansions.value)
    assert "never by size" in str(expansions.value)
    assert "inputs.a:1" in str(inputs.value) and "inputs.b:1" in str(inputs.value)


def test_explicit_shared_axis_ids_pair_inputs_and_distinct_ids_do_not() -> None:
    def explicit(axis_id):
        return {
            "name": axis_id + "_in",
            "kind": "float",
            "axes": [{"id": axis_id, "kind": "sample"}],
        }

    shared = [
        {"name": "a", "kind": "float", "axes": [{"id": "cams", "kind": "sample"}]},
        {"name": "b", "kind": "float", "axes": [{"id": "cams", "kind": "sample"}]},
    ]
    pair = [step("scale", "pair", value="$inputs.a", factor="$inputs.b")]

    plan = _compile(pair, inputs=shared)
    with pytest.raises(LineageError):
        _compile(
            [
                step(
                    "scale", "pair", value="$inputs.cams_in", factor="$inputs.others_in"
                )
            ],
            inputs=[explicit("cams"), explicit("others")],
        )

    assert [binding.mode for binding in plan.step(("pair",)).bindings] == [
        "element",
        "element",
    ]


def test_group_consumes_the_last_axis_and_preserve_keeps_it() -> None:
    plan = _compile(
        [
            EXPAND,
            step("scale", "scale_child", value="$steps.expand.child", factor=10),
            step(
                "sum_children",
                "sum",
                parent="$inputs.values",
                children="$steps.scale_child.scaled",
            ),
            step(
                "scale",
                "rejoin",
                value="$steps.sum.shifted",
                factor="$steps.expand.child",
            ),
            step("collapse", "collapse", data="$steps.expand.child"),
        ]
    )

    summed = plan.step(("sum",))
    assert axes_of(summed.invocation_layout) == ["inputs"]
    assert [(b.field, b.mode) for b in summed.bindings] == [
        ("parent", "element"),
        ("children", "group"),
    ]
    assert axes_of(summed.outputs["total"].layout) == ["inputs"]
    assert summed.outputs["shifted"].transform == "preserve"
    assert axes_of(summed.outputs["shifted"].layout) == ["inputs", "expand:child"]
    assert [b.mode for b in plan.step(("rejoin",)).bindings] == ["element", "element"]
    assert axes_of(plan.step(("collapse",)).invocation_layout) == ["inputs"]


def test_group_over_a_different_level_than_the_invocation_is_rejected() -> None:
    steps = [
        EXPAND,
        step(
            "sum_children", "sum", parent="$steps.expand.child", children="$inputs.deep"
        ),
    ]

    with pytest.raises(LineageError) as info:
        _compile(
            steps,
            inputs=[
                batch_input("values", kind=["float"]),
                batch_input("deep", depth=2),
            ],
        )

    assert info.value.field_path in (("children",), ("parent",))


def test_scalar_group_leaves_are_cast_per_parent_with_a_step_owned_axis() -> None:
    plan = _compile(
        [
            EXPAND,
            step(
                "named_groups",
                "named",
                reference="$inputs.values",
                groups={"label": "$inputs.label", "fixed": "$inputs.label"},
            ),
            step(
                "sum_children",
                "direct",
                parent="$inputs.values",
                children="$inputs.label",
            ),
            step("collapse", "scalar", data="$inputs.label"),
        ],
        inputs=[
            batch_input("values", kind=["float"]),
            parameter("label", default=1.5, kind=["float"]),
        ],
    )

    named = plan.step(("named",))
    casts = [binding.cast_layout for binding in named.bindings_for("groups")]
    assert [binding.mode for binding in named.bindings_for("groups")] == [
        "constant_group"
    ] * 2
    assert [axes_of(layout) for layout in casts] == [
        ["inputs", "named/groups/cast"]
    ] * 2
    assert axes_of(named.outputs["kept"].layout) == ["inputs", "named/groups/cast"]
    assert plan.axis_origin("named/groups/cast").kind == "cast"
    direct = plan.step(("direct",))
    assert direct.binding_for("children").mode == "constant_group"
    assert axes_of(direct.outputs["shifted"].layout) == [
        "inputs",
        "direct/children/cast",
    ]
    scalar = plan.step(("scalar",))
    assert axes_of(scalar.invocation_layout) == []
    assert axes_of(scalar.binding_for("data").cast_layout) == ["scalar/data/cast"]


def test_compound_groups_preserve_only_one_shared_group_layout() -> None:
    same = _compile(
        [
            EXPAND,
            step(
                "named_groups",
                "named",
                reference="$inputs.values",
                groups={"a": "$steps.expand.child", "b": "$steps.expand.label"},
            ),
        ]
    )
    mixed = [
        EXPAND,
        step(
            "named_groups",
            "named",
            reference="$inputs.values",
            groups={"a": "$steps.expand.child", "b": "$inputs.values"},
        ),
    ]

    with pytest.raises(LineageError):
        _compile(mixed)

    named = same.step(("named",))
    assert [b.mode for b in named.bindings_for("groups")] == ["group", "group"]
    assert axes_of(named.outputs["kept"].layout) == ["inputs", "expand:child"]
    assert axes_of(named.outputs["summary"].layout) == ["inputs"]


def test_list_of_batch_selectors_and_mixed_dicts_decide_delivery_per_step() -> None:
    plan = _compile(
        [
            step("scale", "a", value="$inputs.values"),
            step("scale", "b", value="$inputs.values"),
            step(
                "consensus",
                "consensus",
                predictions=["$steps.a.scaled", "$steps.b.scaled"],
            ),
            step(
                "consensus", "scalar_consensus", predictions=["$inputs.f", "$inputs.f"]
            ),
            step(
                "csv", "varying", columns={"camera": "north", "count": "$inputs.values"}
            ),
            step(
                "csv", "scalar_only", columns={"camera": "north", "count": "$inputs.f"}
            ),
        ],
        inputs=[batch_input("values", kind=["float"]), parameter("f", default=7)],
    )

    consensus = plan.step(("consensus",))
    assert [(b.field_path, b.mode, b.batch) for b in consensus.bindings] == [
        (("predictions", 0), "element", "always"),
        (("predictions", 1), "element", "always"),
    ]
    assert consensus.delivers_batches
    assert plan.step(("scalar_consensus",)).delivers_batches, "always casts constants"
    varying = plan.step(("varying",))
    assert [(b.field_path, b.mode, b.batch) for b in varying.bindings] == [
        (("columns", "count"), "element", "if_varying")
    ]
    assert varying.delivers_batches
    assert plan.step(("scalar_only",)).delivers_batches is False
    assert varying.params.columns["camera"] == "north"


def test_batch_accepting_block_requires_constant_per_call_fields() -> None:
    plan = _compile(
        [
            step("batch_scale", "ok", value="$inputs.values", factor="$inputs.f"),
            step("batch_scale", "cast", value="$inputs.f"),
        ],
        inputs=[batch_input("values", kind=["float"]), parameter("f", default=2)],
    )

    with pytest.raises(LineageError) as info:
        _compile(
            [
                step(
                    "batch_scale",
                    "bad",
                    value="$inputs.values",
                    factor="$inputs.values",
                )
            ]
        )

    assert info.value.field_path == ("factor",)
    assert [b.mode for b in plan.step(("ok",)).bindings] == ["element", "constant"]
    cast = plan.step(("cast",))
    assert axes_of(cast.invocation_layout) == [] and cast.delivers_batches


def test_input_free_sources_create_independent_roots() -> None:
    plan = _compile(
        [
            step("source", "left"),
            step("source", "right"),
            step("constant", "constant"),
            step(
                "scale",
                "mapped",
                value="$steps.left.items",
                factor="$steps.constant.value",
            ),
            step("collapse", "total", data="$steps.left.items"),
            step(
                "scale",
                "per_input",
                value="$inputs.values",
                factor="$steps.constant.value",
            ),
        ]
    )

    left = plan.step(("left",))
    assert axes_of(left.invocation_layout) == [] and left.bindings == ()
    assert axes_of(left.outputs["items"].layout) == ["left:generated"]
    assert plan.axis_origin("left:generated").kind == "expand"
    mapped = plan.step(("mapped",))
    assert axes_of(mapped.invocation_layout) == ["left:generated"]
    assert [b.mode for b in mapped.bindings] == ["element", "constant"]
    assert axes_of(plan.step(("total",)).invocation_layout) == []
    assert [b.mode for b in plan.step(("per_input",)).bindings] == [
        "element",
        "constant",
    ]
    with pytest.raises(LineageError):
        _compile(
            [
                step("source", "left"),
                step("source", "right"),
                step(
                    "scale",
                    "pair",
                    value="$steps.left.items",
                    factor="$steps.right.items",
                ),
            ]
        )


def test_nested_constant_reaches_an_item_field_as_a_constant_source() -> None:
    child = workflow(
        [step("scale", "s", value="$inputs.x", factor="$inputs.k")],
        {"y": "$steps.s.scaled"},
        inputs=[batch_input("x", kind=["float"]), parameter("k", default=4)],
    )
    plan = _compile(
        [
            {
                "type": "inner_workflow",
                "name": "child",
                "workflow_definition": child,
                "parameter_bindings": {"x": "$inputs.values"},
            }
        ]
    )

    factor = plan.step(("child", "s")).binding_for("factor")
    assert (origin_of(plan, factor.source), factor.mode) == (Constant(4), "constant")
    value = plan.step(("child", "s")).binding_for("value").source
    assert origin_of(plan, value) == InputPort("values")


def test_native_image_workflow_layouts_and_literal_errors() -> None:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "images"}],
        "steps": [
            {
                "name": "crop",
                "type": "v2/crop",
                "image": "$inputs.images",
                "regions": [[40, 40, 100, 100], [120, 10, 180, 70]],
            },
            {
                "name": "keep",
                "type": "v2/has_brightness",
                "image": "$steps.crop.crops",
                "minimum": 100,
            },
            {
                "name": "gate",
                "type": "v2/continue_if",
                "condition": "$steps.keep.keep",
                "next_steps": ["$steps.invert"],
            },
            {"name": "invert", "type": "v2/invert", "image": "$steps.crop.crops"},
            {
                "name": "mosaic",
                "type": "v2/mosaic",
                "images": "$steps.invert.image",
                "tile_size": 48,
                "background": 128,
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "mosaic", "selector": "$steps.mosaic.image"}
        ],
    }

    plan = compile_workflow(definition, catalogue=create_catalogue())
    bad = {
        **definition,
        "steps": [{**definition["steps"][0], "regions": [[10, 0, 0, 10]]}],
    }

    crop = plan.step(("crop",))
    assert [axis.kind for axis in crop.outputs["crops"].layout.axes] == [
        "sample",
        "dynamic_nesting",
    ]
    assert axes_of(crop.outputs["summary"].layout) == ["inputs"]
    assert crop.outputs["crops"].source_field == "image"
    assert axes_of(plan.step(("invert",)).invocation_layout) == [
        "inputs",
        "crop:regions",
    ]
    assert [gate.controller for gate in plan.step(("invert",)).gates] == [("gate",)]
    mosaic = plan.step(("mosaic",))
    assert mosaic.binding_for("images").mode == "group"
    assert axes_of(mosaic.outputs["image"].layout) == ["inputs"]
    with pytest.raises(ParamsValidationError) as info:
        compile_workflow(bad, catalogue=create_catalogue())
    assert info.value.step_path == ("crop",) and info.value.field_path[0] == "regions"
