"""RunResult entries and V1-shaped rows."""

from roboflow_workflows.execution_engine.v2.data import Axis, EntryLayout

from .blocks import ContinueIf, Echo, Expand, Scale, Source, tagged_kind
from .plans import BATCH, SCALAR, PlanBuilder, nested

GENERATED = EntryLayout(axes=(Axis(id="source/generated", kind="dynamic_nesting"),))


def test_scalar_outputs_make_a_single_row() -> None:
    plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(Scale, "scale", value="$inputs.value")
        .output("scaled", "$steps.scale.scaled")
        .output("value", "$inputs.value")
        .build()
    )

    rows = plan.create_session().run({"value": 4}).rows()

    assert rows == [{"scaled": 8, "value": 4}]


def test_wildcard_with_different_layouts_keeps_one_entry_per_port() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(
            Expand, "expand", at=BATCH, value="$inputs.values", count="$inputs.values"
        )
        .output("everything", "$steps.expand.*")
        .build()
    )

    result = plan.create_session().run({"values": [1, 2]})

    assert result.selections == {
        "everything": {
            "$steps.expand.children": "everything/children",
            "$steps.expand.count": "everything/count",
        }
    }
    assert result.outputs.layout["everything/children"].axis_ids == (
        "N",
        "expand/items",
    )
    assert result.outputs.layout["everything/count"].axis_ids == ("N",)
    assert result.rows() == [
        {"everything": {"children": [1], "count": 1}},
        {"everything": {"children": [2, 3], "count": 2}},
    ]


def test_wildcard_with_shared_nested_layout_zips_ports_at_the_leaves() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Expand, "expand", at=BATCH, value="$inputs.values", count=2)
        .step(
            Expand,
            "inner",
            at=nested("expand/items"),
            value="$steps.expand.children",
            count=0,
        )
        .output("inner", "$steps.inner.*")
        .build()
    )

    rows = plan.create_session().run({"values": [1]}).rows()

    assert rows == [
        {
            "inner": [
                {"children": [], "count": 0},
                {"children": [], "count": 0},
            ]
        }
    ]


def test_generated_axis_is_copied_whole_into_every_input_row() -> None:
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(Source, "source", values=[5.0, 6.0])
        .step(Scale, "scale", at=BATCH, value="$inputs.values")
        .output("scaled", "$steps.scale.scaled")
        .output("generated", "$steps.source.generated")
        .build()
    )
    session = plan.create_session()

    rows = session.run({"values": [1, 2]}).rows()
    empty_input_rows = session.run({"values": []}).rows()

    assert rows == [
        {"scaled": 2, "generated": [5.0, 6.0]},
        {"scaled": 4, "generated": [5.0, 6.0]},
    ]
    assert empty_input_rows == [{"generated": [5.0, 6.0]}]


def test_generated_axis_without_input_axis_is_one_row() -> None:
    plan = (
        PlanBuilder()
        .step(Source, "source", values=[5.0, 6.0])
        .step(Scale, "scale", at=GENERATED, value="$steps.source.generated", factor=1)
        .output("scaled", "$steps.scale.scaled")
        .build()
    )

    rows = plan.create_session().run({}).rows()

    assert plan.axis_origin("source/generated").kind == "expand"
    assert rows == [{"scaled": [5.0, 6.0]}]


def test_kind_hooks_convert_with_output_options_and_serialize_on_request() -> None:
    tagged = tagged_kind()
    plan = (
        PlanBuilder(kinds=(tagged,))
        .input("items", BATCH, kinds=("tagged",))
        .step(Echo, "echo", at=BATCH, value="$inputs.items")
        .output("raw", "$inputs.items", options={"scale": 10})
        .build()
    )
    result = plan.create_session().run({"items": [1, None]})

    assert result.outputs.data["raw"][0] == ("tag", 1)
    assert result.rows() == [{"raw": ("tag", 10)}, {"raw": None}]
    assert result.rows(serialize=True) == [{"raw": {"tagged": 10}}, {"raw": None}]


def test_filtered_scalar_output_is_none_and_absent_from_the_buffer() -> None:
    plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(ContinueIf, "gate", value="$inputs.value", next_steps=["$steps.echo"])
        .step(Echo, "echo", gates=(("gate", "$steps.echo"),), value="$inputs.value")
        .step(Echo, "after", value="$steps.echo.value")
        .output("after", "$steps.after.value")
        .build()
    )
    session = plan.create_session()

    result = session.run({"value": 0})

    assert result.statuses == {"after": "filtered"}
    assert result.filtered_paths == {"after": ((),)}
    assert "after" not in result.outputs.data
    assert result.rows() == [{"after": None}]
    assert session.instances[("after",)].calls == []


def test_none_payload_of_a_scalar_output_is_delivered_not_filtered() -> None:
    plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(Echo, "echo", value="$inputs.value")
        .step(Echo, "after", value="$steps.echo.value")
        .output("after", "$steps.after.value")
        .build()
    )
    session = plan.create_session()

    result = session.run({"value": None})

    assert result.statuses == {"after": "complete"}
    assert result.outputs.data["after"] is None
    assert session.instances[("after",)].calls == [{"value": None}]
