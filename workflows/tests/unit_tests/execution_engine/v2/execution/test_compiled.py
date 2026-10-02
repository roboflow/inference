"""End to end: real ``compile_workflow`` plans run by the executor.

Scenarios mirror the V1 reference cases (development/workflows-2.0/
02-sequential-parity/reference) with local fixture blocks. Values and call
counts are asserted; deliberate corrections of V1 defects are labelled.
"""

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    StepExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)

from . import blocks

CATALOGUE = Catalogue(
    [
        blocks.Collect,
        blocks.Consensus,
        blocks.ContinueIf,
        blocks.Counter,
        blocks.Csv,
        blocks.Deferred,
        blocks.Echo,
        blocks.Expand,
        blocks.Failing,
        blocks.FirstNonEmpty,
        blocks.Mutate,
        blocks.NamedGroups,
        blocks.Notice,
        blocks.Route,
        blocks.Scale,
        blocks.Source,
        blocks.StitchAndTranslate,
        blocks.Sum,
    ],
    kinds=(FLOAT_KIND, INTEGER_KIND, STRING_KIND),
)


def workflow(inputs, steps, outputs=()):
    definition = {
        "version": "2.0",
        "inputs": list(inputs),
        "steps": list(steps),
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs
        ],
    }

    return definition


def batch(name, kind="float"):
    return {"type": "WorkflowBatchInput", "name": name, "kind": [kind]}


def parameter(name, **extra):
    return {"type": "WorkflowParameter", "name": name, **extra}


def counts(session):
    """Calls per step path, rendered as ``"child/step"``."""
    rendered = {
        "/".join(path): len(instance.calls)
        for path, instance in session.instances.items()
        if instance.calls
    }

    return rendered


def compiled_session(definition):
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session()

    return session


def test_literal_default_parameter_and_step_selector_in_one_field() -> None:
    session = compiled_session(
        workflow(
            [batch("values"), parameter("f", default_value=10)],
            [
                {
                    "type": "test/scale@v1",
                    "name": "literal",
                    "value": "$inputs.values",
                    "factor": 3,
                },
                {"type": "test/scale@v1", "name": "default", "value": "$inputs.values"},
                {
                    "type": "test/scale@v1",
                    "name": "from_param",
                    "value": "$inputs.values",
                    "factor": "$inputs.f",
                },
                {
                    "type": "test/scale@v1",
                    "name": "from_step",
                    "value": "$inputs.values",
                    "factor": "$steps.literal.scaled",
                },
            ],
            [
                ("literal", "$steps.literal.scaled"),
                ("default", "$steps.default.scaled"),
                ("from_param", "$steps.from_param.scaled"),
                ("from_step", "$steps.from_step.scaled"),
            ],
        )
    )

    rows = session.run({"values": [1, 2]}).rows()

    assert rows == [
        {"literal": 3, "default": 2, "from_param": 10, "from_step": 3},
        {"literal": 6, "default": 4, "from_param": 20, "from_step": 12},
    ]
    assert counts(session) == {
        "literal": 2,
        "default": 2,
        "from_param": 2,
        "from_step": 2,
    }


@pytest.mark.parametrize("b", [10, [10]])
def test_runtime_input_broadcast(b) -> None:
    session = compiled_session(
        workflow(
            [batch("a"), batch("b")],
            [
                {
                    "type": "test/scale@v1",
                    "name": "s",
                    "value": "$inputs.a",
                    "factor": "$inputs.b",
                }
            ],
            [("s", "$steps.s.scaled")],
        )
    )

    rows = session.run({"a": [1, 2, 3], "b": b}).rows()

    assert rows == [{"s": 10}, {"s": 20}, {"s": 30}]


def test_runtime_input_length_mismatch_fails_before_any_step() -> None:
    session = compiled_session(
        workflow(
            [batch("labels", "string"), batch("values", "string")],
            [{"type": "test/echo@v1", "name": "echo", "value": "$inputs.values"}],
        )
    )

    with pytest.raises(WorkflowInputError, match="same length"):
        session.run({"labels": ["x", "y"], "values": ["a", "b", "c"]})

    assert counts(session) == {}


def test_empty_mask_beside_nonempty_mask_makes_zero_calls() -> None:
    # OBS-V1-01 correction: V1 calls the sink three times here.
    session = compiled_session(
        workflow(
            [batch("items")],
            [
                {
                    "type": "test/continue_if@v1",
                    "name": "above_one",
                    "value": "$inputs.items",
                    "threshold": 1,
                    "next_steps": ["$steps.notice"],
                },
                {
                    "type": "test/continue_if@v1",
                    "name": "above_hundred",
                    "value": "$inputs.items",
                    "threshold": 100,
                    "next_steps": ["$steps.notice"],
                },
                {"type": "test/notice@v1", "name": "notice"},
            ],
        )
    )

    session.run({"items": [0, 2, 3, 4]})

    assert counts(session) == {"above_one": 4, "above_hundred": 4}


def test_shallow_gate_over_deeper_data_admits_descendants() -> None:
    # OBS-V1-03 correction: V1 compiles this and makes zero echo calls.
    session = compiled_session(
        workflow(
            [batch("items", "*")],
            [
                {
                    "type": "test/expand@v1",
                    "name": "expand",
                    "value": "$inputs.items",
                    "count": "$inputs.items",
                },
                {
                    "type": "test/continue_if@v1",
                    "name": "parent_gate",
                    "value": "$inputs.items",
                    "threshold": 1,
                    "next_steps": ["$steps.echo"],
                },
                {
                    "type": "test/echo@v1",
                    "name": "echo",
                    "value": "$steps.expand.children",
                },
            ],
            [("selected", "$steps.echo.value")],
        )
    )

    rows = session.run({"items": [1, 2, 0]}).rows()

    assert rows == [{"selected": [None]}, {"selected": [2, 3]}, {"selected": []}]
    assert counts(session) == {"expand": 3, "parent_gate": 3, "echo": 2}


def test_switch_routes_and_recovery_join_sees_every_index() -> None:
    session = compiled_session(
        workflow(
            [batch("items", "string")],
            [
                {
                    "type": "test/route@v1",
                    "name": "route",
                    "value": "$inputs.items",
                    "cases": {"1": "$steps.left", "2": "$steps.right"},
                },
                {"type": "test/echo@v1", "name": "left", "value": "$inputs.items"},
                {"type": "test/echo@v1", "name": "right", "value": "$inputs.items"},
                {
                    "type": "test/first_non_empty@v1",
                    "name": "merge",
                    "data": ["$steps.left.value", "$steps.right.value"],
                },
            ],
            [("merged", "$steps.merge.value")],
        )
    )

    rows = session.run({"items": ["1", "2", "3"]}).rows()

    assert rows == [{"merged": "1"}, {"merged": "2"}, {"merged": None}]
    assert counts(session) == {"route": 3, "left": 1, "right": 1, "merge": 3}


def test_gate_on_a_nested_workflow_suppresses_every_child_root() -> None:
    # OBS-V1-02 correction: V1 gates only the first inlined child step.
    child = workflow(
        [],
        [
            {"type": "test/notice@v1", "name": "first"},
            {"type": "test/notice@v1", "name": "second", "message": "second"},
        ],
    )
    session = compiled_session(
        workflow(
            [parameter("value")],
            [
                {
                    "type": "test/continue_if@v1",
                    "name": "gate",
                    "value": "$inputs.value",
                    "next_steps": ["$steps.inner"],
                },
                {
                    "type": "inner_workflow",
                    "name": "inner",
                    "workflow_definition": child,
                    "parameter_bindings": {},
                },
            ],
        )
    )

    session.run({"value": 0})
    denied = counts(session)
    session.run({"value": 1})

    assert denied == {"gate": 1}
    assert counts(session) == {"gate": 2, "inner/first": 1, "inner/second": 1}


def test_genuine_empty_expansion_reaches_the_reducer() -> None:
    # V1-QUIRK-EMPTY-EXPANSION: V1 skips the reducer and returns null sums.
    session = compiled_session(
        workflow(
            [batch("values")],
            [
                {
                    "type": "test/expand@v1",
                    "name": "expand",
                    "value": "$inputs.values",
                    "count": 0,
                },
                {
                    "type": "test/sum@v1",
                    "name": "sum",
                    "values": "$steps.expand.children",
                },
            ],
            [("children", "$steps.expand.children"), ("sum", "$steps.sum.total")],
        )
    )

    rows = session.run({"values": [1, 20]}).rows()

    assert rows == [{"children": [], "sum": 0}, {"children": [], "sum": 0}]
    assert counts(session) == {"expand": 2, "sum": 2}


def test_source_axis_and_wildcard_output_rows() -> None:
    session = compiled_session(
        workflow(
            [batch("values")],
            [
                {"type": "test/source@v1", "name": "source", "values": [5.0, 6.0]},
                {
                    "type": "test/expand@v1",
                    "name": "expand",
                    "value": "$inputs.values",
                    "count": 1,
                },
            ],
            [
                ("generated", "$steps.source.generated"),
                ("everything", "$steps.expand.*"),
            ],
        )
    )

    result = session.run({"values": [1, 2]})

    assert result.rows() == [
        {"generated": [5.0, 6.0], "everything": {"children": [1], "count": 1}},
        {"generated": [5.0, 6.0], "everything": {"children": [2], "count": 1}},
    ]
    assert set(result.selections["everything"].values()) == {
        "everything/children",
        "everything/count",
    }


def test_state_persists_per_session_and_futures_are_ready() -> None:
    definition = workflow(
        [batch("values")],
        [
            {"type": "test/counter@v1", "name": "count", "value": "$inputs.values"},
            {"type": "test/deferred@v1", "name": "deferred", "value": "$inputs.values"},
            {
                "type": "test/scale@v1",
                "name": "consumer",
                "value": "$steps.deferred.doubled",
            },
        ],
        [("count", "$steps.count.count"), ("consumed", "$steps.consumer.scaled")],
    )
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session()

    first = session.run({"values": [1, 2]}).rows()
    second = session.run({"values": [3]}).rows()
    other = plan.create_session().run({"values": [4]}).rows()

    assert first == [{"count": 1, "consumed": 4}, {"count": 2, "consumed": 8}]
    assert second == [{"count": 3, "consumed": 12}]
    assert other == [{"count": 1, "consumed": 16}]


def test_consensus_list_and_mixed_csv_call_shapes() -> None:
    session = compiled_session(
        workflow(
            [batch("a"), batch("b"), parameter("x")],
            [
                {
                    "type": "test/consensus@v1",
                    "name": "vote",
                    "predictions": ["$inputs.a", "$inputs.b"],
                },
                {
                    "type": "test/csv@v1",
                    "name": "static",
                    "columns": {"x": "$inputs.x", "label": "l"},
                },
                {
                    "type": "test/csv@v1",
                    "name": "mixed",
                    "columns": {"a": "$inputs.a", "label": "l"},
                },
            ],
            [("votes", "$steps.vote.votes"), ("static", "$steps.static.row")],
        )
    )

    rows = session.run({"a": [1, 2], "b": [10, 20], "x": 7}).rows()

    assert rows == [
        {"votes": [1, 10], "static": {"x": 7, "label": "l"}},
        {"votes": [2, 20], "static": {"x": 7, "label": "l"}},
    ]
    assert counts(session) == {"vote": 1, "static": 1, "mixed": 1}


def test_compound_group_scalar_leaf_is_cast_under_each_parent() -> None:
    session = compiled_session(
        workflow(
            [batch("parents"), parameter("label")],
            [
                {
                    "type": "test/named_groups@v1",
                    "name": "named",
                    "parent": "$inputs.parents",
                    "groups": {"selected": "$inputs.label", "fixed": "$inputs.label"},
                }
            ],
            [("sizes", "$steps.named.sizes")],
        )
    )

    rows = session.run({"parents": [10, 20], "label": "chosen"}).rows()

    calls = session.instances[("named",)].calls
    assert [call["groups"]["selected"].indices for call in calls] == [
        ((0, 0),),
        ((1, 0),),
    ]
    assert rows == [
        {"sizes": {"selected": ["chosen"], "fixed": ["chosen"]}},
        {"sizes": {"selected": ["chosen"], "fixed": ["chosen"]}},
    ]


def test_nested_step_error_carries_scope_path_index_and_cause() -> None:
    child = workflow(
        [batch("x", "*")],
        [{"type": "test/failing@v1", "name": "check", "value": "$inputs.x"}],
        [("value", "$steps.check.value")],
    )
    handled = []
    plan = compile_workflow(
        workflow(
            [batch("values", "*")],
            [
                {
                    "type": "inner_workflow",
                    "name": "child",
                    "workflow_definition": child,
                    "parameter_bindings": {"x": "$inputs.values"},
                }
            ],
            [("value", "$steps.child.value")],
        ),
        catalogue=CATALOGUE,
    )
    session = plan.create_session(error_handler=handled.append)

    with pytest.raises(
        StepExecutionError, match=r"child/check at index \[1\]"
    ) as caught:
        session.run({"values": [1, -1]})

    assert handled == [caught.value]
    assert caught.value.step_path == ("child", "check")
    assert caught.value.index == (1,)
    assert isinstance(caught.value.__cause__, ValueError)


def test_nested_default_is_a_fresh_copy_per_run_shared_by_its_consumers() -> None:
    child = workflow(
        [parameter("payload", default_value={"count": 0})],
        [
            {"type": "test/mutate@v1", "name": "change", "payload": "$inputs.payload"},
            {"type": "test/echo@v1", "name": "read", "value": "$inputs.payload"},
        ],
        [("seen", "$steps.read.value")],
    )
    definition = workflow(
        [],
        [
            {
                "type": "inner_workflow",
                "name": "child",
                "workflow_definition": child,
                "parameter_bindings": {},
            }
        ],
        [("seen", "$steps.child.seen")],
    )
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session()

    runs = [session.run({}).rows(), session.run({}).rows()]
    other_session = plan.create_session().run({}).rows()

    # The echo runs after the mutation in the same run and sees it.
    assert runs == [[{"seen": {"count": 1}}], [{"seen": {"count": 1}}]]
    assert other_session == [{"seen": {"count": 1}}]
    assert child["inputs"][0]["default_value"] == {"count": 0}
