"""Decision 021: nested workflow inputs are prepared once per run.

literal/default (Constant)   private copy, first successful kind decoder,
                             kind check; shared by every consumer and
                             direct output of this run
port or outer child input    already prepared: kinds checked, never
                             decoded or copied; layout, metadata and
                             filtered positions unchanged
"""

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import (
    Batch,
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.execution_engine.v2.errors import WorkflowInputError
from roboflow_workflows.execution_engine.v2.kinds import DICTIONARY_KIND, INTEGER_KIND

from . import blocks
from .blocks import DECODED, counting_kind

COUNTED = counting_kind()
CATALOGUE = Catalogue(
    [blocks.Echo, blocks.Mutate, blocks.ContinueIf, blocks.ContextProbe],
    kinds=(COUNTED, INTEGER_KIND, DICTIONARY_KIND),
)


@pytest.fixture(autouse=True)
def clear_decoder_log():
    DECODED.clear()
    yield
    DECODED.clear()


def workflow(inputs=(), steps=(), outputs=()):
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


def parameter(name, kind="*", **extra):
    return {"type": "WorkflowParameter", "name": name, "kind": [kind], **extra}


def batch(name, kind="*"):
    return {"type": "WorkflowBatchInput", "name": name, "kind": [kind]}


def nested(name, child, bindings=None):
    step = {
        "type": "inner_workflow",
        "name": name,
        "workflow_definition": child,
        "parameter_bindings": bindings or {},
    }

    return step


def echo(name, value):
    return {"type": "test/echo@v1", "name": name, "value": value}


def session_for(definition):
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session()

    return session


def calls(session, *path):
    return session.instances[path].calls


def test_child_default_is_decoded_once_per_run_and_shared_by_all_uses() -> None:
    child = workflow(
        [parameter("v", "counted", default_value="7")],
        [echo("first", "$inputs.v"), echo("second", "$inputs.v")],
        [("first", "$steps.first.value"), ("direct", "$inputs.v")],
    )
    plan = compile_workflow(
        workflow(
            steps=[nested("child", child)],
            outputs=[
                ("first", "$steps.child.first"),
                ("direct", "$steps.child.direct"),
            ],
        ),
        catalogue=CATALOGUE,
    )
    session = plan.create_session()

    first_run = session.run({})
    second_run = session.run({})
    other_session = plan.create_session().run({})

    first, second = calls(session, "child", "first"), calls(session, "child", "second")
    assert DECODED == ["7", "7", "7"]
    assert first[0]["value"] is second[0]["value"]
    assert first_run.outputs.data["direct"] is first[0]["value"]
    assert second_run.outputs.data["direct"] is not first_run.outputs.data["direct"]
    assert first_run.rows() == [{"first": {"decoded": 7}, "direct": {"decoded": 7}}]
    # Direct forwarding uses the child's declared kind hooks; the echo output
    # is a wildcard, so its payload is serialized unchanged.
    assert other_session.rows(serialize=True) == [
        {"first": {"decoded": 7}, "direct": "#7"}
    ]


def test_literal_binding_is_decoded_and_mutation_stays_within_the_run() -> None:
    child = workflow(
        [parameter("payload", "counted")],
        [
            {"type": "test/mutate@v1", "name": "change", "payload": "$inputs.payload"},
            echo("read", "$inputs.payload"),
        ],
        [("seen", "$steps.read.value")],
    )
    session = session_for(
        workflow(
            steps=[nested("child", child, {"payload": "8"})],
            outputs=[("seen", "$steps.child.seen")],
        )
    )

    runs = [session.run({}).rows(), session.run({}).rows()]

    assert runs == [[{"seen": {"decoded": 8, "count": 1}}]] * 2


def test_wildcard_source_is_checked_against_the_child_kind() -> None:
    child = workflow(
        [parameter("v", "integer")],
        [echo("echo", "$inputs.v")],
        [("value", "$steps.echo.value")],
    )
    session = session_for(
        workflow(
            [parameter("value")],
            [nested("child", child, {"v": "$inputs.value"})],
            [("value", "$steps.child.value")],
        )
    )

    rows = session.run({"value": 3}).rows()
    with pytest.raises(WorkflowInputError, match=r"\$steps\.child: \$inputs\.v"):
        session.run({"value": "wrong-type"})

    assert rows == [{"value": 3}]
    assert len(calls(session, "child", "echo")) == 1


def test_direct_forwarding_without_consumer_still_checks_the_child_kind() -> None:
    child = workflow(
        [parameter("v", "integer"), parameter("other", default_value=0)],
        [echo("unrelated", "$inputs.other")],
        [("direct", "$inputs.v")],
    )
    session = session_for(
        workflow(
            [parameter("value")],
            [nested("child", child, {"v": "$inputs.value"})],
            [("direct", "$steps.child.direct")],
        )
    )

    rows = session.run({"value": 4}).rows()
    with pytest.raises(WorkflowInputError, match="is not a valid"):
        session.run({"value": "wrong-type"})

    assert rows == [{"direct": 4}]


def test_selected_payload_is_checked_but_never_decoded_or_copied() -> None:
    child = workflow(
        [parameter("v", "counted")],
        [echo("echo", "$inputs.v")],
        [("value", "$steps.echo.value")],
    )
    session = session_for(
        workflow(
            [parameter("value")],
            [nested("child", child, {"v": "$inputs.value"})],
            [("value", "$steps.child.value")],
        )
    )
    prepared = {"decoded": 5}

    session.run({"value": prepared})
    with pytest.raises(WorkflowInputError, match="is not a valid"):
        session.run({"value": "7"})

    assert DECODED == []
    assert calls(session, "child", "echo")[0]["value"] is prepared


def test_deep_alias_chain_decodes_once_and_checks_every_level() -> None:
    grandchild = workflow(
        [parameter("y", "dictionary")],
        [echo("echo", "$inputs.y")],
        [("echoed", "$steps.echo.value"), ("direct", "$inputs.y")],
    )
    child = workflow(
        [parameter("x", "counted")],
        [nested("inner", grandchild, {"y": "$inputs.x"})],
        [("echoed", "$steps.inner.echoed"), ("direct", "$steps.inner.direct")],
    )
    session = session_for(
        workflow(
            steps=[nested("outer", child, {"x": "9"})],
            outputs=[
                ("echoed", "$steps.outer.echoed"),
                ("direct", "$steps.outer.direct"),
            ],
        )
    )

    result = session.run({})

    assert DECODED == ["9"]
    assert result.rows() == [{"echoed": {"decoded": 9}, "direct": {"decoded": 9}}]
    assert (
        result.outputs.data["direct"]
        is calls(session, "outer", "inner", "echo")[0]["value"]
    )


def test_deep_alias_chain_rejection_names_the_inner_boundary() -> None:
    grandchild = workflow(
        [parameter("y", "integer"), parameter("other", default_value=0)],
        [echo("unrelated", "$inputs.other")],
        [("direct", "$inputs.y")],
    )
    child = workflow(
        [parameter("x")],
        [nested("inner", grandchild, {"y": "$inputs.x"})],
        [("direct", "$steps.inner.direct")],
    )
    session = session_for(
        workflow(
            [parameter("value")],
            [nested("outer", child, {"x": "$inputs.value"})],
            [("direct", "$steps.outer.direct")],
        )
    )

    with pytest.raises(WorkflowInputError, match=r"\$steps\.outer/inner: \$inputs\.y"):
        session.run({"value": "text"})


def test_sparse_source_keeps_indices_metadata_and_names_the_failing_index() -> None:
    child = workflow(
        [batch("items", "integer")],
        [echo("echo", "$inputs.items")],
        [("items", "$steps.echo.value")],
    )
    session = session_for(
        workflow(
            [batch("items")],
            [nested("child", child, {"items": "$inputs.items"})],
            [("items", "$steps.child.items")],
        )
    )
    camera = SampleContext("camera")
    items = InputValue(
        Batch([1, 3], indices=[(0,), (2,)]),
        metadata=EntryMetadata(sample={(2,): camera}),
    )

    result = session.run({"items": items})
    with pytest.raises(WorkflowInputError, match=r"at index \[1\]"):
        session.run({"items": [1, "bad"]})

    assert result.rows() == [{"items": 1}, {"items": None}, {"items": 3}]
    assert result.outputs.metadata["items"].sample_at((2,)) == camera


def test_filtered_positions_are_skipped_by_child_checks_and_stay_filtered() -> None:
    child = workflow(
        [batch("items", "integer")],
        [echo("echo", "$inputs.items")],
        [("items", "$steps.echo.value")],
    )
    session = session_for(
        workflow(
            [batch("items")],
            [
                {
                    "type": "test/continue_if@v1",
                    "name": "gate",
                    "value": "$inputs.items",
                    "threshold": 0,
                    "next_steps": ["$steps.admitted"],
                },
                echo("admitted", "$inputs.items"),
                nested("child", child, {"items": "$steps.admitted.value"}),
            ],
            [("items", "$steps.child.items")],
        )
    )

    result = session.run({"items": [2, 0, 5]})

    assert result.rows() == [{"items": 2}, {"items": None}, {"items": 5}]
    assert result.filtered_paths["items"] == ((1,),)


def test_nested_step_context_carries_its_scope_path() -> None:
    child = workflow(
        [parameter("v")],
        [{"type": "test/context_probe@v1", "name": "probe", "value": "$inputs.v"}],
    )
    session = session_for(
        workflow([parameter("value")], [nested("child", child, {"v": "$inputs.value"})])
    )

    session.run({"value": 1})

    (call,) = calls(session, "child", "probe")
    assert call["context"].step_path == ("child", "probe")
