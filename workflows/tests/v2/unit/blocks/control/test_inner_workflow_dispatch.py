"""Local wrapper exercised through the real compiler and execution session."""

from copy import deepcopy

import pytest
from roboflow_workflows.v2.blocks.control.inner_workflow_dispatch import (
    inner_workflow_step,
)
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.compilation import WorkflowReference, compile_workflow
from roboflow_workflows.v2.engine.errors import (
    CycleError,
    NestedWorkflowError,
    WorkflowCompileError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.support.compilation.fixture_blocks import (
    CATALOGUE,
    batch_input,
    parameter,
    step,
    workflow,
)


def child():
    return workflow(
        [step("scale", "scale", value="$inputs.x", factor="$inputs.factor")],
        {"y": "$steps.scale.scaled"},
        inputs=[batch_input("x", kind=["float"]), parameter("factor", default=3)],
    )


def parent(nested):
    return workflow([nested], {"y": "$steps.child.y"})


def test_wrapper_real_session_matches_hand_written_embedded_step():
    wrapped = inner_workflow_step(
        "child", workflow_definition=child(), parameter_bindings={"x": "$inputs.values"}
    )
    manual = {**wrapped}
    del manual["execution_mode"]
    for declaration in (wrapped, manual):
        plan = compile_workflow(parent(declaration), catalogue=CATALOGUE)
        assert [item.path for item in plan.steps] == [("child", "scale")]
        with plan.create_session() as session:
            assert session.run({"values": [2.0, 5.0]}).rows() == [
                {"y": 6.0},
                {"y": 15.0},
            ]


def test_saved_reference_version_cache_and_input_copy():
    calls = []
    store = child()
    snapshot = deepcopy(store)

    def resolver(reference):
        calls.append(reference)
        return store

    reference = WorkflowReference("local", "triple", "3")
    children = [
        inner_workflow_step(
            name, reference=reference, parameter_bindings={"x": "$inputs.values"}
        )
        for name in ("left", "right")
    ]
    definition = workflow(children, {"a": "$steps.left.y", "b": "$steps.right.y"})
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, reference_resolver=resolver
    )
    assert calls == [reference]
    assert store == snapshot
    with plan.create_session() as session:
        assert session.run({"values": [4.0]}).rows() == [{"a": 12.0, "b": 12.0}]


def test_builder_detaches_mutable_data_without_serializing_secrets():
    definition = child()
    secret = Secret("invented-secret-value", name="test")
    bindings = {"x": [secret]}
    built = inner_workflow_step(
        "child", workflow_definition=definition, parameter_bindings=bindings
    )
    definition["steps"].clear()
    bindings["x"].clear()
    assert built["workflow_definition"]["steps"]
    assert built["parameter_bindings"]["x"][0] is secret
    assert "invented-secret-value" not in repr(built)


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"workflow_definition": {}},
        {"workflow_definition": child(), "reference": WorkflowReference("local", "x")},
        {"reference": WorkflowReference("", "x")},
        {"workflow_definition": child(), "parameter_bindings": ["bad"]},
    ],
)
def test_invalid_sources_and_bindings_fail_before_execution(fields):
    with pytest.raises(WorkflowCompileError):
        inner_workflow_step("child", **fields)


def test_missing_resolver_and_bad_child_bindings_are_compile_errors():
    saved = inner_workflow_step("child", reference=WorkflowReference("local", "x"))
    with pytest.raises(NestedWorkflowError, match="received no reference_resolver"):
        compile_workflow(parent(saved), catalogue=CATALOGUE)
    inline = inner_workflow_step("child", workflow_definition=child())
    with pytest.raises(NestedWorkflowError, match="miss required child inputs"):
        compile_workflow(parent(inline), catalogue=CATALOGUE)


def test_depth_boundary_uses_existing_engine_limit():
    leaf = child()
    middle = workflow(
        [
            inner_workflow_step(
                "child",
                workflow_definition=leaf,
                parameter_bindings={"x": "$inputs.values"},
            )
        ],
        {"y": "$steps.child.y"},
    )
    outer = parent(
        inner_workflow_step(
            "child",
            workflow_definition=middle,
            parameter_bindings={"values": "$inputs.values"},
        )
    )
    compile_workflow(
        outer, catalogue=CATALOGUE, options=CompileOptions(max_nested_depth=2)
    )
    with pytest.raises(NestedWorkflowError, match="depth 2 exceeds the limit of 1"):
        compile_workflow(
            outer, catalogue=CATALOGUE, options=CompileOptions(max_nested_depth=1)
        )


def test_saved_cycle_retains_the_existing_engine_error():
    saved = inner_workflow_step("child", reference=WorkflowReference("local", "loop"))
    definition = workflow([saved], inputs=[])
    with pytest.raises(CycleError, match="local/loop -> local/loop"):
        compile_workflow(
            definition, catalogue=CATALOGUE, reference_resolver=lambda ref: definition
        )


def test_secret_input_crosses_child_scope_without_conversion():
    from roboflow_workflows.v2.data.secrets import SECRET_KIND
    from roboflow_workflows.v2.engine.catalogue import Catalogue

    definition = workflow(
        [step("echo", "echo", value="$inputs.x")],
        {"y": "$steps.echo.value"},
        inputs=[parameter("x", kind=["secret"])],
    )
    wrapped = inner_workflow_step(
        "child", workflow_definition=definition, parameter_bindings={"x": "$inputs.x"}
    )
    outer = workflow(
        [wrapped], {"y": "$steps.child.y"}, inputs=[parameter("x", kind=["secret"])]
    )
    catalogue = Catalogue.merge(CATALOGUE, Catalogue([], kinds=[SECRET_KIND]))
    secret = Secret("invented-local-secret", name="test")
    plan = compile_workflow(outer, catalogue=catalogue)
    with plan.create_session() as session:
        rows = session.run({"x": secret}).rows()
    assert rows[0]["y"] is secret
    assert "invented-local-secret" not in repr(rows)
