"""Decision 023: every block call and its result readiness run in a context.

The context names the step, block type, session, run and the call's logical
indices (one index per call, every admitted index for a batch-delivering
call). It is reset on exit, also after errors and around nested engine calls.
"""

import pytest
from roboflow_workflows.execution_engine.v2.context import (
    NoExecutionContextError,
    get_execution_context,
)
from roboflow_workflows.execution_engine.v2.errors import StepExecutionError

from .blocks import ContextProbe, ContextProbeMany, DeferredContext, RunsInner
from .plans import BATCH, SCALAR, PlanBuilder, nested


def contexts(session, name):
    return [call["context"] for call in session.instances[(name,)].calls]


def probe_plan(*, fail=False):
    plan = (
        PlanBuilder()
        .input("values", BATCH)
        .step(ContextProbe, "probe", at=BATCH, value="$inputs.values", fail=fail)
        .output("value", "$steps.probe.value")
        .build()
    )

    return plan


def test_each_call_sees_its_step_session_run_and_index() -> None:
    session = probe_plan().create_session()

    result = session.run({"values": ["a", "b"]})

    seen = contexts(session, "probe")
    assert [context.indices for context in seen] == [((0,),), ((1,),)]
    assert {context.step_path for context in seen} == {("probe",)}
    assert {context.block_type for context in seen} == {"test/context_probe@v1"}
    assert {context.session_id for context in seen} == {session.session_id}
    assert {context.run_id for context in seen} == {result.run_id}
    with pytest.raises(NoExecutionContextError):
        get_execution_context()


def test_batch_delivering_call_sees_every_index_across_parents() -> None:
    plan = (
        PlanBuilder()
        .input("values", nested("c"))
        .step(ContextProbeMany, "many", at=nested("c"), values="$inputs.values")
        .build()
    )
    session = plan.create_session()

    session.run({"values": [[1, 2], [3]]})

    (context,) = contexts(session, "many")
    assert context.indices == ((0, 0), (0, 1), (1, 0))


def test_context_is_reset_after_a_failing_call() -> None:
    session = probe_plan(fail=True).create_session()

    with pytest.raises(StepExecutionError, match="probe failure"):
        session.run({"values": ["a"]})

    assert contexts(session, "probe")[0].indices == ((0,),)
    with pytest.raises(NoExecutionContextError):
        get_execution_context()


def test_futures_are_resolved_inside_the_call_context() -> None:
    plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(DeferredContext, "deferred", value="$inputs.value")
        .output("step", "$steps.deferred.step")
        .build()
    )

    rows = plan.create_session().run({"value": 1}).rows()

    assert rows == [{"step": ("deferred",)}]


def test_nested_engine_call_restores_the_outer_context() -> None:
    inner_plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(ContextProbe, "probe", value="$inputs.value")
        .output("value", "$steps.probe.value")
        .build()
    )
    inner = inner_plan.create_session()
    outer_plan = (
        PlanBuilder()
        .input("value", SCALAR)
        .step(RunsInner, "outer", value="$inputs.value")
        .output("inner", "$steps.outer.inner")
        .build()
    )
    outer = outer_plan.create_session(resources={"inner": inner})

    rows = outer.run({"value": "x"}).rows()

    (call,) = outer.instances[("outer",)].calls
    (inner_context,) = contexts(inner, "probe")
    assert rows == [{"inner": [{"value": "x"}]}]
    assert call["before"] == call["after"]
    assert call["before"].step_path == ("outer",)
    assert inner_context.step_path == ("probe",)
    assert inner_context.session_id == inner.session_id
