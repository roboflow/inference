"""Phase and implementation declarations, result readiness and ``run_phases``.

Everything here runs without the engine: blocks are declared, called directly
and their graphs executed by the generic primitive.
"""

import gc
import inspect
import weakref
from concurrent.futures import CancelledError, Future
from typing import Any, List

import pytest
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    WorkOperation,
)
from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    NoExecutionContextError,
    use_execution_context,
)
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
    Selected,
    Selection,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DeclarationError,
)
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.phases import (
    PhaseFailure,
    PhaseGraph,
    PhaseSpec,
    phase,
    read_phase_graph,
    run_phases,
)
from roboflow_workflows.execution_engine.v2.readiness import resolve_futures


def done(value: Any) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


def failed(error: Exception) -> Future:
    future: Future = Future()
    future.set_exception(error)
    return future


class Holder:
    """A private intermediate that tests can watch with a weakref."""

    def __init__(self, value: float) -> None:
        self.value = value


class Diamond(Block):
    """base -> (left, right) -> total; every phase call is recorded."""

    type = "test/phases/diamond@v1"
    outputs = {"total": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)
        offset: float = 0.0

    def __init__(self) -> None:
        self.calls: List[str] = []

    @phase
    def base(self, *, value):
        self.calls.append("base")
        return Holder(value)

    @phase
    def left(self, *, base):
        self.calls.append("left")
        return base.value * 2

    @phase
    def right(self, *, base, offset):
        self.calls.append("right")
        return base.value + offset

    @phase
    def total(self, *, left, right):
        self.calls.append("total")
        return {"total": left + right}

    def run(self, *, value, offset):
        base = self.base(value=value)
        return self.total(
            left=self.left(base=base), right=self.right(base=base, offset=offset)
        )


def graph_of(block_class: type) -> PhaseGraph:
    graph = spec_of(block_class).implementations[0].phases
    assert graph is not None
    return graph


def declare(**body: Any) -> type:
    """Declare a one-field block class from a body; returns the class."""
    namespace = {
        "type": "test/phases/declared@v1",
        "outputs": {"y": Output(FLOAT_KIND)},
        "Params": type(
            "Params", (BlockParams,), {"__annotations__": {"x": Ref(FLOAT_KIND)}}
        ),
        "run": lambda self, *, x: {"y": x},
        **body,
    }
    return type("Declared", (Block,), namespace)


# Phase graphs ---------------------------------------------------------------


def test_diamond_graph_is_ordered_described_and_bound_by_signature() -> None:
    graph = graph_of(Diamond)

    assert [spec.name for spec in graph.phases] == ["base", "left", "right", "total"]
    assert graph.result == "total"
    assert graph.phases[2] == PhaseSpec(
        name="right", parameters=("base", "offset"), upstream=("base",)
    )
    assert graph.phases[2].external == ("offset",)
    assert graph.describe()["phases"][0] == {
        "name": "base",
        "parameters": ["value"],
        "upstream": [],
    }


def test_ordinary_block_is_its_own_default_implementation_with_its_phases() -> None:
    spec = spec_of(Diamond)

    (implementation,) = spec.implementations
    assert implementation.name == "default"
    assert implementation.implementation_class is Diamond
    assert implementation.requires == frozenset()
    assert implementation.resources == spec.resources == ()
    assert spec.describe()["implementations"][0]["phases"]["result"] == "total"


def test_unphased_block_has_no_graph() -> None:
    assert spec_of(declare()).implementations[0].phases is None


def test_phase_order_is_topological_and_otherwise_follows_declaration() -> None:
    class Late(Block):
        type = "test/phases/late@v1"
        outputs = {"y": Output(FLOAT_KIND)}

        class Params(BlockParams):
            x: Ref(FLOAT_KIND)

        @phase
        def joined(self, *, second, first):
            return {"y": first + second}

        @phase
        def second(self, *, x):
            return x

        @phase
        def first(self, *, x):
            return x

        def run(self, *, x):
            return self.joined(second=self.second(x=x), first=self.first(x=x))

    assert [spec.name for spec in graph_of(Late).phases] == [
        "second",
        "first",
        "joined",
    ]


def test_phase_must_be_defined_under_its_own_name() -> None:
    with pytest.raises(DeclarationError, match="phase 'a' is function '<lambda>'"):
        declare(a=phase(lambda self, *, x: {"y": x}))


def _phase(name: str, source: str):
    namespace: dict = {}
    exec(f"def {name}{source}", namespace)
    return phase(namespace[name])


@pytest.mark.parametrize(
    "phases, message",
    [
        (
            [_phase("y_value", "(self, *, z): return {'y': z}")],
            "parameter 'z' names neither a Params field nor a phase",
        ),
        (
            [_phase("y_value", "(self, *, x=1.0): return {'y': x}")],
            "parameter 'x' has a default",
        ),
        (
            [_phase("y_value", "(self, **kwargs): return {}")],
            "parameter 'kwargs' must be a named keyword parameter",
        ),
        (
            [_phase("y_value", "(self, *args): return {}")],
            "parameter 'args' must be a named keyword parameter",
        ),
        (
            [_phase("y_value", "(self, x, /): return {}")],
            "parameter 'x' must be a named keyword parameter",
        ),
        (
            [_phase("x", "(self): return 1")],
            "phase 'x' has the name of a Params field",
        ),
        (
            [
                _phase("a", "(self, *, b): return b"),
                _phase("b", "(self, *, a): return a"),
            ],
            "phases form a cycle: a -> b -> a",
        ),
        (
            [_phase("a", "(self, *, a): return a")],
            "phases form a cycle: a -> a",
        ),
        (
            [
                _phase("a", "(self, *, x): return x"),
                _phase("b", "(self, *, x): return x"),
            ],
            r"phases \['a', 'b'\] feed no other phase",
        ),
    ],
)
def test_invalid_phase_graphs_are_rejected_when_the_class_is_created(
    phases, message
) -> None:
    with pytest.raises(DeclarationError, match=message):
        declare(**{function.__name__: function for function in phases})


@pytest.mark.parametrize("name", ["metadata", "outputs", "execution_context", "run"])
def test_phase_shadowing_an_engine_attribute_is_rejected_first(name) -> None:
    with pytest.raises(DeclarationError, match=rf"phase\(s\) \['{name}'\] shadow"):
        declare(**{name: _phase(name, "(self, *, x): return {'y': x}")})


def test_overrides_keep_phase_status_and_are_validated_in_effective_form() -> None:
    class Halved(Diamond):
        type = "test/phases/halved@v1"

        def left(self, *, base):  # undecorated override stays a phase
            self.calls.append("half")
            return base.value / 2

    block = Halved()
    graph = graph_of(Halved)

    assert [spec.name for spec in graph.phases] == ["base", "left", "right", "total"]
    with pytest.raises(PhaseFailure) as caught:
        block.left(base=None)
    assert caught.value.phase == "left"

    with pytest.raises(DeclarationError, match="'other' names neither"):

        class Broken(Diamond):
            type = "test/phases/broken@v1"

            def left(self, *, base, other):
                return 0.0

    with pytest.raises(DeclarationError, match="overridden by staticmethod"):

        class Static(Diamond):
            type = "test/phases/static@v1"
            left = staticmethod(lambda base: 0.0)


def test_read_phase_graph_accepts_any_owner_and_external_names() -> None:
    class Plain:
        @phase
        def doubled(self, *, arrivals):
            return [item * 2 for item in arrivals]

        @phase
        def summary(self, *, doubled, limit):
            return sum(doubled[:limit])

    graph = read_phase_graph(Plain, external=("arrivals", "limit"), fail=ValueError)

    assert graph.result == "summary"
    assert run_phases(Plain(), graph, {"arrivals": [1, 2, 3], "limit": 2}) == 6
    assert read_phase_graph(object, external=(), fail=ValueError) is None
    with pytest.raises(ValueError, match="'limit' names neither"):
        read_phase_graph(Plain, external=("arrivals",), fail=ValueError)


# Phase calls and readiness ----------------------------------------------------


def test_phase_stays_directly_callable_with_its_signature() -> None:
    block = Diamond()

    assert str(inspect.signature(Diamond.right)) == "(self, *, base, offset)"
    assert block.run(value=3.0, offset=1.0) == {"total": 10.0}
    assert block.calls == ["base", "left", "right", "total"]


def test_phase_resolves_futures_before_the_caller_sees_them() -> None:
    class Deferred(Diamond):
        type = "test/phases/deferred@v1"

        def left(self, *, base):
            return done(base.value * 2)

        def total(self, *, left, right):
            return {"total": done(left + right)}

    assert Deferred().run(value=3.0, offset=1.0) == {"total": 10.0}


def test_phase_failure_names_the_phase_and_keeps_the_original_cause() -> None:
    class Failing(Diamond):
        type = "test/phases/failing@v1"

        def right(self, *, base, offset):
            raise ValueError("negative")

    with pytest.raises(PhaseFailure) as caught:
        Failing().run(value=1.0, offset=0.0)

    assert caught.value.phase == "right"
    assert str(caught.value) == "phase 'right' failed: ValueError: negative"
    assert isinstance(caught.value.__cause__, ValueError)


def test_failed_future_and_inner_phase_failure_are_attributed() -> None:
    class Outer:
        @phase
        def inner(self, *, x):
            return failed(KeyError("lost"))

        @phase
        def outer(self, *, x):
            return self.inner(x=x)

    with pytest.raises(PhaseFailure) as caught:
        Outer().outer(x=1)

    assert caught.value.phase == "inner"
    assert isinstance(caught.value.__cause__, KeyError)


async def _never_awaited() -> int:
    return 1


@pytest.mark.parametrize(
    "wrap",
    [
        lambda coroutine: coroutine,
        lambda coroutine: {"private": [coroutine]},
        lambda coroutine: done(coroutine),
        lambda coroutine: Batch.of([1.0, coroutine]),
    ],
    ids=["top-level", "nested", "future-result", "batch"],
)
def test_phase_rejects_and_closes_coroutines_anywhere_in_its_result(wrap) -> None:
    coroutine = _never_awaited()

    class Async:
        @phase
        def value(self):
            return wrap(coroutine)

    with pytest.raises(
        PhaseFailure, match="coroutine, which V2 does not await"
    ) as caught:
        Async().value()

    assert caught.value.phase == "value"
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


class _Pause:
    def __await__(self):
        yield "paused"


def test_started_coroutine_is_rejected_but_left_to_its_owner() -> None:
    cleaned: List[bool] = []

    async def caller_owned():
        try:
            await _Pause()
        finally:
            cleaned.append(True)

    coroutine = caller_owned()
    coroutine.send(None)

    class Borrowing:
        @phase
        def value(self):
            return {"borrowed": coroutine}

    with pytest.raises(PhaseFailure, match="coroutine, which V2 does not await"):
        Borrowing().value()

    # Not advanced, closed or cancelled: the caller still owns it.
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_SUSPENDED
    assert cleaned == []
    coroutine.close()
    assert cleaned == [True]


def test_async_phase_is_rejected_when_decorated() -> None:
    with pytest.raises(TypeError, match="phases are synchronous"):

        @phase
        async def value(self):
            return 1


def test_readiness_keeps_aliases_and_leaves_borrowed_containers_unchanged() -> None:
    payload = Holder(1.0)
    shared = [done(payload)]
    batch = Batch.of([done(2.0)])
    result = {
        "left": shared,
        "right": shared,
        "a": batch,
        "b": batch,
        "plain": [payload],
    }

    ready = resolve_futures(result)

    assert ready["left"] is ready["right"] and ready["left"] == [payload]
    assert ready["left"][0] is payload
    assert ready["a"] is ready["b"] and list(ready["a"]) == [2.0]
    assert ready["a"].indices == batch.indices
    assert ready["plain"] is result["plain"]
    assert isinstance(shared[0], Future) and isinstance(batch.content[0], Future)
    assert resolve_futures(result["plain"]) is result["plain"]


def test_ordinary_readiness_passes_coroutines_through() -> None:
    coroutine = _never_awaited()

    assert resolve_futures({"value": coroutine})["value"] is coroutine
    coroutine.close()


# run_phases -------------------------------------------------------------------


def test_run_phases_runs_each_phase_once_and_matches_run() -> None:
    block = Diamond()
    seen: List[str] = []

    result = run_phases(
        block, graph_of(Diamond), {"value": 3.0, "offset": 1.0}, on_phase=seen.append
    )

    assert result == block.run(value=3.0, offset=1.0) == {"total": 10.0}
    assert seen == ["base", "left", "right", "total"]
    assert block.calls == ["base", "left", "right", "total"] * 2


def test_run_phases_releases_intermediates_after_their_last_consumer() -> None:
    watched: List[weakref.ref] = []

    class Watching(Diamond):
        type = "test/phases/watching@v1"

        def base(self, *, value):
            holder = Holder(value)
            watched.append(weakref.ref(holder))
            return holder

        def total(self, *, left, right):
            gc.collect()
            return {"total": watched[-1]() is None}

    block = Watching()

    first = run_phases(block, graph_of(Watching), {"value": 1.0, "offset": 0.0})
    second = run_phases(block, graph_of(Watching), {"value": 2.0, "offset": 0.0})

    # base fed left and right only, so it was gone before the result phase.
    assert first == second == {"total": True}
    # Nothing about the calls was stored on the instance.
    assert list(vars(block)) == ["calls"]


def test_run_phases_clears_its_state_when_a_phase_fails() -> None:
    watched: List[weakref.ref] = []

    class LateFailure(Block):
        type = "test/phases/late_failure@v1"
        outputs = {"y": Output(FLOAT_KIND)}

        class Params(BlockParams):
            x: Ref(FLOAT_KIND)

        @phase
        def kept(self, *, x):
            holder = Holder(x)
            watched.append(weakref.ref(holder))
            return holder

        @phase
        def check(self, *, x):
            raise RuntimeError("late")

        @phase
        def joined(self, *, kept, check):
            return {"y": kept.value}

        def run(self, *, x):
            return self.joined(kept=self.kept(x=x), check=self.check(x=x))

    block = LateFailure()
    with pytest.raises(PhaseFailure) as caught:
        run_phases(block, graph_of(LateFailure), {"x": 1.0})
    gc.collect()

    # The traceback is still held, but the executor kept no intermediate.
    assert caught.value.phase == "check"
    assert watched[0]() is None


# Failures keep the failure, not the result (G3-1) ---------------------------------


def test_failed_future_in_a_returned_result_does_not_keep_the_result_alive() -> None:
    watched: List[weakref.ref] = []
    original = ValueError("model server went away")
    borrowed = [failed(original)]

    class Worker:
        @phase
        def result(self):
            private = Holder(1.0)
            watched.append(weakref.ref(private))
            return {"private": [private], "left": borrowed, "right": borrowed}

    with pytest.raises(PhaseFailure) as caught:
        Worker().result()
    gc.collect()

    # The exception is still held, the phase returned normally: no user frame
    # holds the result, and the framework holds none either.
    assert watched[0]() is None
    assert caught.value.__cause__ is original
    assert isinstance(borrowed[0], Future)


def test_values_held_by_the_raising_phase_itself_stay_with_its_frame() -> None:
    watched: List[weakref.ref] = []

    class Worker:
        @phase
        def result(self):
            private = Holder(1.0)
            watched.append(weakref.ref(private))
            raise ValueError(f"cannot use {private.value}")

    with pytest.raises(PhaseFailure) as caught:
        Worker().result()
    gc.collect()

    # Expected: the user's own frame is in the traceback, with its locals.
    assert watched[0]() is not None
    del caught
    gc.collect()
    assert watched[0]() is None


def test_run_phases_failure_keeps_neither_inputs_nor_results_and_recovers() -> None:
    watched: List[weakref.ref] = []

    class Remote(Block):
        type = "test/phases/remote@v1"
        outputs = {"y": Output(FLOAT_KIND)}

        class Params(BlockParams):
            x: Ref(FLOAT_KIND)

        @phase
        def prepared(self, *, x):
            holder = Holder(x)
            watched.append(weakref.ref(holder))
            return holder

        @phase
        def answered(self, *, prepared):
            reply = Holder(prepared.value)
            watched.append(weakref.ref(reply))
            if prepared.value < 0:
                return {"y": failed(ValueError("negative")), "reply": reply}
            return {"y": done(prepared.value), "reply": reply}

        @phase
        def output(self, *, answered):
            return {"y": answered["y"]}

        def run(self, *, x):
            return self.output(answered=self.answered(prepared=self.prepared(x=x)))

    block = Remote()
    with pytest.raises(PhaseFailure) as caught:
        run_phases(block, graph_of(Remote), {"x": -1.0})
    gc.collect()

    assert caught.value.phase == "answered"
    assert [reference() for reference in watched] == [None, None]
    assert run_phases(block, graph_of(Remote), {"x": 2.0}) == {"y": 2.0}
    assert block.run(x=3.0) == {"y": 3.0}


def test_failed_future_beside_awaitables_closes_only_unstarted_ones() -> None:
    unstarted = _never_awaited()

    async def caller_owned():
        await _Pause()

    started = caller_owned()
    started.send(None)
    result = {"waiting": [unstarted, started], "failed": failed(KeyError("lost"))}

    with pytest.raises(KeyError):
        resolve_futures(result, reject_awaitables=True)

    assert inspect.getcoroutinestate(unstarted) == inspect.CORO_CLOSED
    assert inspect.getcoroutinestate(started) == inspect.CORO_SUSPENDED
    started.close()


def test_cancelled_future_raises_cancelled_error() -> None:
    cancelled: Future = Future()
    cancelled.cancel()

    with pytest.raises(CancelledError):
        resolve_futures({"value": cancelled})


# Result wrappers (Selected / Selection) ----------------------------------------


def test_readiness_keeps_aliases_across_and_inside_result_wrappers() -> None:
    shared = [done(2.0)]
    same = Selected((1,))
    result = {
        "a": Selected((0,), shared),
        "b": Selected((0,), shared),
        "many": Selection([(0,), (1,)], values=[shared, shared]),
        "same": same,
        "members": Selection([(1,)]),
    }

    ready = resolve_futures(result)

    a, b, many = ready["a"], ready["b"], ready["many"]
    assert a.value is b.value is many.values[0] is many.values[1]
    assert a.value == [2.0] and (a.index, many.indices) == ((0,), ((0,), (1,)))
    a.value.append("seen")
    assert many.values[1] == [2.0, "seen"]
    # Borrowed storage and wrappers without futures are untouched.
    assert isinstance(shared[0], Future) and result["a"].value is shared
    assert ready["same"] is same and ready["members"] is result["members"]


class _Chooser:
    """A sink phase returning whatever wrapper ``make`` builds."""

    def __init__(self, make) -> None:
        self.make = make

    @phase
    def chosen(self):
        return {"result": self.make()}


def test_direct_phase_resolves_wrapper_payloads() -> None:
    ready = _Chooser(lambda: Selected((0,), done(3.0))).chosen()
    many = _Chooser(lambda: Selection([(0,)], values=[done(4.0)])).chosen()

    assert ready["result"].value == 3.0
    assert many["result"].values == (4.0,)


@pytest.mark.parametrize(
    "wrap",
    [
        lambda coroutine: Selected((0,), coroutine),
        lambda coroutine: Selected((0,), done([coroutine])),
        lambda coroutine: Selection([(0,)], values=[{"nested": coroutine}]),
    ],
    ids=["selected", "selected-future", "selection-nested"],
)
def test_phase_rejects_coroutines_inside_wrappers(wrap) -> None:
    unstarted = _never_awaited()

    with pytest.raises(PhaseFailure, match="coroutine") as caught:
        _Chooser(lambda: wrap(unstarted)).chosen()

    assert caught.value.phase == "chosen"
    assert inspect.getcoroutinestate(unstarted) == inspect.CORO_CLOSED


def test_started_coroutine_inside_a_wrapper_stays_with_its_owner() -> None:
    async def caller_owned():
        await _Pause()

    started = caller_owned()
    started.send(None)

    with pytest.raises(PhaseFailure, match="coroutine"):
        _Chooser(lambda: Selected((0,), started)).chosen()

    assert inspect.getcoroutinestate(started) == inspect.CORO_SUSPENDED
    started.close()


def test_failed_future_inside_a_wrapper_names_the_phase() -> None:
    original = ValueError("lost")

    with pytest.raises(PhaseFailure) as caught:
        _Chooser(lambda: Selection([(0,)], values=[failed(original)])).chosen()

    assert caught.value.phase == "chosen" and caught.value.__cause__ is original


def test_run_phases_reports_missing_arguments() -> None:
    with pytest.raises(ContractError, match=r"misses argument\(s\) \['offset'\]"):
        run_phases(Diamond(), graph_of(Diamond), {"value": 1.0})


# Implementations ----------------------------------------------------------------


class _SharedPhases(Implementation):
    """Unlisted helper base: needs no name and may hold part of a graph."""

    @phase
    def doubled(self, *, value):
        return value * 2


class Fast(_SharedPhases):
    name = "fast"
    requires = ("cpu", "fast")

    def __init__(self, *, accelerator, factor: float = 1.0) -> None:
        self.accelerator = accelerator
        self.factor = factor

    @phase
    def scaled(self, *, doubled):
        return {"scaled": doubled * self.factor}

    def run(self, *, value):
        return self.scaled(doubled=self.doubled(value=value))

    @classmethod
    def discover_work_operations(cls, params):
        return [WorkOperation.MODEL_INFERENCE]


class Portable(Implementation):
    name = "portable"
    requires = ("cpu",)

    def run(self, *, value):
        return {"scaled": value * 2}


class Scaled(Block):
    """Logical contract with two implementations."""

    type = "test/phases/scaled@v1"
    outputs = {"scaled": Output(FLOAT_KIND)}
    mutates = ()
    implementations = (Fast, Portable)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    @classmethod
    def discover_dependent_resources(cls, params):
        return []


def test_contract_lists_implementations_with_their_own_resources_and_graphs() -> None:
    spec = spec_of(Scaled)
    fast, portable = spec.implementations

    # The contract is never constructed: no resources of its own.
    assert spec.resources is None
    assert "resources" not in spec.describe()
    assert (fast.name, fast.implementation_class, fast.requires) == (
        "fast",
        Fast,
        frozenset({"cpu", "fast"}),
    )
    assert [(item.name, item.required) for item in fast.resources] == [
        ("accelerator", True),
        ("factor", False),
    ]
    assert [item.name for item in fast.phases.phases] == ["doubled", "scaled"]
    assert portable.resources == () and portable.phases is None
    assert spec.describe()["implementations"][1] == {
        "name": "portable",
        "class": f"{__name__}.Portable",
        "requires": ["cpu"],
        "resources": [],
        "phases": None,
    }
    assert Fast(accelerator=None).run(value=2.0) == Portable().run(value=2.0)


def test_workload_hooks_of_the_selected_implementation_fall_back_to_the_contract() -> (
    None
):
    spec = spec_of(Scaled)
    params = spec.validate_params({"value": "$inputs.v"})
    fast, portable = spec.implementations

    with_fast = spec.describe_workload(params, node_id="$steps.s", implementation=fast)
    with_portable = spec.describe_workload(
        params, node_id="$steps.s", implementation=portable
    )

    assert with_fast.operations.items == [WorkOperation.MODEL_INFERENCE]
    assert with_fast.operations.complete
    # Neither the contract nor Portable declares operations: still unknown.
    assert not with_portable.operations.complete
    assert isinstance(with_portable.dependencies, Discovery)
    assert with_portable.dependencies.complete and with_fast.dependencies.complete


def test_implementations_share_the_execution_context_reader() -> None:
    implementation = Portable()
    context = ExecutionContext(("s",), "test/phases/scaled@v1", "session")

    with pytest.raises(NoExecutionContextError):
        implementation.execution_context
    with use_execution_context(context):
        assert implementation.execution_context is context


def contract(*implementations: type, **body: Any) -> type:
    namespace = {
        "type": "test/phases/contract@v1",
        "outputs": {"scaled": Output(FLOAT_KIND)},
        "Params": type(
            "Params", (BlockParams,), {"__annotations__": {"value": Ref(FLOAT_KIND)}}
        ),
        "implementations": implementations,
        **body,
    }
    return type("Contract", (Block,), namespace)


def implementation(name: str = "impl", **body: Any) -> type:
    namespace = {"name": name, "run": lambda self, *, value: {"scaled": value}, **body}
    return type(f"Impl_{name}", (Implementation,), namespace)


@pytest.mark.parametrize(
    "build, message",
    [
        (
            lambda: contract(implementation(), run=lambda self, *, value: {}),
            "only the logical contract",
        ),
        (
            lambda: contract(implementation(), __init__=lambda self, *, model: None),
            "only the logical contract",
        ),
        (
            lambda: contract(
                implementation(), joined=_phase("joined", "(self, *, value): return {}")
            ),
            "only the logical contract",
        ),
        (
            lambda: contract(implementation(outputs={"other": Output(FLOAT_KIND)})),
            r"restates contract attribute\(s\) \['outputs'\]",
        ),
        (
            lambda: contract(implementation(Params=BlockParams, mutates=("value",))),
            r"restates contract attribute\(s\) \['Params', 'mutates'\]",
        ),
        (
            lambda: contract(implementation(), implementation()),
            r"implementations repeat name\(s\) \['impl'\]",
        ),
        (
            lambda: contract(implementation(name="has space")),
            "name must be letters",
        ),
        (
            lambda: contract(implementation(requires="cpu")),
            "requires must be a tuple",
        ),
        (
            lambda: contract(implementation(run=lambda self, *, other: {})),
            "run\\(\\) requires 'other', which is not a Params field",
        ),
        (lambda: contract(_SharedPhases), "name must be letters"),
        (lambda: contract(Diamond), "not an Implementation"),
        (
            lambda: contract(implementation(), implementations="x"),
            "implementations must be a tuple",
        ),
    ],
)
def test_invalid_contracts_and_implementations_are_rejected(build, message) -> None:
    with pytest.raises(DeclarationError, match=message):
        build()


def test_implementation_phase_names_are_checked_early() -> None:
    with pytest.raises(
        DeclarationError, match=r"Implementation class Bad: phase\(s\) \['requires'\]"
    ):

        class Bad(Implementation):
            @phase
            def requires(self, *, value):
                return value
