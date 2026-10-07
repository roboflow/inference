"""Public ``ManagedState`` semantics, run identically on memory and real Redis."""

import threading
from typing import Mapping, Optional, Tuple

import pytest
from roboflow_workflows.execution_engine.v2.context import (
    ExecutionContext,
    use_execution_context,
)
from roboflow_workflows.execution_engine.v2.data import SampleContext
from roboflow_workflows.execution_engine.v2.state import (
    MANAGED_STATE_RESOURCE,
    MISSING,
    InMemoryStateBackend,
    ManagedState,
    StateBackendError,
    StateOverflowError,
    StateScopeError,
    StateTypeError,
    StateValueError,
)
from roboflow_workflows.execution_engine.v2.state.codec import (
    INT64_MAX,
    INT64_MIN,
    decode_value,
    global_storage_key,
    machine_storage_key,
    source_storage_key,
)


@pytest.fixture(params=["memory", "redis"])
def make_state(request, namespace):
    backends = []

    def factory(namespace_override: Optional[str] = None) -> ManagedState:
        if request.param == "memory":
            backend = backends[0] if backends else InMemoryStateBackend()
        else:
            from roboflow_workflows.execution_engine.v2.state.redis import (
                RedisStateBackend,
            )

            server = request.getfixturevalue("redis_server")
            backend = RedisStateBackend(server.url)
        backends.append(backend)
        state = ManagedState(backend, namespace=namespace_override or namespace)

        return state

    yield factory

    for backend in backends:
        backend.close()


class _CallScope:
    """Engine side of a call, reduced to the source lookup state uses."""

    def __init__(self, samples: Mapping[Tuple[int, ...], Optional[str]]) -> None:
        self._samples = {
            index: None if source is None else SampleContext(source)
            for index, source in samples.items()
        }

    def sample_at(self, index):
        return self._samples[tuple(index)]


def _call(samples, *, batched=False):
    context = ExecutionContext(
        ("count",),
        "demo/count",
        "session-1",
        run_id="run-1",
        indices=tuple(samples),
        batched=batched,
        call_scope=_CallScope(samples),
    )

    return use_execution_context(context)


def test_missing_differs_from_null(make_state):
    scope = make_state().global_

    scope.set("null", None)

    assert scope.get("absent") is None
    assert scope.get("absent", MISSING) is MISSING
    assert scope.get("null", MISSING) is None
    assert scope.get("absent", default=5) == 5


def test_set_only_if_absent_and_delete(make_state):
    scope = make_state().global_

    assert scope.set("k", 1, only_if_absent=True) is True
    assert scope.set("k", 2, only_if_absent=True) is False
    assert scope.set("k", 3) is True
    assert scope.get("k") == 3
    assert scope.delete("k") is True
    assert scope.delete("k") is False


def test_values_are_copied(make_state):
    scope = make_state().global_
    value = {"items": [1, 2]}

    scope.set("k", value)
    value["items"].append(3)
    read = scope.get("k")
    read["items"].append(4)

    assert scope.get("k") == {"items": [1, 2]}


def test_incr_counts_from_zero_and_accepts_negative_amounts(make_state):
    scope = make_state().global_

    assert scope.incr("c") == 1
    assert scope.incr("c", 10) == 11
    assert scope.incr("c", -20) == -9
    assert scope.get("c") == -9


@pytest.mark.parametrize("stored", [None, 1.0, "1", True, [1], {"a": 1}])
def test_incr_on_non_integer_raises_type_error_and_keeps_value(make_state, stored):
    scope = make_state().global_
    scope.set("k", stored)

    with pytest.raises(StateTypeError):
        scope.incr("k")

    assert scope.get("k", MISSING) == stored
    assert type(scope.get("k")) is type(stored)


@pytest.mark.parametrize(
    "start, amount", [(INT64_MAX, 1), (INT64_MIN, -1), (0, INT64_MIN), (-1, INT64_MIN)]
)
def test_incr_overflow_raises_and_keeps_value(make_state, start, amount):
    scope = make_state().global_
    scope.set("k", start)

    if start + amount < INT64_MIN or start + amount > INT64_MAX:
        with pytest.raises(StateOverflowError):
            scope.incr("k", amount)
        assert scope.get("k") == start
    else:
        assert scope.incr("k", amount) == start + amount


@pytest.mark.parametrize("amount", [True, 1.5, INT64_MAX + 1])
def test_invalid_amount_is_rejected_before_the_backend(make_state, amount):
    scope = make_state().global_

    with pytest.raises(StateValueError):
        scope.incr("k", amount)

    assert scope.get("k", MISSING) is MISSING


def test_compare_and_set_with_missing(make_state):
    scope = make_state().global_

    assert scope.compare_and_set("k", MISSING, "idle") is True
    assert scope.compare_and_set("k", MISSING, "other") is False
    assert scope.compare_and_set("k", "busy", "alert") is False
    assert scope.compare_and_set("k", "idle", "alert") is True
    assert scope.get("k") == "alert"
    assert scope.compare_and_set("k", "alert", MISSING) is True
    assert scope.get("k", MISSING) is MISSING
    assert scope.compare_and_set("k", None, 1) is False


def test_compare_and_set_equality_is_canonical(make_state):
    scope = make_state().global_
    scope.set("k", {"b": [1, 2], "a": None})

    assert scope.compare_and_set("k", 1, 2) is False
    assert scope.compare_and_set("k", {"a": None, "b": [1, 2.0]}, 2) is False
    assert scope.compare_and_set("k", {"a": None, "b": [1, 2]}, 1) is True
    assert scope.compare_and_set("k", 1.0, 2) is False
    assert scope.compare_and_set("k", True, 2) is False
    assert scope.compare_and_set("k", 1, 2) is True


def test_rejected_values_change_nothing(make_state):
    scope = make_state().global_
    scope.set("k", 1)

    with pytest.raises(StateValueError):
        scope.set("k", (1, 2))
    with pytest.raises(StateValueError):
        scope.compare_and_set("k", 1, float("nan"))
    with pytest.raises(StateValueError):
        scope.get("")

    assert scope.get("k") == 1


def test_global_and_source_scopes_are_isolated(make_state):
    state = make_state()

    state.global_.set("k", "global")
    state.for_source("cam_a").set("k", "a")
    state.for_source("cam_b").incr("k")

    assert state.global_.get("k") == "global"
    assert state.for_source("cam_a").get("k") == "a"
    assert state.for_source("cam_b").get("k") == 1
    assert state.for_source("cam_c").get("k", MISSING) is MISSING


def test_namespaces_are_isolated_with_unusual_names(make_state, namespace):
    first = make_state(f"{namespace}:s:x")
    second = make_state(f"{namespace}")

    first.global_.set("k", 1)
    second.for_source("x").set("g:k", 2)
    second.global_.set("s:x:g:k", 3)

    assert first.global_.get("k") == 1
    assert second.for_source("x").get("g:k") == 2
    assert second.global_.get("s:x:g:k") == 3
    assert second.global_.get("k", MISSING) is MISSING


def test_same_backend_and_namespace_share_state(make_state, namespace):
    writer = make_state(namespace)
    reader = make_state(namespace)

    writer.for_source("cam_a").incr("seen", 3)

    assert reader.for_source("cam_a").get("seen") == 3


def _stored(state, key, source_id=None):
    # Raw backend text: proves a value is really stored, not computed by a view.
    if source_id is None:
        storage_key = global_storage_key(state.namespace, key)
    else:
        storage_key = source_storage_key(state.namespace, source_id, key)

    stored = state.backend.get(storage_key)

    return stored


def test_declared_defaults_are_stored_initial_values(make_state):
    raw = make_state()
    state = raw.with_defaults(
        global_={"total": 100, "mode": "idle"}, source={"enabled": True, "n": 0}
    )

    assert _stored(raw, "total") == "100"
    assert _stored(raw, "n", "cam_a") is None
    assert state.for_source("cam_a").get("enabled") is True
    assert _stored(raw, "n", "cam_a") == "0"
    assert _stored(raw, "n", "cam_b") is None
    assert raw.global_.get("mode") == "idle"
    assert state.global_.set("mode", "x", only_if_absent=True) is False
    assert state.global_.compare_and_set("mode", MISSING, "x") is False
    assert state.global_.compare_and_set("mode", "idle", "alert") is True
    assert state.global_.incr("total") == 101
    assert state.for_source("cam_b").incr("n", 2) == 2

    with pytest.raises(StateTypeError):
        state.for_source("cam_a").incr("enabled")
    assert state.for_source("cam_a").get("enabled") is True

    assert state.global_.delete("total") is True
    assert state.global_.get("total", MISSING) is MISSING
    assert state.global_.incr("total") == 1
    assert state.global_.delete("total") is True
    assert state.global_.set("total", 5, only_if_absent=True) is True


def test_views_of_one_service_initialize_each_key_once(make_state):
    raw = make_state()
    first = raw.with_defaults(global_={"n": 7}, source={"m": 1})
    first.global_.delete("n")
    first.for_source("cam_a").delete("m")

    second = raw.with_defaults(global_={"n": 7, "k": 2}, source={"m": 1})
    third = first.with_defaults(global_={"k": 99}, source={"z": 5})

    assert second.global_.get("n", MISSING) is MISSING
    assert second.global_.get("k") == 2
    assert third.global_.get("k") == 2
    assert second.for_source("cam_a").get("m", MISSING) is MISSING
    assert third.for_source("cam_a").get("m", MISSING) is MISSING
    assert third.for_source("cam_a").get("z") == 5
    assert third.for_source("cam_b").get("m") == 1
    assert raw.for_source("cam_c").get("m", MISSING) is MISSING


def test_new_service_on_the_same_namespace_seeds_only_absent_defaults(make_state):
    first = make_state().with_defaults(global_={"n": 7, "k": 1})
    first.global_.delete("n")
    first.global_.set("k", 3)

    second = make_state().with_defaults(global_={"n": 7, "k": 1})

    assert second.global_.get("n") == 7
    assert second.global_.get("k") == 3


def test_source_defaults_seed_through_the_current_call(make_state):
    state = make_state().with_defaults(source={"seen": 10})

    with _call({(0,): "cam_a"}):
        first = state.source.incr("seen")
    with _call({(0,): "cam_b"}, batched=True):
        other = state.at((0,)).incr("seen", 5)

    assert (first, other) == (11, 15)


def test_concurrent_first_use_seeds_before_any_operation(make_state):
    raw = make_state()
    state = raw.with_defaults(source={"n": 100})
    barrier = threading.Barrier(8)
    results = []

    def work():
        barrier.wait()
        results.append(state.for_source("cam_a").incr("n"))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == list(range(101, 109))


def test_initialize_once_shares_bookkeeping_with_defaults(make_state):
    raw = make_state()
    view = raw.with_defaults(global_={"n": 1})
    record = machine_storage_key(raw.namespace, "cam_a", "door")

    view.initialize_once(record, {"state": "idle", "version": 0})
    raw.backend.delete(record)
    raw.initialize_once(record, {"state": "idle", "version": 0})
    raw.initialize_once(global_storage_key(raw.namespace, "n"), 5)

    assert raw.backend.get(record) is None
    assert raw.global_.get("n") == 1

    restarted = make_state()
    restarted.initialize_once(record, {"state": "idle", "version": 0})

    assert decode_value(restarted.backend.get(record)) == {
        "state": "idle",
        "version": 0,
    }


_INT64_EDGES = [
    0,
    2**53 - 1,
    2**53,
    2**53 + 1,
    -(2**53) - 1,
    9007199254740993,
    INT64_MAX,
    INT64_MAX - 1,
    INT64_MIN,
    INT64_MIN + 1,
]


# Overflowing pairs are covered by test_incr_overflow_raises_and_keeps_value.
_EXACT_INCREMENTS = [
    (start, amount)
    for start in _INT64_EDGES
    for amount in [0, 1, -1, 2**53 + 1]
    if INT64_MIN <= start + amount <= INT64_MAX
]


@pytest.mark.parametrize("start, amount", _EXACT_INCREMENTS)
@pytest.mark.parametrize("initialized", ["set", "global-default", "source-default"])
def test_incr_returns_exactly_the_stored_value(make_state, start, amount, initialized):
    expected = start + amount
    raw = make_state()
    if initialized == "set":
        raw.global_.set("n", start)
        scope, source_id = raw.global_, None
    elif initialized == "global-default":
        scope, source_id = raw.with_defaults(global_={"n": start}).global_, None
    else:
        scope = raw.with_defaults(source={"n": start}).for_source("cam_a")
        source_id = "cam_a"

    returned = scope.incr("n", amount)

    assert type(returned) is int
    assert returned == expected
    assert scope.get("n") == expected
    assert _stored(raw, "n", source_id) == str(expected)


@pytest.mark.parametrize("amount", [INT64_MAX, INT64_MIN, 2**53 + 1, -(2**53) - 1])
def test_incr_of_a_missing_key_returns_exactly_the_amount(make_state, amount):
    state = make_state()

    returned = state.global_.incr("n", amount)

    assert returned == amount
    assert state.global_.get("n") == amount


def test_invalid_declared_defaults_are_rejected(make_state):
    state = make_state()

    with pytest.raises(StateValueError):
        state.with_defaults(global_={"k": (1,)})
    with pytest.raises(StateValueError):
        state.with_defaults(source={"": 1})


def test_concurrent_increments_are_exact(make_state):
    state = make_state()
    workers, per_worker = 8, 250
    barrier = threading.Barrier(workers)
    errors = []

    def work():
        try:
            barrier.wait()
            for _ in range(per_worker):
                state.global_.incr("total")
                state.for_source("cam_a").incr("total", 2)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert state.global_.get("total") == workers * per_worker
    assert state.for_source("cam_a").get("total") == 2 * workers * per_worker


def test_concurrent_compare_and_set_has_one_winner(make_state):
    state = make_state()
    state.for_source("cam_a").set("machine", "idle")
    contenders = 16
    barrier = threading.Barrier(contenders)
    wins = []

    def contend(name):
        barrier.wait()
        if state.for_source("cam_a").compare_and_set("machine", "idle", name):
            wins.append(name)

    threads = [
        threading.Thread(target=contend, args=(f"alert-{n}",))
        for n in range(contenders)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(wins) == 1
    assert state.for_source("cam_a").get("machine") == wins[0]


def test_source_scope_resolves_the_current_call(make_state):
    state = make_state()
    source = state.source

    with _call({(0,): "cam_a"}):
        source.incr("seen")
    with _call({(1,): "cam_b"}):
        source.incr("seen", 5)
    with _call({(2,): "cam_a"}):
        same_source = source.incr("seen")

    assert same_source == 2
    assert state.for_source("cam_b").get("seen") == 5
    assert state.global_.get("seen", MISSING) is MISSING


@pytest.mark.parametrize(
    "samples",
    [
        {(0,): "cam_a", (1,): "cam_a"},
        {(0,): "cam_a", (1,): "cam_b"},
        {(0,): "cam_a"},
        {},
    ],
    ids=["same-source", "mixed", "one-member", "empty"],
)
def test_source_scope_rejects_every_batch_delivering_call(make_state, samples):
    state = make_state()

    with _call(samples, batched=True):
        with pytest.raises(StateScopeError, match="batch-delivering"):
            state.source.incr("k")
        with pytest.raises(StateScopeError):
            state.source.source_id

    assert state.for_source("cam_a").get("k", MISSING) is MISSING


def test_source_scope_rejects_missing_sources_and_contexts(make_state):
    state = make_state()

    with pytest.raises(StateScopeError, match="only inside"):
        state.source.get("k")
    with _call({(0,): None}):
        with pytest.raises(StateScopeError, match="no single source"):
            state.source.incr("k")
    with _call({}):
        with pytest.raises(StateScopeError, match="constructor"):
            state.source.incr("k")


def test_at_selects_a_member_of_a_batch_call(make_state):
    state = make_state()
    samples = {(0, 2): "cam_a", (1, 5): "cam_b", (1, 6): None}

    with _call(samples, batched=True):
        state.at((0, 2)).incr("seen")
        state.at([1, 5]).incr("seen", 2)
        with pytest.raises(StateScopeError, match="names no index"):
            state.at((0, 0))
        with pytest.raises(StateScopeError, match="no single source"):
            state.at((1, 6))
    with _call({(3,): "cam_a"}):
        state.at((3,)).incr("seen")

    assert state.for_source("cam_a").get("seen") == 2
    assert state.for_source("cam_b").get("seen") == 2
    with pytest.raises(StateScopeError, match="only inside"):
        state.at((0, 2))


def test_call_without_engine_sources_has_no_source_scope():
    state = ManagedState()
    context = ExecutionContext(("count",), "demo/count", "s", indices=((0,),))

    with use_execution_context(context):
        with pytest.raises(StateScopeError, match="no single source"):
            state.source.get("k")


def test_default_state_is_private_and_needs_no_singleton():
    first, second = ManagedState(), ManagedState()

    first.global_.set("k", 1)

    assert second.global_.get("k", MISSING) is MISSING
    assert first.namespace != second.namespace
    assert MANAGED_STATE_RESOURCE == "managed_state"


def test_close_only_closes_an_owned_backend():
    shared = InMemoryStateBackend()
    borrowed = ManagedState(shared, namespace="n")
    borrowed_view = borrowed.with_defaults(global_={"k": 1})
    owned = ManagedState()
    owned_view = owned.with_defaults(global_={"k": 1})

    borrowed_view.close()
    borrowed.close()
    owned_view.close()
    owned_view.global_.incr("k")

    assert borrowed.global_.get("k") == 1
    assert owned.global_.get("k") == 2

    owned.close()

    with pytest.raises(StateBackendError, match="closed"):
        owned_view.global_.get("k")


def test_invalid_namespace_is_rejected():
    with pytest.raises(StateValueError):
        ManagedState(namespace="")
    with pytest.raises(StateValueError):
        ManagedState().for_source("")
