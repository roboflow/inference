"""Operators inside active runs: derived pulses, lifecycle and attribution.

Sources are the scripted feeds of ``test_active_runtime``. Results are
compared per output group, so they never depend on how independent readers
interleave; where an order matters, the feed gates it.
"""

import threading
from typing import Any, List, Optional

import pytest
from pydantic import Field
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import ActiveRunError
from roboflow_workflows.execution_engine.v2.operators import (
    Operator,
    OperatorParams,
    OperatorPort,
    OperatorPulse,
)
from roboflow_workflows.execution_engine.v2.operators.alignment import Align
from roboflow_workflows.execution_engine.v2.operators.window import Window
from roboflow_workflows.execution_engine.v2.plan import PulseKey

from tests.unit_tests.execution_engine.v2.execution.blocks import (
    ContinueIf,
    Counter,
    Echo,
    Failing,
    Scale,
)
from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    CATALOGUE,
    UNTIL_STOP,
    WAIT,
    Collector,
    Events,
    Harness,
    Log,
    active,
    emit,
    group,
    source,
    step,
)

RECORDS: List[tuple] = []
"""Lifecycle calls of every ``Recorder`` instance, in order."""


class Recorder(Operator):
    """Emits every present arrival as its own pulse and records its lifecycle.

    Like a minimal third-party operator, it maintains no counters itself.
    """

    type = "test/recorder@v1"
    input_roles = ("input",)

    class Params(OperatorParams):
        fail_in: Optional[str] = Field(
            default=None, description="Lifecycle method that raises."
        )

    @classmethod
    def plan_ports(cls, name, params, inputs):
        return {"echo": OperatorPort(*inputs[0].kinds)}

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        self._raise_in("init")
        RECORDS.append((self.name, "init", self))

    def push(self, arrivals):
        self._raise_in("push")
        RECORDS.append((self.name, "push", [arrival.pulse for arrival in arrivals]))
        pulses = [
            OperatorPulse(ports={"echo": arrival.entry}, causes=(arrival.pulse,))
            for arrival in arrivals
            if not arrival.entry.is_effectively_filtered()
        ]
        return pulses

    def end_input(self, name):
        RECORDS.append((self.name, "end_input", name))
        return []

    def finish(self, reason):
        self._raise_in("finish")
        RECORDS.append((self.name, "finish", reason))
        return []

    def close(self):
        RECORDS.append((self.name, "close"))
        self._raise_in("close")

    def _raise_in(self, method: str) -> None:
        if self.params.fail_in == method:
            raise RuntimeError(f"{method} failed")


OPERATOR_CATALOGUE = Catalogue.merge(
    CATALOGUE, Catalogue([], operators=[Align, Window, Recorder])
)


@pytest.fixture(autouse=True)
def records():
    RECORDS.clear()
    yield RECORDS
    RECORDS.clear()


class OperatorHarness(Harness):
    """``Harness`` compiling against the catalogue that also has operators."""

    def __init__(self, definition: dict, feeds: dict, *, observer: Any = None) -> None:
        self.log = Log()
        self.started = threading.Event()
        self.plan = compile_workflow(definition, catalogue=OPERATOR_CATALOGUE)
        self.session = self.plan.create_session(
            resources={"feeds": feeds, "log": self.log, "started": self.started},
            observer=observer,
        )
        self.run = None


def with_operators(definition: dict, *operators: dict) -> dict:
    definition["operators"] = list(operators)

    return definition


def operator(kind: type, name: str, **fields: Any) -> dict:
    return {"type": kind.type, "name": name, **fields}


def pulse(domain: str, sequence: int, run: Any) -> PulseKey:
    return PulseKey(active_run_id=run.run_id, source=domain, sequence=sequence)


def values(result, field: str) -> Any:
    return result.rows()[0][field]


# Derived pulses ----------------------------------------------------------------


def test_alignment_pairs_different_rates_beside_an_unaligned_branch() -> None:
    feeds = {
        "a": [emit(value=float(index), pts=100 * index) for index in range(3)],
        "b": [emit(value=10.0 * index, pts=50 * index) for index in range(5)],
    }
    definition = with_operators(
        active(
            [source("a"), source("b")],
            [
                step(Scale, "double_pair", value="$operators.pair.b"),
                step(Scale, "double_a", value="$sources.a.value"),
            ],
            [
                group(
                    "P",
                    "$operators.pair.a",
                    a="$operators.pair.a",
                    b="$operators.pair.b",
                    doubled="$steps.double_pair.scaled",
                ),
                group("A", "$sources.a.value", doubled="$steps.double_a.scaled"),
            ],
        ),
        operator(
            Align,
            "pair",
            inputs={"a": "$sources.a.value", "b": "$sources.b.value"},
            clock="media",
        ),
    )
    harness = OperatorHarness(definition, feeds)
    collected = Collector()

    run = harness.start(collected.handlers("P", "A"))

    assert run.wait(WAIT)
    assert collected.rows("P") == [
        {"a": 0.0, "b": 0.0, "doubled": 0.0},
        {"a": 1.0, "b": 20.0, "doubled": 40.0},
        {"a": 2.0, "b": 40.0, "doubled": 80.0},
    ]
    assert collected.rows("A") == [{"doubled": 0.0}, {"doubled": 2.0}, {"doubled": 4.0}]
    results = collected.results["P"]
    assert [result.source for result in results] == ["pair"] * 3
    assert [result.pulse.sequence for result in results] == [0, 1, 2]
    assert results[1].causes == (pulse("a", 1, run), pulse("b", 2, run))
    assert collected.results["A"][0].causes == ()
    counters = run.operator_counters["pair"]
    assert (counters.emitted, counters.processed, counters.delivered) == (3, 3, 3)
    assert (counters.finished, counters.closed, counters.cancelled) == (True, True, 0)
    assert (counters.arrivals, counters.evicted) == (8, 5)


def test_admission_is_released_while_a_window_retains_its_rows() -> None:
    feeds = {"s": [emit(value=float(index), pts=index) for index in range(7)]}
    definition = with_operators(
        active(
            [source("s")],
            [],
            [group("W", "$operators.clip.x", x="$operators.clip.x")],
        ),
        operator(Window, "clip", size=3, collect={"x": "$sources.s.value"}),
    )
    harness = OperatorHarness(definition, feeds)
    collected = Collector()

    run = harness.start(collected.handlers("W"), admission_bound=1)

    assert run.wait(WAIT)
    assert collected.rows("W") == [{"x": [0.0, 1.0, 2.0]}, {"x": [3.0, 4.0, 5.0]}]
    window = collected.results["W"][1]
    assert window.outputs.layout["x"].axis_ids == ("operators.clip:t",)
    assert window.outputs.metadata["x"].temporal_at(()) is None
    assert window.causes == tuple(pulse("s", index, run) for index in (3, 4, 5))
    counters = run.operator_counters["clip"]
    assert (counters.peak_retained, counters.partial_dropped) == (3, 1)
    assert run.counters["s"].processed == 7


def test_eof_finishes_each_operator_once_and_propagates_after_final_pulses() -> None:
    feeds = {
        "a": [emit(value=float(index), pts=100 * index) for index in range(3)],
        "b": [emit(value=10.0 * index, pts=100 * index) for index in range(2)],
    }
    definition = with_operators(
        active(
            [source("a"), source("b")],
            [],
            [group("W", "$operators.clip.a", a="$operators.clip.a")],
        ),
        operator(
            Align,
            "pair",
            inputs={"a": "$sources.a.value", "b": "$sources.b.value"},
            clock="media",
            missing="partial",
        ),
        operator(
            Window, "clip", size=2, partial="emit", collect={"a": "$operators.pair.a"}
        ),
    )
    harness = OperatorHarness(definition, feeds)
    collected = Collector()

    run = harness.start(collected.handlers("W"))

    assert run.wait(WAIT)
    assert collected.rows("W") == [{"a": [0.0, 1.0]}, {"a": [2.0]}]
    assert collected.results["W"][1].causes == (pulse("pair", 2, run),)
    assert run.operator_counters["pair"].emitted == 3
    assert all(counters.finished for counters in run.operator_counters.values())


def test_stop_drains_admitted_pulses_then_applies_the_partial_policy(records) -> None:
    feeds = {"s": [emit(value=1.0), emit(value=2.0), UNTIL_STOP, emit(value=3.0)]}
    definition = with_operators(
        active(
            [source("s")],
            [],
            [
                group("S", "$sources.s.value", value="$sources.s.value"),
                group("W", "$operators.clip.x", x="$operators.clip.x"),
                group("R", "$operators.rec.echo", echo="$operators.rec.echo"),
            ],
        ),
        operator(
            Window, "clip", size=3, partial="emit", collect={"x": "$sources.s.value"}
        ),
        operator(Recorder, "rec", inputs={"value": "$sources.s.value"}),
    )
    harness = OperatorHarness(definition, feeds)
    collected = Collector()

    def on_source(result) -> None:
        collected.handler("S")(result)
        if len(collected.results["S"]) == 2:
            harness.session.stop()

    run = harness.start({"S": on_source, **collected.handlers("W", "R")})

    assert run.wait(WAIT)
    assert collected.rows("W") == [{"x": [1.0, 2.0]}]
    assert collected.rows("R") == [{"echo": 1.0}, {"echo": 2.0}]
    assert ("rec", "finish", "stop") in records
    assert run.counters["s"].unadmitted == 1
    assert run.state == "finished"


def test_operator_pulses_get_fresh_state_and_drive_shared_work_once() -> None:
    feeds = {"s": [emit(value=1.0), emit(value=-1.0), emit(value=2.0)]}
    definition = with_operators(
        active(
            [source("s")],
            [
                step(
                    ContinueIf,
                    "positive",
                    value="$operators.rec.echo",
                    next_steps=["$steps.count"],
                ),
                step(Counter, "count", value="$operators.rec.echo"),
            ],
            [
                group("One", "$operators.rec.echo", count="$steps.count.count"),
                group(
                    "Two",
                    "$operators.rec.echo",
                    echo="$operators.rec.echo",
                    count="$steps.count.count",
                ),
            ],
        ),
        operator(Recorder, "rec", inputs={"value": "$sources.s.value"}),
    )
    harness = OperatorHarness(definition, feeds)
    collected = Collector()

    run = harness.start(collected.handlers("One", "Two"))

    assert run.wait(WAIT)
    assert collected.rows("One") == [{"count": 1}, {"count": None}, {"count": 2}]
    assert collected.rows("Two") == [
        {"echo": 1.0, "count": 1},
        {"echo": -1.0, "count": None},
        {"echo": 2.0, "count": 2},
    ]
    assert len(harness.instance("count").calls) == 2


def test_gated_windows_skip_filtered_rows_and_link_their_source_pulses() -> None:
    feeds = {"s": [emit(value=value) for value in (1.0, -1.0, 2.0, -2.0, 3.0, -3.0)]}
    definition = with_operators(
        active(
            [source("s")],
            [
                step(
                    ContinueIf,
                    "positive",
                    value="$sources.s.value",
                    next_steps=["$steps.kept"],
                ),
                step(Echo, "kept", value="$sources.s.value"),
            ],
            [group("W", "$operators.clip.x", x="$operators.clip.x")],
        ),
        operator(
            Window, "clip", size=2, partial="emit", collect={"x": "$steps.kept.value"}
        ),
    )
    harness = OperatorHarness(definition, feeds)
    collected = Collector()

    run = harness.start(collected.handlers("W"))

    assert run.wait(WAIT)
    assert collected.rows("W") == [{"x": [1.0, 2.0]}, {"x": [3.0]}]
    assert [result.causes for result in collected.results["W"]] == [
        (pulse("s", 0, run), pulse("s", 2, run)),
        (pulse("s", 4, run),),
    ]
    assert run.operator_counters["clip"].dropped == 3


# Failures ----------------------------------------------------------------------


def test_exceeding_a_bound_fails_with_operator_attribution_and_no_partial_output(
    records,
) -> None:
    feeds = {
        "a": [emit(value=float(index), pts=index) for index in range(3)],
        "b": [UNTIL_STOP],
    }
    definition = with_operators(
        active(
            [source("a"), source("b")],
            [],
            [group("W", "$operators.clip.echo", echo="$operators.clip.echo")],
        ),
        operator(
            Align,
            "pair",
            inputs={"a": "$sources.a.value", "b": "$sources.b.value"},
            clock="media",
            max_pending=1,
        ),
        operator(Recorder, "rec", inputs={"value": "$operators.pair.a"}),
        operator(
            Window,
            "clip",
            size=5,
            partial="emit",
            collect={"echo": "$operators.rec.echo"},
        ),
    )
    events = Events()
    finished: List[tuple] = []
    events.on_operator_finished = lambda *, operator, error: finished.append(
        (operator, None if error is None else error.stage)
    )
    harness = OperatorHarness(definition, feeds, observer=events)
    collected = Collector()

    run = harness.start(collected.handlers("W"))

    with pytest.raises(ActiveRunError, match="above max_pending=1") as raised:
        run.wait(WAIT)
    error = raised.value
    assert (error.stage, error.operator, error.source, error.pulse) == (
        "operator",
        "pair",
        "a",
        1,
    )
    assert "pair" in str(error) and "input 'a'" in str(error)
    assert collected.results == {}
    assert [item for item in records if item[1] in ("finish", "close")] == [
        ("rec", "close")
    ]
    assert finished == [("pair", "operator"), ("rec", None), ("clip", None)]
    assert all(counters.closed for counters in run.operator_counters.values())
    assert not any(counters.finished for counters in run.operator_counters.values())
    assert harness.log.count("close", "a") == harness.log.count("close", "b") == 1


def test_a_step_failing_in_an_operator_pulse_names_step_operator_and_pulse() -> None:
    feeds = {"s": [emit(value=1.0), emit(value=-1.0), emit(value=2.0)]}
    definition = with_operators(
        active(
            [source("s")],
            [step(Failing, "check", value="$operators.rec.echo")],
            [group("R", "$operators.rec.echo", checked="$steps.check.value")],
        ),
        operator(Recorder, "rec", inputs={"value": "$sources.s.value"}),
    )
    harness = OperatorHarness(definition, feeds)
    collected = Collector()

    run = harness.start(collected.handlers("R"))

    with pytest.raises(ActiveRunError, match="negative -1.0") as raised:
        run.wait(WAIT)
    error = raised.value
    assert (error.stage, error.operator, error.pulse, error.step_path) == (
        "step",
        "rec",
        1,
        ("check",),
    )
    assert error.source is None
    assert collected.rows("R") == [{"checked": 1.0}]
    counters = run.operator_counters["rec"]
    assert (counters.emitted, counters.processed, counters.cancelled) == (2, 1, 1)
    assert run.counters["s"].cancelled >= 1


def test_a_handler_failing_on_an_operator_group_is_attributed_to_the_operator() -> None:
    feeds = {"s": [emit(value=1.0)]}
    definition = with_operators(
        active(
            [source("s")],
            [],
            [group("R", "$operators.rec.echo", echo="$operators.rec.echo")],
        ),
        operator(Recorder, "rec", inputs={"value": "$sources.s.value"}),
    )
    harness = OperatorHarness(definition, feeds)

    def broken(result) -> None:
        raise KeyError("handler bug")

    run = harness.start({"R": broken})

    with pytest.raises(ActiveRunError, match="handler bug") as raised:
        run.wait(WAIT)
    assert (raised.value.stage, raised.value.operator, raised.value.group) == (
        "handler",
        "rec",
        "R",
    )


def test_incompatible_clocks_fail_the_run_at_the_operator() -> None:
    feeds = {"a": [emit(value=1.0, pts=0)], "b": [emit(value=2.0, pts=0)]}
    definition = with_operators(
        active([source("a"), source("b")], [], []),
        operator(
            Align,
            "pair",
            inputs={"a": "$sources.a.value", "b": "$sources.b.value"},
            clock="capture-clock",
        ),
    )
    harness = OperatorHarness(definition, feeds)

    run = harness.start({})

    with pytest.raises(
        ActiveRunError, match="not the declared clock 'capture-clock'"
    ) as raised:
        run.wait(WAIT)
    assert (raised.value.stage, raised.value.operator) == ("operator", "pair")


@pytest.mark.parametrize("method", ["push", "finish", "close"])
def test_every_operator_is_closed_exactly_once_whatever_fails(records, method) -> None:
    feeds = {"s": [emit(value=1.0)]}
    definition = with_operators(
        active([source("s")], [], []),
        operator(Recorder, "first", inputs={"value": "$sources.s.value"}),
        operator(
            Recorder, "second", inputs={"value": "$sources.s.value"}, fail_in=method
        ),
    )
    harness = OperatorHarness(definition, feeds)

    run = harness.start({})

    with pytest.raises(ActiveRunError, match=f"{method}.* failed") as raised:
        run.wait(WAIT)
    assert (raised.value.stage, raised.value.operator) == ("operator", "second")
    closes = [item[0] for item in records if item[1] == "close"]
    assert sorted(closes) == ["first", "second"]
    assert harness.log.count("close", "s") == 1


def test_a_failing_operator_constructor_fails_start_and_closes_the_built_ones(
    records,
) -> None:
    definition = with_operators(
        active([source("s")], [], []),
        operator(Recorder, "first", inputs={"value": "$sources.s.value"}),
        operator(
            Recorder, "second", inputs={"value": "$sources.s.value"}, fail_in="init"
        ),
    )
    harness = OperatorHarness(definition, {"s": []})

    with pytest.raises(
        ActiveRunError, match="constructor of Recorder failed"
    ) as raised:
        harness.start({})

    assert (raised.value.stage, raised.value.operator) == ("start", "second")
    assert [item[:2] for item in records] == [("first", "init"), ("first", "close")]
    assert harness.log == []


def test_a_restart_constructs_fresh_operators_and_retains_nothing(records) -> None:
    feeds = {"s": [emit(value=float(index)) for index in range(4)]}
    definition = with_operators(
        active(
            [source("s")],
            [],
            [group("W", "$operators.clip.x", x="$operators.clip.x")],
        ),
        operator(Window, "clip", size=3, collect={"x": "$sources.s.value"}),
        operator(Recorder, "rec", inputs={"value": "$sources.s.value"}),
    )
    harness = OperatorHarness(definition, feeds)
    first, second = Collector(), Collector()

    assert harness.start(first.handlers("W")).wait(WAIT)
    feeds["s"] = [emit(value=10.0 + index) for index in range(3)]
    run = harness.start(second.handlers("W"))

    assert run.wait(WAIT)
    assert first.rows("W") == [{"x": [0.0, 1.0, 2.0]}]
    assert second.rows("W") == [{"x": [10.0, 11.0, 12.0]}]
    assert second.sequences("W") == [0]
    instances = [item[2] for item in records if item[1] == "init"]
    assert len(instances) == 2 and instances[0] is not instances[1]
    assert [item[1] for item in records if item[1] in ("init", "close")] == [
        "init",
        "close",
        "init",
        "close",
    ]


def test_operator_pulses_are_observed_like_source_pulses() -> None:
    feeds = {"s": [emit(value=1.0)]}
    definition = with_operators(
        active(
            [source("s")],
            [],
            [group("R", "$operators.rec.echo", echo="$operators.rec.echo")],
        ),
        operator(Recorder, "rec", inputs={"value": "$sources.s.value"}),
    )
    events = Events()
    harness = OperatorHarness(definition, feeds, observer=events)

    run = harness.start({"R": lambda result: None})

    assert run.wait(WAIT)
    pulses = [event for event in events.events if event[0].startswith("pulse")]
    assert pulses == [
        ("pulse_started", "s", 0),
        ("pulse_started", "rec", 0),
        ("pulse_finished", "rec", 0, True),
        ("pulse_finished", "s", 0, True),
    ]
    assert ("group_delivered", "R", 0) in events.events


def test_operator_counters_reach_the_host_while_sources_block() -> None:
    gate = threading.Event()
    feeds = {"s": [emit(value=1.0), gate, emit(value=2.0)]}
    definition = with_operators(
        active(
            [source("s")],
            [],
            [group("R", "$operators.rec.echo", echo="$operators.rec.echo")],
        ),
        operator(Recorder, "rec", inputs={"value": "$sources.s.value"}),
    )
    harness = OperatorHarness(definition, feeds)
    seen: List[int] = []

    def on_result(result) -> None:
        seen.append(harness.run.operator_counters["rec"].processed)
        gate.set()

    run = harness.start({"R": on_result})

    assert run.wait(WAIT)
    assert seen == [0, 1]
    assert run.operator_counters["rec"].processed == 2


def test_a_failing_reader_closes_operators_without_a_partial_emission(records) -> None:
    delivered = threading.Event()
    feeds = {"s": [emit(value=1.0), delivered, RuntimeError("camera unplugged")]}
    definition = with_operators(
        active(
            [source("s")],
            [],
            [
                group("W", "$operators.clip.x", x="$operators.clip.x"),
                group("R", "$operators.rec.echo", echo="$operators.rec.echo"),
            ],
        ),
        operator(
            Window, "clip", size=3, partial="emit", collect={"x": "$sources.s.value"}
        ),
        operator(Recorder, "rec", inputs={"value": "$sources.s.value"}),
    )
    harness = OperatorHarness(definition, feeds)
    collected = Collector()

    def on_echo(result) -> None:
        collected.handler("R")(result)
        delivered.set()  # only now may the reader fail

    run = harness.start({"R": on_echo, **collected.handlers("W")})

    with pytest.raises(ActiveRunError, match="camera unplugged") as raised:
        run.wait(WAIT)
    assert (raised.value.stage, raised.value.source) == ("read", "s")
    assert collected.rows("R") == [{"echo": 1.0}]
    assert "W" not in collected.results
    counters = run.operator_counters["clip"]
    assert (counters.finished, counters.closed, counters.emitted) == (False, True, 0)
    assert [item[1] for item in records if item[0] == "rec"] == [
        "init",
        "push",
        "close",
    ]


# Common accounting -------------------------------------------------------------


def accounted(counters) -> bool:
    return counters.emitted == counters.processed + counters.cancelled


def test_a_pair_built_inside_a_failing_push_was_never_emitted() -> None:
    # The follower at 110 decides the leader at 100 (pairing 99) inside the
    # same push that then exceeds max_pending=1: the pair never leaves push.
    leader_allowed, follower_allowed = threading.Event(), threading.Event()
    feeds = {
        "a": [leader_allowed, emit(value=100.0, pts=100), UNTIL_STOP],
        "b": [emit(value=99.0, pts=99), follower_allowed, emit(value=110.0, pts=110)],
    }
    definition = with_operators(
        active(
            [source("a"), source("b")],
            [],
            [
                group("A", "$sources.a.value", a="$sources.a.value"),
                group("B", "$sources.b.value", b="$sources.b.value"),
                group("P", "$operators.pair.a", a="$operators.pair.a"),
            ],
        ),
        operator(
            Align,
            "pair",
            inputs={"a": "$sources.a.value", "b": "$sources.b.value"},
            clock="media",
            tolerance_ms=10,
            max_pending=1,
        ),
    )
    harness = OperatorHarness(definition, feeds)
    paired: List[Any] = []

    run = harness.start(
        {
            "A": lambda result: follower_allowed.set(),
            "B": lambda result: leader_allowed.set(),
            "P": paired.append,
        },
        admission_bound=1,
    )

    with pytest.raises(ActiveRunError, match="above max_pending=1") as raised:
        run.wait(WAIT)
    assert (raised.value.stage, raised.value.operator) == ("operator", "pair")
    counters = run.operator_counters["pair"]
    assert (counters.arrivals, counters.emitted, counters.processed) == (3, 0, 0)
    assert counters.cancelled == 0 and counters.closed
    assert paired == []


def test_returned_pulses_after_a_failing_one_are_cancelled_once() -> None:
    feeds = {"s": [emit(value=-1.0)]}
    definition = with_operators(
        active(
            [source("s")],
            [step(Failing, "check", value="$operators.rec.echo")],
            [group("R", "$operators.rec.echo", checked="$steps.check.value")],
        ),
        # Two inputs from one domain: one push returns two pulses.
        operator(
            Recorder, "rec", inputs={"v": "$sources.s.value", "w": "$sources.s.value"}
        ),
    )
    events = Events()
    errors: List[Any] = []
    events.on_pulse_finished = lambda *, run_id, source, pulse, error: errors.append(
        (source, pulse.sequence, error)
    )
    harness = OperatorHarness(definition, feeds, observer=events)

    run = harness.start({"R": lambda result: None})

    with pytest.raises(ActiveRunError, match="negative -1.0") as raised:
        run.wait(WAIT)
    counters = run.operator_counters["rec"]
    assert (counters.arrivals, counters.emitted) == (2, 2)
    assert (counters.processed, counters.cancelled) == (0, 2)
    assert run.counters["s"].cancelled == 1
    # The operator pulse reports its own failure; the source pulse that fed it
    # reports the same error, and the second pulse never started.
    assert errors == [("rec", 0, raised.value), ("s", 0, raised.value)]
    assert raised.value.suppressed == ()


def test_a_custom_operator_gets_common_counters_without_maintaining_them() -> None:
    feeds = {"s": [emit(value=1.0), emit(), emit(value=3.0)]}
    definition = with_operators(
        active(
            [source("s")],
            [],
            [group("R", "$operators.rec.echo", echo="$operators.rec.echo")],
        ),
        operator(Recorder, "rec", inputs={"value": "$sources.s.value"}),
    )
    harness = OperatorHarness(definition, feeds)
    collected = Collector()

    run = harness.start(collected.handlers("R"))

    assert run.wait(WAIT)
    counters = run.operator_counters["rec"]
    assert (counters.arrivals, counters.emitted, counters.processed) == (3, 2, 2)
    assert (counters.delivered, counters.cancelled) == (2, 0)
    assert (counters.finished, counters.closed) == (True, True)
    assert (counters.filtered, counters.dropped) == (0, 0)  # policy counters untouched


@pytest.mark.parametrize("failing", ["on_pulse_started", "handler"])
def test_observer_and_handler_failures_on_operator_pulses_keep_accounting(
    failing,
) -> None:
    feeds = {"s": [emit(value=1.0), emit(value=2.0)]}
    definition = with_operators(
        active(
            [source("s")],
            [],
            [group("R", "$operators.rec.echo", echo="$operators.rec.echo")],
        ),
        operator(Recorder, "rec", inputs={"value": "$sources.s.value"}),
    )
    events = Events()
    if failing == "on_pulse_started":

        def started(*, run_id, source, pulse):
            if source == "rec":
                raise RuntimeError("observer bug")

        events.on_pulse_started = started
    harness = OperatorHarness(definition, feeds, observer=events)

    def handler(result) -> None:
        if failing == "handler":
            raise RuntimeError("handler bug")

    run = harness.start({"R": handler})

    with pytest.raises(ActiveRunError, match="bug") as raised:
        run.wait(WAIT)
    expected_stage = "observer" if failing == "on_pulse_started" else "handler"
    assert (raised.value.stage, raised.value.operator) == (expected_stage, "rec")
    assert raised.value.pulse == 0
    for counters in (run.operator_counters["rec"], run.counters["s"]):
        assert counters.processed == 0
    assert accounted(run.operator_counters["rec"])
    assert run.counters["s"].admitted == (
        run.counters["s"].processed + run.counters["s"].cancelled
    )
