"""``ActiveRun.apply_update`` with a reset candidate: new processing, same sources.

A ``Hold`` step blocks a pulse, or a feed waits on an event, so each reset
lands at a boundary the test controls. Every wait is bounded by ``WAIT``;
no test sleeps to order threads.
"""

import gc
import threading
import time
import weakref
from typing import Any, Dict, List, Tuple

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    EventEmissionError,
    GraphUpdateError,
    IncompatibleUpdateError,
    UpdateConflictError,
    UpdateTimeoutError,
)
from roboflow_workflows.execution_engine.v2.kinds import STRING_KIND
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions

from tests.unit_tests.execution_engine.v2.reactions.test_machines_runtime import (
    CATALOGUE as MACHINE_CATALOGUE,
)
from tests.unit_tests.execution_engine.v2.reactions.test_machines_runtime import (
    Journal,
    inspection,
)
from tests.unit_tests.execution_engine.v2.updates.active_blocks import (
    END,
    WAIT,
    Collector,
    Count,
    Echo,
    Feed,
    Hold,
    Log,
    Pair,
    Record,
    Scale,
    Tracker,
    Updater,
    active,
    emit,
    group,
    handler,
    step,
)


class BrokenPair(Pair):
    """Its constructor raises."""

    type = "test/reset_broken_pair@v1"

    def __init__(self, **arguments: Any) -> None:
        raise RuntimeError("no pair today")


class MemoryPair(Pair):
    """Its constructor fails to allocate."""

    type = "test/reset_memory_pair@v1"

    def __init__(self, **arguments: Any) -> None:
        raise MemoryError("cannot allocate the window")


class SlowPair(Pair):
    """Its ``close`` waits for ``SlowPair.release``, then raises if ``failing``."""

    type = "test/reset_slow_pair@v1"
    entered = threading.Event()
    release = threading.Event()
    failing = False

    def close(self) -> None:
        super().close()
        SlowPair.entered.set()
        assert SlowPair.release.wait(WAIT)
        if SlowPair.failing:
            raise RuntimeError("close failed")


# (what, time.monotonic()) of every Stamped and StampedPair construction.
BUILT: List[Tuple[str, float]] = []


class Stamped(Count):
    """``Count`` that stamps its construction into ``BUILT``."""

    type = "test/reset_stamped@v1"

    def __init__(self, *, log: Log) -> None:
        super().__init__(log=log)
        BUILT.append(("step", time.monotonic()))


class StampedPair(Pair):
    """``Pair`` that stamps its construction into ``BUILT``."""

    type = "test/reset_stamped_pair@v1"

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        BUILT.append(("operator", time.monotonic()))


CATALOGUE = Catalogue(
    [Count, Scale, Hold, Tracker, Echo, Record, Stamped],
    sources=[Feed],
    operators=[Pair, BrokenPair, MemoryPair, SlowPair, StampedPair],
    kinds=[STRING_KIND],
)
PIPELINED = PipelineOptions(max_in_flight=2)
MODES = pytest.mark.parametrize(
    "pipeline", [None, PIPELINED], ids=["serial", "pipelined"]
)
COUNTED = [
    step(Hold, "hold", value="$sources.a.value"),
    step(Count, "count", value="$steps.hold.value"),
]
COUNTS = group("counts", "$sources.a.value", count="$steps.count.count")
PAIRS = group("pairs", "$operators.pair.first", first="$operators.pair.first")


def paired(operator: type = Pair, source: str = "a", **sections: Any) -> dict:
    """``Hold`` and ``Count`` on ``a``, a pair window on ``source`` and its group."""
    pair = {
        "type": operator.type,
        "name": "pair",
        "inputs": {"value": f"$sources.{source}.value"},
    }
    sources = sorted({"a", source})
    definition = active(sources, COUNTED, [COUNTS, PAIRS], operators=[pair], **sections)

    return definition


def counted_only() -> dict:
    return active(["a"], COUNTED, [COUNTS])


class Scene:
    """A session over scripted feeds, with one started run."""

    def __init__(self, definition: dict, feeds: Dict[str, list]) -> None:
        self.log = Log()
        self.holds = {
            name: {"entered": threading.Event(), "release": threading.Event()}
            for name in ("hold", "record")
        }
        self.session = compile_workflow(definition, catalogue=CATALOGUE).create_session(
            resources={"feeds": feeds, "log": self.log, "holds": self.holds}
        )
        self.collected = Collector()
        self.run: Any = None

    def start(self, pipeline: Any, *groups: str) -> Any:
        options = {"pipeline": pipeline} if pipeline is not None else {}
        self.run = self.session.start(
            handlers=self.collected.handlers(*groups), admission_bound=1, **options
        )

        return self.run

    def let_through(self) -> None:
        """Wait for the first held call, then let every call through."""
        assert self.holds["hold"]["entered"].wait(WAIT)
        self.holds["hold"]["release"].set()

    def reset(self, definition: dict) -> Any:
        return self.session.prepare_update(
            compile_workflow(definition, catalogue=CATALOGUE), reset=True
        )

    def by_sequence(self, name: str) -> List[Any]:
        return sorted(
            self.collected.results.get(name, ()), key=lambda r: r.pulse.sequence
        )


@pytest.fixture(autouse=True)
def slow_pair() -> Any:
    SlowPair.entered.clear()
    SlowPair.release.clear()
    SlowPair.failing = False
    yield
    SlowPair.release.set()


@MODES
def test_reset_replaces_operators_and_steps_and_drops_the_partial_window(
    pipeline,
) -> None:
    resume = threading.Event()
    feed = [emit(1), emit(2), emit(3), resume, emit(4), emit(5), emit(6), emit(7), END]
    scene = Scene(paired(), {"a": feed})
    run = scene.start(pipeline, "counts", "pairs")
    scene.let_through()
    scene.collected.wait_for("counts", 3)
    old_pair = run._executor.operators["pair"].instance
    assert len(old_pair.buffer) == 1

    receipt = run.apply_update(scene.reset(paired()))
    resume.set()
    assert run.wait(WAIT)

    assert receipt.reset and receipt.processing_version == 1
    assert receipt.frontiers == {"a": 3, "pair": 1}
    assert receipt.cleanup.wait(WAIT) and receipt.cleanup.state == "done"
    assert receipt.called_at <= receipt.cut_at <= receipt.drained_at
    assert receipt.drained_at <= receipt.resumed_at
    assert receipt.admission_pause_seconds == receipt.resumed_at - receipt.cut_at
    assert receipt.drain_after_cut_seconds == receipt.drained_at - receipt.cut_at
    assert receipt.call_to_resume_seconds == receipt.resumed_at - receipt.called_at
    # The legacy durations count from the reservation, after the run's build.
    assert receipt.run_build_seconds > 0
    assert receipt.paused_seconds == pytest.approx(
        receipt.call_to_resume_seconds - receipt.run_build_seconds, abs=1e-9
    )
    assert receipt.admission_pause_seconds <= receipt.paused_seconds
    assert old_pair.lifecycle == [("close",)]  # never finished: the pair is dropped
    new_pair = run._executor.operators["pair"].instance
    assert new_pair is not old_pair
    assert new_pair.lifecycle == [
        ("end_input", "value"),
        ("finish", "eof", 0),
        ("close",),
    ]
    counts = scene.by_sequence("counts")
    assert [r.pulse.sequence for r in counts] == list(range(7))
    assert [(r.processing_version, r.rows()[0]["count"]) for r in counts] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ]
    pairs = scene.by_sequence("pairs")
    assert [r.pulse.sequence for r in pairs] == [0, 1, 2]
    assert [[c.sequence for c in r.causes] for r in pairs] == [[0, 1], [3, 4], [5, 6]]
    assert [r.processing_version for r in pairs] == [0, 1, 1]
    assert [e[0] for e in scene.log if e[1:2] == ("a",)].count("open") == 1


@MODES
def test_a_removed_and_reintroduced_operator_numbers_on(pipeline) -> None:
    first, second = threading.Event(), threading.Event()
    feed = [emit(1), emit(2), first, emit(3), emit(4), second, emit(5), emit(6), END]
    scene = Scene(paired(), {"a": feed})
    run = scene.start(pipeline, "counts", "pairs")
    scene.let_through()
    scene.collected.wait_for("counts", 2)

    removed = run.apply_update(scene.reset(counted_only()))
    first.set()
    scene.collected.wait_for("counts", 4)
    assert removed.cleanup.wait(WAIT)
    back = run.apply_update(
        scene.reset(paired()), handlers={"pairs": scene.collected.handler("pairs")}
    )
    second.set()
    assert run.wait(WAIT)

    assert back.frontiers == {"a": 4, "pair": 1}
    pairs = scene.by_sequence("pairs")
    assert [r.pulse.sequence for r in pairs] == [0, 1]
    assert [[c.sequence for c in r.causes] for r in pairs] == [[0, 1], [4, 5]]
    assert [r.processing_version for r in pairs] == [0, 2]


@MODES
def test_three_resets_leave_no_old_processing_alive(pipeline) -> None:
    gates = [threading.Event() for _ in range(3)]
    feed = [emit(1), gates[0], emit(2), gates[1], emit(3), gates[2], emit(4), END]
    scene = Scene(paired(), {"a": feed})
    run = scene.start(pipeline, "counts", "pairs")
    scene.let_through()
    kept: List[Any] = []
    old: List[Any] = []
    for position, gate in enumerate(gates):
        scene.collected.wait_for("counts", position + 1)
        old.append(weakref.ref(run._executor.operators["pair"].instance))
        old.append(weakref.ref(scene.session.generation.instances[("count",)]))
        candidate = scene.reset(paired())
        receipt = run.apply_update(candidate)
        assert receipt.cleanup.wait(WAIT) and receipt.cleanup.state == "done"
        kept.append((candidate, receipt))
        gate.set()
    assert run.wait(WAIT)
    gc.collect()

    assert [ref() for ref in old] == [None] * len(old)
    assert scene.session.processing_version == 3


@MODES
def test_a_source_that_ended_before_the_cut_ends_the_new_operators(pipeline) -> None:
    resume = threading.Event()  # source a cannot end before the cut
    feeds = {"a": [emit(1), emit(2), resume, END], "b": [emit(10), END]}
    scene = Scene(paired(source="b"), feeds)
    run = scene.start(pipeline, "counts")
    assert scene.holds["hold"]["entered"].wait(WAIT)
    scene.log.wait_for(lambda: ("read", "b", None) in scene.log)
    old_pair = run._executor.operators["pair"].instance
    updater = Updater(run, scene.reset(paired(source="b")))
    scene.holds["hold"]["release"].set()
    receipt = updater.result()
    resume.set()
    assert run.wait(WAIT)

    assert receipt.processing_version == 1
    assert old_pair.lifecycle[-1] == ("close",)
    new_pair = run._executor.operators["pair"].instance
    assert new_pair.lifecycle == [
        ("end_input", "value"),
        ("finish", "eof", 0),
        ("close",),
    ]
    assert run.counters["b"].admitted == 1


@MODES
def test_a_source_ending_during_the_pause_ends_the_new_operators(pipeline) -> None:
    hold_b = threading.Event()
    feeds = {"a": [emit(1), emit(2), END], "b": [emit(10), hold_b, END]}
    scene = Scene(paired(source="b"), feeds)
    run = scene.start(pipeline, "counts")
    assert scene.holds["hold"]["entered"].wait(WAIT)
    updater = Updater(run, scene.reset(paired(source="b")))
    assert not updater.finished.wait(0.05)
    hold_b.set()
    scene.log.wait_for(lambda: ("read", "b", None) in scene.log)
    old_pair = run._executor.operators["pair"].instance
    scene.holds["hold"]["release"].set()
    updater.result()
    assert run.wait(WAIT)

    assert old_pair.lifecycle == [("close",)]
    new_pair = run._executor.operators["pair"].instance
    assert new_pair.lifecycle == [
        ("end_input", "value"),
        ("finish", "eof", 0),
        ("close",),
    ]


@MODES
def test_failed_operator_construction_keeps_the_old_processing(pipeline) -> None:
    resume = threading.Event()
    scene = Scene(paired(), {"a": [emit(1), emit(2), resume, emit(3), emit(4), END]})
    run = scene.start(pipeline, "counts", "pairs")
    scene.let_through()
    scene.collected.wait_for("counts", 2)
    candidate = scene.reset(paired(BrokenPair))

    with pytest.raises(GraphUpdateError, match="no pair today"):
        run.apply_update(candidate)
    resume.set()
    assert run.wait(WAIT)

    assert candidate.state == "prepared"
    candidate.discard()
    assert [r.processing_version for r in scene.by_sequence("counts")] == [0] * 4
    assert len(scene.by_sequence("pairs")) == 2


@MODES
def test_a_drain_timeout_rejects_the_reset_and_retires_nothing(pipeline) -> None:
    scene = Scene(paired(), {"a": [emit(1), emit(2), END]})
    run = scene.start(pipeline, "counts", "pairs")
    assert scene.holds["hold"]["entered"].wait(WAIT)
    old_pair = run._executor.operators["pair"].instance
    candidate = scene.reset(paired())

    with pytest.raises(UpdateTimeoutError):
        run.apply_update(candidate, timeout=0.05)
    scene.holds["hold"]["release"].set()
    assert run.wait(WAIT)

    assert candidate.state == "prepared"
    assert run._executor.operators["pair"].instance is old_pair
    assert old_pair.lifecycle == [
        ("end_input", "value"),
        ("finish", "eof", 0),
        ("close",),
    ]
    assert scene.session.processing_version == 0
    assert [r.rows()[0]["count"] for r in scene.by_sequence("counts")] == [1, 2]


@MODES
def test_a_stop_during_the_drain_rejects_the_reset(pipeline) -> None:
    scene = Scene(paired(), {"a": [emit(1), emit(2), emit(3), END]})
    run = scene.start(pipeline, "counts")
    assert scene.holds["hold"]["entered"].wait(WAIT)
    updater = Updater(run, scene.reset(paired()))
    assert not updater.finished.wait(0.05)
    run.stop()
    scene.holds["hold"]["release"].set()

    with pytest.raises(UpdateConflictError, match="stopping"):
        updater.result()
    assert run.wait(WAIT)
    assert scene.session.processing_version == 0
    assert updater.update.state == "prepared"


def test_a_stale_reset_candidate_is_discarded_before_anything_pauses() -> None:
    resume = threading.Event()
    scene = Scene(paired(), {"a": [emit(1), resume, emit(2), END]})
    run = scene.start(None, "counts")
    scene.let_through()
    scene.collected.wait_for("counts", 1)
    first, second = scene.reset(paired()), None
    run.apply_update(first).cleanup.wait(WAIT)
    second = scene.session.prepare_update(
        compile_workflow(paired(), catalogue=CATALOGUE), reset=True
    )
    stale = first

    with pytest.raises(UpdateConflictError):
        run.apply_update(stale)
    run.apply_update(second)
    resume.set()
    assert run.wait(WAIT)
    assert scene.session.processing_version == 2


@MODES
def test_a_pending_cleanup_refuses_only_another_reset(pipeline) -> None:
    resume, last = threading.Event(), threading.Event()
    feed = [emit(1), emit(2), resume, emit(3), emit(4), last, END]
    scene = Scene(paired(SlowPair), {"a": feed})
    run = scene.start(pipeline, "counts", "pairs")
    scene.let_through()
    scene.collected.wait_for("counts", 2)

    receipt = run.apply_update(scene.reset(paired(SlowPair)))
    assert SlowPair.entered.wait(WAIT)
    assert receipt.cleanup.state == "pending" and not receipt.cleanup.wait(0)
    resume.set()
    scene.collected.wait_for("counts", 4)  # the new graph runs meanwhile
    blocked = scene.reset(paired(SlowPair))
    with pytest.raises(UpdateConflictError, match="still retires"):
        run.apply_update(blocked)
    SlowPair.release.set()
    assert receipt.cleanup.wait(WAIT)
    last.set()
    assert run.wait(WAIT)

    assert (receipt.cleanup.state, receipt.cleanup.errors) == ("done", ())
    assert blocked.state == "prepared"
    assert [r.processing_version for r in scene.by_sequence("counts")] == [0, 0, 1, 1]


def test_a_failing_cleanup_is_reported_and_the_reset_stands() -> None:
    SlowPair.failing = True
    SlowPair.release.set()
    resume = threading.Event()  # the run cannot end before the cut
    scene = Scene(paired(SlowPair), {"a": [emit(1), emit(2), resume, END]})
    run = scene.start(None, "counts")
    assert scene.holds["hold"]["entered"].wait(WAIT)
    updater = Updater(run, scene.reset(counted_only()))
    scene.holds["hold"]["release"].set()
    receipt = updater.result()
    resume.set()
    assert run.wait(WAIT)

    assert receipt.cleanup.wait(WAIT) and receipt.cleanup.state == "failed"
    assert receipt.cleanup.errors == (
        "closing operator 'pair': RuntimeError: close failed",
    )
    assert receipt.graph_version == 1 and scene.session.processing_version == 1
    assert receipt.cleanup_failures == receipt.cleanup.errors
    assert receipt.describe()["cleanup"] == {
        "state": "failed",
        "errors": list(receipt.cleanup.errors),
    }


def test_a_signal_reaching_the_replaced_reactions_is_rejected() -> None:
    reacting = paired()
    reacting["signals"] = [{"name": "poke", "fields": {}}]
    resume = threading.Event()  # the run cannot end before the cut
    scene = Scene(reacting, {"a": [emit(1), emit(2), resume, END]})
    run = scene.start(None, "counts")
    assert scene.holds["hold"]["entered"].wait(WAIT)
    old_reactions = run._reactions
    updater = Updater(run, scene.reset(reacting))
    scene.holds["hold"]["release"].set()
    updater.result()

    with pytest.raises(EventEmissionError, match="processing reset replaced"):
        old_reactions.signal("poke", {}, sample=None, temporal=None)
    assert run._reactions is not old_reactions
    resume.set()
    assert run.wait(WAIT)


@pytest.mark.parametrize("operator", [BrokenPair, MemoryPair], ids=["raises", "memory"])
def test_a_failed_allocation_before_the_cut_keeps_the_old_processing(
    operator,
) -> None:
    resume = threading.Event()
    scene = Scene(paired(), {"a": [emit(1), emit(2), resume, emit(3), emit(4), END]})
    run = scene.start(PIPELINED, "counts", "pairs")
    scene.let_through()
    scene.collected.wait_for("counts", 2)

    with pytest.raises(GraphUpdateError, match="the run keeps its graph"):
        run.apply_update(scene.reset(paired(operator)))
    resume.set()
    assert run.wait(WAIT)

    assert scene.session.processing_version == 0
    assert len(scene.by_sequence("counts")) == 4
    assert len(scene.by_sequence("pairs")) == 2


def reacting(definition: dict) -> dict:
    """``definition`` plus a ``ping`` signal whose async handler records its note."""
    definition = dict(definition)
    definition["signals"] = [{"name": "ping", "fields": {"note": ["string"]}}]
    definition["handlers"] = [handler("audit", "$signals.ping")]
    definition["outputs"] = [
        *definition["outputs"],
        group("audits", "$handlers.audit", out="$handlers.audit.out"),
    ]

    return definition


def notes(scene: Scene) -> List[Any]:
    return [item[1] for item in scene.log if item[0] == "record"]


def test_a_reset_drains_queued_handler_work_under_the_old_processing() -> None:
    resume = threading.Event()
    scene = Scene(reacting(paired()), {"a": [emit(1), resume, emit(2), END]})
    run = scene.start(None, "counts", "audits")
    scene.let_through()
    scene.collected.wait_for("counts", 1)
    run.signal("ping", note="first")
    assert scene.holds["record"]["entered"].wait(WAIT)
    run.signal("ping", note="second")  # queued behind the held handler run
    replaced = run._reactions

    updater = Updater(run, scene.reset(reacting(paired())))
    assert not updater.finished.wait(0.05), "the reset did not wait for handlers"
    scene.holds["record"]["release"].set()
    updater.result()
    run.signal("ping", note="third")
    resume.set()
    assert run.wait(WAIT)

    assert run._reactions is not replaced
    assert notes(scene) == ["first", "second", "third"]
    audits = scene.collected.results["audits"]
    assert [result.processing_version for result in audits] == [0, 0, 1]


def test_a_reset_drains_a_handler_cascade_started_before_the_cut() -> None:
    # async decide -> machine setter -> decided -> async audit, as in
    # reactions/test_cascades; each handler holds until the test lets it go.
    journal = Journal()
    decide_held = journal.hold("pick", 0.9)
    audit_held = journal.hold("audit", "approved")
    gate = threading.Event()
    definition = inspection(decide="async", audit="async")
    session = compile_workflow(definition, catalogue=MACHINE_CATALOGUE).create_session(
        resources={"journal": journal, "feeds": {"cam_a": [0.9, gate, 0.3]}}
    )
    audited = []
    journal.on_record = lambda label, value: (
        audited.append((value, session.processing_version))
        if label == "audit"
        else None
    )
    run = session.start()
    assert journal.started("pick", 0.9).wait(WAIT)
    reset = session.prepare_update(
        compile_workflow(definition, catalogue=MACHINE_CATALOGUE), reset=True
    )

    updater = Updater(run, reset)
    assert not updater.finished.wait(0.05), "the reset did not wait for decide"
    decide_held.set()  # its setter emits ``decided`` during the drain
    assert journal.started("audit", "approved").wait(WAIT)
    assert not updater.finished.wait(0.05), "the reset did not wait for the cascade"
    audit_held.set()
    receipt = updater.result()
    gate.set()
    assert run.wait(WAIT)

    assert receipt.reset and receipt.processing_version == 1
    assert receipt.cleanup.wait(WAIT) and receipt.cleanup.state == "done"
    # The cascade finished under the old processing; the new machine started
    # idle, so the next frame went through a whole review again.
    assert audited == [("approved", 0), ("rejected", 1)]
    assert run.machine_state("inspection_state", source_id="cam_a") == ("rejected", 2)


def test_a_signal_stalled_in_its_payload_check_is_rejected_after_the_reset(
    monkeypatch,
) -> None:
    # The feed waits on ``resume``, so the run cannot end before the cut.
    resume = threading.Event()
    scene = Scene(reacting(paired()), {"a": [emit(1), resume, emit(2), END]})
    scene.holds["record"]["release"].set()
    run = scene.start(None, "counts", "audits")
    assert scene.holds["hold"]["entered"].wait(WAIT)
    stalled_event = run._reactions._reactions.signals["ping"].event
    checking, proceed = threading.Event(), threading.Event()
    check_payload = type(stalled_event).check_payload

    def stall(event: Any, name: str, fields: Any) -> None:
        if event is stalled_event:
            checking.set()
            assert proceed.wait(WAIT)
        check_payload(event, name, fields)

    monkeypatch.setattr(type(stalled_event), "check_payload", stall)
    outcome: Dict[str, BaseException] = {}

    def send() -> None:
        try:
            run.signal("ping", note="stalled")
        except BaseException as raised:
            outcome["error"] = raised

    sender = threading.Thread(target=send, daemon=True)
    sender.start()
    assert checking.wait(WAIT)
    updater = Updater(run, scene.reset(reacting(paired())))
    scene.holds["hold"]["release"].set()
    updater.result()  # not admitted yet, so the drain does not wait for it
    proceed.set()
    sender.join(WAIT)
    resume.set()
    assert run.wait(WAIT)

    assert isinstance(outcome.get("error"), EventEmissionError)
    assert "processing reset replaced" in str(outcome["error"])
    assert notes(scene) == []


def controlled() -> dict:
    steps = [
        step(Hold, "hold", value="$sources.a.value"),
        step(Tracker, "tracker", value="$steps.hold.value"),
    ]
    groups = [group("ticks", "$sources.a.value", ticks="$steps.tracker.ticks")]
    controls = {
        "analysis": {
            "type": "enable",
            "steps": ["$steps.tracker"],
            "state": "reset_on_enable",
            "suspends_effects": True,
        }
    }

    return active(["a"], steps, groups, controls=controls)


@MODES
def test_control_writes_land_wholly_before_or_after_the_reset(pipeline) -> None:
    resume = threading.Event()
    scene = Scene(controlled(), {"a": [emit(1), emit(2), resume, emit(3), END]})
    panel = scene.session.controls
    run = scene.start(pipeline, "ticks")
    assert scene.holds["hold"]["entered"].wait(WAIT)
    old_tracker = scene.session.instances[("tracker",)]
    updater = Updater(run, scene.reset(controlled()))
    assert not updater.finished.wait(0.05)

    panel.update(analysis=False)  # during the drain: before the commit
    scene.holds["hold"]["release"].set()
    updater.result()
    after_commit = (panel.version, dict(panel.current.enabled), panel.current.epochs)
    panel.update(analysis=True)  # after the commit: resets the new tracker
    resume.set()
    assert run.wait(WAIT)

    assert scene.session.controls is panel
    assert after_commit == (2, {"analysis": False}, {"analysis": 0})
    new_tracker = scene.session.instances[("tracker",)]
    assert new_tracker is not old_tracker
    assert (old_tracker.resets, new_tracker.resets) == (0, 1)


def test_a_live_run_whose_sources_all_ended_cannot_reset_until_it_finished() -> None:
    scene = Scene(paired(), {"a": [emit(1), END]})
    run = scene.start(None, "counts")
    assert scene.holds["hold"]["entered"].wait(WAIT)
    scene.log.wait_for(lambda: ("read", "a", None) in scene.log)
    plan = compile_workflow(paired(), catalogue=CATALOGUE)

    blocked = scene.session.assess_update(plan)
    with pytest.raises(IncompatibleUpdateError, match="sources_ended"):
        scene.session.prepare_update(plan, reset=True)
    scene.holds["hold"]["release"].set()
    assert run.wait(WAIT)
    receipt = scene.session.update(plan, reset=True)

    assert [blocker.reason for blocker in blocked.blockers] == ["sources_ended"]
    assert scene.session.assess_update(plan).blockers == ()
    assert receipt.processing_version == 1


def test_a_recording_run_rules_a_reset_out_before_preparation() -> None:
    scene = Scene(paired(), {"a": [emit(1), END]})
    run = scene.start(None, "counts")
    assert scene.holds["hold"]["entered"].wait(WAIT)
    plan = compile_workflow(paired(), catalogue=CATALOGUE)
    # A recording run, without writing files: only the run's capture is read.
    run._capture, capture = object(), run._capture

    reasons = [blocker.reason for blocker in scene.session.assess_update(plan).blockers]
    with pytest.raises(IncompatibleUpdateError, match="recording_run"):
        scene.session.prepare_update(plan, reset=True)
    run._capture = capture
    scene.holds["hold"]["release"].set()
    assert run.wait(WAIT)

    assert reasons[:1] == ["recording_run"]


# Contract fixes: ownership, publication and retirement bounds -----------------


@MODES
def test_reset_constructors_all_run_before_the_cut(pipeline) -> None:
    pair = {
        "type": StampedPair.type,
        "name": "pair",
        "inputs": {"value": "$sources.a.value"},
    }
    steps = [
        step(Hold, "hold", value="$sources.a.value"),
        step(Stamped, "count", value="$steps.hold.value"),
    ]
    stamped = active(["a"], steps, [COUNTS], operators=[pair])
    resume = threading.Event()
    scene = Scene(stamped, {"a": [emit(1), resume, emit(2), END]})
    run = scene.start(pipeline, "counts")
    scene.let_through()
    scene.collected.wait_for("counts", 1)
    BUILT.clear()

    receipt = run.apply_update(scene.reset(stamped))
    resume.set()
    assert run.wait(WAIT)

    assert sorted(what for what, _ in BUILT) == ["operator", "step"]
    assert all(stamp < receipt.cut_at for _, stamp in BUILT)
    assert receipt.cut_at <= receipt.resumed_at


def test_a_handler_cannot_lose_the_value_it_shares_with_a_source() -> None:
    definition = active(
        ["a"],
        [step(Echo, "echo", value="$sources.a.value")],
        [group("values", "$sources.a.value", value="$steps.echo.out")],
        signals=[{"name": "ping", "fields": {"note": ["string"]}}],
        handlers=[handler("audit", "$signals.ping")],
    )
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    log = Log()
    session = plan.create_session(
        resources={"feeds": {"a": []}, "log": log, "holds": {}}
    )
    audit = session.handler_sessions[("audit",)]
    assert audit.resources[("record",)]["log"].value is log

    assessment = session.assess_update(plan, resources={"log": Log()})
    with pytest.raises(IncompatibleUpdateError, match="source_resource_overridden"):
        session.prepare_update(plan, reset=True, resources={"log": Log()})
    holds_only = session.assess_update(plan, resources={"holds": {}})

    assert [(b.reason, b.name) for b in assessment.blockers] == [
        ("source_resource_overridden", "provided:log")
    ]
    assert assessment.reset is None
    assert session.handler_sessions[("audit",)] is audit
    assert holds_only.blockers == () and holds_only.reset is not None


def test_a_handler_cannot_lose_a_source_value_it_holds_under_another_key() -> None:
    definition = active(
        ["a"],
        [step(Echo, "echo", value="$sources.a.value")],
        [group("values", "$sources.a.value", value="$steps.echo.out")],
        signals=[{"name": "ping", "fields": {"note": ["string"]}}],
        handlers=[handler("audit", "$signals.ping")],
    )
    catalogue = Catalogue.merge(
        Catalogue([Echo, Record], namespace="demo", kinds=[STRING_KIND]),
        Catalogue(sources=[Feed]),
    )
    plan = compile_workflow(definition, catalogue=catalogue)
    log = Log()
    session = plan.create_session(
        resources={"feeds": {"a": []}, "log": log, "demo.log": log, "holds": {}}
    )
    audit = session.handler_sessions[("audit",)]
    assert audit.resources[("record",)]["log"].source == "provided:demo.log"

    assessment = session.assess_update(plan, resources={"demo.log": Log()})
    kept = session.assess_update(plan, resources={"demo.log": log})

    assert [(b.reason, b.name) for b in assessment.blockers] == [
        ("source_resource_overridden", "provided:demo.log")
    ]
    assert kept.blockers == () and kept.reset is not None


@MODES
def test_signals_reach_new_reactions_only_after_their_publication(
    pipeline, monkeypatch
) -> None:
    resume = threading.Event()
    scene = Scene(reacting(paired()), {"a": [emit(1), resume, emit(2), END]})
    scene.holds["record"]["release"].set()
    run = scene.start(pipeline, "counts", "audits")
    scene.let_through()
    scene.collected.wait_for("counts", 1)
    old_reactions = run._reactions
    inside, finish = threading.Event(), threading.Event()
    restart_domains = run._driver.restart_domains

    def held_install() -> None:
        # One assignment of the install; the new reactions are reachable.
        restart_domains()
        inside.set()
        assert finish.wait(WAIT)

    monkeypatch.setattr(run._driver, "restart_domains", held_install)
    updater = Updater(run, scene.reset(reacting(paired())))
    assert inside.wait(WAIT)
    exposed = run._reactions is not old_reactions
    versions = (scene.session.graph_version, scene.session.processing_version)
    with pytest.raises(EventEmissionError, match="graph update of the run"):
        run.signal("ping", note="during")
    finish.set()
    updater.result()
    run.signal("ping", note="after")
    audits = scene.collected.wait_for("audits", 1)
    resume.set()
    assert run.wait(WAIT)

    assert exposed and versions == (0, 0)
    assert [(r.graph_version, r.processing_version) for r in audits] == [(1, 1)]
    assert notes(scene) == ["after"]


def test_a_reset_without_a_cleanup_thread_is_refused_before_the_cut(
    monkeypatch,
) -> None:
    resume = threading.Event()
    scene = Scene(paired(SlowPair), {"a": [emit(1), resume, emit(2), END]})
    run = scene.start(None, "counts")
    scene.let_through()
    scene.collected.wait_for("counts", 1)
    old_pair = run._executor.operators["pair"].instance
    candidate = scene.reset(counted_only())
    start = threading.Thread.start

    def without_cleanup_threads(thread: threading.Thread) -> None:
        if thread.name.startswith("workflows-v2-cleanup-"):
            raise RuntimeError("can't start new thread")
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", without_cleanup_threads)
    with pytest.raises(GraphUpdateError, match="no thread starts"):
        run.apply_update(candidate)
    monkeypatch.undo()
    refused = (
        scene.session.processing_version,
        candidate.state,
        list(old_pair.lifecycle),
    )
    receipt = run.apply_update(candidate)  # returns while the old close blocks
    assert SlowPair.entered.wait(WAIT)
    pending = receipt.cleanup.state
    SlowPair.release.set()
    resume.set()
    assert run.wait(WAIT)

    assert refused == (0, "prepared", [])
    assert pending == "pending"
    assert receipt.cleanup.wait(WAIT) and receipt.cleanup.state == "done"
    assert old_pair.lifecycle == [("close",)]
    assert receipt.cleanup.finished_at >= receipt.resumed_at


def test_one_pending_retirement_per_session_also_across_runs() -> None:
    resume = threading.Event()
    scene = Scene(paired(SlowPair), {"a": [emit(1), resume, emit(2), END]})
    first = scene.start(None, "counts")
    scene.let_through()
    scene.collected.wait_for("counts", 1)
    pending = first.apply_update(scene.reset(counted_only())).cleanup
    assert SlowPair.entered.wait(WAIT)
    resume.set()
    assert first.wait(WAIT) and pending.state == "pending"

    # The idle session, then a new run of it: no other reset meanwhile.
    blocked = scene.reset(counted_only())
    with pytest.raises(UpdateConflictError, match="still retires"):
        scene.session.apply_update(blocked)
    resume_second = threading.Event()
    feeds = scene.session.source_resources["a"]["feeds"].value
    feeds["a"] = [emit(3), resume_second, emit(4), END]
    second = scene.session.start(
        handlers=scene.collected.handlers("counts"), admission_bound=1
    )
    scene.log.wait_for(lambda: ("read", "a", {"value": 3.0}) in scene.log)
    with pytest.raises(UpdateConflictError, match="still retires"):
        second.apply_update(blocked)
    still_prepared = blocked.state
    same = compile_workflow(counted_only(), catalogue=CATALOGUE)
    preserved = second.apply_update(scene.session.prepare_update(same))
    SlowPair.release.set()
    assert pending.wait(WAIT)
    later = second.apply_update(scene.reset(counted_only()))
    resume_second.set()
    assert second.wait(WAIT)

    assert still_prepared == "prepared"
    assert (preserved.reset, preserved.graph_version) == (False, 2)
    assert later.processing_version == 2
    assert later.cleanup.wait(WAIT) and later.cleanup.state == "done"


def test_a_host_can_only_watch_a_hung_cleanup() -> None:
    resume = threading.Event()
    scene = Scene(paired(SlowPair), {"a": [emit(1), resume, emit(2), END]})
    run = scene.start(None, "counts")
    scene.let_through()
    scene.collected.wait_for("counts", 1)
    receipt = run.apply_update(scene.reset(counted_only()))
    assert SlowPair.entered.wait(WAIT)
    resume.set()
    assert run.wait(WAIT) and not receipt.cleanup.wait(0.05)

    for reflex in ("close", "cancel", "release", "reserve"):
        with pytest.raises(AttributeError):
            getattr(receipt.cleanup, reflex)()
    with pytest.raises(AttributeError):
        receipt.cleanup.state = "done"
    blocked = scene.reset(counted_only())
    with pytest.raises(UpdateConflictError, match="still retires"):
        scene.session.apply_update(blocked)
    SlowPair.release.set()

    assert receipt.cleanup.wait(WAIT) and receipt.cleanup.state == "done"
    assert receipt.cleanup.finished_at is not None
    assert scene.session.apply_update(blocked).processing_version == 2


def test_a_reset_names_the_started_handlers_that_do_not_run_again() -> None:
    booting = paired()
    boot = handler("boot", "$system.events.started")
    boot["bindings"] = {"value": "boot"}
    booting["handlers"] = [boot]
    resume = threading.Event()
    scene = Scene(booting, {"a": [emit(1), resume, emit(2), END]})
    scene.holds["record"]["release"].set()
    run = scene.start(None, "counts")
    scene.let_through()
    scene.log.wait_for(lambda: ("record", "boot") in scene.log)
    old_record = scene.session.handler_sessions[("boot",)].instances[("record",)]
    plan = compile_workflow(booting, catalogue=CATALOGUE)

    assessment = scene.session.assess_update(plan)
    receipt = run.apply_update(scene.session.prepare_update(plan, reset=True))
    resume.set()
    assert run.wait(WAIT)
    assert receipt.cleanup.wait(WAIT)

    assert assessment.reset.handlers_on_started == ("$steps.boot",)
    assert assessment.reset.describe()["handlers_on_started"] == ["$steps.boot"]
    new_record = scene.session.handler_sessions[("boot",)].instances[("record",)]
    assert new_record is not old_record
    assert notes(scene) == ["boot"]  # started ran once, before the reset
