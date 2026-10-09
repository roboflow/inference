"""``ActiveRun.apply_update``: the quiescent boundary of a running run.

A ``Hold`` step blocks one pulse while the update is requested, so the
boundary is reached only when the test releases it. Every wait is bounded by
``WAIT`` and ordered by events the test controls; no test sleeps.
"""

import threading
from typing import Any, Dict, List

import pytest
from roboflow_workflows.execution_engine.v2.errors import (
    EventEmissionError,
    UpdateConflictError,
)
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.updates import ActiveUpdateReceipt

from tests.unit_tests.execution_engine.v2.updates.active_blocks import (
    END,
    WAIT,
    Collector,
    Count,
    Echo,
    Hold,
    Log,
    Pair,
    Scale,
    Tracker,
    Updater,
    active,
    compiled,
    emit,
    group,
    handler,
    step,
)

COUNTED = [
    step(Hold, "hold", value="$sources.a.value"),
    step(Count, "count", value="$steps.hold.value"),
]
COUNTS = group("counts", "$sources.a.value", count="$steps.count.count")
SCALED = group("scaled", "$sources.a.value", scaled="$steps.scale.scaled")


def base(**sections: Any) -> dict:
    return active(["a"], COUNTED, [COUNTS], **sections)


def attached(**sections: Any) -> dict:
    """``base`` plus a ``Scale`` consumer of the held value and its group."""
    steps = [*COUNTED, step(Scale, "scale", value="$steps.hold.value")]
    definition = active(["a"], steps, [COUNTS, SCALED], **sections)

    return definition


def gates() -> Dict[str, threading.Event]:
    return {"entered": threading.Event(), "release": threading.Event()}


class Scene:
    """A session of ``definition`` over scripted feeds, with one started run."""

    def __init__(self, definition: dict, feeds: Dict[str, list]) -> None:
        self.log = Log()
        self.holds = {"hold": gates(), "record": gates()}
        self.feeds = feeds
        self.plan = compiled(definition)
        self.session = self.plan.create_session(
            resources={"feeds": feeds, "log": self.log, "holds": self.holds}
        )
        self.collected = Collector()
        self.run: Any = None

    def start(self, *groups: str, **options: Any) -> Any:
        self.run = self.session.start(
            handlers=self.collected.handlers(*groups), **options
        )

        return self.run

    def hold(self, name: str = "hold") -> Dict[str, threading.Event]:
        return self.holds[name]

    def entered(self, name: str = "hold") -> None:
        """Wait until the held step entered a call, then arm it for the next."""
        assert self.hold(name)["entered"].wait(WAIT), "the step was never entered"
        self.hold(name)["entered"].clear()

    def release(self, name: str = "hold") -> None:
        self.hold(name)["release"].set()

    def prepare(self, definition: dict) -> Any:
        return self.session.prepare_update(compiled(definition))

    def paused_update(self, definition: dict, **options: Any) -> Updater:
        """Hold pulse 0, request the update, and check it waits for the hold."""
        self.entered()
        updater = Updater(self.run, self.prepare(definition), **options)
        assert not updater.finished.wait(0.05), "the update did not wait for the drain"

        return updater

    def finish(self) -> None:
        self.release()
        assert self.run.wait(WAIT)


def receipt_ok(receipt: Any, *, graph_version: int = 1) -> None:
    assert isinstance(receipt, ActiveUpdateReceipt)
    assert receipt.graph_version == graph_version
    assert receipt.previous_version == graph_version - 1
    assert 0 <= receipt.drained_seconds <= receipt.paused_seconds
    assert 0 <= receipt.drain_after_cut_seconds <= receipt.admission_pause_seconds
    assert receipt.admission_pause_seconds <= receipt.paused_seconds
    assert receipt.admission_pause_seconds <= receipt.call_to_resume_seconds
    assert receipt.run_build_seconds == 0.0 and receipt.cleanup is None


# 1. Serial: blocked step, queued pulse, parked reader ----------------------


def test_serial_update_lands_between_admitted_and_later_pulses() -> None:
    resume = threading.Event()
    scene = Scene(base(), {"a": [emit(1), emit(2), emit(3), resume, emit(4), END]})
    run = scene.start("counts", admission_bound=2)
    # Pulse 0 is held, pulse 1 is queued, the reader holds a slot for pulse 2.
    updater = scene.paused_update(
        attached(), handlers={"scaled": scene.collected.handler("scaled")}
    )
    scene.log.wait_for(lambda: scene.log.count("read") == 3)
    count = scene.session.instances[("count",)]
    scene.release()
    receipt = updater.result()
    receipt_ok(receipt)
    resume.set()
    assert run.wait(WAIT)

    assert scene.collected.sequences("counts") == [0, 1, 2, 3]
    assert scene.collected.versions("counts") == [0, 0, 1, 1]
    assert scene.collected.sequences("scaled") == [2, 3]
    assert scene.collected.versions("scaled") == [1, 1]
    assert receipt.frontiers == {"a": 2}
    # Retained instance and state, one source open, no new Count.
    assert scene.session.instances[("count",)] is count and count.count == 4
    assert scene.log.count("open") == 1 and scene.log.count("init") == 2
    counters = run.counters["a"]
    assert (counters.read, counters.admitted, counters.unadmitted) == (4, 4, 0)
    assert run.state == "finished" and scene.session.graph_version == 1


def test_candidate_prepared_while_running_constructs_only_the_added_step() -> None:
    scene = Scene(base(), {"a": [emit(1), END]})
    scene.start("counts")
    scene.entered()
    update = scene.prepare(attached())
    assert [item[1] for item in scene.log if item[0] == "init"] == ["count", "scale"]
    assert set(update.instances) == {("scale",)}
    update.discard()
    scene.finish()


# 2. Pipelined depths ---------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_pipelined_update_waits_for_in_flight_work_and_seeds_new_stages(
    depth: int,
) -> None:
    resume = threading.Event()
    items = [emit(value) for value in range(1, 6)]
    scene = Scene(base(), {"a": [*items, resume, emit(6), emit(7), END]})
    run = scene.start(
        "counts", admission_bound=2, pipeline=PipelineOptions(max_in_flight=depth)
    )
    scene.hold()["entered"].wait(WAIT)
    # Pulses beyond 0 are admitted (2 slots) or wait at the parked reader.
    updater = Updater(
        run,
        scene.prepare(attached()),
        handlers={"scaled": scene.collected.handler("scaled")},
    )
    assert not updater.finished.wait(0.05)
    scene.release()
    receipt = updater.result()
    receipt_ok(receipt)
    resume.set()
    assert run.wait(WAIT)

    sequences = scene.collected.sequences("counts")
    assert sequences == list(range(7))
    versions = scene.collected.versions("counts")
    boundary = receipt.frontiers["a"]
    assert versions == [0] * boundary + [1] * (7 - boundary)
    # The new delivery stage started at the boundary: no missing or extra pulse.
    assert scene.collected.sequences("scaled") == list(range(boundary, 7))
    assert scene.collected.versions("scaled") == [1] * (7 - boundary)
    assert run.pipeline_counters.current("executing") == 0
    assert scene.log.count("open") == 1


# 3. Latest policy and EOF accounting ------------------------------------------


def test_latest_pending_stays_unadmitted_during_the_pause_and_promotes_after() -> None:
    burst, more = threading.Event(), threading.Event()
    scene = Scene(
        base(),
        {"a": [emit(1), burst, emit(2), emit(3), emit(4), more, emit(5), emit(6), END]},
    )
    run = scene.start(
        "counts",
        admission_bound=1,
        pipeline=PipelineOptions(max_in_flight=1, overload="latest"),
    )
    scene.entered()  # pulse 0 is emission 1 and holds its one slot
    burst.set()
    scene.log.wait_for(lambda: scene.log.count("wait") == 2)
    # Emissions 2 and 3 were replaced by 4, which is pending, unadmitted.
    updater = Updater(
        run,
        scene.prepare(attached()),
        handlers={"scaled": scene.collected.handler("scaled")},
    )
    assert not updater.finished.wait(0.05)
    before = run.counters["a"]
    assert (before.read, before.admitted, before.dropped) == (4, 1, 2)
    scene.release()
    receipt = updater.result()
    receipt_ok(receipt)
    # Nothing pending was admitted while paused: the frontier is still 1.
    assert receipt.frontiers == {"a": 1}
    more.set()
    assert run.wait(WAIT)

    counters = run.counters["a"]
    assert counters.read == 6 and counters.unadmitted == 0
    assert counters.admitted + counters.dropped == 6
    assert counters.admitted >= 2  # emission 4 promoted after the resume
    assert scene.collected.versions("counts") == [0] + [1] * (counters.admitted - 1)
    assert scene.collected.sequences("scaled") == list(range(1, counters.admitted))


LATEST = PipelineOptions(max_in_flight=2, overload="latest")


def with_b_scaled(definition: dict) -> dict:
    """``definition`` over sources ``a`` and ``b`` plus a ``Scale`` consumer of ``b``."""
    definition = dict(definition)
    definition["steps"] = [
        *definition["steps"],
        step(Scale, "scale_b", value="$sources.b.value"),
    ]
    definition["outputs"] = [
        *definition["outputs"],
        group("scaled_b", "$sources.b.value", scaled="$steps.scale_b.scaled"),
    ]

    return definition


def test_latest_source_ending_while_paused_promotes_its_pending_emission_after_the_commit() -> (
    None
):
    resume, burst = threading.Event(), threading.Event()
    definition = with_b_scaled(active(["a", "b"], COUNTED, [COUNTS]))
    added = with_b_scaled(
        active(
            ["a", "b"],
            [*COUNTED, step(Scale, "scale", value="$steps.hold.value")],
            [COUNTS, SCALED],
        )
    )
    scene = Scene(
        definition,
        {
            "a": [emit(1), resume, emit(2), END],
            "b": [emit(10), burst, emit(11), emit(12), END],
        },
    )
    run = scene.start("counts", "scaled_b", admission_bound=1, pipeline=LATEST)
    scene.entered()
    scene.collected.wait_for("scaled_b", 1)  # b's first emission ran before the cut
    updater = Updater(
        run,
        scene.prepare(added),
        handlers={"scaled": scene.collected.handler("scaled")},
    )
    eventually(lambda: run._driver._paused)
    burst.set()  # 11 is replaced by 12, which stays pending; then b ends
    scene.log.wait_for(lambda: ("read", "b", None) in scene.log)
    assert not updater.finished.wait(0.05)
    before = run.counters["b"]
    assert (before.read, before.admitted, before.dropped) == (3, 1, 1)
    scene.release()
    receipt = updater.result()
    receipt_ok(receipt)
    assert receipt.frontiers["b"] == 1  # the pending emission was not sequenced
    resume.set()
    assert run.wait(WAIT)

    # Promoted after the resume under the new graph, then the source sealed.
    counters = run.counters["b"]
    assert (counters.admitted, counters.dropped, counters.unadmitted) == (2, 1, 0)
    assert counters.ended and counters.closed
    assert scene.collected.sequences("scaled_b") == [0, 1]
    assert scene.collected.versions("scaled_b") == [0, 1]
    assert scene.collected.versions("counts") == [0, 1]


def test_latest_only_source_ending_while_paused_rejects_and_keeps_its_pending_emission() -> (
    None
):
    burst = threading.Event()
    scene = Scene(base(), {"a": [emit(1), burst, emit(2), emit(3), END]})
    run = scene.start("counts", admission_bound=1, pipeline=LATEST)
    scene.entered()
    updater = Updater(run, scene.prepare(attached()))
    eventually(lambda: run._driver._paused)
    burst.set()
    scene.log.wait_for(lambda: ("read", "a", None) in scene.log)
    scene.release()
    with pytest.raises(UpdateConflictError, match="every source .* ended"):
        updater.result()
    assert run.wait(WAIT)

    # The rejected commit left the panel on the current plan; the pending
    # emission was admitted under it and the run completed.
    assert scene.session.graph_version == 0
    assert scene.session.controls.plan is scene.session.plan
    counters = run.counters["a"]
    assert (
        counters.read,
        counters.admitted,
        counters.dropped,
        counters.unadmitted,
    ) == (
        3,
        2,
        1,
        0,
    )
    assert scene.collected.sequences("counts") == [0, 1]
    assert scene.collected.versions("counts") == [0, 0]
    assert run.state == "finished"


# 4. EOF on each side of the cut --------------------------------------------------


def test_end_of_the_only_source_during_the_pause_rejects_and_completes_the_run() -> (
    None
):
    ending = threading.Event()
    scene = Scene(base(), {"a": [emit(1), ending, END]})
    run = scene.start("counts", admission_bound=1)
    updater = scene.paused_update(attached())
    ending.set()  # the source ends while paused: remembered, not sealed
    scene.log.wait_for(lambda: scene.log.count("close") == 1)
    scene.release()
    with pytest.raises(UpdateConflictError, match="every source .* ended"):
        updater.result()
    assert run.wait(WAIT)

    assert scene.collected.sequences("counts") == [0]
    assert scene.collected.versions("counts") == [0]
    assert scene.session.graph_version == 0 and run.state == "finished"
    assert run.counters["a"].ended and run.counters["a"].closed


def test_source_ended_before_the_cut_lets_the_run_conclude_and_rejects() -> None:
    scene = Scene(base(), {"a": [emit(1), END]})
    run = scene.start("counts", admission_bound=1)
    scene.entered()
    scene.log.wait_for(lambda: scene.log.count("close") == 1)
    eventually(run._driver.readers_finished)
    updater = Updater(run, scene.prepare(attached()))
    assert not updater.finished.wait(0.05)
    scene.release()
    with pytest.raises(UpdateConflictError, match="concluded before|done"):
        updater.result()
    assert run.wait(WAIT)
    assert scene.collected.versions("counts") == [0]
    assert scene.session.graph_version == 0


def eventually(predicate: Any) -> None:
    """Bounded wait for a condition another thread makes true."""
    finished = threading.Event()

    def poll() -> None:
        while not predicate():
            if finished.wait(0.005):
                return
        finished.set()

    threading.Thread(target=poll, daemon=True).start()
    assert finished.wait(WAIT), "the condition never held"


def test_end_of_one_source_during_the_pause_is_sealed_after_the_commit() -> None:
    hold_b = threading.Event()
    definition = active(
        ["a", "b"],
        COUNTED,
        [COUNTS],
        operators=[
            {"type": Pair.type, "name": "pair", "inputs": {"value": "$sources.b.value"}}
        ],
    )
    added = dict(definition)
    added["steps"] = [*COUNTED, step(Scale, "scale", value="$steps.hold.value")]
    added["outputs"] = [COUNTS, SCALED]
    scene = Scene(
        definition, {"a": [emit(1), emit(2), END], "b": [emit(10), hold_b, END]}
    )
    run = scene.start("counts", admission_bound=1)
    updater = scene.paused_update(added)
    hold_b.set()
    # ``b`` ends while paused: its seal, and the operator's finish, wait.
    scene.log.wait_for(lambda: ("read", "b", None) in scene.log)
    pair = run._executor.operators["pair"].instance
    assert pair.lifecycle == []
    scene.release()
    receipt = updater.result()
    receipt_ok(receipt)
    assert run.wait(WAIT)

    assert pair.lifecycle == [("end_input", "value"), ("finish", "eof", 1), ("close",)]
    assert scene.collected.versions("counts") == [0, 1]
    assert (
        run.counters["b"].admitted == 1 and run.operator_counters["pair"].emitted == 0
    )


def test_end_of_one_source_before_the_cut_finishes_its_operator_as_old_work() -> None:
    definition = active(
        ["a", "b"],
        COUNTED,
        [COUNTS],
        operators=[
            {"type": Pair.type, "name": "pair", "inputs": {"value": "$sources.b.value"}}
        ],
    )
    added = dict(definition)
    added["steps"] = [*COUNTED, step(Scale, "scale", value="$steps.hold.value")]
    added["outputs"] = [COUNTS, SCALED]
    scene = Scene(definition, {"a": [emit(1), emit(2), END], "b": [emit(10), END]})
    run = scene.start("counts", admission_bound=1)
    scene.entered()
    # ``b`` ends before the cut: its end is queued behind the held pulse.
    eventually(lambda: "b" in run._driver._finished)
    pair = run._executor.operators["pair"].instance
    assert pair.lifecycle == []
    updater = Updater(run, scene.prepare(added))
    assert not updater.finished.wait(0.05)
    scene.release()
    receipt = updater.result()
    receipt_ok(receipt)
    # The chained end and finish drained before the commit, as old work.
    assert pair.lifecycle[:2] == [("end_input", "value"), ("finish", "eof", 1)]
    assert run.wait(WAIT)

    assert pair.lifecycle == [("end_input", "value"), ("finish", "eof", 1), ("close",)]
    assert scene.collected.versions("counts") == [0, 1]
    assert (
        run.counters["b"].admitted == 1 and run.operator_counters["pair"].emitted == 0
    )


# 5. Reactions: blocked handler, queued event, accepted signal -------------------


def reacting(definition: dict) -> dict:
    definition = dict(definition)
    definition["signals"] = [{"name": "ping", "fields": {"note": ["string"]}}]
    definition["handlers"] = [handler("audit", "$signals.ping")]
    definition["outputs"] = [
        *definition["outputs"],
        group("audits", "$handlers.audit", out="$handlers.audit.out"),
    ]

    return definition


def test_update_waits_for_handler_work_and_signals_reject_meanwhile() -> None:
    resume = threading.Event()
    scene = Scene(reacting(base()), {"a": [emit(1), resume, emit(2), END]})
    run = scene.start("counts", "audits")
    scene.entered()
    run.signal("ping", note="first")
    scene.entered("record")
    run.signal("ping", note="second")  # queued behind the held handler run
    delivered: List[Any] = []
    updater = Updater(
        run, scene.prepare(reacting(attached())), handlers={"audits": delivered.append}
    )
    assert not updater.finished.wait(0.05)
    scene.release()  # the pulse completes; the handler still holds
    assert not updater.finished.wait(0.05)
    with pytest.raises(EventEmissionError, match="graph update"):
        run.signal("ping", note="rejected")
    reactions = run._reactions
    scene.release("record")
    receipt = updater.result()
    receipt_ok(receipt)
    assert run._reactions is reactions
    run.signal("ping", note="third")
    resume.set()
    assert run.wait(WAIT)

    notes = [item[1] for item in scene.log if item[0] == "record"]
    assert notes == ["first", "second", "third"]
    # Handler results are attributed to the run's graph, patched callback included.
    assert len(scene.collected.results["audits"]) == 2
    assert [result.graph_version for result in delivered] == [1]
    assert scene.collected.versions("counts") == [0, 1]


# 6. Unequal source frontiers and a new consumer of the quiet source --------------


def test_new_consumer_of_a_source_without_pulses_starts_at_ordinal_zero() -> None:
    open_b = threading.Event()
    definition = active(["a", "b"], COUNTED, [COUNTS])
    added = active(
        ["a", "b"],
        [*COUNTED, step(Scale, "scale_b", value="$sources.b.value")],
        [COUNTS, group("scaled_b", "$sources.b.value", scaled="$steps.scale_b.scaled")],
    )
    scene = Scene(
        definition,
        {
            "a": [emit(1), emit(2), emit(3), emit(4), END],
            "b": [open_b, emit(10), emit(11), END],
        },
    )
    run = scene.start(
        "counts", admission_bound=1, pipeline=PipelineOptions(max_in_flight=2)
    )
    scene.entered()
    # a advances past pulse 0 while b has not emitted at all.
    scene.release()
    scene.collected.wait_for("counts", 2)
    updater = Updater(
        run,
        scene.prepare(added),
        handlers={"scaled_b": scene.collected.handler("scaled_b")},
    )
    receipt = updater.result()
    receipt_ok(receipt)
    assert receipt.frontiers["b"] == 0 and receipt.frontiers["a"] >= 2
    open_b.set()
    assert run.wait(WAIT)

    assert scene.collected.sequences("scaled_b") == [0, 1]
    assert scene.collected.versions("scaled_b") == [1, 1]
    assert scene.collected.sequences("counts") == [0, 1, 2, 3]


# 7. A partial operator window survives the boundary -------------------------------


def test_partial_pair_window_completes_after_the_update_with_old_causes() -> None:
    resume = threading.Event()
    definition = active(
        ["a"],
        COUNTED,
        [COUNTS],
        operators=[
            {"type": Pair.type, "name": "pair", "inputs": {"value": "$sources.a.value"}}
        ],
    )
    added = dict(definition)
    added["outputs"] = [
        COUNTS,
        group("pairs", "$operators.pair.first", first="$operators.pair.first"),
    ]
    scene = Scene(definition, {"a": [emit(1), emit(2), emit(3), resume, emit(4), END]})
    run = scene.start(
        "counts", admission_bound=1, pipeline=PipelineOptions(max_in_flight=2)
    )
    scene.entered()
    scene.release()
    scene.collected.wait_for("counts", 3)
    pair = run._executor.operators["pair"].instance
    assert len(pair.buffer) == 1  # pulse 2 waits for its partner
    updater = Updater(
        run, scene.prepare(added), handlers={"pairs": scene.collected.handler("pairs")}
    )
    receipt = updater.result()
    receipt_ok(receipt)
    assert receipt.frontiers == {"a": 3, "pair": 1}
    resume.set()
    assert run.wait(WAIT)

    assert run._executor.operators["pair"].instance is pair
    pairs = scene.collected.results["pairs"]
    assert [result.pulse.sequence for result in pairs] == [1]
    assert [cause.sequence for cause in pairs[0].causes] == [2, 3]
    assert pairs[0].graph_version == 1
    assert run.operator_counters["pair"].emitted == 2


# 8. Controls: a write during the drain, a pending reset, separate versions -------


def controlled(consumer: bool) -> dict:
    steps = [
        step(Hold, "hold", value="$sources.a.value"),
        step(Tracker, "tracker", value="$steps.hold.value"),
    ]
    groups = [group("ticks", "$sources.a.value", ticks="$steps.tracker.ticks")]
    if consumer:
        steps.append(step(Scale, "scale", value="$steps.hold.value"))
        groups.append(SCALED)
    definition = active(
        ["a"],
        steps,
        groups,
        controls={
            "analysis": {
                "type": "enable",
                "steps": ["$steps.tracker"],
                "state": "reset_on_enable",
                "suspends_effects": True,
            }
        },
    )

    return definition


def test_control_writes_during_the_drain_keep_their_version_and_pending_reset() -> None:
    resume = threading.Event()
    scene = Scene(controlled(False), {"a": [emit(1), emit(2), resume, emit(3), END]})
    run = scene.start("ticks", admission_bound=1)
    updater = scene.paused_update(controlled(True))
    scene.session.controls.update(analysis=False)
    scene.session.controls.update(analysis=True)  # a reset is pending for pulse 1
    assert scene.session.controls.version == 2
    tracker = scene.session.instances[("tracker",)]
    scene.release()
    receipt = updater.result()
    receipt_ok(receipt)
    assert scene.session.controls.version == 2
    assert scene.session.controls.current.epochs == {"analysis": 2}
    assert scene.session.controls.plan is scene.session.plan
    resume.set()
    assert run.wait(WAIT)

    assert tracker.resets == 1
    ticks = [result.rows()[0]["ticks"] for result in scene.collected.results["ticks"]]
    assert len(ticks) == 3
    assert scene.collected.versions("ticks") == [0, 1, 1]


def chained(echo: bool) -> dict:
    """``controlled`` plus a prunable ``Echo`` of the tracker: it joins the closure."""
    definition = controlled(False)
    if echo:
        definition["steps"] = [
            *definition["steps"],
            step(Echo, "echo", value="$steps.tracker.ticks"),
        ]
        definition["outputs"] = [
            *definition["outputs"],
            group("echoes", "$sources.a.value", out="$steps.echo.out"),
        ]

    return definition


def field_of(scene: Scene, name: str, field: str) -> List[int]:
    return [result.rows()[0][field] for result in scene.collected.results[name]]


def test_pure_consumer_joining_the_closure_is_suspended_with_it_and_never_reset() -> (
    None
):
    resume, again = threading.Event(), threading.Event()
    scene = Scene(
        chained(False), {"a": [emit(1), emit(2), resume, emit(3), again, emit(4), END]}
    )
    run = scene.start("ticks", admission_bound=1)
    updater = scene.paused_update(
        chained(True), handlers={"echoes": scene.collected.handler("echoes")}
    )
    scene.session.controls.update(analysis=False)
    scene.session.controls.update(analysis=True)  # a reset is pending for the tracker
    tracker = scene.session.instances[("tracker",)]
    assert tracker.resets == 0  # pulse 0 is held: the reset waits for pulse 1
    scene.release()
    receipt = updater.result()
    receipt_ok(receipt)
    # Only a prunable step can join a closure by reading (a stateful reader
    # must be listed, which changes the declaration): it is pure, so the
    # reset guards stay as they were and the pending reset is the tracker's.
    control = scene.session.plan.controls.enable_controls["analysis"]
    assert ("echo",) in control.closure and control.state_classes[("echo",)] == "pure"
    assert scene.session.activity(("echo",)) is None
    assert scene.session.activity(("tracker",)) is not None
    resume.set()
    scene.collected.wait_for("echoes", 2)
    assert tracker.resets == 1
    scene.session.controls.update(analysis=False)  # suspends the echo with the tracker
    assert ("echo",) in scene.session.controls.current.omitted_steps
    again.set()
    assert run.wait(WAIT)

    assert tracker.resets == 1 and scene.session.controls.version == 3
    assert field_of(scene, "ticks", "ticks") == [1, 1, 2]
    assert field_of(scene, "echoes", "out") == [1, 2]
    assert scene.collected.versions("echoes") == [1, 1]
