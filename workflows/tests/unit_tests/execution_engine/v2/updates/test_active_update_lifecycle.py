"""``ActiveRun.apply_update`` against the run's lifecycle and other updates.

Timeouts, stops, cancellations, failures, competing and stale candidates
and self-waits all leave the old graph usable or the run terminal, never
half-switched. Every wait is bounded by ``WAIT`` and ordered by events.
"""

import contextlib
import threading
from typing import Any, Dict, Iterator, List

import pytest
from roboflow_workflows.execution_engine.v2.active.runtime import _UpdateToken
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import (
    ActiveRunError,
    ContractError,
    EventEmissionError,
    GraphUpdateError,
    IncompatibleUpdateError,
    SessionBusyError,
    UpdateConflictError,
    UpdateTimeoutError,
)
from roboflow_workflows.execution_engine.v2.kinds import STRING_KIND, Kind
from roboflow_workflows.execution_engine.v2.pipelining.options import PipelineOptions
from roboflow_workflows.execution_engine.v2.updates import APPLIED, DISCARDED, PREPARED

from tests.unit_tests.execution_engine.v2.updates.active_blocks import (
    END,
    WAIT,
    Collector,
    Count,
    Feed,
    Hold,
    Log,
    Scale,
    Updater,
    active,
    compiled,
    emit,
    group,
    handler,
    step,
)
from tests.unit_tests.execution_engine.v2.updates.test_active_update_boundary import (
    COUNTS,
    SCALED,
    Scene,
    attached,
    base,
    reacting,
)


@pytest.mark.parametrize("pipeline", [None, PipelineOptions(max_in_flight=2)])
def test_timeout_lifts_the_pause_and_keeps_the_old_graph_and_the_candidate(
    pipeline,
) -> None:
    resume = threading.Event()
    scene = Scene(base(), {"a": [emit(1), emit(2), resume, emit(3), END]})
    run = scene.start("counts", admission_bound=1, pipeline=pipeline)
    scene.entered()
    update = scene.prepare(attached())
    with pytest.raises(UpdateTimeoutError, match="keeps its current graph"):
        run.apply_update(update, timeout=0.2)
    assert update.state == PREPARED and scene.session.graph_version == 0
    assert run._update_token is None
    scene.release()
    scene.collected.wait_for("counts", 2)
    # The candidate is still good: the same run applies it once the hold is
    # gone; the stale barrier of the timed-out attempt is passed over.
    receipt = run.apply_update(update, timeout=WAIT)
    assert receipt.graph_version == 1 and update.state == APPLIED
    assert receipt.frontiers == {"a": 2}
    resume.set()
    assert run.wait(WAIT)

    assert scene.collected.sequences("counts") == [0, 1, 2]
    assert scene.collected.versions("counts") == [0, 0, 1]
    assert run.counters["a"].read == 3 and run.counters["a"].unadmitted == 0


def test_stop_during_the_drain_wins_and_the_run_finishes_with_the_old_graph() -> None:
    scene = Scene(base(), {"a": [emit(1), emit(2), emit(3), END]})
    run = scene.start("counts", admission_bound=1)
    updater = scene.paused_update(attached())
    run.stop()
    with pytest.raises(UpdateConflictError, match="stopping"):
        updater.result()
    scene.release()
    assert run.wait(WAIT)

    assert run.state == "finished" and scene.session.graph_version == 0
    # The admitted pulse completed; the stopped reader admitted nothing more.
    assert scene.collected.versions("counts") == [0]
    counters = run.counters["a"]
    assert counters.admitted == 1 and counters.unadmitted >= 1


def test_cancel_during_the_drain_wins_and_nothing_reopens_admission() -> None:
    scene = Scene(base(), {"a": [emit(1), emit(2), emit(3), END]})
    run = scene.start(
        "counts", admission_bound=1, pipeline=PipelineOptions(max_in_flight=1)
    )
    updater = scene.paused_update(attached())
    run.cancel()
    with pytest.raises(UpdateConflictError, match="cancelled"):
        updater.result()
    scene.release()
    assert run.wait(WAIT)

    assert run.state == "cancelled" and scene.session.graph_version == 0
    assert (
        run.counters["a"].processed + run.counters["a"].cancelled
        == run.counters["a"].admitted
    )


def test_failure_during_the_drain_fails_the_run_and_rejects_the_update() -> None:
    scene = Scene(base(), {"a": [emit(1), emit(2), END]})
    run = scene.start("counts", admission_bound=1)
    scene.hold()["fail"] = threading.Event()
    updater = scene.paused_update(attached())
    scene.hold()["fail"].set()
    scene.release()
    with pytest.raises(UpdateConflictError, match="failed"):
        updater.result()
    with pytest.raises(ActiveRunError):
        run.wait(WAIT)
    assert run.state == "failed" and scene.session.graph_version == 0


def test_competing_update_is_rejected_while_one_is_in_progress() -> None:
    scene = Scene(base(), {"a": [emit(1), emit(2), END]})
    run = scene.start("counts", admission_bound=1)
    updater = scene.paused_update(attached())
    other = scene.prepare(attached())
    with pytest.raises(UpdateConflictError, match="already has a graph update"):
        run.apply_update(other)
    assert other.state == PREPARED
    scene.release()
    receipt = updater.result()
    assert receipt.graph_version == 1
    # The other candidate is stale now; applying it discards it.
    with pytest.raises(UpdateConflictError, match="prepared from graph version 0"):
        run.apply_update(other)
    assert other.state == DISCARDED
    assert run.wait(WAIT)


def test_discarded_and_applied_candidates_are_rejected_before_any_pause() -> None:
    scene = Scene(base(), {"a": [emit(1), END]})
    run = scene.start("counts", admission_bound=1)
    scene.entered()
    update = scene.prepare(attached())
    update.discard()
    with pytest.raises(UpdateConflictError, match="discarded"):
        run.apply_update(update)
    other_session = compiled(base()).create_session(
        resources={"feeds": {}, "log": Log(), "holds": {}}
    )
    foreign = other_session.prepare_update(compiled(attached()))
    with pytest.raises(UpdateConflictError, match="prepared for session"):
        run.apply_update(foreign)
    assert run._update_token is None
    scene.finish()


def test_idle_apply_and_close_are_refused_while_the_run_is_active() -> None:
    scene = Scene(base(), {"a": [emit(1), END]})
    run = scene.start("counts", admission_bound=1)
    scene.entered()
    update = scene.prepare(attached())
    with pytest.raises(SessionBusyError):
        scene.session.apply_update(update)
    with pytest.raises(ContractError, match="unfinished active run"):
        scene.session.close()
    scene.finish()
    with pytest.raises(UpdateConflictError, match="done"):
        run.apply_update(update)
    assert update.state == PREPARED
    scene.session.apply_update(update)
    scene.session.close()


def test_incompatible_or_failing_preparation_never_touches_the_run() -> None:
    scene = Scene(base(), {"a": [emit(1), END]})
    run = scene.start("counts", admission_bound=1)
    scene.entered()
    changed = active(
        ["a"],
        [step(Hold, "hold", value="$sources.a.value")],
        [group("counts", "$sources.a.value", value="$steps.hold.value")],
    )
    with pytest.raises(IncompatibleUpdateError):
        scene.prepare(changed)
    assert run.state == "running" and run._update_token is None
    scene.finish()
    assert scene.collected.versions("counts") == [0]


def test_recording_runs_reject_an_active_update_before_pausing(tmp_path) -> None:
    definition = base(
        inputs=[{"type": "WorkflowParameter", "name": "capture_dir"}],
        recording={
            "type": "file",
            "directory": "$inputs.capture_dir",
            "groups": ["counts"],
        },
    )
    scene = Scene(definition, {"a": [emit(1), END]})
    run = scene.session.start(
        inputs={"capture_dir": str(tmp_path)},
        handlers=scene.collected.handlers("counts"),
        admission_bound=1,
    )
    scene.run = run
    scene.entered()
    added = attached(
        inputs=[{"type": "WorkflowParameter", "name": "capture_dir"}],
        recording={
            "type": "file",
            "directory": "$inputs.capture_dir",
            "groups": ["counts"],
        },
    )
    update = scene.prepare(added)
    with pytest.raises(GraphUpdateError, match="records output groups"):
        run.apply_update(update)
    assert update.state == PREPARED and run._update_token is None
    scene.finish()


def test_invalid_handlers_and_timeouts_are_rejected_before_pausing() -> None:
    scene = Scene(base(), {"a": [emit(1), END]})
    run = scene.start("counts", admission_bound=1)
    scene.entered()
    update = scene.prepare(attached())
    with pytest.raises(ContractError, match="unknown output groups"):
        run.apply_update(update, handlers={"nope": print})
    with pytest.raises(ContractError, match="must be callable"):
        run.apply_update(update, handlers={"scaled": 1})
    with pytest.raises(ContractError, match="timeout"):
        run.apply_update(update, timeout=0)
    with pytest.raises(ContractError, match="timeout"):
        run.apply_update(update, timeout=None)
    assert update.state == PREPARED and run._update_token is None
    scene.finish()


def test_handler_patch_keeps_replaces_and_drops_callbacks() -> None:
    resume = threading.Event()
    held = group("held", "$sources.a.value", value="$steps.hold.value")
    with_held = active(["a"], base()["steps"], [COUNTS, held])
    scene = Scene(with_held, {"a": [emit(1), resume, emit(2), END]})
    run = scene.start("counts", "held", admission_bound=1)
    scene.entered()
    scene.release()
    scene.collected.wait_for("held", 1)
    replaced: List[Any] = []
    receipt = run.apply_update(
        scene.prepare(with_held), handlers={"held": replaced.append}
    )
    assert receipt.graph_version == 1 and set(run._handlers) == {"counts", "held"}
    # Drop the ``held`` group from the plan (its step stays): the callback goes.
    receipt = run.apply_update(scene.prepare(base()))
    assert receipt.graph_version == 2 and set(run._handlers) == {"counts"}
    resume.set()
    assert run.wait(WAIT)
    assert scene.collected.versions("counts") == [0, 2]
    assert scene.collected.versions("held") == [0] and replaced == []


# Self-waits from threads of the run ------------------------------------------------


def test_apply_update_from_a_group_handler_source_read_or_block_is_rejected() -> None:
    errors: Dict[str, BaseException] = {}
    more = threading.Event()
    scene = Scene(base(), {"a": [emit(1), more, emit(2), END]})
    update_holder: Dict[str, Any] = {}

    def from_handler(result: Any) -> None:
        try:
            scene.run.apply_update(update_holder["update"])
        except ContractError as raised:
            errors["handler"] = raised

    def from_reader(name: str) -> None:
        if "update" in update_holder:
            try:
                scene.run.apply_update(update_holder["update"])
            except ContractError as raised:
                errors.setdefault("reader", raised)

    def from_block() -> None:
        if "update" in update_holder:
            try:
                scene.run.apply_update(update_holder["update"])
            except ContractError as raised:
                errors.setdefault("block", raised)

    scene.hold()["hook"] = from_block
    scene.session = scene.plan.create_session(
        resources={
            "feeds": scene.feeds,
            "log": scene.log,
            "holds": scene.holds,
            "hook": from_reader,
        }
    )
    run = scene.session.start(handlers={"counts": from_handler}, admission_bound=1)
    scene.run = run
    scene.entered()
    update_holder["update"] = scene.prepare(attached())
    more.set()  # the reader reads emission 2 with the candidate at hand
    scene.release()
    assert run.wait(WAIT)
    assert "handler" in errors and "wait for itself" in str(errors["handler"])
    assert "reader" in errors and "wait for itself" in str(errors["reader"])
    assert "block" in errors and "wait for itself" in str(errors["block"])
    assert run.state == "finished" and update_holder["update"].state == PREPARED


def test_apply_update_from_a_reaction_thread_is_rejected_also_during_a_drain() -> None:
    errors: List[BaseException] = []
    resume = threading.Event()
    scene = Scene(reacting(base()), {"a": [emit(1), resume, END]})
    collected: Dict[str, Any] = {}

    def from_reaction(result: Any) -> None:
        try:
            scene.run.apply_update(collected["update"])
        except ContractError as raised:
            errors.append(raised)

    run = scene.start("counts", admission_bound=1)
    scene.entered()
    collected["update"] = scene.prepare(reacting(attached()))
    # Patch the handler-group callback in through a first update, then signal.
    run.signal("ping", note="before")
    scene.entered("record")
    updater = Updater(run, collected["update"], handlers={"audits": from_reaction})
    assert not updater.finished.wait(0.05)
    scene.release()
    scene.release("record")
    receipt = updater.result()
    assert receipt.graph_version == 1
    collected["update"] = scene.prepare(reacting(attached()))
    run.signal("ping", note="after")  # its callback tries to update the run
    scene.log.wait_for(lambda: ("record", "after") in scene.log)
    resume.set()
    assert run.wait(WAIT)
    assert len(errors) == 1 and "wait for itself" in str(errors[0])
    assert collected["update"].state == PREPARED


# Pause ownership, and the terminal decision against the publication ---------------


@pytest.mark.parametrize("pipeline", [None, PipelineOptions(max_in_flight=2)])
def test_a_stale_token_cannot_resume_another_update_pause(pipeline) -> None:
    scene = Scene(reacting(base()), {"a": [emit(1), END]})
    run = scene.start("counts", admission_bound=1, pipeline=pipeline)
    scene.entered()
    first, second = _UpdateToken(), _UpdateToken()
    run._driver.pause(first)
    assert run._reactions.pause_ingress(first)
    run._driver.resume(second)  # not its pause: nothing changes
    run._reactions.resume_ingress(second)
    assert run._driver._paused
    with pytest.raises(EventEmissionError, match="graph update"):
        run.signal("ping", note="paused")
    run._driver.resume(first)
    run._reactions.resume_ingress(first)
    assert not run._driver._paused
    run.signal("ping", note="resumed")
    scene.release("record")
    scene.finish()
    assert ("record", "resumed") in scene.log


def test_handler_results_are_built_only_for_subscribed_groups() -> None:
    resume = threading.Event()
    scene = Scene(reacting(base()), {"a": [emit(1), resume, END]})
    run = scene.start("counts", admission_bound=1)  # nobody receives ``audits``
    built: List[str] = []
    group_result = run._reactions._group_result

    def counting(state: Any, group: Any, **arguments: Any) -> Any:
        built.append(group.name)
        return group_result(state, group, **arguments)

    run._reactions._group_result = counting
    scene.entered()
    scene.release()
    run.signal("ping", note="unsubscribed")
    scene.entered("record")
    scene.release("record")
    scene.log.wait_for(lambda: ("record", "unsubscribed") in scene.log)
    assert run._reactions.wait_quiescent(WAIT, interrupted=lambda: False)
    assert built == []  # the handler ran; no result was constructed
    delivered: List[Any] = []
    receipt = run.apply_update(
        scene.prepare(reacting(attached())), handlers={"audits": delivered.append}
    )
    assert receipt.graph_version == 1
    run.signal("ping", note="subscribed")
    scene.log.wait_for(lambda: ("record", "subscribed") in scene.log)
    resume.set()
    assert run.wait(WAIT)

    assert built == ["audits"]
    assert [result.graph_version for result in delivered] == [1]
    notes = [item[1] for item in scene.log if item[0] == "record"]
    assert notes == ["unsubscribed", "subscribed"]


def test_stop_between_the_drain_and_the_publication_rejects_the_update() -> None:
    scene = Scene(base(), {"a": [emit(1), emit(2), END]})
    run = scene.start("counts", admission_bound=1)
    updater = scene.paused_update(attached())
    panel = scene.session.controls
    rebasing = panel._rebasing
    seen: List[int] = []

    @contextlib.contextmanager
    def stop_during_commit(plan: Any, **options: Any) -> Iterator[Any]:
        # The panel built its snapshot for the plan; the run is stopped
        # before its last check and the publication, which share one hold
        # of the run's lock.
        with rebasing(plan, **options) as adopt:
            if not seen:
                seen.append(scene.session.graph_version)
                run.stop()
            yield adopt

    panel._rebasing = stop_during_commit
    scene.release()
    with pytest.raises(UpdateConflictError, match="stopping"):
        updater.result()
    assert run.wait(WAIT)

    assert seen == [0] and scene.session.graph_version == 0
    assert panel.plan is scene.session.plan and panel.version == 0
    assert scene.collected.versions("counts") == [0]


def test_a_failing_control_rebuild_rejects_the_update_and_changes_nothing() -> None:
    resume = threading.Event()
    scene = Scene(base(), {"a": [emit(1), resume, emit(2), END]})
    run = scene.start("counts", admission_bound=1)
    panel = scene.session.controls
    build = panel._build
    handlers = dict(run._handlers)
    scene.entered()
    update = scene.prepare(attached())

    def failing_build(*args: Any, **options: Any) -> Any:
        raise RuntimeError("the snapshot could not be built")

    panel._build = failing_build
    scene.release()
    with pytest.raises(RuntimeError, match="could not be built"):
        run.apply_update(update, timeout=WAIT)

    # Nothing of the commit landed: the graph, the panel and the run's
    # bindings are as before, and the candidate can still be applied.
    assert update.state == PREPARED and scene.session.graph_version == 0
    assert panel.plan is scene.session.plan and panel.version == 0
    assert run._handlers == handlers and run._update_token is None
    panel._build = build
    receipt = run.apply_update(update, timeout=WAIT)
    assert receipt.graph_version == 1 and panel.plan is scene.session.plan
    resume.set()
    assert run.wait(WAIT)
    assert scene.collected.versions("counts") == [0, 1]


@pytest.mark.parametrize("pipeline", [None, PipelineOptions(max_in_flight=2)])
def test_commit_blocked_by_a_slow_control_codec_times_out_and_retries(
    pipeline,
) -> None:
    # A control write decodes its value with the kind's custom codec while
    # it holds the panel. The update's deadline covers that wait: it times
    # out while the codec still blocks, leaves the old graph and the
    # candidate usable, and the same candidate applies once the write landed.
    decode_entered, decode_release, more = (threading.Event() for _ in range(3))

    def decode(value: Any) -> float:
        if decode_entered.is_set() or value != 1.0:
            decode_entered.set()
            assert decode_release.wait(WAIT)

        return float(value)

    scalar = Kind(
        "slow_scalar", validate=lambda v: type(v) is float, deserialize=decode
    )
    catalogue = Catalogue(
        [Count, Hold, Scale], sources=[Feed], kinds=[STRING_KIND, scalar]
    )
    sections = dict(
        inputs=[
            {
                "type": "WorkflowParameter",
                "name": "threshold",
                "kind": [scalar.name],
                "default_value": 1.0,
            }
        ],
        controls={"threshold": {"type": "input", "input": "$inputs.threshold"}},
    )
    before = compile_workflow(base(**sections), catalogue=catalogue)
    after = compile_workflow(attached(**sections), catalogue=catalogue)
    scene = Scene(base(), {"a": [emit(1), more, emit(2), END]})
    scene.plan = before
    scene.session = before.create_session(
        resources={"feeds": scene.feeds, "log": scene.log, "holds": scene.holds}
    )
    panel = scene.session.controls
    run = scene.start("counts", admission_bound=1, pipeline=pipeline)
    scene.entered()
    scene.release()
    scene.collected.wait_for("counts", 1)
    scene.log.wait_for(lambda: scene.log.count("wait") == 1)
    update = scene.session.prepare_update(after)
    written: Dict[str, Any] = {}
    control_done = threading.Event()

    def write_control() -> None:
        try:
            written["receipt"] = panel.update(threshold=2.0)
        except BaseException as error:
            written["error"] = error
        finally:
            control_done.set()

    threading.Thread(target=write_control, daemon=True).start()
    try:
        assert decode_entered.wait(WAIT)
        updater = Updater(run, update, timeout=0.2)
        with pytest.raises(UpdateTimeoutError, match="the control panel"):
            updater.result()
        # The codec still blocks the control write; the update gave up alone.
        assert not control_done.is_set()
        assert update.state == PREPARED and scene.session.graph_version == 0
        assert panel.plan is before and panel.version == 0
        assert run._update_token is None

        decode_release.set()
        assert control_done.wait(WAIT), written
        assert written["receipt"].version == 1
        receipt = run.apply_update(update, timeout=WAIT)
        assert receipt.graph_version == 1 and update.state == APPLIED
        assert panel.plan is after and panel.version == 1
        assert panel.current.values["threshold"] == 2.0
        more.set()
        assert run.wait(WAIT)
        assert scene.collected.versions("counts") == [0, 1]
    finally:
        decode_release.set()
        more.set()
        assert run.wait(WAIT)
