"""Unit tests of the shared sink runtime: delivery, background effects, cooldown."""

import logging
import threading
from typing import Optional

import pytest
from roboflow_workflows.v2.blocks.sinks import runtime
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    FULL_MESSAGE,
    BackgroundEffects,
    Cooldown,
    EffectFailure,
    deliver,
    disabled_sink_message,
)

from tests.v2.support.sinks.loopback import ManualClock


def _ok():
    return False, "done"


def _boom():
    raise RuntimeError("disk full")


class _Recorder:
    """Effect that waits for an optional gate, then records that it ran."""

    def __init__(self, gate: Optional[threading.Event] = None) -> None:
        self.gate = gate
        self.runs = 0

    def __call__(self):
        if self.gate is not None:
            self.gate.wait(5)
        self.runs += 1
        return False, "done"


def test_effect_runs_now_without_background_effects() -> None:
    effect = _Recorder()

    outcome = deliver(effect, fire_and_forget=True, sink_effects=None, label="t")

    assert outcome == (False, "done")
    assert effect.runs == 1


@pytest.fixture
def no_label_warned_yet(monkeypatch, caplog) -> None:
    """Empty the process-wide record of warned labels; capture WARNINGs."""
    monkeypatch.setattr(runtime, "_INLINE_WARNED_LABELS", set())
    caplog.set_level(logging.WARNING, logger=runtime.LOGGER.name)


def test_inline_fire_and_forget_warns_once_per_label(
    no_label_warned_yet, caplog
) -> None:
    for label in ("v2/a@v1", "v2/a@v1", "v2/b@v1", "v2/a@v1"):
        deliver(_ok, fire_and_forget=True, sink_effects=None, label=label)

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert messages[0].startswith("v2/a@v1: fire_and_forget runs inline")
    assert messages[1].startswith("v2/b@v1: fire_and_forget runs inline")
    assert "'sink_effects'" in messages[0]


def test_no_inline_warning_when_caller_waits_or_a_runner_exists(
    no_label_warned_yet, caplog
) -> None:
    deliver(_ok, fire_and_forget=False, sink_effects=None, label="t")
    with BackgroundEffects() as effects:
        deliver(_ok, fire_and_forget=True, sink_effects=effects, label="t")

    assert caplog.records == []


def test_concurrent_first_inline_calls_warn_exactly_once(
    no_label_warned_yet, caplog
) -> None:
    # given: callers that all reach deliver together, as parallel steps may
    callers = 8
    together = threading.Barrier(callers, timeout=5)

    def call() -> None:
        together.wait()
        deliver(_ok, fire_and_forget=True, sink_effects=None, label="v2/a@v1")

    threads = [threading.Thread(target=call) for _ in range(callers)]

    # when
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    # then: the lock lets exactly one of them log
    assert not any(thread.is_alive() for thread in threads)
    assert len(caplog.records) == 1


def test_effect_runs_now_when_caller_waits() -> None:
    with BackgroundEffects() as effects:
        outcome = deliver(_boom, fire_and_forget=False, sink_effects=effects, label="t")

        assert outcome == (True, "RuntimeError: disk full")
        assert effects.stats().submitted == 0


def test_background_effect_completes_after_drain() -> None:
    effect = _Recorder()
    with BackgroundEffects() as effects:
        outcome = deliver(effect, fire_and_forget=True, sink_effects=effects, label="t")

        assert outcome == (False, BACKGROUND_MESSAGE)
        assert effects.drain(timeout=5)
        assert effect.runs == 1
        assert effects.stats().completed == 1


def test_background_failures_are_counted_kept_logged_and_reported(caplog) -> None:
    caplog.set_level(logging.WARNING)
    reported = []
    with BackgroundEffects(kept_failures=1, on_failure=reported.append) as effects:
        effects.deliver(_boom, label="v2/a@v1")
        effects.deliver(lambda: (True, "HTTP 500"), label="v2/b@v1")
        effects.drain(timeout=5)

        stats = effects.stats()

    assert (stats.failed, stats.completed, stats.pending) == (2, 0, 0)
    assert effects.failures() == (EffectFailure(label="v2/b@v1", message="HTTP 500"),)
    assert [failure.label for failure in reported] == ["v2/a@v1", "v2/b@v1"]
    assert "disk full" in caplog.text and "HTTP 500" in caplog.text


def test_a_raising_failure_callback_is_logged_not_propagated(caplog) -> None:
    def broken_callback(failure: EffectFailure) -> None:
        raise ValueError("callback bug")

    with BackgroundEffects(on_failure=broken_callback) as effects:
        effects.deliver(_boom, label="t")
        effects.drain(timeout=5)

    assert effects.stats().failed == 1
    assert "on_failure of BackgroundEffects raised" in caplog.text


def test_full_queue_runs_the_effect_inline_by_default() -> None:
    gate = threading.Event()
    second = _Recorder()
    with BackgroundEffects(max_pending=1) as effects:
        effects.deliver(_Recorder(gate), label="t")

        outcome = deliver(second, fire_and_forget=True, sink_effects=effects, label="t")
        gate.set()

    assert outcome == (False, "done")
    assert second.runs == 1
    assert effects.stats().ran_inline == 1


def test_full_queue_can_reject_visibly_instead() -> None:
    gate = threading.Event()
    second = _Recorder()
    with BackgroundEffects(max_pending=1, when_full="reject") as effects:
        effects.deliver(_Recorder(gate), label="t")

        outcome = deliver(second, fire_and_forget=True, sink_effects=effects, label="t")
        gate.set()

    assert outcome == (True, FULL_MESSAGE)
    assert second.runs == 0
    assert effects.stats().rejected == 1


@pytest.mark.parametrize(
    "when_full, expected, runs, counted",
    [
        ("run_inline", (False, "done"), 1, "ran_inline"),
        ("reject", (True, FULL_MESSAGE), 0, "rejected"),
    ],
)
def test_direct_delivery_applies_and_counts_the_full_queue_policy(
    when_full: str, expected: tuple, runs: int, counted: str
) -> None:
    # given: a host or sink that calls BackgroundEffects without module deliver
    gate = threading.Event()
    second = _Recorder()
    with BackgroundEffects(max_pending=1, when_full=when_full) as effects:
        admitted = effects.deliver(_Recorder(gate), label="t")

        # when: the queue is full
        outcome = effects.deliver(second, label="t")
        gate.set()

    # then: the object itself applied when_full and counted it once
    stats = effects.stats()
    assert admitted == (False, BACKGROUND_MESSAGE)
    assert outcome == expected
    assert second.runs == runs
    assert (stats.submitted, stats.ran_inline + stats.rejected) == (1, 1)
    assert getattr(stats, counted) == 1


def test_close_drains_pending_then_refuses_new_effects() -> None:
    gate = threading.Event()
    pending = _Recorder(gate)
    effects = BackgroundEffects()
    effects.deliver(pending, label="t")
    threading.Timer(0.05, gate.set).start()

    effects.close()

    assert pending.runs == 1
    with pytest.raises(RuntimeError, match="closed"):
        effects.deliver(_ok, label="t")


def test_timed_out_close_abandons_queued_effects() -> None:
    gate = threading.Event()
    running, queued = _Recorder(gate), _Recorder()
    effects = BackgroundEffects()
    effects.deliver(running, label="t")
    effects.deliver(queued, label="t")

    effects.close(timeout=0.05)
    gate.set()

    assert queued.runs == 0
    assert effects.stats().abandoned == 1


def test_when_full_must_be_known() -> None:
    with pytest.raises(ValueError, match="when_full"):
        BackgroundEffects(when_full="drop")


@pytest.mark.parametrize("limit", ["max_workers", "max_pending", "kept_failures"])
@pytest.mark.parametrize("value", [0, True, 1.5])
def test_limits_must_be_positive_integers(limit: str, value) -> None:
    with pytest.raises(ValueError, match=limit):
        BackgroundEffects(**{limit: value})


def test_cooldown_opens_one_window_at_a_time() -> None:
    clock = ManualClock()
    cooldown = Cooldown(clock)

    first = cooldown.try_start(5)
    clock.advance(4.9)
    inside = cooldown.try_start(5)
    clock.advance(0.1)
    boundary = cooldown.try_start(5)

    assert (first, inside, boundary) == (True, False, True)


def test_cooling_reports_the_window_without_starting_one() -> None:
    clock = ManualClock()
    cooldown = Cooldown(clock)

    before_any = cooldown.cooling(5)
    cooldown.try_start(5)
    clock.advance(4.9)
    inside = cooldown.cooling(5)
    clock.advance(0.1)
    after = cooldown.cooling(5)

    assert (before_any, inside, after) == (False, True, False)
    assert cooldown.try_start(5)


def test_zero_cooldown_never_throttles() -> None:
    cooldown = Cooldown(ManualClock())

    assert all(cooldown.try_start(0) for _ in range(3))


def test_disabled_messages_match_v1() -> None:
    assert disabled_sink_message(by_policy=True) == (
        "Sink was disabled by workflow execution policy"
    )
    assert disabled_sink_message(by_policy=False) == (
        "Sink was disabled by parameter `disable_sink`"
    )
