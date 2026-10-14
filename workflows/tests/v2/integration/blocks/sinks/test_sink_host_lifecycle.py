"""Host-owned ``BackgroundEffects`` around an active run of a webhook sink.

The host nests three blocks; they exit in reverse order::

    with BackgroundEffects(...) as effects:                       # exit 3
        with (
            plan.create_session(resources={..., "sink_effects": effects})
                as session,                                       # exit 2
            session.start(handlers=...) as run,                   # exit 1
        ):
            run.wait()

    scripted source ─▶ roboflow_core/webhook_sink@v1 ─▶ group "out" (error_status, message)
                            └ webhook_client: in-process recording fake

Order is proved with events, not clocks: the client holds every request
until the test sets ``release``. ``_released`` sets it on every exit path,
so a failed assertion cannot leave a context waiting for a held request.
``WAIT`` only bounds a deadlock; no test asserts a duration.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import pytest
from roboflow_workflows.v2.blocks.sinks import runtime
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    FULL_MESSAGE,
    BackgroundEffects,
)
from roboflow_workflows.v2.blocks.sinks.webhook_sink.v1 import SENT_MESSAGE, WebhookSinkBlock
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow

from tests.v2.support.runtime.active import (
    WAIT,
    Collector,
    Log,
    Scripted,
    active,
    emit,
    group,
    source,
)
from tests.v2.support.sinks.loopback import RecordingWebhookClient

PULSES = 3
HOOK = WebhookSinkBlock.type
PLAN = compile_workflow(
    active(
        [source("frames")],
        [
            {
                "type": HOOK,
                "name": "hook",
                "url": "http://hooks.test/frames",
                "method": "POST",
                "json_payload": {"value": "$sources.frames.value"},
                "cooldown_seconds": 0,
            }
        ],
        [
            group(
                "out",
                "$sources.frames.value",
                error_status="$steps.hook.error_status",
                message="$steps.hook.message",
            )
        ],
    ),
    catalogue=Catalogue([WebhookSinkBlock], sources=[Scripted]),
)


class _EnteringClient(RecordingWebhookClient):
    """Recording client that also signals when a send has started."""

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.entered = threading.Event()

    def __call__(self, request: Any) -> None:
        self.entered.set()
        super().__call__(request)


@contextmanager
def _released(gate: threading.Event) -> Iterator[None]:
    try:
        yield
    finally:
        gate.set()


def _resources(
    client: RecordingWebhookClient, effects: Optional[BackgroundEffects] = None
) -> Dict[str, Any]:
    started = threading.Event()
    # Nothing to hold before the run handle exists: sources may open at once.
    started.set()
    resources = {
        "feeds": {"frames": [emit(value=float(n)) for n in range(PULSES)]},
        "log": Log(),
        "started": started,
        "webhook_client": client,
    }
    if effects is not None:
        resources["sink_effects"] = effects

    return resources


def _outcomes(collected: Collector) -> list:
    outcomes = [(row["error_status"], row["message"]) for row in collected.rows("out")]

    return outcomes


@pytest.fixture(autouse=True)
def _effect_threads_stop() -> Iterator[None]:
    """Leaving ``with effects`` stops its worker threads in every test."""
    before = set(threading.enumerate())
    yield
    leftover = [
        thread
        for thread in set(threading.enumerate()) - before
        if thread.name.startswith("v2-sink-effects") and thread.is_alive()
    ]
    assert leftover == [], leftover


def test_active_run_finishes_before_its_background_requests_are_released() -> None:
    release = threading.Event()
    client = RecordingWebhookClient(gate=release)
    collected = Collector()

    with BackgroundEffects(max_pending=8) as effects, _released(release):
        with (
            PLAN.create_session(resources=_resources(client, effects)) as session,
            session.start(handlers=collected.handlers("out")) as run,
        ):
            # when: the run finishes while the client holds every request
            finished = run.wait(WAIT)
            requests_during_run = len(client.requests)
            during_run = effects.stats()

        # the session closed, but it did not drain or close the host's runner
        after_session = effects.stats()
        extra = effects.deliver(lambda: (False, "host"), label="host")

    # then: the run never waited for a request
    assert finished
    assert requests_during_run == 0
    assert during_run.pending == PULSES
    assert _outcomes(collected) == [(False, BACKGROUND_MESSAGE)] * PULSES
    assert after_session.pending == PULSES
    assert extra == (False, BACKGROUND_MESSAGE)
    # and leaving the effects block drained the 3 requests and the extra effect
    stats = effects.stats()
    assert len(client.requests) == PULSES
    assert (stats.submitted, stats.completed, stats.pending) == (4, 4, 0)


def test_without_sink_effects_fire_and_forget_holds_the_run(
    monkeypatch, caplog
) -> None:
    # given: no runner, so every send runs inline in the step
    monkeypatch.setattr(runtime, "_INLINE_WARNED_LABELS", set())
    caplog.set_level(logging.WARNING, logger=runtime.LOGGER.name)
    release = threading.Event()
    client = _EnteringClient(gate=release)
    collected = Collector()

    with (
        PLAN.create_session(resources=_resources(client)) as session,
        session.start(handlers=collected.handlers("out")) as run,
        _released(release),
    ):
        # when: the first send has started and is held
        entered = client.entered.wait(WAIT)
        outcomes_while_held = _outcomes(collected)
        done_while_held = run.wait(timeout=0)
        release.set()
        finished = run.wait(WAIT)

    # then: the run could not finish, nor deliver, until the send returned
    assert entered
    assert outcomes_while_held == []
    assert done_while_held is False
    assert finished
    assert _outcomes(collected) == [(False, SENT_MESSAGE)] * PULSES
    assert len(client.requests) == PULSES
    # and one WARNING names the block type, not one per send
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1
    assert warnings[0].startswith(f"{HOOK}: fire_and_forget runs inline")


def test_background_failures_reach_the_host_not_the_run() -> None:
    client = RecordingWebhookClient(error=ConnectionError("endpoint down"))
    received = []
    collected = Collector()

    with BackgroundEffects(on_failure=received.append) as effects:
        with (
            PLAN.create_session(resources=_resources(client, effects)) as session,
            session.start(handlers=collected.handlers("out")) as run,
        ):
            finished = run.wait(WAIT)

    stats = effects.stats()
    assert finished
    assert _outcomes(collected) == [(False, BACKGROUND_MESSAGE)] * PULSES
    assert (stats.failed, stats.completed) == (PULSES, 0)
    assert [failure.label for failure in effects.failures()] == [HOOK] * PULSES
    assert all("endpoint down" in failure.message for failure in effects.failures())
    assert received == list(effects.failures())


def test_full_runner_rejects_visibly_inside_an_active_run() -> None:
    # given: one slot, held by the first request for the whole run
    release = threading.Event()
    client = RecordingWebhookClient(gate=release)
    collected = Collector()

    with (
        BackgroundEffects(max_pending=1, when_full="reject") as effects,
        _released(release),
    ):
        with (
            PLAN.create_session(resources=_resources(client, effects)) as session,
            session.start(handlers=collected.handlers("out")) as run,
        ):
            finished = run.wait(WAIT)

    # then: pulses 2 and 3 report the refusal; only the first request is sent
    stats = effects.stats()
    assert finished
    assert _outcomes(collected) == [(False, BACKGROUND_MESSAGE)] + [
        (True, FULL_MESSAGE)
    ] * (PULSES - 1)
    assert (stats.submitted, stats.rejected, stats.completed) == (1, PULSES - 1, 1)
    assert len(client.requests) == 1


def test_a_processing_reset_keeps_delivering_to_the_host_runner() -> None:
    client = RecordingWebhookClient()

    with BackgroundEffects() as effects:
        with PLAN.create_session(resources=_resources(client, effects)) as session:
            with session.start(handlers=Collector().handlers("out")) as run:
                assert run.wait(WAIT)
            hook_before = session.instances[("hook",)]

            # when: the idle session resets its processing, keeping resources
            receipt = session.apply_update(session.prepare_update(PLAN, reset=True))
            with session.start(handlers=Collector().handlers("out")) as run:
                assert run.wait(WAIT)
            hook_after = session.instances[("hook",)]

    # then: a new webhook instance delivered to the same, still open runner
    stats = effects.stats()
    assert receipt.cleanup_failures == ()
    assert session.processing_version == 1
    assert hook_after is not hook_before
    assert (stats.submitted, stats.completed) == (2 * PULSES, 2 * PULSES)
    assert len(client.requests) == 2 * PULSES
