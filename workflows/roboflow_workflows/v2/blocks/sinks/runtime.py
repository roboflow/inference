"""Shared runtime of V2 sink blocks: outcomes, disabling and background effects.

A sink effect is a zero-argument callable returning ``(error_status, message)``,
V1's sink outcome. A sink block runs it now, or hands it to the host-owned
``BackgroundEffects`` resource when its step asks for fire-and-forget::

    host                         block call                   worker thread
    ────                         ──────────                   ─────────────
    with BackgroundEffects() as effects:
      session(resources={"sink_effects": effects})
                                 deliver(effect, ...)
                                   ├ caller waits, or no
                                   │ sink_effects ─▶ run now
                                   └ effects.deliver(effect)
                                       ├ admitted ──────────▶ effect()
                                       │                        ├ ok     → completed
                                       │                        └ failed → failures(),
                                       │                                   on_failure
                                       └ full: when_full="run_inline" → run now
                                               when_full="reject"     → error_status=True
    # leaving the block drains, then closes

``BackgroundEffects.deliver`` decides and counts every outcome itself, so a
sink, or a host, that calls it directly gets the configured ``when_full``.

The engine never closes block resources, so the host that creates
``BackgroundEffects`` also drains and closes it. An active-run host nests
three blocks; they exit in reverse order::

    with BackgroundEffects(max_pending=64) as effects:
        with (
            plan.create_session(
                resources={"webhook_client": client, "sink_effects": effects}
            ) as session,
            session.start(handlers=handlers) as run,
        ):
            run.wait()
    # exit 1, run:      stops admission, finishes admitted pulses and handlers
    # exit 2, session:  closes engine-owned state; caller resources stay open
    # exit 3, effects:  refuses new effects, drains pending ones, stops threads

The run ends before the effects close, so its last sink call can still
deliver. The run does not wait for an admitted effect. Leaving
``with effects`` waits, without a limit, until every pending effect has
finished: up to ``max_pending`` effects, each limited only by its own
client's timeout (e.g. ``HttpWebhookClient`` passes the step's
``request_timeout``). A host that needs a bounded shutdown calls
``effects.close(timeout=...)`` instead; effects still queued then are
cancelled and counted as ``abandoned``.

There is no default instance: without one, fire-and-forget sinks run
synchronously, as V1 did without a background scheduler or thread pool, and
the first such call of each block type logs one WARNING. With
``max_workers > 1`` the effects of one step may finish out of order, as in V1.
"""

import logging
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Literal, Optional, Set, Tuple

from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Output

LOGGER = logging.getLogger(__name__)

# Labels whose inline fire-and-forget fallback was logged, once per process.
_INLINE_WARNED_LABELS: Set[str] = set()
_INLINE_WARNING_LOCK = threading.Lock()

BACKGROUND_MESSAGE = "Notification sent in the background task"
FULL_MESSAGE = "background effects are full; the effect was not run"

EffectOutcome = Tuple[bool, str]
Effect = Callable[[], EffectOutcome]
WhenFull = Literal["run_inline", "reject"]

ERROR_STATUS_OUTPUT = Output(
    BOOLEAN_KIND,
    description=(
        "True when the effect failed, or full background effects refused it "
        "(when_full='reject'). False when it succeeded, was skipped, or was "
        "handed to background effects. A host permission error stops the run."
    ),
)
MESSAGE_OUTPUT = Output(STRING_KIND, description="Human-readable outcome.")


def disabled_sink_message(*, by_policy: bool) -> str:
    """Return V1's message for a disabled sink.

    Args:
        by_policy: True when the host disabled every sink (``disable_sinks``),
            False when the step's own ``disable_sink`` parameter did.

    Returns:
        The message text.
    """
    if by_policy:
        return "Sink was disabled by workflow execution policy"

    return "Sink was disabled by parameter `disable_sink`"


def run_effect(effect: Effect) -> EffectOutcome:
    """Run one effect now and turn an exception into a failed outcome.

    Args:
        effect: The effect to run.

    Returns:
        The effect's ``(error_status, message)``; ``(True, "<Type>: <text>")``
        when it raises.
    """
    try:
        error_status, message = effect()
    except Exception as error:
        return True, describe_error(error)

    return bool(error_status), str(message)


def describe_error(error: BaseException) -> str:
    """Return ``"<ExceptionType>: <text>"`` for an outcome message.

    Args:
        error: The raised exception.

    Returns:
        The description.
    """
    description = f"{type(error).__name__}: {error}"

    return description


@dataclass(frozen=True)
class EffectFailure:
    """One background effect that failed.

    Args:
        label: Who submitted it, normally the block type.
        message: The failed outcome's message.
    """

    label: str
    message: str


@dataclass(frozen=True)
class EffectStats:
    """Counts of a ``BackgroundEffects`` instance.

    Args:
        submitted: Effects admitted to run in the background.
        completed: Admitted effects that succeeded.
        failed: Admitted effects that raised or returned ``error_status``.
        ran_inline: Effects run by the caller because the queue was full.
        rejected: Effects refused because the queue was full.
        abandoned: Admitted effects cancelled by a timed-out ``close``.
        pending: Admitted effects not finished yet.
    """

    submitted: int
    completed: int
    failed: int
    ran_inline: int
    rejected: int
    abandoned: int
    pending: int


class BackgroundEffects:
    """Host-owned, bounded runner of sink effects whose caller does not wait.

    Pass it to sessions as the ``sink_effects`` resource; one instance may
    serve many steps and sessions. Effects arrive through ``deliver``. At most
    ``max_pending`` effects are queued or running. When full, ``when_full``
    decides: ``run_inline`` makes the
    caller run the effect now (nothing is lost; the caller waits) and
    ``reject`` refuses it (the step reports ``error_status=True``). Every
    background failure is counted, logged at WARNING, kept (the newest
    ``kept_failures``) and passed to ``on_failure``. The host drains and
    closes the instance, e.g. with ``with``; the engine never does.

    Args:
        max_workers: Threads running effects.
        max_pending: Most admitted effects not finished yet.
        when_full: ``run_inline`` or ``reject``.
        kept_failures: Most recent failures kept for ``failures()``.
        on_failure: Called with each background failure, on a worker
            thread. An exception it raises is logged, not propagated.

    Raises:
        ValueError: When a limit is lower than 1 or ``when_full`` is unknown.
    """

    def __init__(
        self,
        *,
        max_workers: int = 1,
        max_pending: int = 64,
        when_full: WhenFull = "run_inline",
        kept_failures: int = 100,
        on_failure: Optional[Callable[[EffectFailure], None]] = None,
    ) -> None:
        for name, value in (
            ("max_workers", max_workers),
            ("max_pending", max_pending),
            ("kept_failures", kept_failures),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
        if when_full not in ("run_inline", "reject"):
            raise ValueError(
                f"when_full must be 'run_inline' or 'reject', got {when_full!r}"
            )

        self._when_full = when_full
        self._max_pending = max_pending
        self._on_failure = on_failure
        self._condition = threading.Condition()
        self._failures: Deque[EffectFailure] = deque(maxlen=kept_failures)
        self._counts: Dict[str, int] = dict.fromkeys(
            ("submitted", "completed", "failed", "ran_inline", "rejected", "abandoned"),
            0,
        )
        self._futures: Set[Future] = set()
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="v2-sink-effects"
        )

    def deliver(self, effect: Effect, *, label: str) -> EffectOutcome:
        """Run one effect in the background, or apply ``when_full`` when full.

        Args:
            effect: Zero-argument callable returning ``(error_status, message)``.
            label: Who delivers it, reported with a failure.

        Returns:
            ``(False, BACKGROUND_MESSAGE)`` when admitted. When
            ``max_pending`` effects are already queued or running: the
            effect's own outcome (``run_inline``, run now by the caller) or
            ``(True, FULL_MESSAGE)`` (``reject``).

        Raises:
            RuntimeError: After ``close``; the host closed it too early.
        """
        with self._condition:
            if self._closed:
                raise RuntimeError(
                    "BackgroundEffects is closed; close it only after its "
                    "sessions stop running"
                )
            admitted = len(self._futures) < self._max_pending
            if admitted:
                self._counts["submitted"] += 1
                # Under the lock, so close() cannot shut the executor in between.
                future = self._executor.submit(self._run, effect, label)
                self._futures.add(future)
                future.add_done_callback(self._forget)
            elif self._when_full == "run_inline":
                self._counts["ran_inline"] += 1
            else:
                self._counts["rejected"] += 1

        if admitted:
            return False, BACKGROUND_MESSAGE

        if self._when_full == "reject":
            return True, FULL_MESSAGE

        outcome = run_effect(effect)

        return outcome

    def drain(self, timeout: Optional[float] = None) -> bool:
        """Wait until no admitted effect is pending.

        Args:
            timeout: Most seconds to wait; ``None`` waits until idle.

        Returns:
            True when idle, False when the timeout passed first.
        """
        with self._condition:
            idle = self._condition.wait_for(lambda: not self._futures, timeout)

        return idle

    def close(self, timeout: Optional[float] = None) -> None:
        """Stop admission, wait for pending effects, then stop the threads.

        Args:
            timeout: Most seconds to wait; ``None`` waits until idle. Effects
                still queued afterwards are cancelled and counted as
                abandoned; running ones finish on their threads.
        """
        with self._condition:
            self._closed = True
        if not self.drain(timeout):
            with self._condition:
                # cancel() runs _forget, which removes from the set: copy it.
                abandoned = [
                    future for future in list(self._futures) if future.cancel()
                ]
                self._counts["abandoned"] += len(abandoned)

        self._executor.shutdown(wait=timeout is None)

    def stats(self) -> EffectStats:
        """Return the current counts."""
        with self._condition:
            stats = EffectStats(pending=len(self._futures), **self._counts)

        return stats

    def failures(self) -> Tuple[EffectFailure, ...]:
        """Return the kept failures, oldest first."""
        with self._condition:
            failures = tuple(self._failures)

        return failures

    def __enter__(self) -> "BackgroundEffects":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _run(self, effect: Effect, label: str) -> None:
        error_status, message = run_effect(effect)
        if not error_status:
            with self._condition:
                self._counts["completed"] += 1
            return

        LOGGER.warning("Background sink effect of %s failed: %s", label, message)
        failure = EffectFailure(label=label, message=message)
        with self._condition:
            self._counts["failed"] += 1
            self._failures.append(failure)
        if self._on_failure is not None:
            try:
                self._on_failure(failure)
            except Exception:
                LOGGER.exception("on_failure of BackgroundEffects raised")

    def _forget(self, future: Future) -> None:
        with self._condition:
            self._futures.discard(future)
            self._condition.notify_all()


def deliver(
    effect: Effect,
    *,
    fire_and_forget: bool,
    sink_effects: Optional[BackgroundEffects],
    label: str,
) -> EffectOutcome:
    """Run an effect now, or hand it to background effects.

    Fire-and-forget without ``sink_effects`` runs now, as in V1. The first
    such call of each ``label`` in the process logs one WARNING, also when
    several threads make it at once; later calls are silent.

    Args:
        effect: The sink's effect.
        fire_and_forget: The step asks not to wait for the effect.
        sink_effects: The host's background runner, if any.
        label: Block type, reported with background failures and the
            inline WARNING.

    Returns:
        The effect's outcome when run now (the caller waits, or there is no
        ``sink_effects``); otherwise ``BackgroundEffects.deliver``'s outcome.

    Raises:
        RuntimeError: When ``sink_effects`` is already closed.
    """
    if not fire_and_forget or sink_effects is None:
        if fire_and_forget:
            _warn_inline_once(label)
        outcome = run_effect(effect)
        return outcome

    outcome = sink_effects.deliver(effect, label=label)

    return outcome


def _warn_inline_once(label: str) -> None:
    with _INLINE_WARNING_LOCK:
        first = label not in _INLINE_WARNED_LABELS
        _INLINE_WARNED_LABELS.add(label)
    if first:
        LOGGER.warning(
            "%s: fire_and_forget runs inline because the host passed no "
            "'sink_effects' resource; pass a BackgroundEffects to send in the "
            "background. Logged once per block type.",
            label,
        )


class Cooldown:
    """V1's per-step cooldown: one shared window for every call of a step.

    Args:
        clock: Seconds now; monotonic by default.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._last: Optional[float] = None

    def cooling(self, cooldown_seconds: float) -> bool:
        """Tell whether the current window still runs, without starting one.

        Lets a sink skip preparing an effect that ``try_start`` would refuse.

        Args:
            cooldown_seconds: Least seconds between two started effects.

        Returns:
            True while cooling down.
        """
        with self._lock:
            active = (
                self._last is not None and self._clock() - self._last < cooldown_seconds
            )

        return active

    def try_start(self, cooldown_seconds: float) -> bool:
        """Start a new window when the previous one has passed.

        Args:
            cooldown_seconds: Least seconds between two started effects.

        Returns:
            True when this call may run its effect; False while cooling down.
        """
        with self._lock:
            now = self._clock()
            if self._last is not None and now - self._last < cooldown_seconds:
                return False

            self._last = now

        return True
