"""Process-wide resource caps for video clips (`Budget`).

One `Budget` (``default_budget()``) is shared by every ring and sink in the
process. It enforces, at admission time:

* ``WORKFLOWS_VIDEO_CLIP_BUDGET_BYTES`` (default 1 GiB) over ring packets
  (including packets held by captures), ingest queues, artifacts and in-flight
  uploads, plus the new capture's estimated size;
* ``WORKFLOWS_VIDEO_CLIP_MAX_PENDING_CAPTURES`` (default 16) admitted captures
  that have not reached a terminal status;
* ``WORKFLOWS_VIDEO_CLIP_MAX_CLIP_BYTES`` (default 512 MiB) on the estimated
  (and later the actual) clip size;

and it holds the process-wide upload slots,
``WORKFLOWS_VIDEO_CLIP_MAX_UPLOAD_CONCURRENCY`` (default 2).
"""

import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Protocol

from roboflow_workflows.core_steps.common.video_clips import (
    WORKFLOWS_VIDEO_CLIP_BUDGET_BYTES,
    WORKFLOWS_VIDEO_CLIP_MAX_CLIP_BYTES,
    WORKFLOWS_VIDEO_CLIP_MAX_PENDING_CAPTURES,
    WORKFLOWS_VIDEO_CLIP_MAX_UPLOAD_CONCURRENCY,
)

REJECT_BUDGET_EXCEEDED = "budget_exceeded"
REJECT_TOO_MANY_PENDING_CAPTURES = "too_many_pending_captures"
REJECT_CLIP_TOO_LARGE = "clip_too_large"


class ByteSource(Protocol):
    """Anything that holds clip bytes the budget must count (rings)."""

    def budget_bytes(self) -> int:
        """Return the bytes currently held."""


@dataclass(frozen=True)
class BudgetUsage:
    """A snapshot of what the budget currently counts.

    Attributes:
        source_bytes: Bytes held by registered sources (rings and queues).
        ticket_bytes: Bytes of admitted artifacts and in-flight uploads.
        pending_captures: Admitted captures not yet terminal.
    """

    source_bytes: int
    ticket_bytes: int
    pending_captures: int

    @property
    def total_bytes(self) -> int:
        """All bytes counted against the byte cap."""
        return self.source_bytes + self.ticket_bytes


class AdmissionTicket:
    """One admitted capture's claim on the budget until its terminal status.

    Args:
        budget: The budget that issued the ticket.
    """

    def __init__(self, budget: "Budget"):
        self._budget = budget
        self._bytes = 0
        self._released = False

    @property
    def bytes(self) -> int:
        """Artifact or in-flight upload bytes counted for this capture."""
        return self._bytes

    @property
    def released(self) -> bool:
        """True once `release()` was called."""
        return self._released

    def set_bytes(self, size_bytes: int) -> None:
        """Count the capture's artifact (and its upload) against the byte cap.

        Args:
            size_bytes: Artifact size in bytes.
        """
        with self._budget._lock:
            if not self._released:
                self._bytes = max(0, int(size_bytes))

    def release(self) -> None:
        """Return the claim to the budget. Idempotent."""
        self._budget._release_ticket(self)


@dataclass(frozen=True)
class AdmissionDecision:
    """Outcome of `Budget.admit`.

    Attributes:
        reason: ``None`` when admitted, else the reason without the
            ``"rejected:"`` prefix.
        ticket: The ticket of an admitted capture.
    """

    reason: Optional[str]
    ticket: Optional[AdmissionTicket]

    @property
    def admitted(self) -> bool:
        """True when the capture was admitted."""
        return self.reason is None


class Budget:
    """Resource caps shared by every ring, sink and uploader in a process.

    Args:
        budget_bytes: Byte cap over rings, queues, artifacts and uploads.
        max_pending_captures: Maximum admitted, not yet terminal captures.
        max_upload_concurrency: Process-wide concurrent uploads.
        max_clip_bytes: Maximum size of one clip.

    Raises:
        ValueError: If any cap is not positive.
    """

    def __init__(
        self,
        *,
        budget_bytes: int = WORKFLOWS_VIDEO_CLIP_BUDGET_BYTES,
        max_pending_captures: int = WORKFLOWS_VIDEO_CLIP_MAX_PENDING_CAPTURES,
        max_upload_concurrency: int = WORKFLOWS_VIDEO_CLIP_MAX_UPLOAD_CONCURRENCY,
        max_clip_bytes: int = WORKFLOWS_VIDEO_CLIP_MAX_CLIP_BYTES,
    ):
        for name, value in (
            ("budget_bytes", budget_bytes),
            ("max_pending_captures", max_pending_captures),
            ("max_upload_concurrency", max_upload_concurrency),
            ("max_clip_bytes", max_clip_bytes),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        self.budget_bytes = budget_bytes
        self.max_pending_captures = max_pending_captures
        self.max_upload_concurrency = max_upload_concurrency
        self.max_clip_bytes = max_clip_bytes
        self._lock = threading.RLock()
        self._sources: "weakref.WeakSet[Any]" = weakref.WeakSet()
        self._tickets: "set[AdmissionTicket]" = set()
        self._upload_slots = threading.BoundedSemaphore(max_upload_concurrency)

    def register_source(self, source: ByteSource) -> None:
        """Count a ring's bytes against the cap (held weakly).

        Args:
            source: Object exposing ``budget_bytes() -> int``.
        """
        with self._lock:
            self._sources.add(source)

    def unregister_source(self, source: ByteSource) -> None:
        """Stop counting a source. Unknown sources are ignored.

        Args:
            source: A previously registered source.
        """
        with self._lock:
            self._sources.discard(source)

    def usage(self) -> BudgetUsage:
        """Return what the budget counts right now.

        Returns:
            The usage snapshot.
        """
        with self._lock:
            source_bytes = sum(source.budget_bytes() for source in list(self._sources))
            ticket_bytes = sum(ticket.bytes for ticket in self._tickets)
            usage = BudgetUsage(
                source_bytes=source_bytes,
                ticket_bytes=ticket_bytes,
                pending_captures=len(self._tickets),
            )

        return usage

    def admit(self, capture: Any) -> AdmissionDecision:
        """Admit or reject a pending capture.

        A rejected capture is released here, so its packets are freed at once.

        Args:
            capture: A `PendingCapture` (needs ``estimated_size_bytes`` and
                ``release()``).

        Returns:
            The decision; carries a ticket when admitted.
        """
        estimated_size = max(0, int(getattr(capture, "estimated_size_bytes", 0)))
        reason = None
        ticket = None
        with self._lock:
            usage = self.usage()
            if estimated_size > self.max_clip_bytes:
                reason = REJECT_CLIP_TOO_LARGE
            elif usage.pending_captures >= self.max_pending_captures:
                reason = REJECT_TOO_MANY_PENDING_CAPTURES
            elif usage.total_bytes + estimated_size > self.budget_bytes:
                reason = REJECT_BUDGET_EXCEEDED
            else:
                ticket = AdmissionTicket(self)
                self._tickets.add(ticket)

        if reason is not None:
            capture.release()
        decision = AdmissionDecision(reason=reason, ticket=ticket)

        return decision

    @contextmanager
    def upload_slot(self) -> Iterator[None]:
        """Hold one of the process-wide upload slots for the duration.

        Yields:
            Nothing; the slot is returned on exit.
        """
        self._upload_slots.acquire()
        try:
            yield None
        finally:
            self._upload_slots.release()

    def _release_ticket(self, ticket: AdmissionTicket) -> None:
        with self._lock:
            ticket._released = True
            ticket._bytes = 0
            self._tickets.discard(ticket)


_DEFAULT_BUDGET: Optional[Budget] = None
_DEFAULT_BUDGET_LOCK = threading.Lock()


def default_budget() -> Budget:
    """Return the process-wide budget, creating it on first use.

    Returns:
        The shared `Budget` configured from the environment.
    """
    global _DEFAULT_BUDGET
    with _DEFAULT_BUDGET_LOCK:
        if _DEFAULT_BUDGET is None:
            _DEFAULT_BUDGET = Budget()
        budget = _DEFAULT_BUDGET

    return budget
