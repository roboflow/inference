from __future__ import annotations

import heapq
import logging
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class _PendingItem:
    """A single pre-processed item waiting to be batched."""

    __slots__ = ("priority", "seq", "tensor", "meta", "future", "nbytes")

    def __init__(
        self,
        priority: int,
        seq: int,
        tensor: Any,
        meta: Any,
        future: Future,
        nbytes: int = 0,
    ) -> None:
        self.priority = priority
        self.seq = seq
        self.tensor = tensor
        self.meta = meta
        self.future = future
        self.nbytes = nbytes

    def __lt__(self, other: _PendingItem) -> bool:
        # Higher priority first, then FIFO within same priority
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.seq < other.seq


class BatchCollector:
    """Accumulates pre-processed items and dispatches them as batches.

    Thread-safe. Accepts items from multiple concurrent ``submit()`` calls
    via ``add()``, groups them by batch size, delay, and byte budget, then
    dispatches through::

        collate → forward → uncollate → post_process

    Each item gets its own ``Future`` resolved with its individual result.

    A single daemon dispatch thread runs the loop:

    1. Wait for at least one item.
    2. Start delay timer.
    3. Collect until batch is full, delay expires, or byte budget exceeded.
    4. Pop up to ``max_size`` items (highest priority first, FIFO within
       same priority), respecting ``max_bytes``.
    5. Run the pipeline, resolve futures.
    6. Go to 1.

    Used by both ``DirectBackend`` and ``SubprocessBackend``.
    """

    def __init__(
        self,
        *,
        forward_fn: Callable[..., Any],
        collate_fn: Optional[Callable[[List[Tuple[Any, Any]]], Any]] = None,
        uncollate_fn: Optional[Callable[[Any, int], List[Any]]] = None,
        post_process_fn: Optional[Callable[[Any, Any], Any]] = None,
        max_size: int,
        max_delay_s: float,
        max_bytes: int = 0,
    ) -> None:
        self._collate = collate_fn or (lambda items: items)
        self._forward = forward_fn
        self._uncollate = uncollate_fn or (lambda results, count: results)
        self._post_process = post_process_fn or (lambda result, meta: result)
        self._max_size = max_size
        self._max_delay_s = max_delay_s
        self._max_bytes = max_bytes

        self._heap: List[_PendingItem] = []
        self._seq = 0
        self._cond = threading.Condition()
        self._running = True

        # Stats (written by dispatch thread, read by anyone via stats())
        self._batches_dispatched = 0
        self._total_batch_items = 0
        self._total_batch_delay_s = 0.0

        self._thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="batch-collector"
        )
        self._thread.start()
        logger.info(
            "BatchCollector started | max_size=%d | max_delay=%.1fms | max_bytes=%s",
            max_size,
            max_delay_s * 1000,
            f"{max_bytes}" if max_bytes > 0 else "unlimited",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        tensor: Any,
        meta: Any,
        priority: int = 0,
        nbytes: int = 0,
    ) -> Future:
        """Add a pre-processed item. Returns a Future for the post-processed result.

        Args:
            nbytes: Size of the item in bytes (for byte-budget batching).
                When 0, the item doesn't count toward the byte budget.
        """
        future: Future = Future()
        with self._cond:
            if not self._running:
                future.set_exception(RuntimeError("BatchCollector is stopped"))
                return future
            item = _PendingItem(priority, self._seq, tensor, meta, future, nbytes)
            self._seq += 1
            heapq.heappush(self._heap, item)
            self._cond.notify()
        return future

    @property
    def queue_depth(self) -> int:
        with self._cond:
            return len(self._heap)

    def queue_depth_by_priority(self) -> Dict[int, int]:
        with self._cond:
            counts: Dict[int, int] = {}
            for item in self._heap:
                counts[item.priority] = counts.get(item.priority, 0) + 1
            return counts

    def stats(self) -> Dict[str, Any]:
        n = self._batches_dispatched
        return {
            "queue_depth": self.queue_depth,
            "queue_depth_by_priority": self.queue_depth_by_priority(),
            "batches_dispatched": n,
            "avg_batch_fill_pct": (
                (self._total_batch_items / (n * self._max_size) * 100) if n else 0.0
            ),
            "avg_batch_delay_ms": (
                (self._total_batch_delay_s / n * 1000) if n else 0.0
            ),
        }

    def stop(self, drain: bool = True) -> None:
        """Stop the dispatch thread.

        Args:
            drain: If True, process remaining items before stopping.
                   If False, fail remaining futures with RuntimeError.
        """
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self._thread.join()

        # Thread is done — safe to touch the heap without locking.
        remaining = [heapq.heappop(self._heap) for _ in range(len(self._heap))]
        if drain and remaining:
            # Respect max_size to avoid OOM on huge drain
            for i in range(0, len(remaining), self._max_size):
                self._process_batch(remaining[i : i + self._max_size])
        else:
            for item in remaining:
                if not item.future.done():
                    item.future.set_exception(RuntimeError("BatchCollector stopped"))

    # ------------------------------------------------------------------
    # Dispatch loop (runs on daemon thread)
    # ------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        while self._running:
            # Phase 1: block until at least one item arrives
            with self._cond:
                while not self._heap and self._running:
                    self._cond.wait()
                if not self._running and not self._heap:
                    break

            # Phase 2: accumulate until batch full or delay expires
            batch_start = time.monotonic()
            deadline = batch_start + self._max_delay_s

            with self._cond:
                while len(self._heap) < self._max_size and self._running:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(timeout=remaining)

                # Pop items, respecting byte budget
                batch: list[_PendingItem] = []
                batch_bytes = 0
                max_pop = min(len(self._heap), self._max_size)
                for _ in range(max_pop):
                    item = self._heap[0]  # peek
                    if (
                        self._max_bytes > 0
                        and batch
                        and batch_bytes + item.nbytes > self._max_bytes
                    ):
                        break  # this item starts the next batch
                    batch.append(heapq.heappop(self._heap))
                    batch_bytes += item.nbytes

            if batch:
                actual_delay = time.monotonic() - batch_start
                with self._cond:
                    self._batches_dispatched += 1
                    self._total_batch_items += len(batch)
                    self._total_batch_delay_s += actual_delay
                self._process_batch(batch)

    def _process_batch(self, batch: List[_PendingItem]) -> None:
        items = [(item.tensor, item.meta) for item in batch]

        try:
            batched_input = self._collate(items)
            batched_output = self._forward(batched_input)
            raw_outputs = self._uncollate(batched_output, len(items))
        except Exception as e:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)
            return

        for item, raw_output in zip(batch, raw_outputs):
            if item.future.done():
                continue  # cancelled (e.g. client disconnected) — skip post_process
            try:
                result = self._post_process(raw_output, item.meta)
                if not item.future.done():
                    item.future.set_result(result)
            except Exception as e:
                if not item.future.done():
                    item.future.set_exception(e)
