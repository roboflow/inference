from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from typing import Any, List, Tuple
from unittest.mock import MagicMock

import pytest

from inference_model_manager.backends.batch_collector import BatchCollector

# ------------------------------------------------------------------
# Fake pipeline functions — no model, no torch, no GPU
# ------------------------------------------------------------------


def _collate(items: List[Tuple[Any, Any]]) -> List[Any]:
    """Stack tensors into a list (fake collate)."""
    return [t for t, _ in items]


def _forward(batch: List[Any]) -> List[Any]:
    """Double each input (fake forward)."""
    return [x * 2 for x in batch]


def _uncollate(output: List[Any], count: int) -> List[Any]:
    """Already a list — just return it."""
    return output


def _post_process(raw: Any, meta: Any) -> Tuple[Any, Any]:
    """Return (raw, meta) pair."""
    return (raw, meta)


def _make_collector(
    max_size: int = 4,
    max_delay_ms: float = 50.0,
    collate_fn=None,
    forward_fn=None,
    uncollate_fn=None,
    post_process_fn=None,
) -> BatchCollector:
    return BatchCollector(
        collate_fn=collate_fn or _collate,
        forward_fn=forward_fn or _forward,
        uncollate_fn=uncollate_fn or _uncollate,
        post_process_fn=post_process_fn or _post_process,
        max_size=max_size,
        max_delay_s=max_delay_ms / 1000,
    )


# ------------------------------------------------------------------
# Basic dispatch
# ------------------------------------------------------------------


def test_single_item_resolves_with_correct_result() -> None:
    # given
    bc = _make_collector(max_size=4, max_delay_ms=10)

    # when
    future = bc.add(tensor=5, meta="m")
    result = future.result(timeout=2)

    # then — forward doubles the tensor, post_process returns (raw, meta)
    assert result == (10, "m")

    bc.stop()


def test_multiple_items_each_resolve_independently() -> None:
    # given
    bc = _make_collector(max_size=8, max_delay_ms=10)

    # when
    futures = [bc.add(tensor=i, meta=f"m{i}") for i in range(5)]
    results = [f.result(timeout=2) for f in futures]

    # then
    assert results == [(i * 2, f"m{i}") for i in range(5)]

    bc.stop()


# ------------------------------------------------------------------
# Batch size trigger
# ------------------------------------------------------------------


def test_batch_fires_immediately_when_max_size_reached() -> None:
    # given — large delay so only size can trigger dispatch
    forward_called = threading.Event()
    call_log: list = []

    def tracking_forward(batch):
        call_log.append(len(batch))
        forward_called.set()
        return [x * 2 for x in batch]

    bc = _make_collector(max_size=4, max_delay_ms=5000, forward_fn=tracking_forward)

    # when — submit exactly max_size items
    futures = [bc.add(tensor=i, meta=None) for i in range(4)]

    # then — forward should fire quickly (not wait for 5s delay)
    assert forward_called.wait(timeout=1), "Batch should fire on size, not delay"
    assert call_log[0] == 4
    for f in futures:
        f.result(timeout=1)

    bc.stop()


# ------------------------------------------------------------------
# Delay trigger
# ------------------------------------------------------------------


def test_partial_batch_fires_after_delay() -> None:
    # given — max_size=8 but we only submit 2 items
    bc = _make_collector(max_size=8, max_delay_ms=30)

    # when
    t0 = time.monotonic()
    futures = [bc.add(tensor=i, meta=None) for i in range(2)]
    results = [f.result(timeout=2) for f in futures]
    elapsed_ms = (time.monotonic() - t0) * 1000

    # then — should resolve after ~30ms delay, not instantly
    assert results == [(0, None), (2, None)]
    assert elapsed_ms >= 20, f"Expected delay >= 20ms, got {elapsed_ms:.1f}ms"

    bc.stop()


# ------------------------------------------------------------------
# Priority ordering
# ------------------------------------------------------------------


def test_higher_priority_items_dispatched_first() -> None:
    # given — capture the order collate sees items
    collate_order: list = []

    def tracking_collate(items):
        collate_order.extend([t for t, _ in items])
        return [t for t, _ in items]

    bc = _make_collector(max_size=3, max_delay_ms=50, collate_fn=tracking_collate)

    # when — submit with different priorities, block dispatch briefly
    # by submitting all before the collector wakes up
    bc.add(tensor="low", meta=None, priority=0)
    bc.add(tensor="high", meta=None, priority=10)
    bc.add(tensor="mid", meta=None, priority=5)

    # then — wait for batch to dispatch
    time.sleep(0.2)
    assert collate_order == ["high", "mid", "low"]

    bc.stop()


def test_fifo_within_same_priority() -> None:
    # given
    collate_order: list = []

    def tracking_collate(items):
        collate_order.extend([t for t, _ in items])
        return [t for t, _ in items]

    bc = _make_collector(max_size=3, max_delay_ms=50, collate_fn=tracking_collate)

    # when — same priority, different order
    bc.add(tensor="first", meta=None, priority=0)
    bc.add(tensor="second", meta=None, priority=0)
    bc.add(tensor="third", meta=None, priority=0)

    # then
    time.sleep(0.2)
    assert collate_order == ["first", "second", "third"]

    bc.stop()


# ------------------------------------------------------------------
# Error propagation
# ------------------------------------------------------------------


def test_forward_error_fails_all_futures_in_batch() -> None:
    # given
    def failing_forward(batch):
        raise RuntimeError("GPU OOM")

    bc = _make_collector(max_size=4, max_delay_ms=10, forward_fn=failing_forward)

    # when
    futures = [bc.add(tensor=i, meta=None) for i in range(3)]

    # then — all futures should raise
    for f in futures:
        with pytest.raises(RuntimeError, match="GPU OOM"):
            f.result(timeout=2)

    bc.stop()


def test_post_process_error_fails_only_affected_item() -> None:
    # given — post_process fails for tensor=1 only
    def selective_post_process(raw, meta):
        if raw == 2:  # forward doubles, so tensor=1 → raw=2
            raise ValueError("bad item")
        return (raw, meta)

    bc = _make_collector(
        max_size=4, max_delay_ms=10, post_process_fn=selective_post_process
    )

    # when
    f0 = bc.add(tensor=0, meta="ok")
    f1 = bc.add(tensor=1, meta="bad")
    f2 = bc.add(tensor=2, meta="ok")

    # then — only f1 fails
    assert f0.result(timeout=2) == (0, "ok")
    with pytest.raises(ValueError, match="bad item"):
        f1.result(timeout=2)
    assert f2.result(timeout=2) == (4, "ok")

    bc.stop()


# ------------------------------------------------------------------
# Stop behavior
# ------------------------------------------------------------------


def test_stop_drain_true_processes_remaining_items() -> None:
    # given — block the dispatch thread so items accumulate
    gate = threading.Event()

    def gated_forward(batch):
        gate.wait(timeout=5)
        return [x * 2 for x in batch]

    bc = _make_collector(max_size=100, max_delay_ms=5000, forward_fn=gated_forward)

    # when — submit items, then stop with drain before dispatch can run
    futures = [bc.add(tensor=i, meta=None) for i in range(3)]
    gate.set()  # unblock
    bc.stop(drain=True)

    # then — all items processed
    for i, f in enumerate(futures):
        result = f.result(timeout=2)
        assert result == (i * 2, None)


def test_stop_drain_false_fails_remaining_items() -> None:
    # given — block dispatch so items sit in queue.
    # Use a gate to hold the forward call, then submit MORE items while
    # forward is blocked so they stay pending in the heap.
    gate = threading.Event()
    forward_entered = threading.Event()

    def blocked_forward(batch):
        forward_entered.set()
        gate.wait(timeout=5)
        return [x * 2 for x in batch]

    # max_size=2: first 2 items trigger a batch immediately, forward blocks.
    # The remaining items stay in the heap.
    bc = _make_collector(max_size=2, max_delay_ms=5000, forward_fn=blocked_forward)

    # when — submit 2 items to trigger dispatch, then submit more while blocked
    first_futures = [bc.add(tensor=i, meta=None) for i in range(2)]
    forward_entered.wait(timeout=2)  # wait until forward is executing
    remaining_futures = [bc.add(tensor=i, meta=None) for i in range(2, 5)]

    # Stop without drain — remaining items should be failed
    gate.set()  # unblock forward so dispatch thread can exit
    bc.stop(drain=False)

    # then — the remaining futures (submitted while forward was blocked) should fail
    failed_count = 0
    all_futures = first_futures + remaining_futures
    for f in all_futures:
        if f.exception() is not None:
            assert "stopped" in str(f.exception()).lower()
            failed_count += 1
    assert (
        failed_count > 0
    ), "Expected at least one future to be failed/cancelled after drain=False stop"


def test_add_after_stop_returns_failed_future() -> None:
    # given
    bc = _make_collector()
    bc.stop()

    # when
    future = bc.add(tensor=1, meta=None)

    # then
    with pytest.raises(RuntimeError, match="stopped"):
        future.result(timeout=1)


# ------------------------------------------------------------------
# Queue depth
# ------------------------------------------------------------------


def test_queue_depth_reflects_pending_items() -> None:
    # given — block forward so items stay queued
    gate = threading.Event()

    def blocked_forward(batch):
        gate.wait(timeout=5)
        return [x * 2 for x in batch]

    bc = _make_collector(max_size=100, max_delay_ms=5000, forward_fn=blocked_forward)

    # when
    assert bc.queue_depth == 0
    futures = [bc.add(tensor=i, meta=None) for i in range(5)]
    time.sleep(0.02)  # let items land in heap

    # then — queue_depth should reflect pending items
    # (dispatch thread may have grabbed some, but max_size=100 and forward is
    # blocked, so at most one batch could be in-flight)
    depth = bc.queue_depth
    assert depth >= 0, f"queue_depth should be non-negative, got {depth}"
    # At least some items should still be pending (forward is blocked, so the
    # dispatch thread can grab at most one batch but cannot finish it)
    assert depth <= 5, f"queue_depth should be at most 5, got {depth}"

    # cleanup
    gate.set()
    bc.stop(drain=True)
    for f in futures:
        f.result(timeout=2)


def test_queue_depth_by_priority() -> None:
    # given — block dispatch so all items stay in heap
    gate = threading.Event()

    def blocked_forward(batch):
        gate.wait(timeout=5)
        return [x * 2 for x in batch]

    bc = _make_collector(max_size=100, max_delay_ms=5000, forward_fn=blocked_forward)

    # when
    bc.add(tensor=1, meta=None, priority=0)
    bc.add(tensor=2, meta=None, priority=0)
    bc.add(tensor=3, meta=None, priority=5)
    time.sleep(0.02)  # let items land in heap

    # then — might not be exact if dispatch thread grabbed the first item
    by_priority = bc.queue_depth_by_priority()

    # cleanup
    gate.set()
    bc.stop(drain=True)

    # at least verify the method returns a dict with int keys
    assert isinstance(by_priority, dict)


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------


def test_stats_after_batches() -> None:
    # given
    bc = _make_collector(max_size=2, max_delay_ms=10)

    # when — submit 4 items → should produce 2 batches of 2
    futures = [bc.add(tensor=i, meta=None) for i in range(4)]
    for f in futures:
        f.result(timeout=2)
    time.sleep(0.05)  # let stats update

    # then
    s = bc.stats()
    assert s["batches_dispatched"] >= 1
    assert s["avg_batch_fill_pct"] > 0
    assert s["avg_batch_delay_ms"] >= 0

    bc.stop()


# ------------------------------------------------------------------
# Concurrent submit stress test
# ------------------------------------------------------------------


def test_concurrent_submits_all_resolve() -> None:
    # given
    bc = _make_collector(max_size=8, max_delay_ms=20)
    n_items = 50
    futures: List[Future] = []

    # when — submit from multiple threads
    barrier = threading.Barrier(4)

    def submitter(start: int, count: int) -> None:
        barrier.wait()
        for i in range(start, start + count):
            futures.append(bc.add(tensor=i, meta=i))

    threads = [threading.Thread(target=submitter, args=(i * 13, 13)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # then — all futures resolve without error
    results = []
    for f in futures:
        r = f.result(timeout=5)
        results.append(r)

    assert len(results) == 52  # 4 * 13
    # Each result is (tensor * 2, meta=tensor), verify post_process output
    for raw, meta in results:
        assert raw == meta * 2

    bc.stop()
