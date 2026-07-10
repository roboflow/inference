"""Unit tests for the CUDA memory reclamation watchdog.

torch (and a CUDA device) is not available in the unit-test environment, so every
test here stubs ``torch`` via ``sys.modules`` or patches the module-level
``cuda_is_available`` / ``reclaim_cuda_memory`` helpers. The behaviour under test is
deliberately torch-free: the guard that keeps the daemon from spinning when there is
no CUDA device is exactly the thing we want covered without a GPU in CI.
"""

import sys
import threading
from unittest import mock

from inference.core.managers import cuda_memory_watchdog as mod
from inference.core.managers.cuda_memory_watchdog import (
    MIN_RECLAMATION_INTERVAL_SECONDS,
    CudaMemoryReclamationWatchdog,
    cuda_is_available,
    reclaim_cuda_memory,
)

MB = 1024 * 1024


def _fake_torch(cuda_available: bool = True, mem_get_info=None) -> mock.MagicMock:
    torch = mock.MagicMock(name="torch")
    torch.cuda.is_available.return_value = cuda_available
    if mem_get_info is not None:
        torch.cuda.mem_get_info.side_effect = mem_get_info
    return torch


# --------------------------- cuda_is_available ---------------------------


def test_cuda_is_available_returns_false_when_torch_not_importable() -> None:
    # given - `import torch` raises ImportError (None entry in sys.modules)
    with mock.patch.dict(sys.modules, {"torch": None}):
        # when / then
        assert cuda_is_available() is False


def test_cuda_is_available_returns_false_when_no_cuda_device() -> None:
    # given - torch present but reports no CUDA device (CPU-only box)
    with mock.patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=False)}):
        # when / then
        assert cuda_is_available() is False


def test_cuda_is_available_returns_true_when_cuda_present() -> None:
    # given
    with mock.patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=True)}):
        # when / then
        assert cuda_is_available() is True


def test_cuda_is_available_returns_false_when_probe_raises() -> None:
    # given - torch.cuda.is_available() itself blows up (broken driver)
    torch = _fake_torch()
    torch.cuda.is_available.side_effect = RuntimeError("driver error")
    with mock.patch.dict(sys.modules, {"torch": torch}):
        # when / then - swallowed, treated as unavailable
        assert cuda_is_available() is False


# --------------------------- reclaim_cuda_memory ---------------------------


def test_reclaim_is_noop_when_cuda_unavailable() -> None:
    # given
    torch = _fake_torch(cuda_available=False)
    with mock.patch.dict(sys.modules, {"torch": torch}):
        # when - must not raise
        reclaim_cuda_memory()
    # then - never touched the allocator
    torch.cuda.empty_cache.assert_not_called()
    torch.cuda.ipc_collect.assert_not_called()


def test_reclaim_calls_empty_cache_and_ipc_collect_when_available() -> None:
    # given - 1000MB free before, 2500MB free after (1500MB reclaimed) of 4000MB total
    torch = _fake_torch(
        cuda_available=True,
        mem_get_info=[(1000 * MB, 4000 * MB), (2500 * MB, 4000 * MB)],
    )
    with mock.patch.dict(sys.modules, {"torch": torch}):
        # when
        reclaim_cuda_memory()
    # then
    torch.cuda.empty_cache.assert_called_once()
    torch.cuda.ipc_collect.assert_called_once()


def test_reclaim_swallows_errors_without_raising() -> None:
    # given - probing free memory raises
    torch = _fake_torch(cuda_available=True)
    torch.cuda.mem_get_info.side_effect = RuntimeError("mem_get_info failed")
    with mock.patch.dict(sys.modules, {"torch": torch}):
        # when / then - error is caught, call returns None
        assert reclaim_cuda_memory() is None
    torch.cuda.empty_cache.assert_not_called()


# ----------------------- CudaMemoryReclamationWatchdog -----------------------


def test_interval_below_minimum_is_clamped() -> None:
    wd = CudaMemoryReclamationWatchdog(interval_seconds=1.0)
    assert wd._interval_seconds == MIN_RECLAMATION_INTERVAL_SECONDS


def test_interval_above_minimum_is_preserved() -> None:
    wd = CudaMemoryReclamationWatchdog(interval_seconds=42.0)
    assert wd._interval_seconds == 42.0


def test_start_does_not_spin_daemon_when_cuda_unavailable() -> None:
    # given - no CUDA device: starting would only wake up to no-op
    with mock.patch.object(mod, "cuda_is_available", return_value=False):
        wd = CudaMemoryReclamationWatchdog(interval_seconds=5.0)
        # when
        wd.start()
    # then - no thread was ever created
    assert wd._thread is None


def test_start_launches_daemon_and_reclaims_then_stop_joins() -> None:
    # given
    reclaimed = threading.Event()
    with mock.patch.object(mod, "cuda_is_available", return_value=True), mock.patch.object(
        mod, "reclaim_cuda_memory", side_effect=lambda: reclaimed.set()
    ):
        wd = CudaMemoryReclamationWatchdog(interval_seconds=5.0)
        # when
        wd.start()
        try:
            # then - the loop runs at least one reclamation cycle promptly
            assert reclaimed.wait(timeout=3.0)
            assert wd._thread is not None
            assert wd._thread.is_alive()
        finally:
            wd.stop(timeout=3.0)
    # and stop() joins and clears the thread handle
    assert wd._thread is None


def test_start_is_idempotent_while_running() -> None:
    with mock.patch.object(mod, "cuda_is_available", return_value=True), mock.patch.object(
        mod, "reclaim_cuda_memory", return_value=None
    ):
        wd = CudaMemoryReclamationWatchdog(interval_seconds=5.0)
        wd.start()
        try:
            first_thread = wd._thread
            # second start() must not replace the running thread
            wd.start()
            assert wd._thread is first_thread
        finally:
            wd.stop(timeout=3.0)


def test_stop_is_quiet_noop_when_never_started() -> None:
    wd = CudaMemoryReclamationWatchdog(interval_seconds=5.0)
    # must not raise even though the daemon never started
    wd.stop()
    assert wd._thread is None
