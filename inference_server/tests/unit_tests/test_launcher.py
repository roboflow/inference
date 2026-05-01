"""Unit tests for launcher.py — launch_inprocess, launch_orchestrated, launch()."""

from __future__ import annotations

import os
import threading

import pytest

from inference_model_manager.model_manager import ModelManager
from inference_server.launcher import (
    _MODE_INPROCESS,
    _MODE_ORCHESTRATED,
    LaunchHandle,
    launch,
    launch_inprocess,
    launch_orchestrated,
)

# ---------------------------------------------------------------------------
# launch_inprocess
# ---------------------------------------------------------------------------


class TestLaunchInprocess:
    def test_returns_model_manager(self) -> None:
        mm = launch_inprocess()
        assert isinstance(mm, ModelManager)

    def test_max_pinned_memory_forwarded(self) -> None:
        mm = launch_inprocess(max_pinned_memory_mb=512)
        assert mm._max_pinned_memory_bytes == 512 * 1024 * 1024

    def test_default_pinned_memory_zero(self) -> None:
        mm = launch_inprocess()
        assert mm._max_pinned_memory_bytes == 0


# ---------------------------------------------------------------------------
# launch_orchestrated
# ---------------------------------------------------------------------------


class TestLaunchOrchestrated:
    def test_returns_launch_handle(self) -> None:
        handle = launch_orchestrated(
            n_slots=4,
            input_mb=1.0,
            mmp_start_timeout=5.0,
        )
        try:
            assert isinstance(handle, LaunchHandle)
        finally:
            handle.shutdown()

    def test_mmp_addr_populated(self) -> None:
        handle = launch_orchestrated(
            n_slots=4,
            input_mb=1.0,
            mmp_start_timeout=5.0,
        )
        try:
            assert handle.mmp_addr is not None
            assert handle.mmp_addr.startswith(("tcp://", "ipc://"))
        finally:
            handle.shutdown()

    def test_shm_name_populated(self) -> None:
        handle = launch_orchestrated(
            n_slots=4,
            input_mb=1.0,
            mmp_start_timeout=5.0,
        )
        try:
            assert handle.shm_name is not None
            assert len(handle.shm_name) > 0
        finally:
            handle.shutdown()

    def test_manager_attribute_is_model_manager(self) -> None:
        handle = launch_orchestrated(
            n_slots=4,
            input_mb=1.0,
            mmp_start_timeout=5.0,
        )
        try:
            assert isinstance(handle.manager, ModelManager)
        finally:
            handle.shutdown()

    def test_mmp_thread_alive_after_launch(self) -> None:
        handle = launch_orchestrated(
            n_slots=4,
            input_mb=1.0,
            mmp_start_timeout=5.0,
        )
        try:
            assert handle._thread.is_alive()
        finally:
            handle.shutdown()

    def test_shutdown_stops_thread(self) -> None:
        handle = launch_orchestrated(
            n_slots=4,
            input_mb=1.0,
            mmp_start_timeout=5.0,
        )
        thread = handle._thread
        handle.shutdown(timeout=10.0)
        assert not thread.is_alive()

    def test_custom_mmp_addr(self) -> None:
        # Use INFERENCE_ZMQ_TRANSPORT=tcp to get a deterministic tcp addr
        handle = launch_orchestrated(
            n_slots=4,
            input_mb=1.0,
            mmp_addr="tcp://127.0.0.1:19876",
            mmp_start_timeout=5.0,
        )
        try:
            assert handle.mmp_addr == "tcp://127.0.0.1:19876"
        finally:
            handle.shutdown()

    def test_repr(self) -> None:
        handle = launch_orchestrated(
            n_slots=4,
            input_mb=1.0,
            mmp_start_timeout=5.0,
        )
        try:
            r = repr(handle)
            assert "mmp_addr=" in r
            assert "shm_name=" in r
        finally:
            handle.shutdown()


# ---------------------------------------------------------------------------
# launch() unified entry point
# ---------------------------------------------------------------------------


class TestLaunch:
    def test_inprocess_returns_model_manager(self) -> None:
        result = launch(_MODE_INPROCESS)
        assert isinstance(result, ModelManager)

    def test_orchestrated_returns_launch_handle(self) -> None:
        handle = launch(
            _MODE_ORCHESTRATED,
            n_slots=4,
            input_mb=1.0,
            mmp_start_timeout=5.0,
        )
        try:
            assert isinstance(handle, LaunchHandle)
        finally:
            handle.shutdown()

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown deployment mode"):
            launch("foobar")

    def test_default_mode_is_inprocess(self, monkeypatch) -> None:
        monkeypatch.delenv("INFERENCE_DEPLOYMENT_MODE", raising=False)
        result = launch()
        assert isinstance(result, ModelManager)

    def test_env_var_selects_inprocess(self, monkeypatch) -> None:
        monkeypatch.setenv("INFERENCE_DEPLOYMENT_MODE", _MODE_INPROCESS)
        result = launch()
        assert isinstance(result, ModelManager)

    def test_env_var_selects_orchestrated(self, monkeypatch) -> None:
        monkeypatch.setenv("INFERENCE_DEPLOYMENT_MODE", _MODE_ORCHESTRATED)
        handle = launch(n_slots=4, input_mb=1.0, mmp_start_timeout=5.0)
        try:
            assert isinstance(handle, LaunchHandle)
        finally:
            handle.shutdown()

    def test_max_pinned_memory_forwarded_inprocess(self) -> None:
        mm = launch(_MODE_INPROCESS, max_pinned_memory_mb=256)
        assert mm._max_pinned_memory_bytes == 256 * 1024 * 1024
