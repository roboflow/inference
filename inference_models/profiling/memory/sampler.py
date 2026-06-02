from __future__ import annotations

import os
import threading
import time
from typing import Any, List, Optional


class NvmlProcessMemorySampler:
    """Poll NVML in a background thread to track peak GPU memory for this process."""

    def __init__(
        self,
        device_index: int,
        *,
        interval_seconds: float,
    ) -> None:
        """Initialize the sampler for one CUDA device.

        Args:
            device_index: CUDA device index passed to NVML.
            interval_seconds: Sleep interval between background polls.
        """
        self._device_index = device_index
        self._interval_seconds = interval_seconds
        self._pid = os.getpid()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak_bytes: Optional[int] = None
        self._source = "unavailable"

        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._available = True
        except Exception:
            self._pynvml = None
            self._handle = None
            self._available = False

    @property
    def source(self) -> str:
        """Return how the latest reading was obtained (``process``, ``device``, etc.)."""
        return self._source

    def snapshot(self) -> Optional[int]:
        """Read current GPU memory for this process or device.

        Returns:
            Bytes used, or ``None`` when NVML is unavailable.
        """
        if not self._available:
            return None

        process_bytes = self._get_process_memory_bytes()
        if process_bytes is not None:
            self._source = "process"

            return process_bytes

        device_bytes = self._get_device_used_bytes()
        if device_bytes is not None:
            self._source = "device"

            return device_bytes

        self._source = "unavailable"

        return None

    def start(self) -> None:
        """Start background polling and seed the running peak from an initial snapshot."""
        self._peak_bytes = self.snapshot()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> Optional[int]:
        """Stop polling and return the peak bytes observed since ``start``.

        Returns:
            Peak bytes, or ``None`` if no successful samples were recorded.
        """
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join()

        final_value = self.snapshot()
        if final_value is not None:
            self._record(final_value)

        return self._peak_bytes

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            value = self.snapshot()
            if value is not None:
                self._record(value)

            time.sleep(self._interval_seconds)

    def _record(self, value: int) -> None:
        if self._peak_bytes is None or value > self._peak_bytes:
            self._peak_bytes = value

    def _get_process_memory_bytes(self) -> Optional[int]:
        process_entries = self._get_nvml_process_entries()
        if process_entries is None:
            return None

        for process in process_entries:
            if getattr(process, "pid", None) != self._pid:
                continue

            used_gpu_memory = getattr(process, "usedGpuMemory", None)
            if used_gpu_memory is None:
                continue

            process_bytes = int(used_gpu_memory)

            return process_bytes

        return None

    def _get_nvml_process_entries(self) -> Optional[List[Any]]:
        entries: List[Any] = []

        for getter_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            getter = getattr(self._pynvml, getter_name, None)
            if getter is None:
                continue

            try:
                entries.extend(getter(self._handle))
            except Exception:
                continue

        if not entries:
            return None

        return entries

    def _get_device_used_bytes(self) -> Optional[int]:
        try:
            meminfo = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            used_bytes = int(meminfo.used)
        except Exception:
            return None

        return used_bytes
