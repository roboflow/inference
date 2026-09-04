"""Capability probe and scoped controller for in-pod NVIDIA MPS experiments.

The default command is read-only.  ``--start`` changes GPU process state and is
intended only for a staging performance pod that exclusively owns one GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional


@dataclass(frozen=True)
class MPSCapability:
    control_binary: Optional[str]
    server_binary: Optional[str]
    nvidia_smi: Optional[str]
    cuda_visible_devices: Optional[str]

    @property
    def available(self) -> bool:
        return bool(self.control_binary and self.server_binary and self.nvidia_smi)


def probe_capability(environ: Mapping[str, str] = os.environ) -> MPSCapability:
    return MPSCapability(
        control_binary=shutil.which("nvidia-cuda-mps-control"),
        server_binary=shutil.which("nvidia-cuda-mps-server"),
        nvidia_smi=shutil.which("nvidia-smi"),
        cuda_visible_devices=environ.get("CUDA_VISIBLE_DEVICES"),
    )


class MPSController:
    """Start one MPS control/server pair in a private directory."""

    def __init__(
        self,
        control_binary: str,
        base_directory: Optional[str] = None,
        active_thread_percentage: Optional[int] = None,
        pinned_device_memory_limit: Optional[str] = None,
    ) -> None:
        if (
            active_thread_percentage is not None
            and not 1 <= active_thread_percentage <= 100
        ):
            raise ValueError("active_thread_percentage must be in [1, 100]")
        self.control_binary = control_binary
        self.base_directory = Path(
            base_directory or tempfile.mkdtemp(prefix="rf-video-mps-")
        )
        self.pipe_directory = self.base_directory / "pipe"
        self.log_directory = self.base_directory / "log"
        self.active_thread_percentage = active_thread_percentage
        self.pinned_device_memory_limit = pinned_device_memory_limit
        self._started = False

    def client_environment(
        self, parent: Mapping[str, str] = os.environ
    ) -> dict[str, str]:
        env = dict(parent)
        env["CUDA_MPS_PIPE_DIRECTORY"] = str(self.pipe_directory)
        env["CUDA_MPS_LOG_DIRECTORY"] = str(self.log_directory)
        if self.active_thread_percentage is not None:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
                self.active_thread_percentage
            )
        if self.pinned_device_memory_limit is not None:
            env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] = self.pinned_device_memory_limit
        return env

    def start(self) -> None:
        self.pipe_directory.mkdir(parents=True, exist_ok=True)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [self.control_binary, "-d"],
            env=self.client_environment(),
            check=True,
            text=True,
            capture_output=True,
        )
        self._started = True

    def query(self, command: str = "get_server_list") -> str:
        completed = subprocess.run(
            [self.control_binary],
            input=f"{command}\n",
            env=self.client_environment(),
            check=True,
            text=True,
            capture_output=True,
        )
        return completed.stdout.strip()

    def stop(self) -> None:
        if not self._started:
            return
        try:
            subprocess.run(
                [self.control_binary],
                input="quit\n",
                env=self.client_environment(),
                check=False,
                text=True,
                capture_output=True,
                timeout=10,
            )
        finally:
            self._started = False

    def __enter__(self) -> "MPSController":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--active-thread-percentage", type=int)
    parser.add_argument("--pinned-device-memory-limit")
    args = parser.parse_args()

    capability = probe_capability()
    result = {"capability": asdict(capability), "available": capability.available}
    if args.start:
        if not capability.available:
            raise SystemExit("NVIDIA MPS tooling is not available in this image")
        with MPSController(
            capability.control_binary,
            active_thread_percentage=args.active_thread_percentage,
            pinned_device_memory_limit=args.pinned_device_memory_limit,
        ) as controller:
            result["server_list"] = controller.query()
            result["client_environment"] = {
                key: value
                for key, value in controller.client_environment().items()
                if key.startswith("CUDA_MPS_")
            }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
