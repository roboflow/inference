"""ZMQ address factory — selects transport based on platform or env override.

Platform defaults (benchmarked):
  Linux:   ipc:// (unix socket)  — ~1μs RTT, kernel shortcut, no TCP stack
  macOS:   tcp://127.0.0.1       — loopback is faster than unix socket on macOS kernel
  Windows: tcp://127.0.0.1       — no ipc:// support

Override with env var:
  INFERENCE_ZMQ_TRANSPORT=ipc   → force unix socket (Linux/macOS only)
  INFERENCE_ZMQ_TRANSPORT=tcp   → force loopback TCP (all platforms)

Per-socket port override (tcp only):
  INFERENCE_ZMQ_PORT_MMPROCESS=15555  (upper-cased socket name)
"""

from __future__ import annotations

import os
import sys
import tempfile

from inference_model_manager.configuration import (
    INFERENCE_ZMQ_PORT_ENV_PREFIX,
    INFERENCE_ZMQ_TRANSPORT_ENV,
)

_DEFAULT_PORTS: dict[str, int] = {
    "mmprocess": 15555,
}


def default_transport() -> str:
    """Return the fastest transport for the current platform."""
    # Linux: unix socket ~1μs RTT beats loopback TCP ~25μs
    # macOS: loopback TCP is faster than unix socket (macOS kernel quirk)
    # Windows: no ipc:// support
    return "ipc" if sys.platform == "linux" else "tcp"


def zmq_addr(name: str, transport: str | None = None) -> str:
    """Return a ZMQ bind/connect address for the given logical socket name.

    Args:
        name:      Logical socket name, e.g. ``"mmprocess"``.
        transport: ``"ipc"`` or ``"tcp"``. If None, reads
                   ``INFERENCE_ZMQ_TRANSPORT`` env var, then platform default.

    Returns:
        ZMQ address string, e.g. ``"ipc:///tmp/inference_mmprocess.ipc"``
        or ``"tcp://127.0.0.1:15555"``.

    Examples::

        zmq_addr("mmprocess")              # platform default
        zmq_addr("mmprocess", "ipc")       # force unix socket
        zmq_addr("mmprocess", "tcp")       # force loopback
    """
    if transport is None:
        transport = os.environ.get(INFERENCE_ZMQ_TRANSPORT_ENV, default_transport())

    if transport == "ipc":
        return f"ipc://{tempfile.gettempdir()}/inference_{name}.ipc"

    # TCP loopback — port from env or registry
    env_key = f"{INFERENCE_ZMQ_PORT_ENV_PREFIX}{name.upper()}"
    default_port = _DEFAULT_PORTS.get(name)
    env_port = os.environ.get(env_key)
    if env_port is not None:
        port = int(env_port)
    elif default_port is not None:
        port = default_port
    else:
        raise ValueError(
            f"No default port for ZMQ socket '{name}'. "
            f"Set {env_key} or add to _DEFAULT_PORTS."
        )
    return f"tcp://127.0.0.1:{port}"
