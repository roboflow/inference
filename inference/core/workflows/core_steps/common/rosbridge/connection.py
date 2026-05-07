"""Refcounted rosbridge connection registry.

A single ``roslibpy.Ros`` instance is shared across all blocks/producers that
target the same ``(host, port, ssl)`` endpoint. The registry returns a
``RosHandle`` per acquire; the underlying WebSocket is closed when the last
handle is released.

``roslibpy`` is imported lazily so the rest of inference can import this
module without the ``ros`` extra installed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

_ROSLIBPY_IMPORT_ERROR_MSG = (
    "roslibpy is required for ROS bridge interop. "
    "Install with: pip install 'inference[ros]'"
)


def _import_roslibpy():
    try:
        import roslibpy  # type: ignore
    except ImportError as e:
        raise ImportError(_ROSLIBPY_IMPORT_ERROR_MSG) from e
    return roslibpy


def normalize_message_type(message_type: str, ros_version: int = 2) -> str:
    """Accept ROS1 (``pkg/Msg``) or ROS2 (``pkg/msg/Msg``) shapes; return the
    form preferred by the running rosbridge.

    rosbridge_server normalizes either form internally, but we prefer to send
    the user-configured shape so bridge logs are readable.
    """
    if message_type.count("/") == 2:
        if ros_version == 1:
            pkg, _msg, name = message_type.split("/")
            return f"{pkg}/{name}"
        return message_type
    if message_type.count("/") == 1 and ros_version == 2:
        pkg, name = message_type.split("/")
        return f"{pkg}/msg/{name}"
    return message_type


@dataclass
class _RegistryEntry:
    ros: Any
    refcount: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class RosHandle:
    """Handle to a shared rosbridge connection. Always release on shutdown."""

    def __init__(
        self,
        registry: "RosbridgeConnectionRegistry",
        key: Tuple[str, int, bool],
    ):
        self._registry = registry
        self._key = key
        self._released = False

    @property
    def ros(self) -> Any:
        if self._released:
            raise RuntimeError("RosHandle already released")
        return self._registry._entry(self._key).ros

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._registry._release(self._key)


class RosbridgeConnectionRegistry:
    """Process-global registry of rosbridge connections, keyed by endpoint."""

    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, int, bool], _RegistryEntry] = {}
        self._guard = threading.Lock()

    def acquire(
        self,
        host: str,
        port: int = 9090,
        ssl: bool = False,
        connect_timeout: float = 10.0,
    ) -> RosHandle:
        key = (host, port, ssl)
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                roslibpy = _import_roslibpy()
                ros = roslibpy.Ros(host=host, port=port, is_secure=ssl)
                entry = _RegistryEntry(ros=ros)
                self._entries[key] = entry
            entry.refcount += 1

        if not entry.ros.is_connected:
            with entry.lock:
                if not entry.ros.is_connected:
                    entry.ros.run(timeout=connect_timeout)
        return RosHandle(self, key)

    def _entry(self, key: Tuple[str, int, bool]) -> _RegistryEntry:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                raise RuntimeError(f"No rosbridge connection registered for {key}")
            return entry

    def _release(self, key: Tuple[str, int, bool]) -> None:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.refcount -= 1
            if entry.refcount <= 0:
                del self._entries[key]
                ros = entry.ros
            else:
                ros = None
        if ros is not None:
            try:
                ros.close()
            except Exception:
                pass

    def reset_for_tests(self) -> None:
        """Clear all entries without closing — for unit tests with mocked Ros."""
        with self._guard:
            self._entries.clear()


_REGISTRY: Optional[RosbridgeConnectionRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> RosbridgeConnectionRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = RosbridgeConnectionRegistry()
    return _REGISTRY
