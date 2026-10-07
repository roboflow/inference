"""In-memory managed state backend.

Each ``InMemoryStateBackend`` object is its own store: there is no
process-wide instance. One lock guards the dictionary; every operation holds
it only for the dictionary access and integer arithmetic, never while user
code, callbacks or I/O run.
"""

import json
import threading
from typing import Dict, Optional

from roboflow_workflows.execution_engine.v2.state.codec import INT64_MAX, INT64_MIN
from roboflow_workflows.execution_engine.v2.state.errors import (
    StateBackendError,
    StateOverflowError,
    StateTypeError,
)

__all__ = ["InMemoryStateBackend"]


class InMemoryStateBackend:
    """Thread-safe dictionary of encoded state values for one process.

    Values are the canonical encoded strings from the codec, exactly as Redis
    stores them, so both backends compare and count identically. Share data
    between sessions by sharing this object (or a ``ManagedState`` using it).
    """

    def __init__(self) -> None:
        self._values: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._closed = False

    def get(self, key: str) -> Optional[str]:
        """Return the stored text of ``key`` or ``None`` when missing."""
        with self._lock:
            self._check_open()
            stored = self._values.get(key)

        return stored

    def set(self, key: str, value: str, *, only_if_absent: bool) -> bool:
        """Store ``value``; with ``only_if_absent`` only when ``key`` is missing.

        Returns:
            Whether the value was stored.
        """
        with self._lock:
            self._check_open()
            if only_if_absent and key in self._values:
                return False
            self._values[key] = value

        return True

    def delete(self, key: str) -> bool:
        """Remove ``key``.

        Returns:
            Whether a stored value existed.
        """
        with self._lock:
            self._check_open()
            existed = self._values.pop(key, None) is not None

        return existed

    def incr(self, key: str, amount: int) -> int:
        """Add ``amount`` to the integer stored under ``key``; missing counts as 0.

        Args:
            key: Storage key.
            amount: Signed 64-bit increment.

        Returns:
            The new integer.

        Raises:
            StateTypeError: When the current value is not an integer.
            StateOverflowError: When the result leaves the signed 64-bit range.
        """
        with self._lock:
            self._check_open()
            current = self._values.get(key)
            current_number = 0 if current is None else _stored_integer(current)
            result = current_number + amount
            if not INT64_MIN <= result <= INT64_MAX:
                raise StateOverflowError(
                    f"incr by {amount} would leave the signed 64-bit range"
                )
            self._values[key] = str(result)

        return result

    def compare_and_set(
        self,
        key: str,
        expected: Optional[str],
        new: Optional[str],
    ) -> bool:
        """Replace the value of ``key`` when it equals ``expected``.

        Args:
            key: Storage key.
            expected: Encoded expected value; ``None`` expects a missing key.
            new: Encoded new value; ``None`` deletes the key.

        Returns:
            Whether the comparison matched and the change was applied.
        """
        with self._lock:
            self._check_open()
            current = self._values.get(key)
            if current != expected:
                return False
            if new is None:
                self._values.pop(key, None)
            else:
                self._values[key] = new

        return True

    def close(self) -> None:
        """Drop all values; later operations raise ``StateBackendError``."""
        with self._lock:
            self._closed = True
            self._values.clear()

    def _check_open(self) -> None:
        if self._closed:
            raise StateBackendError("in-memory state backend is closed")


def _stored_integer(stored: str) -> int:
    value = json.loads(stored)
    if type(value) is not int:
        found = "null" if value is None else type(value).__name__
        raise StateTypeError(f"incr needs an integer value, found {found}")

    return value
