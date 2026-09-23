"""Process-local `WorkflowsCache` used when the host injects nothing.

The server always overrides it with `workflows_core.cache` - its Redis-backed
(or in-process) singleton - at every composition root, so this only ever runs
standalone. Deliberately small: no background thread, no size bound, no sorted
sets. Expiry is lazy; the server's `MemoryCache` also sweeps in a daemon thread,
which a library default must not start, but the observable `get` behaviour is
the same.

`if expire:` matches `MemoryCache._set_unlocked` exactly - a falsy expire
(None, 0, 0.0) means "no expiry", not "expire immediately".
"""

import time
from threading import Lock
from typing import Any, Dict, Optional


class InMemoryWorkflowsCache:
    def __init__(self) -> None:
        self._values: Dict[str, Any] = {}
        self._deadlines: Dict[str, float] = {}
        self._guard = Lock()

    def get(self, key: str) -> Any:
        with self._guard:
            deadline = self._deadlines.get(key)
            if deadline is not None and deadline < time.time():
                self._values.pop(key, None)
                self._deadlines.pop(key, None)
                return None
            return self._values.get(key)

    def set(self, key: str, value: Any, expire: Optional[float] = None) -> None:
        with self._guard:
            self._values[key] = value
            if expire:
                self._deadlines[key] = time.time() + expire
            else:
                self._deadlines.pop(key, None)
