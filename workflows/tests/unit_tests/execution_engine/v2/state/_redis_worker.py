"""Independent process using managed state on Redis (spawned by the tests).

Usage: ``python _redis_worker.py <url> <namespace> <incr|cas> <count> <name>``.
The worker announces itself, waits for the shared start signal, then:

* ``incr``: increments the global and ``cam_a`` key ``total`` ``count`` times;
* ``cas``: tries once to move ``cam_a`` key ``machine`` from ``idle`` to
  ``<name>`` and prints ``won`` or ``lost``.
"""

import sys
import time

from roboflow_workflows.execution_engine.v2.state import ManagedState
from roboflow_workflows.execution_engine.v2.state.redis import RedisStateBackend

_START_TIMEOUT_SECONDS = 30.0


def main() -> None:
    url, namespace, mode, count, name = sys.argv[1:]
    state = ManagedState(RedisStateBackend(url), namespace=namespace)
    control = state.global_
    control.incr("ready")
    deadline = time.monotonic() + _START_TIMEOUT_SECONDS
    while control.get("go") is not True:
        if time.monotonic() > deadline:
            raise SystemExit("start signal never arrived")
        time.sleep(0.001)

    if mode == "incr":
        for _ in range(int(count)):
            state.global_.incr("total")
            state.for_source("cam_a").incr("total")
    else:
        won = state.for_source("cam_a").compare_and_set("machine", "idle", name)
        print("won" if won else "lost")
    state.backend.close()


if __name__ == "__main__":
    main()
