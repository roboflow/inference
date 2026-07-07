"""Supervisor entrypoint: runs the ModelManagerProcess next to uvicorn.

Started by the container entrypoint when LEGACY_MMP_ADAPTER_ENABLED is set.
INFERENCE_MMP_ADDR must be set in the container env so both this supervisor
and the uvicorn workers agree on the ZMQ address; the SHM pool name is
discovered by workers via the T_SHM_INFO handshake.
"""

import logging
import os
import signal
import threading

from inference_server import configuration
from inference_server.launcher import launch_orchestrated

logger = logging.getLogger("mmp_supervisor")


def main() -> None:
    mmp_addr = os.environ.get(configuration.INFERENCE_MMP_ADDR_ENV)
    if not mmp_addr:
        raise SystemExit(
            f"{configuration.INFERENCE_MMP_ADDR_ENV} must be set; uvicorn workers "
            "read it to reach the MMP started by this supervisor"
        )
    handle = launch_orchestrated(
        mmp_addr=mmp_addr,
        n_slots=configuration.SERVER_N_SLOTS,
        input_mb=configuration.SERVER_INPUT_MB,
        decoder=configuration.SERVER_DECODER,
        batch_max_size=configuration.SERVER_BATCH_MAX_SIZE,
        batch_max_wait_ms=configuration.SERVER_BATCH_MAX_WAIT_MS,
        idle_timeout_s=configuration.SERVER_MODEL_IDLE_TIMEOUT_S,
    )
    logger.info("MMP ready: addr=%s shm=%s", handle.mmp_addr, handle.shm_name)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    while not stop.wait(timeout=5.0):
        if not handle.is_alive():
            logger.error("MMP thread died; exiting so the container restarts")
            raise SystemExit(1)
    handle.shutdown()


if __name__ == "__main__":
    main()
