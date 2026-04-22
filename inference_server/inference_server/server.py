"""Integrated server entry point.

Starts ModelManagerProcess in a background thread, injects the MMP address
and SHM name into the environment, then launches uvicorn workers.
Each uvicorn worker connects to MMP via ZMQ and attaches the shared SHM pool.

Usage::

    # Plain HTTP (dev/local):
    python -m inference_server.server

    # HTTPS with self-signed cert (remote machine):
    SSL_CERTFILE=/path/c.pem SSL_KEYFILE=/path/c.key python -m inference_server.server

    # Or via start_server.sh (generates cert automatically):
    ./start_server.sh

Environment variables::

    HOST                    Bind host (default: 0.0.0.0)
    PORT                    Bind port (default: 8443)
    NUM_WORKERS             uvicorn worker processes (default: 4)
    SSL_CERTFILE            Path to TLS certificate (PEM). Enables HTTPS if set.
    SSL_KEYFILE             Path to TLS private key (PEM). Required when SSL_CERTFILE set.

    INFERENCE_N_SLOTS       SHM pool slot count (default: 256)
    INFERENCE_INPUT_MB      Bytes per slot input area, MB (default: 25.0)
    INFERENCE_RESULT_MB     Bytes per slot result area, MB (default: 4.0)

    INFERENCE_DECODER       Image decoder for worker subprocesses.
                            "imagecodecs" (default, CPU) or "nvjpeg" (GPU decode).
    INFERENCE_BATCH_MAX_SIZE    Max images per worker batch (default: 0 = use model's max).
    INFERENCE_BATCH_MAX_WAIT_MS Max ms to wait for a full batch (default: 5.0).
    NVIDIA_MPS              Set to "1" to start NVIDIA MPS before launching.
                            MPS daemon is guaranteed to be stopped on exit
                            (even on crash / SIGKILL of this process — via atexit
                            + signal handlers).

    LOG_LEVEL               uvicorn log level (default: warning)
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NVIDIA MPS management
# ---------------------------------------------------------------------------

_mps_started = False


def _start_mps() -> None:
    global _mps_started
    logger.info("Starting NVIDIA MPS daemon…")
    subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
    _mps_started = True
    atexit.register(_stop_mps)
    # Catch common termination signals so MPS is cleaned up even on kill
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        prev = signal.getsignal(sig)

        def _handler(s, f, _prev=prev, _sig=sig):
            _stop_mps()
            if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                _prev(s, f)
            else:
                sys.exit(128 + _sig)

        signal.signal(sig, _handler)
    logger.info("NVIDIA MPS daemon started")


def _stop_mps() -> None:
    global _mps_started
    if not _mps_started:
        return
    _mps_started = False
    logger.info("Stopping NVIDIA MPS daemon…")
    try:
        subprocess.run(
            ["bash", "-c", "echo quit | nvidia-cuda-mps-control"],
            timeout=10,
        )
        logger.info("NVIDIA MPS daemon stopped")
    except Exception:
        logger.warning("Failed to stop MPS daemon cleanly", exc_info=True)


def _preload_models(mmp_addr: str, preload_spec: str) -> None:
    """Send T_LOAD for each model in comma-separated spec, wait for T_OK.

    Format: "model_id:api_key,model_id:api_key,..." or just "model_id,model_id,..."
    (api_key defaults to ROBOFLOW_API_KEY env var if omitted per-model).
    """
    import struct
    import uuid

    import zmq

    T_LOAD = b"\x20"
    T_OK = b"\x40"
    T_ERROR = b"\xFF"

    default_key = os.environ.get("ROBOFLOW_API_KEY", "")
    models = []
    for entry in preload_spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            mid, key = entry.rsplit(":", 1)
        else:
            mid, key = entry, default_key
        models.append((mid.strip(), key.strip()))

    if not models:
        return

    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(mmp_addr)

    try:
        for model_id, api_key in models:
            req_id = uuid.uuid4().int & 0xFFFF_FFFF_FFFF_FFFF
            mid_bytes = model_id.encode()
            key_bytes = api_key.encode()
            payload = (
                struct.pack(">QH", req_id, len(mid_bytes))
                + mid_bytes
                + struct.pack(">H", len(key_bytes))
                + key_bytes
            )
            sock.send_multipart([T_LOAD, payload])
            logger.info("Preload: sent T_LOAD for '%s'", model_id)

            # Wait for T_OK or T_ERROR (up to 120s per model)
            if sock.poll(timeout=120_000):
                frames = sock.recv_multipart()
                msg_type = frames[0]
                if msg_type == T_OK:
                    logger.info("Preload: '%s' load accepted", model_id)
                elif msg_type == T_ERROR:
                    logger.error("Preload: '%s' load failed", model_id)
                else:
                    logger.warning(
                        "Preload: '%s' unexpected reply: %r", model_id, msg_type
                    )
            else:
                logger.error("Preload: '%s' no response from MMP within 120s", model_id)
    finally:
        sock.close()
        ctx.term()

    logger.info("Preload: %d model(s) requested", len(models))


def main() -> None:
    from inference_server.launcher import launch_orchestrated

    # ── MPS ────────────────────────────────────────────────────────────────
    if os.environ.get("NVIDIA_MPS", "").strip() == "1":
        _start_mps()

    # ── MMP config ─────────────────────────────────────────────────────────
    n_slots = int(os.environ.get("INFERENCE_N_SLOTS", "256"))
    input_mb = float(os.environ.get("INFERENCE_INPUT_MB", "25.0"))
    decoder = os.environ.get("INFERENCE_DECODER", "imagecodecs")
    batch_max_size = int(os.environ.get("INFERENCE_BATCH_MAX_SIZE", "0"))
    batch_max_wait = float(os.environ.get("INFERENCE_BATCH_MAX_WAIT_MS", "5.0"))
    idle_timeout = float(os.environ.get("INFERENCE_MODEL_IDLE_TIMEOUT_S", "300.0"))

    # ── HTTP / TLS config ──────────────────────────────────────────────────
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8443"))
    workers = int(os.environ.get("NUM_WORKERS", "4"))
    ssl_cert = os.environ.get("SSL_CERTFILE")
    ssl_key = os.environ.get("SSL_KEYFILE")
    log_level = os.environ.get("LOG_LEVEL", "warning").lower()

    if ssl_cert and not ssl_key:
        logger.error("SSL_CERTFILE set but SSL_KEYFILE is missing — aborting")
        sys.exit(1)

    # ── Start MMP ──────────────────────────────────────────────────────────
    logger.info(
        "Starting MMP: slots=%d data=%.0fMB",
        n_slots,
        input_mb,
    )
    handle = launch_orchestrated(
        n_slots=n_slots,
        input_mb=input_mb,
        decoder=decoder,
        batch_max_size=batch_max_size,
        batch_max_wait_ms=batch_max_wait,
        idle_timeout_s=idle_timeout,
    )
    logger.info("MMP ready: addr=%s  shm=%s", handle.mmp_addr, handle.shm_name)

    # Inject into env so uvicorn worker processes pick them up at import time
    os.environ["INFERENCE_MMP_ADDR"] = handle.mmp_addr
    os.environ["INFERENCE_SHM_NAME"] = handle.shm_name
    os.environ["INFERENCE_SHM_DATA_SIZE"] = str(int(input_mb * 1024 * 1024))

    # ── Preload models ────────────────────────────────────────────────────
    preload = os.environ.get("INFERENCE_PRELOAD_MODELS", "").strip()
    if preload:
        _preload_models(handle.mmp_addr, preload)

    # ── Start uvicorn ──────────────────────────────────────────────────────
    scheme = "https" if ssl_cert else "http"
    logger.info(
        "Starting uvicorn: %s://%s:%d  workers=%d",
        scheme,
        host,
        port,
        workers,
    )

    uvicorn_kwargs: dict = dict(
        host=host,
        port=port,
        workers=workers,
        loop="uvloop",
        http="httptools",
        log_level=log_level,
        access_log=False,
    )
    if ssl_cert:
        uvicorn_kwargs["ssl_certfile"] = ssl_cert
        uvicorn_kwargs["ssl_keyfile"] = ssl_key

    uvicorn.run("inference_server.app:app", **uvicorn_kwargs)


if __name__ == "__main__":
    main()
