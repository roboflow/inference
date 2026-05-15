from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict

from profiling.memory.onnx_worker import worker_main


def _run_worker(queue: mp.Queue, data: Dict[str, Any]) -> None:
    message = worker_main(data)

    queue.put(message)


def run_onnx_profile_subprocess(
    payload: Dict[str, Any],
    *,
    mp_context: str = "spawn",
) -> Dict[str, Any]:
    """Spawn an isolated ONNX Runtime worker process and return the result dict."""
    ctx = mp.get_context(mp_context)

    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_run_worker,
        args=(queue, payload),
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        raise RuntimeError(
            f"ONNX profiling worker exited with code {proc.exitcode}; "
            "see stderr from the worker process if present."
        )

    try:
        msg = queue.get_nowait()
    except Exception as exc:
        raise RuntimeError("ONNX profiling worker produced no result message.") from exc

    if not msg.get("ok"):
        error_message = msg.get("error") or "ONNX profiling worker failed."

        raise RuntimeError(error_message)

    result = msg["result"]

    return result
