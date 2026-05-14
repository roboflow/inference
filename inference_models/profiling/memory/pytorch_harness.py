from __future__ import annotations

import json
import multiprocessing as mp
from typing import Any, Dict

from profiling.memory.pytorch_worker import worker_main


def run_pytorch_profile_subprocess(
    payload: Dict[str, Any],
    *,
    mp_context: str = "spawn",
) -> Dict[str, Any]:
    """Spawn an isolated worker process and return the JSON-ready result dict.

    A separate process reduces allocator cache contamination across scenario sweeps,
    matching the workflow described in ``docs/description.md``.
    """
    ctx = mp.get_context(mp_context)

    def _target(queue: mp.Queue, data: Dict[str, Any]) -> None:
        message = worker_main(data)

        queue.put(message)

    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_target,
        args=(queue, payload),
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        raise RuntimeError(
            f"Profiling worker exited with code {proc.exitcode}; "
            "see stderr from the worker process if present."
        )

    try:
        msg = queue.get_nowait()
    except Exception as exc:
        raise RuntimeError("Profiling worker produced no result message.") from exc

    if not msg.get("ok"):
        error_message = msg.get("error") or "Profiling worker failed without detail."

        raise RuntimeError(error_message)

    result = msg["result"]

    return result


def dump_result_json(
    result: Dict[str, Any],
    *,
    path: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
