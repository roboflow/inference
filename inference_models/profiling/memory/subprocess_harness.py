from __future__ import annotations

import importlib
import json
import multiprocessing as mp
from typing import Any, Dict


def _run_worker(
    queue: mp.Queue,
    worker_module: str,
    worker_entry: str,
    data: Dict[str, Any],
) -> None:
    module = importlib.import_module(worker_module)
    worker_main = getattr(module, worker_entry)
    message = worker_main(data)

    queue.put(message)


def run_profile_subprocess(
    payload: Dict[str, Any],
    *,
    worker_module: str,
    worker_entry: str = "worker_main",
    harness_label: str = "Profiling",
    mp_context: str = "spawn",
) -> Dict[str, Any]:
    """Spawn an isolated worker process and return the JSON-ready result dict.

    ``worker_module`` / ``worker_entry`` must resolve to a top-level callable
    importable under ``spawn`` (no nested closures).

    A separate process reduces allocator cache contamination across scenario sweeps,
    matching the workflow described in ``docs/description.md``.
    """
    ctx = mp.get_context(mp_context)

    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_run_worker,
        args=(queue, worker_module, worker_entry, payload),
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        raise RuntimeError(
            f"{harness_label} worker exited with code {proc.exitcode}; "
            "see stderr from the worker process if present."
        )

    try:
        msg = queue.get_nowait()
    except Exception as exc:
        raise RuntimeError(
            f"{harness_label} worker produced no result message."
        ) from exc

    if not msg.get("ok"):
        error_message = (
            msg.get("error") or f"{harness_label} worker failed without detail."
        )

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
