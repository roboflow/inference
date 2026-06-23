"""Tiny benchmark trace helper shared by server and MMP hot paths."""

from __future__ import annotations

import json
import logging
import os
import time
from itertools import count
from typing import Any

_TRACE_EVERY = int(os.environ.get("INFERENCE_BENCHMARK_TRACE_EVERY", "0") or 0)
_COUNTER = count(1)


def sampled(key: int | None = None) -> bool:
    if _TRACE_EVERY <= 0:
        return False
    if key is None:
        key = next(_COUNTER)
    return key % _TRACE_EVERY == 0


def ms(start: float, end: float | None = None) -> float:
    return round(((time.monotonic() if end is None else end) - start) * 1000, 3)


def log(logger: logging.Logger, event: str, **fields: Any) -> None:
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "bench_trace %s",
            json.dumps({"event": event, **fields}, separators=(",", ":"), default=str),
        )
