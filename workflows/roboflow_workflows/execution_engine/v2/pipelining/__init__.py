"""Bounded pipelining of V2 runs (opt-in; serial execution stays the default).

    options.py   PipelineOptions: max_in_flight and overload policy
    stages.py    stage gates, coordination of one run, counters
    workers.py   fixed worker pool without a queue, run-owned threads
    passive.py   session.pipeline(): bounded submissions of passive runs

Only ``options`` is imported eagerly, so ``plan`` can name ``PipelineOptions``
without loading the scheduling modules. The other public names load on first
access.
"""

import importlib
from typing import Any

from roboflow_workflows.execution_engine.v2.pipelining.options import (
    OverloadPolicy,
    PipelineOptions,
)

__all__ = [
    "OverloadPolicy",
    "PassivePipeline",
    "PipelineAbortedError",
    "PipelineCounters",
    "PipelineFullError",
    "PipelineOptions",
]

_LAZY = {
    "PipelineCounters": "stages",
    "PassivePipeline": "passive",
    "PipelineAbortedError": "passive",
    "PipelineFullError": "passive",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)

    return value
