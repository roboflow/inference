"""Bounded pipelining of V2 runs (opt-in; serial execution stays the default).

    options.py   PipelineOptions: P pulse teams, W workers per team, overload
                 policy and a pending-operator-pulse count safeguard
    stages.py    stage gates, coordination of one run, counters
    workers.py   reusable run-owned worker teams without a task queue
    teams.py     ready graph units and dependency-driven branch progress
    passive.py   session.pipeline(): bounded submissions of passive runs

Only ``options`` is imported eagerly, so ``plan`` can name ``PipelineOptions``
without loading the scheduling modules. The other public names load on first
access.
"""

import importlib
from typing import Any

from roboflow_workflows.v2.engine.pipelining.options import (
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
