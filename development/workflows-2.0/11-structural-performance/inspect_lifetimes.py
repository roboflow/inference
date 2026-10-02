"""Do completed V2 runs release their payloads before cyclic collection?

A lifetime diagnostic, not a speed measurement::

    session = compile(passthrough: WorkflowBatchInput value -> output value)
    gc.disable()                        # this process only, restored below
    repeat --runs times:
        payload = _Payload(); keep weakref(payload)
        result = session.run({"value": [payload]}); drop result and payload
    count payloads still alive          # held by reference cycles if > 0
    gc.collect(); count again           # 0 here: nothing else keeps them
    gc.enable() if it was enabled

Automatic cyclic GC is switched off only so that payloads kept alive by
reference cycles stay visible. It is not a performance setting. The JSON
names the engine modules that build and read run entries, with their sha256,
so a result names the engine it came from. An engine that frees entries
without the cyclic collector reports 0 alive before collection.

    python inspect_lifetimes.py --runs 32
"""

import gc
import hashlib
import json
import weakref
from pathlib import Path
from typing import Any, Dict

import click
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.execution import entries, outputs

PASSTHROUGH_WORKFLOW = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowBatchInput", "name": "value"}],
    "steps": [],
    "outputs": [{"type": "JsonField", "name": "value", "selector": "$inputs.value"}],
}


class _Payload:
    """A plain object whose lifetime a weak reference can observe."""


@click.command()
@click.option(
    "--runs",
    type=click.IntRange(
        min=1,
        max=10000,
    ),
    default=32,
    show_default=True,
    help="Completed runs to inspect.",
)
def main(runs: int) -> None:
    """Print how many payloads of completed runs stay alive, as JSON."""
    report = inspect_lifetimes(runs=runs)

    click.echo(json.dumps(report, indent=2))


def inspect_lifetimes(*, runs: int) -> Dict[str, Any]:
    """Run the passthrough workflow ``runs`` times with automatic GC off.

    Args:
        runs: Number of independent completed runs.

    Returns:
        ``payloads_alive_after_results_dropped`` (before ``gc.collect``),
        ``payloads_alive_after_cyclic_collection``, the engine sources and
        a note. GC is restored to its previous state before returning.

    Raises:
        RuntimeError: When a run does not return its own payload.
    """
    session = compile_workflow(
        PASSTHROUGH_WORKFLOW, catalogue=Catalogue([])
    ).create_session()
    gc.collect()
    automatic_gc = gc.isenabled()
    gc.disable()
    try:
        references = [_one_run(session) for _ in range(runs)]
        before_collection = sum(reference() is not None for reference in references)
        gc.collect()
        after_collection = sum(reference() is not None for reference in references)
    finally:
        if automatic_gc:
            gc.enable()

    report = {
        "runs": runs,
        "payloads_alive_after_results_dropped": before_collection,
        "payloads_alive_after_cyclic_collection": after_collection,
        "engine_sources": _engine_sources(),
        "note": "Lifetime diagnostic only; does not measure FPS or explain live stalls.",
    }

    return report


def _one_run(session: Any) -> "weakref.ref[_Payload]":
    # Only the weak reference leaves this frame; result and payload are dropped.
    payload = _Payload()
    reference = weakref.ref(payload)
    result = session.run({"value": [payload]})
    if result.rows()[0]["value"] is not payload:
        raise RuntimeError("the passthrough run did not return its own payload")

    return reference


def _engine_sources() -> Dict[str, Dict[str, str]]:
    sources = {}
    for module in (entries, outputs):
        path = Path(module.__file__)
        sources[module.__name__] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    return sources


if __name__ == "__main__":
    main()
