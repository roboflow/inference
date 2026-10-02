"""Synthetic CPU scheduling benchmark; no native decode, model, or inference.

Run with a working Inference development environment from the repository root:
    python -m development.benchmark_parallel_workflow_startup
"""

import json
from statistics import median
from time import perf_counter, sleep
from unittest.mock import patch

from inference.core.interfaces.stream import inference_pipeline as pipeline_module
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.workflows.execution_engine.v1.compiler import core as compiler


class _Manager(dict):
    max_size = 1

    def add_model(self, model_id, api_key, **kwargs):
        sleep(0.2)
        self[model_id] = object()


class _Source:
    def start(self):
        sleep(0.15)

    def terminate(self, **kwargs):
        pass


def _run(*, parallel, run):
    source, manager = _Source(), _Manager()
    definition = {
        "version": "1.0",
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": f"detect_{parallel}_{run}",
                "images": "$inputs.image",
                "model_id": "inert/1",
            }
        ],
        "outputs": [],
    }
    graph = compiler.prepare_execution_graph

    def delayed_graph(**kwargs):
        sleep(0.1)
        return graph(**kwargs)

    with patch.object(
        pipeline_module, "prepare_video_sources", return_value=[source]
    ), patch.object(compiler, "prepare_execution_graph", delayed_graph):
        start = perf_counter()
        pipeline = InferencePipeline.init_with_workflow(
            video_reference="inert",
            workflow_specification=definition,
            model_manager=manager,
            parallel_startup=parallel,
        )
        if not parallel:
            # Mimic the default ordering without executing any workflow/frame.
            source.start()
            manager.add_model(model_id="inert/1", api_key="inert")
        elapsed = perf_counter() - start
        timings = pipeline.startup_phase_timings
        pipeline.join()
        return {"elapsed_seconds": elapsed, "phases": timings}


def main():
    """Print measured synthetic scheduling timings as JSON."""
    _run(parallel=False, run="warm_imports")
    serial, parallel = [], []
    for run in range(5):
        serial.append(_run(parallel=False, run=run))
        parallel.append(_run(parallel=True, run=run))
    print(
        json.dumps(
            {
                "evidence": "synthetic CPU sleeps with real workflow compilation; no GPU/native/model work",
                "delays_seconds": {"source": 0.15, "models": 0.2, "graph": 0.1},
                "serial_median_seconds": median(
                    item["elapsed_seconds"] for item in serial
                ),
                "parallel_median_seconds": median(
                    item["elapsed_seconds"] for item in parallel
                ),
                "serial": serial,
                "parallel": parallel,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
