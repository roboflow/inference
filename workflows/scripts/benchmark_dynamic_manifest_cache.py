"""Measure dynamic manifest retention without models, images or HTTP transport.

Run from the repository root with its Python environment, for example:
PYTHONPATH=workflows:inference_models python workflows/scripts/benchmark_dynamic_manifest_cache.py --output /tmp/manifest-cache.jsonl

Each simulated request compiles three tiny custom blocks through the real
dynamic block compiler and parses their manifests. No compilation objects or
parsed results survive the iteration; only weak references are retained for
measurement, and dead references are pruned at each sample. Optional explicit
GC distinguishes live objects from RSS retained by the allocator. Run in
separate processes before and after a change. This is not a full HTTP benchmark.
"""

import gc
import json
import platform
import time
import weakref
from pathlib import Path

import click
import psutil
import pydantic
import pydantic_core
from roboflow_workflows.execution_engine.introspection import (
    blocks_loader,
    schema_parser,
)
from roboflow_workflows.execution_engine.v1.compiler.syntactic_parser import (
    build_workflow_definition_entity,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
    compile_dynamic_blocks,
)


def _definitions():
    definitions = []
    for index in range(3):
        definitions.append(
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": f"MemoryProbe{index}",
                    "inputs": {
                        "value": {
                            "type": "DynamicInputDefinition",
                            "value_types": ["string"],
                            "selector_types": ["input_parameter", "step_output"],
                            "selector_data_kind": {
                                "input_parameter": ["string"],
                                "step_output": ["string"],
                            },
                        }
                    },
                    "outputs": {
                        "output": {
                            "type": "DynamicOutputDefinition",
                            "kind": ["string"],
                        }
                    },
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": (
                        'def run(self, value):\n    return {"output": value}'
                    ),
                },
            }
        )

    return definitions


def _compile_and_parse(definitions, *, include_workflow_model):
    blocks = compile_dynamic_blocks(dynamic_blocks_definitions=definitions)
    if include_workflow_model:
        build_workflow_definition_entity(available_blocks=blocks)
    references = []
    for block in blocks:
        schema_parser.parse_block_manifest(manifest_type=block.manifest_class)
        references.append(weakref.ref(block.manifest_class))

    return references


def _sample(*, requests, references, started, process, collect, phase="running"):
    rss_before_gc = process.memory_info().rss
    if collect:
        gc.collect()
    references[:] = [reference for reference in references if reference() is not None]
    record = {
        "phase": phase,
        "requests": requests,
        "generated_manifests": requests * 3,
        "live_manifests": len(references),
        "rss_before_gc_mib": round(rss_before_gc / 2**20, 3),
        "rss_mib": round(process.memory_info().rss / 2**20, 3),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "manifest_cache": schema_parser.parse_block_manifest.cache_info()._asdict(),
        "schema_cache": blocks_loader._cached_model_json_schema.cache_info()._asdict(),
        "outputs_cache": blocks_loader._cached_describe_outputs.cache_info()._asdict(),
    }

    return record


@click.command()
@click.option(
    "--requests",
    type=click.IntRange(min=1),
    default=10000,
    show_default=True,
)
@click.option(
    "--sample-every",
    type=click.IntRange(min=1),
    default=500,
    show_default=True,
)
@click.option(
    "--collect-at-sample/--no-collect-at-sample",
    default=False,
    show_default=True,
    help="Force cyclic garbage collection at each measurement.",
)
@click.option(
    "--include-workflow-model",
    is_flag=True,
    help="Also build the workflow validation model around the dynamic blocks.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Write measurements as JSON Lines to this file.",
)
def main(
    requests: int,
    sample_every: int,
    collect_at_sample: bool,
    include_workflow_model: bool,
    output: Path,
) -> None:
    """Compile repeated definitions and record cache ownership and process RSS.

    Args:
        requests: Number of simulated requests, each creating three manifests.
        sample_every: Number of requests between measurements.
        collect_at_sample: Whether to force GC before every measurement.
        include_workflow_model: Whether to exercise workflow validation-model creation.
        output: Destination for environment metadata and measurement records.
    """
    definitions = _definitions()
    # Warm imports and static kind registries before measuring the baseline.
    _compile_and_parse(definitions, include_workflow_model=include_workflow_model)
    blocks_loader.clear_caches()
    schema_parser.clear_cache()
    process = psutil.Process()
    references = []
    started = time.monotonic()
    with output.open("w") as stream:

        def emit(record):
            line = json.dumps(record)
            stream.write(line + "\n")
            stream.flush()
            click.echo(line)

        emit(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pydantic": pydantic.__version__,
                "pydantic_core": pydantic_core.__version__,
                "requests": requests,
                "sample_every": sample_every,
                "collect_at_sample": collect_at_sample,
                "include_workflow_model": include_workflow_model,
            }
        )
        emit(
            _sample(
                requests=0,
                references=references,
                started=started,
                process=process,
                collect=True,
            )
        )
        for count in range(1, requests + 1):
            references.extend(
                _compile_and_parse(
                    definitions, include_workflow_model=include_workflow_model
                )
            )
            if count % sample_every == 0 or count == requests:
                emit(
                    _sample(
                        requests=count,
                        references=references,
                        started=started,
                        process=process,
                        collect=collect_at_sample,
                    )
                )

        blocks_loader.clear_caches()
        schema_parser.clear_cache()
        emit(
            _sample(
                requests=requests,
                references=references,
                started=started,
                process=process,
                collect=True,
                phase="after_cache_clear",
            )
        )


if __name__ == "__main__":
    main()
