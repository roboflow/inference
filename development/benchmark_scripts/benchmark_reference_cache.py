"""Measure cold rendering and warm reference-cache latency and allocations.

Run the same script, dependencies and arguments against both revisions. Input
generation, output hashing and tracemalloc are excluded from timing samples.
The URL control measures key lookup with a stub renderer, without network I/O.
"""

import base64
import dataclasses
import hashlib
import json
import os
import statistics
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import click
import cv2
import numpy as np
from roboflow_workflows import configuration

CONFIGURATIONS = {
    "small": (640, 360, 1, 3),
    "hd": (1920, 1080, 1, 30),
    "large": (4000, 3000, 10, 5),
    "dense": (4000, 3000, 10, 80),
    "url": (640, 360, 10, 80),
}


def _distribution(samples):
    ordered = sorted(samples)
    return {
        "p50_ms": statistics.median(samples) * 1000,
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000,
        "samples_ms": [sample * 1000 for sample in samples],
    }


def _emit(case, status, **metadata):
    path = os.environ.get("AGENT_TEST_SCENARIOS")
    if path:
        with Path(path).open("a") as stream:
            stream.write(
                json.dumps({"case_id": case, "status": status, **metadata}) + "\n"
            )


def _measure(refs, *, provider, size, cold_repetitions, warm_repetitions):
    width, height, count, label_count = CONFIGURATIONS[size]
    rng = np.random.default_rng(3363)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    payload = base64.b64encode(encoded).decode()
    labels = [f"class-{i}" for i in range(label_count)]
    examples = [
        {
            "image": {
                "type": "url" if size == "url" else "base64",
                "value": (
                    f"https://example.com/{j}.jpg"
                    if size == "url"
                    else payload.encode().decode()
                ),
            },
            "annotations": [
                {
                    "x": width * (i + 1) / (label_count + 1),
                    "y": height / 2 + j,
                    "width": width / 20,
                    "height": height / 10,
                    "class": label,
                }
                for i, label in enumerate(labels)
            ],
        }
        for j in range(count)
    ]
    cache = refs.ReferencePartsCache()
    render = (
        refs.gemini_reference_parts_for
        if provider == "gemini"
        else refs.openai_reference_parts_for
    )

    def run():
        if size == "url":
            return refs._reference_parts(
                examples,
                labels=labels,
                rendering=provider,
                cache=cache,
                render=lambda _: [{"text": "URL control"}],
            )
        return render(examples, labels=labels, cache=cache)

    run()  # Exclude first codec/import initialization.
    cold = []
    with patch.object(
        refs, "load_reference_example", wraps=refs.load_reference_example
    ) as load:
        for _ in range(cold_repetitions):
            cache.clear()
            start = time.perf_counter()
            parts = run()
            cold.append(time.perf_counter() - start)
        expected_loads = 0 if size == "url" else cold_repetitions * count
        assert load.call_count == expected_loads
        warm = []
        for _ in range(warm_repetitions):
            start = time.perf_counter()
            assert run() is parts
            warm.append(time.perf_counter() - start)
        # Trace a separate warm request, not the cold build or input creation.
        tracemalloc.start()
        try:
            assert run() is parts
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert load.call_count == expected_loads, "warm request decoded an image"
    assert not cache._pending
    return {
        "provider": provider,
        "configuration": size,
        "width": width,
        "height": height,
        "examples": count,
        "labels": label_count,
        "input_string_bytes": sum(
            len(example["image"]["value"]) for example in examples
        ),
        "cold": _distribution(cold),
        "warm": _distribution(warm),
        "warm_traced_current_bytes": current,
        "warm_traced_peak_bytes": peak,
        "retained_string_bytes": cache._size,
        "reference_loads": load.call_count,
        "output_sha256": hashlib.sha256(
            json.dumps(parts, sort_keys=True).encode()
        ).hexdigest(),
        "source": refs.__file__,
        "tensor": os.environ.get("ENABLE_TENSOR_DATA_REPRESENTATION", "False"),
    }


@click.command()
@click.option(
    "--provider",
    type=click.Choice(["gemini", "openai"]),
    required=True,
)
@click.option(
    "--size",
    type=click.Choice(list(CONFIGURATIONS)),
    required=True,
)
@click.option(
    "--cold-repetitions",
    type=click.IntRange(min=1),
    default=3,
    show_default=True,
)
@click.option(
    "--warm-repetitions",
    type=click.IntRange(min=1),
    default=25,
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
)
def main(provider, size, cold_repetitions, warm_repetitions, output):
    """Benchmark one configuration; emit registry events when tracking is enabled.

    Args:
        provider: Reference prompt renderer to exercise.
        size: Input dimensions, reference count and label density preset.
        cold_repetitions: Measured builds after clearing the cache.
        warm_repetitions: Measured lookups after the cache is populated.
        output: Destination for machine-readable measurements.
    """
    tensor = (
        os.environ.get("ENABLE_TENSOR_DATA_REPRESENTATION", "False").lower() == "true"
    )
    base = configuration.default_configuration()
    configuration.configure_process(
        dataclasses.replace(
            base,
            tensor=dataclasses.replace(
                base.tensor,
                representation_enabled=tensor,
                image_tensor_device=configuration.resolve_image_tensor_device(tensor),
            ),
        )
    )
    from roboflow_workflows.core_steps.common import vlm_reference_examples as refs

    case = f"{provider}-{size}"
    _emit(case, "running")
    start = time.perf_counter()
    try:
        result = _measure(
            refs,
            provider=provider,
            size=size,
            cold_repetitions=cold_repetitions,
            warm_repetitions=warm_repetitions,
        )
        output.write_text(json.dumps(result, indent=2))
    except Exception as error:
        _emit(
            case,
            "failed",
            detail=str(error),
            duration_seconds=time.perf_counter() - start,
        )
        raise
    _emit(
        case,
        "passed",
        measurements=result,
        duration_seconds=time.perf_counter() - start,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
