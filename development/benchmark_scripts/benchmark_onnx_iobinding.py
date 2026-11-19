"""Compare complete tensor I/O binding paths on static, known-input ONNX fixtures.

Run in the parallel test Docker image, with --gpus all for CUDA. CPU mode is a
local harness check, not GPU performance evidence. Both paths use the same
session/input, warm up equally, and alternate execution order between samples.
Timing includes binding, execution, CPU output retrieval and owned float32
conversion. Model/session creation, input upload and correctness checks are
outside timing. These synthetic Neg models measure output-handling overhead,
not end-to-end model throughput. No credentials or downloaded weights needed.
"""

import json
import platform
import time
from pathlib import Path

import click
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

from inference.core.utils.onnx import _run_session_with_tensor_iobinding

OUTPUT_SHAPES = {
    "small": (100, 6),
    "detection": (8400, 85),
    "mask": (32, 160, 160),
}


def _run_numpy_allocated_reference(session, *, input_name, input_data):
    """Original tensor binding path from main at 378a680570ff8eb6ab1de66fc7fa9b7a96e0dd97.

    Kept as the before-allocation reference. Static, same-dtype fixtures ensure
    both implementations are valid. Each call includes the final float32 copy.
    """
    binding = session.io_binding()
    predictions = []
    dtype = None
    for output in session.get_outputs():
        if dtype is None:
            dtype = np.float16 if "16" in output.type else np.float32
        prediction = np.empty(output.shape, dtype=dtype)
        binding.bind_output(
            name=output.name,
            device_type="cpu",
            device_id=0,
            element_type=dtype,
            shape=output.shape,
            buffer_ptr=prediction.ctypes.data,
        )
        predictions.append(prediction)

    input_data = input_data.contiguous()
    binding.bind_input(
        name=input_name,
        device_type=input_data.device.type,
        device_id=input_data.device.index if input_data.device.index is not None else 0,
        element_type=dtype,
        shape=input_data.shape,
        buffer_ptr=input_data.data_ptr(),
    )
    binding.synchronize_inputs()
    session.run_with_iobinding(binding)
    predictions = [prediction.astype(np.float32) for prediction in predictions]

    return predictions


def _make_case(*, shape, dtype, device):
    element_type = TensorProto.FLOAT16 if dtype == "float16" else TensorProto.FLOAT
    graph = helper.make_graph(
        [helper.make_node("Neg", ["images"], ["predictions"])],
        "static_output_allocation",
        [helper.make_tensor_value_info("images", element_type, shape)],
        [helper.make_tensor_value_info("predictions", element_type, shape)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9
    )
    onnx.checker.check_model(model)
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if device == "cuda":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=[provider]
    )
    assert provider in session.get_providers(), session.get_providers()
    values = ((np.arange(np.prod(shape), dtype=np.float32) % 97) / 32).astype(dtype)
    values = values.reshape(shape)
    inputs = torch.from_numpy(values).to(device)
    expected = -values.astype(np.float32)

    return session, inputs, expected


def _check_predictions(predictions, *, expected):
    assert len(predictions) == 1
    assert predictions[0].dtype == np.float32
    assert predictions[0].flags.owndata
    np.testing.assert_array_equal(predictions[0], expected)


def _check_cpu_output_view(session, *, inputs):
    """Verify the installed runtime avoids a copy for CPU OrtValue.numpy()."""
    binding = session.io_binding()
    binding.bind_cpu_input("images", inputs.cpu().numpy())
    binding.bind_output("predictions", "cpu")
    session.run_with_iobinding(binding)
    output = binding.get_outputs()[0]
    view = output.numpy()
    copied = binding.copy_outputs_to_cpu()[0]
    assert output.device_name() == "cpu"
    assert view.ctypes.data == output.data_ptr()
    assert copied.ctypes.data != output.data_ptr()
    np.testing.assert_array_equal(copied, view)


def _measure_case(*, shape, dtype, device, warmup, iterations):
    session, inputs, expected = _make_case(shape=shape, dtype=dtype, device=device)
    _check_cpu_output_view(session, inputs=inputs)
    runners = {
        "numpy_allocation": _run_numpy_allocated_reference,
        "ort_allocation": _run_session_with_tensor_iobinding,
    }
    samples = {name: [] for name in runners}
    for runner in runners.values():
        for _ in range(warmup):
            predictions = runner(session, input_name="images", input_data=inputs)
        _check_predictions(predictions, expected=expected)
        del predictions

    names = list(runners)
    for iteration in range(iterations):
        order = names if iteration % 2 == 0 else names[::-1]
        for name in order:
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter_ns()
            predictions = runners[name](session, input_name="images", input_data=inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            samples[name].append(elapsed_ms)
            del predictions

    # Verify the measured paths again after allocator reuse, outside timing.
    for runner in runners.values():
        predictions = runner(session, input_name="images", input_data=inputs)
        _check_predictions(predictions, expected=expected)
        del predictions

    timings = {
        name: {
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "samples_ms": values,
        }
        for name, values in samples.items()
    }
    result = {
        "shape": shape,
        "dtype": dtype,
        "output_mib": expected.nbytes / (1024 * 1024),
        "cpu_numpy_view_verified": True,
        "timings": timings,
        "p50_change_percent": 100
        * (
            timings["ort_allocation"]["p50_ms"] / timings["numpy_allocation"]["p50_ms"]
            - 1
        ),
    }

    return result


@click.command()
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    show_default=True,
)
@click.option(
    "--warmup",
    type=click.IntRange(min=1),
    default=30,
    show_default=True,
)
@click.option(
    "--iterations",
    type=click.IntRange(min=2),
    default=200,
    show_default=True,
)
@click.option(
    "--batch-size",
    "batch_sizes",
    type=click.IntRange(min=1),
    multiple=True,
    default=(1, 8, 32),
    show_default=True,
)
@click.option(
    "--output-kind",
    "output_kinds",
    type=click.Choice(list(OUTPUT_SHAPES)),
    multiple=True,
    default=tuple(OUTPUT_SHAPES),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
)
def _main(device, warmup, iterations, batch_sizes, output_kinds, output):
    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA is required; no CPU fallback"
        assert "CUDAExecutionProvider" in ort.get_available_providers()

    metadata = {
        "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "onnxruntime": ort.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "warmup_per_path": warmup,
        "iterations_per_path": iterations,
        "order": "alternating AB/BA",
        "timed_scope": "complete tensor binding helper including owned float32 outputs",
        "cuda_sync": "before timer start and before timer stop",
    }
    click.echo(json.dumps(metadata))
    rows = [
        "| Output | Batch | Dtype | Float32 MiB | NumPy p50/p95 ms | ORT p50/p95 ms | p50 change |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    cases = []
    for kind in output_kinds:
        for batch_size in batch_sizes:
            for dtype in ("float16", "float32"):
                result = _measure_case(
                    shape=(batch_size, *OUTPUT_SHAPES[kind]),
                    dtype=dtype,
                    device=device,
                    warmup=warmup,
                    iterations=iterations,
                )
                result["output_kind"] = kind
                cases.append(result)
                before = result["timings"]["numpy_allocation"]
                after = result["timings"]["ort_allocation"]
                row = (
                    f"| {kind} | {batch_size} | {dtype} | {result['output_mib']:.3f} | "
                    f"{before['p50_ms']:.3f}/{before['p95_ms']:.3f} | "
                    f"{after['p50_ms']:.3f}/{after['p95_ms']:.3f} | "
                    f"{result['p50_change_percent']:+.1f}% |"
                )
                rows.append(row)
                click.echo(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"metadata": metadata, "cases": cases}, indent=2) + "\n"
    )
    output.with_suffix(".md").write_text(
        "Synthetic static Neg fixtures; lower latency is better.\n\n"
        + json.dumps(metadata, indent=2)
        + "\n\n"
        + "\n".join(rows)
        + "\n"
    )


if __name__ == "__main__":
    _main()
