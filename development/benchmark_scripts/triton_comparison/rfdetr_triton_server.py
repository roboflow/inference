# pyright: reportMissingImports=false
"""Prepare and serve an RF-DETR TensorRT package with Triton.

The module expects a Roboflow TRT model package directory containing
``engine.plan``, ``trt_config.json``, and ``inference_config.json``. It creates a
Triton model repository with the plan copied to ``<model-name>/1/model.plan``,
writes ``config.pbtxt`` from the TensorRT engine metadata, and then starts
``tritonserver``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Optional

import click


_DEFAULT_MODEL_NAME = "rfdetr"
_DEFAULT_MODEL_PACKAGE_DIR = "/models/rfdetr-trt-package"
_DEFAULT_MODEL_REPOSITORY_DIR = "/triton-model-repository"
_PLAN_FILE_NAME = "engine.plan"
_TRITON_PLAN_FILE_NAME = "model.plan"


def _read_json(path: Path) -> dict[str, Any]:
    decoded_json = json.loads(path.read_text())
    if not isinstance(decoded_json, dict):
        raise ValueError(f"Expected JSON object in {path}.")

    return decoded_json


def _load_trt_engine(engine_path: Path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine_bytes = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise ValueError(f"Could not deserialize TensorRT engine: {engine_path}")

    return engine


def _triton_dtype(trt_dtype: Any) -> str:
    import tensorrt as trt

    dtype_mapping = {
        trt.DataType.FLOAT: "TYPE_FP32",
        trt.DataType.HALF: "TYPE_FP16",
        trt.DataType.INT8: "TYPE_INT8",
        trt.DataType.INT32: "TYPE_INT32",
        trt.DataType.BOOL: "TYPE_BOOL",
    }
    if hasattr(trt.DataType, "INT64"):
        dtype_mapping[trt.DataType.INT64] = "TYPE_INT64"

    triton_dtype = dtype_mapping[trt_dtype]

    return triton_dtype


def _tensor_shape(engine: Any, tensor_name: str) -> list[int]:
    raw_shape = list(engine.get_tensor_shape(tensor_name))
    shape = [int(dimension) for dimension in raw_shape]

    return shape


def _tensor_mode_name(engine: Any, tensor_name: str) -> str:
    import tensorrt as trt

    tensor_mode = engine.get_tensor_mode(tensor_name)
    if tensor_mode == trt.TensorIOMode.INPUT:
        mode_name = "input"
    elif tensor_mode == trt.TensorIOMode.OUTPUT:
        mode_name = "output"
    else:
        raise ValueError(f"Unsupported TensorRT IO mode for tensor {tensor_name}.")

    return mode_name


def _inspect_engine(engine_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    engine = _load_trt_engine(engine_path=engine_path)
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for index in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(index)
        tensor_metadata = {
            "name": tensor_name,
            "data_type": _triton_dtype(engine.get_tensor_dtype(tensor_name)),
            "dims": _tensor_shape(engine=engine, tensor_name=tensor_name),
        }
        mode_name = _tensor_mode_name(engine=engine, tensor_name=tensor_name)
        if mode_name == "input":
            inputs.append(tensor_metadata)
        else:
            outputs.append(tensor_metadata)

    if not inputs:
        raise ValueError(f"No TensorRT inputs found in {engine_path}.")
    if not outputs:
        raise ValueError(f"No TensorRT outputs found in {engine_path}.")

    return inputs, outputs


def _configured_max_batch_size(
    *,
    trt_config: dict[str, Any],
    inferred_input_dims: list[int],
    override: Optional[int],
) -> int:
    if override is not None:
        return override

    dynamic_batch_size_max = trt_config.get("dynamic_batch_size_max")
    if dynamic_batch_size_max is not None:
        resolved_max_batch_size = int(dynamic_batch_size_max)

        return resolved_max_batch_size

    if inferred_input_dims and inferred_input_dims[0] == -1:
        return 1

    return 0


def _triton_dims(
    *,
    tensor_dims: list[int],
    max_batch_size: int,
) -> list[int]:
    if max_batch_size > 0:
        dims = tensor_dims[1:]
    else:
        dims = tensor_dims

    return dims


def _format_dims(dims: list[int]) -> str:
    rendered_dims = ", ".join(str(dimension) for dimension in dims)
    formatted_dims = f"[ {rendered_dims} ]"

    return formatted_dims


def _format_io_block(
    *,
    block_name: str,
    tensors: list[dict[str, Any]],
    max_batch_size: int,
) -> str:
    entries = []
    for tensor in tensors:
        dims = _triton_dims(
            tensor_dims=tensor["dims"],
            max_batch_size=max_batch_size,
        )
        entry = (
            "  {\n"
            f"    name: \"{tensor['name']}\"\n"
            f"    data_type: {tensor['data_type']}\n"
            f"    dims: {_format_dims(dims)}\n"
            "  }\n"
        )
        entries.append(entry)

    rendered_entries = ",\n".join(entries)
    rendered_block = f"{block_name} [\n{rendered_entries}\n]"

    return rendered_block


def _write_triton_config(
    *,
    model_repository_dir: Path,
    model_name: str,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    max_batch_size: int,
    instance_count: int,
) -> Path:
    model_dir = model_repository_dir / model_name
    config_path = model_dir / "config.pbtxt"
    config = (
        f"name: \"{model_name}\"\n"
        "platform: \"tensorrt_plan\"\n"
        f"max_batch_size: {max_batch_size}\n"
        f"{_format_io_block(block_name='input', tensors=inputs, max_batch_size=max_batch_size)}\n"
        f"{_format_io_block(block_name='output', tensors=outputs, max_batch_size=max_batch_size)}\n"
        "instance_group [\n"
        "  {\n"
        f"    count: {instance_count}\n"
        "    kind: KIND_GPU\n"
        "  }\n"
        "]\n"
    )
    config_path.write_text(config)

    return config_path


def _prepare_model_repository(
    *,
    model_package_dir: Path,
    model_repository_dir: Path,
    model_name: str,
    max_batch_size: Optional[int],
    instance_count: int,
) -> Path:
    engine_path = model_package_dir / _PLAN_FILE_NAME
    trt_config_path = model_package_dir / "trt_config.json"
    inference_config_path = model_package_dir / "inference_config.json"
    for required_path in (engine_path, trt_config_path, inference_config_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required model package file missing: {required_path}")

    trt_config = _read_json(path=trt_config_path)
    model_dir = model_repository_dir / model_name
    version_dir = model_dir / "1"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    version_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(engine_path, version_dir / _TRITON_PLAN_FILE_NAME)
    inputs, outputs = _inspect_engine(engine_path=engine_path)
    resolved_max_batch_size = _configured_max_batch_size(
        trt_config=trt_config,
        inferred_input_dims=inputs[0]["dims"],
        override=max_batch_size,
    )
    config_path = _write_triton_config(
        model_repository_dir=model_repository_dir,
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        max_batch_size=resolved_max_batch_size,
        instance_count=instance_count,
    )
    shutil.copy2(inference_config_path, model_dir / "inference_config.json")
    shutil.copy2(trt_config_path, model_dir / "trt_config.json")

    print(f"[triton-server] model repository prepared: {model_repository_dir}", flush=True)
    print(f"[triton-server] model config written: {config_path}", flush=True)

    return model_repository_dir


def _build_triton_command(
    *,
    model_repository_dir: Path,
    strict_model_config: bool,
    http_port: int,
    grpc_port: int,
    metrics_port: int,
    log_verbose: int,
    extra_args: tuple[str, ...],
) -> list[str]:
    strict_model_config_value = "true" if strict_model_config else "false"
    command = [
        "tritonserver",
        f"--model-repository={model_repository_dir}",
        f"--strict-model-config={strict_model_config_value}",
        f"--http-port={http_port}",
        f"--grpc-port={grpc_port}",
        f"--metrics-port={metrics_port}",
    ]
    if log_verbose > 0:
        command.append(f"--log-verbose={log_verbose}")
    command.extend(extra_args)

    return command


def _serve_triton(command: list[str]) -> None:
    print(f"[triton-server] starting: {' '.join(command)}", flush=True)
    os.execvp(command[0], command)


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
)
@click.option(
    "--model-package-dir",
    envvar="MODEL_PACKAGE_DIR",
    default=_DEFAULT_MODEL_PACKAGE_DIR,
    show_default=True,
    help="RF-DETR TRT package directory containing engine.plan.",
)
@click.option(
    "--model-repository-dir",
    envvar="TRITON_MODEL_REPOSITORY_DIR",
    default=_DEFAULT_MODEL_REPOSITORY_DIR,
    show_default=True,
    help="Directory where the Triton model repository will be materialized.",
)
@click.option(
    "--model-name",
    envvar="TRITON_MODEL_NAME",
    default=_DEFAULT_MODEL_NAME,
    show_default=True,
    help="Triton model name exposed by the server.",
)
@click.option(
    "--max-batch-size",
    type=click.IntRange(
        min=0,
    ),
    default=None,
    help="Override Triton max_batch_size. By default, read from trt_config.json.",
)
@click.option(
    "--instance-count",
    type=click.IntRange(
        min=1,
    ),
    envvar="TRITON_INSTANCE_COUNT",
    default=1,
    show_default=True,
    help="Number of GPU model instances in Triton config.pbtxt.",
)
@click.option(
    "--strict-model-config/--no-strict-model-config",
    envvar="TRITON_STRICT_MODEL_CONFIG",
    default=True,
    show_default=True,
)
@click.option(
    "--http-port",
    type=click.IntRange(
        min=1,
        max=65535,
    ),
    envvar="TRITON_HTTP_PORT",
    default=8000,
    show_default=True,
)
@click.option(
    "--grpc-port",
    type=click.IntRange(
        min=1,
        max=65535,
    ),
    envvar="TRITON_GRPC_PORT",
    default=8001,
    show_default=True,
)
@click.option(
    "--metrics-port",
    type=click.IntRange(
        min=1,
        max=65535,
    ),
    envvar="TRITON_METRICS_PORT",
    default=8002,
    show_default=True,
)
@click.option(
    "--log-verbose",
    type=click.IntRange(
        min=0,
    ),
    envvar="TRITON_LOG_VERBOSE",
    default=0,
    show_default=True,
)
@click.pass_context
def _main(
    ctx: click.Context,
    model_package_dir: str,
    model_repository_dir: str,
    model_name: str,
    max_batch_size: Optional[int],
    instance_count: int,
    strict_model_config: bool,
    http_port: int,
    grpc_port: int,
    metrics_port: int,
    log_verbose: int,
) -> None:
    repository_dir = _prepare_model_repository(
        model_package_dir=Path(model_package_dir),
        model_repository_dir=Path(model_repository_dir),
        model_name=model_name,
        max_batch_size=max_batch_size,
        instance_count=instance_count,
    )
    command = _build_triton_command(
        model_repository_dir=repository_dir,
        strict_model_config=strict_model_config,
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
        log_verbose=log_verbose,
        extra_args=tuple(ctx.args),
    )
    _serve_triton(command=command)


if __name__ == "__main__":
    try:
        _main()
    except Exception as error:
        print(f"[triton-server] failed: {error}", file=sys.stderr, flush=True)
        raise
