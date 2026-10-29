"""Experimental TensorRT build and execution for native SAM3 text prompts."""

import json
import time
from pathlib import Path
from threading import RLock
from typing import Dict

import torch

from inference_models.models.sam3.trt_export import (
    DECODER_OUTPUTS,
    IMAGE_SIZE,
    PARTS,
    TEXT_LENGTH,
    TEXT_OUTPUTS,
    VISION_OUTPUTS,
    file_sha256,
)


def _tensorrt():
    import tensorrt

    return tensorrt


def _environment() -> Dict:
    trt = _tensorrt()
    return {
        "tensorrt_version": trt.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_capability": list(torch.cuda.get_device_capability()),
    }


def build_sam3_engines(
    export_dir: Path,
    engine_dir: Path,
    fp16: bool = False,
    workspace_gib: int = 8,
) -> Dict:
    """Build static TensorRT engines with detailed inspection metadata.

    Args:
        export_dir (Path): Directory with native ONNX exports and their manifest.
        engine_dir (Path): Empty destination for engines and build measurements.
        fp16 (bool): Enable FP16 tactics while retaining FP32 Softmax.
        workspace_gib (int): Builder workspace limit in GiB per engine.

    Returns:
        Dict: Engine provenance, checksums, and per-engine build durations.

    Raises:
        ValueError: If the manifest, destination, or shapes are unsupported.
        RuntimeError: If TensorRT cannot parse or build an engine.
    """
    trt = _tensorrt()
    manifest = json.loads((export_dir / "export.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported SAM3 export manifest")
    elif workspace_gib < 1:
        raise ValueError("workspace_gib must be positive")
    elif engine_dir.exists() and any(engine_dir.iterdir()):
        raise ValueError(
            "Build into an empty directory to avoid mixing engine versions"
        )

    engine_dir.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    engines = {}
    build_seconds = {}
    for part in PARTS:
        started = time.perf_counter()
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str((export_dir / f"{part}.onnx").resolve())):
            errors = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"Cannot parse SAM3 {part}:\n{errors}")
        for index in range(network.num_inputs):
            tensor = network.get_input(index)
            if any(dimension < 1 for dimension in tensor.shape):
                raise ValueError(
                    f"Expected static, nonempty input: {tensor.name} {tensor.shape}"
                )
        configuration = builder.create_builder_config()
        configuration.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        configuration.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_gib << 30
        )
        configuration.clear_flag(trt.BuilderFlag.TF32)
        if fp16:
            configuration.set_flag(trt.BuilderFlag.FP16)
            configuration.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            # Softmax precision is sensitive in the split-export reference.
            for index in range(network.num_layers):
                layer = network.get_layer(index)
                if layer.type == trt.LayerType.SOFTMAX:
                    layer.precision = trt.float32
                    layer.set_output_type(0, trt.float32)
        serialized = builder.build_serialized_network(network, configuration)
        if serialized is None:
            raise RuntimeError(f"TensorRT failed to build SAM3 {part}")
        path = engine_dir / f"{part}.engine"
        path.write_bytes(bytes(serialized))
        engines[part] = file_sha256(path)
        build_seconds[part] = time.perf_counter() - started

    manifest.update(
        environment=_environment(),
        precision="fp16" if fp16 else "fp32",
        profiling_verbosity="detailed",
        engines_sha256=engines,
        build_wall_seconds=build_seconds,
    )
    (engine_dir / "engines.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class _TensorRTEngine:
    def __init__(self, path: Path, logger):
        trt = _tensorrt()
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot load {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Cannot create TensorRT execution context for {path}")
        # Detailed NVTX annotations distort enqueue latency during normal timing.
        self.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        self.names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.inputs = {
            name
            for name in self.names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        }
        self.dtypes = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
        }

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        trt = _tensorrt()
        if set(inputs) != self.inputs:
            raise ValueError(f"Expected inputs {self.inputs}, received {set(inputs)}")
        bindings = {}
        device = torch.device("cuda", torch.cuda.current_device())
        for name in self.names:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.dtypes[self.engine.get_tensor_dtype(name)]
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"Expected GPU binding: {name}")
            elif name in self.inputs:
                tensor = inputs[name]
                if (
                    tuple(tensor.shape) != shape
                    or tensor.dtype != dtype
                    or tensor.device != device
                ):
                    raise ValueError(f"{name} requires {shape}, {dtype} on {device}")
                bindings[name] = tensor.contiguous()
            else:
                bindings[name] = torch.empty(shape, dtype=dtype, device=device)
            if not self.context.set_tensor_address(name, bindings[name].data_ptr()):
                raise RuntimeError(f"Cannot bind SAM3 tensor {name}")
        stream = torch.cuda.current_stream()
        try:
            if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("SAM3 TensorRT execution failed")
        finally:
            # Bindings must stay alive until the asynchronous execution finishes.
            stream.synchronize()
        return {
            name: tensor for name, tensor in bindings.items() if name not in self.inputs
        }


class SplitSAM3TensorRT:
    """Single-image, single-text-query pipeline. Returns native raw output tensors."""

    def __init__(self, engine_dir: Path, model_package: Path):
        trt = _tensorrt()
        self.manifest = json.loads((engine_dir / "engines.json").read_text())
        expected = {
            "schema_version": 1,
            "sam3_version": "0.1.4",
            "image_size": IMAGE_SIZE,
            "text_length": TEXT_LENGTH,
            "batch_size": 1,
            "prompt_mode": "text_only",
            "environment": _environment(),
            "weights_sha256": file_sha256(model_package / "weights.pt"),
            "tokenizer_sha256": file_sha256(
                model_package / "bpe_simple_vocab_16e6.txt.gz"
            ),
        }
        for name, value in expected.items():
            if self.manifest.get(name) != value:
                raise ValueError(
                    f"SAM3 engine mismatch for {name}; rebuild for this package and GPU stack"
                )
        self.logger = trt.Logger(trt.Logger.WARNING)
        self._lock = RLock()
        self._device = torch.cuda.current_device()
        self.engines = {}
        for part in PARTS:
            path = engine_dir / f"{part}.engine"
            if file_sha256(path) != self.manifest["engines_sha256"][part]:
                raise ValueError(f"SAM3 engine checksum mismatch: {part}")
            self.engines[part] = _TensorRTEngine(path, self.logger)

    def __call__(
        self, images: torch.Tensor, input_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if torch.cuda.current_device() != self._device:
            raise ValueError("Use the CUDA device on which this pipeline was loaded")
        with self._lock:
            vision = self.engines["vision"]({"images": images})
            text = self.engines["text"]({"input_ids": input_ids})
            inputs = {name: vision[name] for name in VISION_OUTPUTS}
            inputs.update({name: text[name] for name in TEXT_OUTPUTS})
            output = self.engines["decoder"](inputs)
        return {name: output[name] for name in DECODER_OUTPUTS}
