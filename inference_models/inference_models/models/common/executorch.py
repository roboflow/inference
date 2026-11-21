"""Load and run ExecuTorch programs (.pte model packages)."""

import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from packaging.version import Version

from inference_models.errors import MissingDependencyError, ModelRuntimeError
from inference_models.models.common.torch import generate_batch_chunks

# The TensorRT delegate ships in its own wheel and registers with the runtime when imported.
TENSORRT_DELEGATE_MODULE = "torch_tensorrt_executorch_runtime"
# Program metadata devices and dtypes, and `execute(..., clone_outputs=...)`, first shipped in ExecuTorch 1.6.
MIN_EXECUTORCH_VERSION = Version("1.6.0")


def load_executorch_program(
    program_path: str,
    *,
    data_path: Optional[str] = None,
) -> Any:
    """Load an ExecuTorch program without initializing its methods.

    Registers the TensorRT delegate first when its package is installed, so programs
    lowered with Torch-TensorRT load the same way as XNNPACK or CUDA programs.
    Delegates are initialized later, by ``program.load_method(...)``.

    Args:
        program_path (str): Path to the ``.pte`` file.
        data_path (str, optional): Path to a ``.ptd`` file holding data the program
            keeps outside the ``.pte``, such as the weights of a CUDA export.

    Returns:
        executorch.runtime.Program: The loaded program.

    Raises:
        MissingDependencyError: If ExecuTorch is missing or older than 1.6.
    """
    try:
        from executorch.runtime import Runtime
    except ImportError as import_error:
        raise MissingDependencyError(
            message="Running a model with the ExecuTorch backend requires ExecuTorch. Install it with "
            '`pip install "torch_tensorrt[executorch]"` for programs lowered to TensorRT, or with '
            "`pip install executorch` for programs that run on XNNPACK. See: "
            "https://inference-models.roboflow.com/getting-started/backends/#executorch",
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
        ) from import_error

    executorch_version = Version(importlib.metadata.version("executorch"))
    if executorch_version.release < MIN_EXECUTORCH_VERSION.release:
        raise MissingDependencyError(
            message=f"Running a model with the ExecuTorch backend requires ExecuTorch {MIN_EXECUTORCH_VERSION} or "
            f"newer (Python 3.11+), but {executorch_version} is installed. See: "
            "https://inference-models.roboflow.com/getting-started/backends/#executorch",
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
        )
    if importlib.util.find_spec(TENSORRT_DELEGATE_MODULE) is not None:
        importlib.import_module(TENSORRT_DELEGATE_MODULE)
    program = Runtime.get().load_program(Path(program_path), data_path=data_path)

    return program


def get_executorch_input_spec(
    program: Any,
    *,
    method_name: str,
) -> Tuple[Tuple[int, ...], torch.device, str]:
    """Return the static shape, device and dtype name of a method's first input.

    Reads the program's metadata, so it works before the method is loaded.

    Args:
        program (executorch.runtime.Program): Loaded program.
        method_name (str): Name of the method.

    Returns:
        tuple: Input shape the program was exported with, the device it expects the
        input on (the CPU unless the program was exported with device-resident
        inputs), and the input dtype name, such as ``"float32"``.
    """
    input_meta = program.metadata(method_name).input_tensor_meta(0)
    input_shape = tuple(input_meta.sizes())
    input_device = torch.device(input_meta.device())
    input_dtype_name = input_meta.dtype_name()

    return input_shape, input_device, input_dtype_name


def run_executorch_method_with_static_batch_size(
    method: Any,
    *,
    inputs: torch.Tensor,
    batch_size: int,
    input_device: torch.device,
    stream: Optional[torch.cuda.Stream] = None,
) -> List[torch.Tensor]:
    """Run a float32 method exported for one batch size on a batch of any size.

    The batch is split into chunks of ``batch_size``; a short last chunk is padded
    and its padding is dropped from the outputs.

    Args:
        method (executorch.runtime.Method): Loaded method.
        inputs (torch.Tensor): Input batch.
        batch_size (int): Batch size the program was exported with.
        input_device (torch.device): Device the program expects its input on.
        stream (torch.cuda.Stream, optional): Stream for the torch side of the call
            (input copies, padding and concatenation) when ``inputs`` are on a GPU.
            Defaults to the current stream. It is synchronized after the outputs
            are assembled; ExecuTorch synchronizes the device around calls with GPU
            inputs or outputs.

    Returns:
        list[torch.Tensor]: Outputs in program order, on the devices the program
        returns them on.

    Raises:
        ModelRuntimeError: If the program returns no outputs.
    """
    if inputs.device.type != "cuda" and input_device.type != "cuda":
        results = _run_executorch_method_in_chunks(
            method, inputs=inputs, batch_size=batch_size, input_device=input_device
        )

        return results

    cuda_device = inputs.device if inputs.device.type == "cuda" else input_device
    if stream is None:
        stream = torch.cuda.current_stream(cuda_device)
    with torch.cuda.stream(stream):
        results = _run_executorch_method_in_chunks(
            method, inputs=inputs, batch_size=batch_size, input_device=input_device
        )
    stream.synchronize()

    return results


def _run_executorch_method_in_chunks(
    method: Any,
    *,
    inputs: torch.Tensor,
    batch_size: int,
    input_device: torch.device,
) -> List[torch.Tensor]:
    # The runtime rejects device-resident inputs that are not laid out as exported.
    inputs = inputs.to(device=input_device, dtype=torch.float32).contiguous()
    chunk_outputs = []
    for chunk, padding_size in generate_batch_chunks(inputs, chunk_size=batch_size):
        # Cloned outputs survive the next chunk's execution, which reuses the method's memory.
        outputs = method.execute([chunk], clone_outputs=True)
        valid_size = batch_size - padding_size
        chunk_outputs.append([output[:valid_size] for output in outputs])
    if not chunk_outputs or not chunk_outputs[0]:
        raise ModelRuntimeError(
            message="ExecuTorch program returned no outputs.",
            help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
        )

    results = [torch.cat(parts, dim=0) for parts in zip(*chunk_outputs)]

    return results
