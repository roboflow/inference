from typing import List, Tuple

import torch
from inference_exp.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_exp.logger import LOGGER
from inference_exp.models.common.trt.common import InferenceTRTLogger

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not TRT tools required to run models with TRT backend - this error means that some additional "
        f"dependencies are not installed in the environment. If you run the `inference-exp` library directly in your "
        f"Python program, make sure the following extras of the package are installed: `trt10` - installation can only "
        f"succeed for Linux and Windows machines with Cuda 12 installed. Jetson devices, should have TRT 10.x "
        f"installed for all builds with Jetpack 6. "
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error


def get_engine_inputs_and_outputs(
    engine: trt.ICudaEngine,
) -> Tuple[List[str], List[str]]:
    inputs: List[str] = []
    outputs: List[str] = []
    num_bindings = engine.num_bindings
    for i in range(num_bindings):
        name = engine.get_binding_name(i)
        if engine.binding_is_input(i):
            inputs.append(name)
        else:
            outputs.append(name)
    return inputs, outputs


def execute_trt_engine(
    pre_processed_images: torch.Tensor,
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    device: torch.device,
    input_name: str,
    outputs: List[str],
) -> List[torch.Tensor]:
    batch_size = pre_processed_images.shape[0]
    try:
        input_index = engine.get_binding_index(input_name)
    except Exception as error:
        raise ModelRuntimeError(
            message=f"Could not find input binding '{input_name}' in TRT engine.",
            help_url="https://todo",
        ) from error
    if input_index < 0:
        raise ModelRuntimeError(
            message=f"Input binding '{input_name}' not found in TRT engine.",
            help_url="https://todo",
        )
    status = context.set_binding_shape(input_index, tuple(pre_processed_images.shape))
    if not status:
        raise ModelRuntimeError(
            message="Failed to set TRT model input shape during forward pass from the model.",
            help_url="https://todo",
        )
    num_bindings = engine.num_bindings
    bindings: List[int] = [0] * num_bindings
    results: List[torch.Tensor] = []
    bindings[input_index] = pre_processed_images.data_ptr()
    for output in outputs:
        output_index = engine.get_binding_index(output)
        if output_index < 0:
            raise ModelRuntimeError(
                message=f"Output binding '{output}' not found in TRT engine.",
                help_url="https://todo",
            )
        output_tensor_shape = tuple(context.get_binding_shape(output_index))
        output_tensor_type = trt_dtype_to_torch(engine.get_binding_dtype(output_index))
        result = torch.empty(
            (batch_size,) + output_tensor_shape[1:],
            dtype=output_tensor_type,
            device=device,
        )
        bindings[output_index] = result.data_ptr()
        results.append(result)
    stream = torch.cuda.Stream(device=device)
    status = context.execute_async_v2(
        bindings=bindings,
        stream_handle=stream.cuda_stream,
    )
    if not status:
        raise ModelRuntimeError(
            message="Failed to complete inference from TRT model",
            help_url="https://todo",
        )
    stream.synchronize()
    return results


def trt_dtype_to_torch(trt_dtype):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }[trt_dtype]


def load_model(
    model_path: str,
    engine_host_code_allowed: bool = False,
) -> trt.ICudaEngine:
    if engine_host_code_allowed:
        LOGGER.warning(
            f"Used load_model(...) with `engine_host_code_allowed=True` which is not supported for TRT 8.X "
            f"and will be  ignored."
        )
    try:
        local_logger = InferenceTRTLogger(with_memory=True)
        with open(model_path, "rb") as f, trt.Runtime(local_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                logger_traces = local_logger.get_memory()
                logger_traces_str = "\n".join(
                    f"[{severity}] {msg}" for severity, msg in logger_traces
                )
                raise CorruptedModelPackageError(
                    message="Could not load TRT engine due to runtime error. This error is usually caused "
                    "by model package incompatibility with runtime environment. If you selected model with "
                    "specific model package to be run - verify that your environment is compatible with your "
                    "package. If the package was selected automatically by the library - this error indicate bug. "
                    "You can help us solving this problem describing the issue: "
                    "https://github.com/roboflow/inference/issues\nBelow you can find debug information provided "
                    f"by TRT runtime, which may be helpful:\n{logger_traces_str}",
                    help_url="https://todo",
                )
            return engine
    except OSError as error:
        raise CorruptedModelPackageError(
            message="Could not load TRT engine - file not found. This error may be caused by "
            "corrupted model package or invalid model path that was provided. If you "
            "initialized the model manually, running the code locally - make sure that provided "
            "path is correct. Otherwise, contact Roboflow to solve the problem: "
            "https://github.com/roboflow/inference/issues",
            help_url="https://todo",
        ) from error
