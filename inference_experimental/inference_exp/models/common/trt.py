from typing import List, Tuple

import torch
from inference_exp.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_exp.logger import LOGGER
from inference_exp.models.common.roboflow.model_packages import TRTConfig

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

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="TODO",
        help_url="https://todo",
    ) from import_error


class InferenceTRTLogger(trt.ILogger):

    def __init__(self, with_memory: bool = False):
        super().__init__()
        self._memory: List[Tuple[trt.ILogger.Severity, str]] = []
        self._with_memory = with_memory

    def log(self, severity: trt.ILogger.Severity, msg: str) -> None:
        if self._with_memory:
            self._memory.append((severity, msg))
        severity_str = str(severity)
        if severity_str == str(trt.Logger.VERBOSE):
            log_function = LOGGER.debug
        elif severity_str is str(trt.Logger.INFO):
            log_function = LOGGER.info
        elif severity_str is str(trt.Logger.WARNING):
            log_function = LOGGER.warning
        else:
            log_function = LOGGER.error
        log_function(msg)

    def get_memory(self) -> List[Tuple[trt.ILogger.Severity, str]]:
        return self._memory


TRT_LOGGER = InferenceTRTLogger()


def get_engine_inputs_and_outputs(
    engine: trt.ICudaEngine,
) -> Tuple[List[str], List[str]]:
    num_inputs = engine.num_io_tensors
    inputs = []
    outputs = []
    for i in range(num_inputs):
        name = engine.get_tensor_name(i)
        io_mode = engine.get_tensor_mode(name)
        if io_mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        elif io_mode == trt.TensorIOMode.OUTPUT:
            outputs.append(name)
    return inputs, outputs


def infer_from_trt_engine(
    pre_processed_images: torch.Tensor,
    trt_config: TRTConfig,
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    device: torch.device,
    input_name: str,
    outputs: List[str],
) -> List[torch.Tensor]:
    if trt_config.static_batch_size is not None:
        return infer_from_trt_engine_with_batch_size_boundaries(
            pre_processed_images=pre_processed_images,
            engine=engine,
            context=context,
            device=device,
            input_name=input_name,
            outputs=outputs,
            min_batch_size=trt_config.static_batch_size,
            max_batch_size=trt_config.static_batch_size,
        )
    return infer_from_trt_engine_with_batch_size_boundaries(
        pre_processed_images=pre_processed_images,
        engine=engine,
        context=context,
        device=device,
        input_name=input_name,
        outputs=outputs,
        min_batch_size=trt_config.dynamic_batch_size_min,
        max_batch_size=trt_config.dynamic_batch_size_max,
    )


def infer_from_trt_engine_with_batch_size_boundaries(
    pre_processed_images: torch.Tensor,
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    device: torch.device,
    input_name: str,
    outputs: List[str],
    min_batch_size: int,
    max_batch_size: int,
) -> List[torch.Tensor]:
    if pre_processed_images.shape[0] <= max_batch_size:
        reminder = min_batch_size - pre_processed_images.shape[0]
        if reminder > 0:
            pre_processed_images = torch.cat(
                (
                    pre_processed_images,
                    torch.zeros(
                        (reminder,) + pre_processed_images.shape[1:],
                        dtype=pre_processed_images.dtype,
                        device=pre_processed_images.device,
                    ),
                ),
                dim=0,
            )
        results = execute_trt_engine(
            pre_processed_images=pre_processed_images,
            engine=engine,
            context=context,
            device=device,
            input_name=input_name,
            outputs=outputs,
        )
        if reminder > 0:
            results = [r[:-reminder] for r in results]
        return results
    all_results = []
    for _ in outputs:
        all_results.append([])
    for i in range(0, pre_processed_images.shape[0], max_batch_size):
        batch = pre_processed_images[i : i + max_batch_size].contiguous()
        reminder = min_batch_size - batch.shape[0]
        if reminder > 0:
            batch = torch.cat(
                (
                    pre_processed_images,
                    torch.zeros(
                        (reminder,) + batch.shape[1:],
                        dtype=pre_processed_images.dtype,
                        device=pre_processed_images.device,
                    ),
                ),
                dim=0,
            )
        results = execute_trt_engine(
            pre_processed_images=batch,
            engine=engine,
            context=context,
            device=device,
            input_name=input_name,
            outputs=outputs,
        )
        if reminder > 0:
            results = [r[:-reminder] for r in results]
        for partial_result, all_result_element in zip(results, all_results):
            all_result_element.append(partial_result)
    return [torch.cat(e, dim=0).contiguous() for e in all_results]


def execute_trt_engine(
    pre_processed_images: torch.Tensor,
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    device: torch.device,
    input_name: str,
    outputs: List[str],
) -> List[torch.Tensor]:
    batch_size = pre_processed_images.shape[0]
    results = []
    for output in outputs:
        output_tensor_shape = engine.get_tensor_shape(output)
        output_tensor_type = trt_dtype_to_torch(engine.get_tensor_dtype(output))
        result = torch.empty(
            (batch_size,) + output_tensor_shape[1:],
            dtype=output_tensor_type,
            device=device,
        )
        context.set_tensor_address(output, result.data_ptr())
        results.append(result)
    context.set_input_shape(input_name, tuple(pre_processed_images.shape))
    context.set_tensor_address(input_name, pre_processed_images.data_ptr())
    stream = torch.cuda.Stream(device=device)
    status = context.execute_async_v3(stream_handle=stream.cuda_stream)
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
    try:
        local_logger = InferenceTRTLogger(with_memory=True)
        with open(model_path, "rb") as f, trt.Runtime(local_logger) as runtime:
            runtime.engine_host_code_allowed = engine_host_code_allowed
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
