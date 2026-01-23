from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch

from inference_models.errors import (
    CorruptedModelPackageError,
    MissingDependencyError,
    ModelRuntimeError,
)
from inference_models.logger import LOGGER
from inference_models.models.common.roboflow.model_packages import TRTConfig

try:
    import tensorrt as trt
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not TRT tools required to run models with TRT backend - this error means that some additional "
        f"dependencies are not installed in the environment. If you run the `inference-models` library directly in your "
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

import pycuda.driver as cuda
@dataclass
class TRTCudaGraphState:
    cuda_graph: cuda.GraphExec
    cuda_stream: torch.cuda.Stream
    input_pointer: int
    input_shape: Tuple[int, ...]
    output_pointers: List[int]
    output_shapes: List[Tuple[int, ...]]

    def has_changed_shape(self, input_shape: Tuple[int, ...], output_shapes: List[Tuple[int, ...]]) -> bool:
        return self.input_shape != input_shape or self.output_shapes != output_shapes

def get_trt_engine_inputs_and_outputs(
    engine: trt.ICudaEngine,
) -> Tuple[List[str], List[str]]:
    """Extract input and output tensor names from a TensorRT engine.

    Inspects a TensorRT engine to determine which tensors are inputs and which
    are outputs. This is useful for setting up inference execution contexts and
    understanding the engine's interface.

    Args:
        engine: TensorRT CUDA engine (ICudaEngine) to inspect.

    Returns:
        Tuple of (input_names, output_names) where:
            - input_names: List of input tensor names
            - output_names: List of output tensor names

    Examples:
        Inspect TensorRT engine:

        >>> from inference_models.developer_tools import (
        ...     load_trt_model,
        ...     get_trt_engine_inputs_and_outputs
        ... )
        >>>
        >>> engine = load_trt_model("model.plan")
        >>> inputs, outputs = get_trt_engine_inputs_and_outputs(engine)
        >>>
        >>> print(f"Inputs: {inputs}")   # ['images']
        >>> print(f"Outputs: {outputs}") # ['output0', 'output1']

        Use for setting up inference:

        >>> inputs, outputs = get_trt_engine_inputs_and_outputs(engine)
        >>> context = engine.create_execution_context()
        >>>
        >>> # Set input tensor
        >>> input_name = inputs[0]
        >>> context.set_input_shape(input_name, (1, 3, 640, 640))

    Note:
        - Requires TensorRT to be installed
        - Works with TensorRT 10.x engines
        - Tensor names are defined during engine building/export

    See Also:
        - `load_trt_model()`: Load TensorRT engine from file
        - `infer_from_trt_engine()`: Run inference with TensorRT engine
    """
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
    use_cuda_graph: bool = False,
    trt_cuda_graph_state: Optional[TRTCudaGraphState] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], TRTCudaGraphState]:
    """Run inference using a TensorRT engine.

    Executes inference on preprocessed images using a TensorRT engine and execution
    context. Handles both static and dynamic batch sizes, automatically splitting
    large batches if needed.

    Args:
        pre_processed_images: Preprocessed input tensor on CUDA device.
            Shape: (batch_size, channels, height, width).

        trt_config: TensorRT configuration object containing batch size settings
            and other engine-specific parameters.

        engine: TensorRT CUDA engine (ICudaEngine) to use for inference.

        context: TensorRT execution context (IExecutionContext) for running inference.

        device: PyTorch CUDA device to use for inference.

        input_name: Name of the input tensor in the TensorRT engine.

        outputs: List of output tensor names to retrieve from the engine.

    Returns:
        List of output tensors from the TensorRT engine, in the order specified
        by the outputs parameter.

    Examples:
        Run TensorRT inference:

        >>> from inference_models.developer_tools import (
        ...     load_trt_model,
        ...     get_trt_engine_inputs_and_outputs,
        ...     infer_from_trt_engine
        ... )
        >>> from inference_models.models.common.roboflow.model_packages import (
        ...     parse_trt_config
        ... )
        >>> import torch
        >>>
        >>> # Load engine and config
        >>> engine = load_trt_model("model.plan")
        >>> trt_config = parse_trt_config("trt_config.json")
        >>> context = engine.create_execution_context()
        >>>
        >>> # Get input/output names
        >>> inputs, outputs = get_trt_engine_inputs_and_outputs(engine)
        >>>
        >>> # Prepare input
        >>> images = torch.randn(1, 3, 640, 640, device="cuda:0")
        >>>
        >>> # Run inference
        >>> results = infer_from_trt_engine(
        ...     pre_processed_images=images,
        ...     trt_config=trt_config,
        ...     engine=engine,
        ...     context=context,
        ...     device=torch.device("cuda:0"),
        ...     input_name=inputs[0],
        ...     outputs=outputs
        ... )

        Handle large batches:

        >>> # Large batch will be automatically split
        >>> large_batch = torch.randn(100, 3, 640, 640, device="cuda:0")
        >>>
        >>> results = infer_from_trt_engine(
        ...     pre_processed_images=large_batch,
        ...     trt_config=trt_config,
        ...     engine=engine,
        ...     context=context,
        ...     device=torch.device("cuda:0"),
        ...     input_name=inputs[0],
        ...     outputs=outputs
        ... )
        >>> # Results are automatically concatenated

    Note:
        - Requires TensorRT and PyCUDA to be installed
        - Input must be on CUDA device
        - Automatically handles batch size constraints from trt_config
        - Uses asynchronous execution with CUDA streams

    Raises:
        ModelRuntimeError: If inference execution fails.

    See Also:
        - `load_trt_model()`: Load TensorRT engine from file
        - `get_trt_engine_inputs_and_outputs()`: Get engine tensor names
    """
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
            use_cuda_graph=use_cuda_graph,
            trt_cuda_graph_state=trt_cuda_graph_state,
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
        use_cuda_graph=use_cuda_graph,
        trt_cuda_graph_state=trt_cuda_graph_state,
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
    use_cuda_graph: bool = False,
    trt_cuda_graph_state: Optional[TRTCudaGraphState] = None,
) -> Tuple[List[torch.Tensor], TRTCudaGraphState]:
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
        results, trt_cuda_graph_state = execute_trt_engine(
            pre_processed_images=pre_processed_images,
            engine=engine,
            context=context,
            device=device,
            input_name=input_name,
            outputs=outputs,
            use_cuda_graph=use_cuda_graph,
            trt_cuda_graph_state=trt_cuda_graph_state,
        )
        if reminder > 0:
            results = [r[:-reminder] for r in results]
        return results, trt_cuda_graph_state
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
        results, trt_cuda_graph_state = execute_trt_engine(
            pre_processed_images=batch,
            engine=engine,
            context=context,
            device=device,
            input_name=input_name,
            outputs=outputs,
            use_cuda_graph=use_cuda_graph,
            trt_cuda_graph_state=trt_cuda_graph_state,
        )
        if reminder > 0:
            results = [r[:-reminder] for r in results]
        for partial_result, all_result_element in zip(results, all_results):
            all_result_element.append(partial_result)
    return [torch.cat(e, dim=0).contiguous() for e in all_results], trt_cuda_graph_state


def execute_trt_engine(
    pre_processed_images: torch.Tensor,
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    device: torch.device,
    input_name: str,
    outputs: List[str],
    use_cuda_graph: bool = False,
    trt_cuda_graph_state: Optional[TRTCudaGraphState] = None,
) -> Tuple[List[torch.Tensor], Optional[TRTCudaGraphState]]:

    if use_cuda_graph:

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
        status = context.set_input_shape(input_name, tuple(pre_processed_images.shape))
        if not status:
            raise ModelRuntimeError(
                message="Failed to set TRT model input shape during forward pass from the model.",
                help_url="https://todo",
            )
        status = context.set_tensor_address(input_name, pre_processed_images.data_ptr())
        if not status:
            raise ModelRuntimeError(
                message="Failed to set input tensor data pointer during forward pass from the model.",
                help_url="https://todo",
            )
        stream = torch.cuda.Stream(device=device)
        status = context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not status:
            raise ModelRuntimeError(
                message="Failed to complete inference from TRT model",
                help_url="https://todo",
            )
        stream.synchronize()
        return results, None  

    else:

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
        status = context.set_input_shape(input_name, tuple(pre_processed_images.shape))
        if not status:
            raise ModelRuntimeError(
                message="Failed to set TRT model input shape during forward pass from the model.",
                help_url="https://todo",
            )
        status = context.set_tensor_address(input_name, pre_processed_images.data_ptr())
        if not status:
            raise ModelRuntimeError(
                message="Failed to set input tensor data pointer during forward pass from the model.",
                help_url="https://todo",
            )
        stream = torch.cuda.Stream(device=device)
        status = context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not status:
            raise ModelRuntimeError(
                message="Failed to complete inference from TRT model",
                help_url="https://todo",
            )
        stream.synchronize()
        return results, None


def execute_trt_engine_with_cuda_graph(
    pre_processed_images: torch.Tensor,
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    device: torch.device,
    input_name: str,
    outputs: List[str],
) -> Tuple[List[torch.Tensor], TRTCudaGraphState]:
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
    status = context.set_input_shape(input_name, tuple(pre_processed_images.shape))
    if not status:
        raise ModelRuntimeError(
            message="Failed to set TRT model input shape during forward pass from the model.",
            help_url="https://todo",
        )
    status = context.set_tensor_address(input_name, pre_processed_images.data_ptr())
    if not status:
        raise ModelRuntimeError(
            message="Failed to set input tensor data pointer during forward pass from the model.",
            help_url="https://todo",
        )
    stream = torch.cuda.Stream(device=device)
    status = context.execute_async_v3(stream_handle=stream.cuda_stream)
    if not status:
        raise ModelRuntimeError(
            message="Failed to complete inference from TRT model",
            help_url="https://todo",
        )
    stream.synchronize()
    return results, None


def trt_dtype_to_torch(trt_dtype):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }[trt_dtype]


def load_trt_model(
    model_path: str,
    engine_host_code_allowed: bool = False,
) -> trt.ICudaEngine:
    """Load a TensorRT engine from a serialized engine file.

    Deserializes a TensorRT engine from a .plan file and returns the engine
    object ready for inference. Handles errors during deserialization and
    provides detailed error messages.

    Args:
        model_path: Path to the serialized TensorRT engine file (.plan).

        engine_host_code_allowed: Allow the engine to execute host code.
            **Security risk** - only enable if you trust the engine source.
            Default: False.

    Returns:
        TensorRT CUDA engine (ICudaEngine) ready for creating execution contexts
        and running inference.

    Raises:
        CorruptedModelPackageError: If the engine file cannot be loaded due to:
            - File not found
            - Incompatible TensorRT version
            - Incompatible CUDA version
            - Corrupted engine file
            - Runtime deserialization errors

        MissingDependencyError: If TensorRT is not installed.

    Examples:
        Load TensorRT engine:

        >>> from inference_models.developer_tools import load_trt_model
        >>>
        >>> engine = load_trt_model("model.plan")
        >>> print(f"Engine loaded: {engine.name}")
        >>>
        >>> # Create execution context
        >>> context = engine.create_execution_context()

        Load engine with host code allowed:

        >>> # Only if you trust the engine source!
        >>> engine = load_trt_model(
        ...     "custom_model.plan",
        ...     engine_host_code_allowed=True
        ... )

        Complete inference setup:

        >>> from inference_models.developer_tools import (
        ...     load_trt_model,
        ...     get_trt_engine_inputs_and_outputs
        ... )
        >>>
        >>> # Load engine
        >>> engine = load_trt_model("yolov8n.plan")
        >>>
        >>> # Get input/output info
        >>> inputs, outputs = get_trt_engine_inputs_and_outputs(engine)
        >>> print(f"Inputs: {inputs}")
        >>> print(f"Outputs: {outputs}")
        >>>
        >>> # Create context for inference
        >>> context = engine.create_execution_context()

    Note:
        - Requires TensorRT to be installed (TensorRT 10.x recommended)
        - Engine files are platform and TensorRT version specific
        - Engines built on one GPU architecture may not work on another
        - Engine files typically have .plan or .engine extension
        - Provides detailed error messages from TensorRT runtime

    See Also:
        - `get_trt_engine_inputs_and_outputs()`: Inspect engine inputs/outputs
        - `infer_from_trt_engine()`: Run inference with loaded engine
    """
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
