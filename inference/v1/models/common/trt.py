from typing import List

import tensorrt as trt
import torch

from inference.v1.errors import ModelRuntimeError

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def execute_trt_engine(
    pre_processed_image: torch.Tensor,
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    device: torch.device,
    input_name: str,
    outputs: List[str],
) -> List[torch.Tensor]:
    batch_size = pre_processed_image.shape[0]
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
    context.set_input_shape(input_name, tuple(pre_processed_image.shape))
    context.set_tensor_address(input_name, pre_processed_image.data_ptr())
    stream = torch.cuda.Stream(device=device)
    status = context.execute_async_v3(stream_handle=stream.cuda_stream)
    if not status:
        raise ModelRuntimeError("Failed to complete inference from TRT model")
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


def load_model(model_path: str, logger: trt.Logger = TRT_LOGGER) -> trt.ICudaEngine:
    with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
