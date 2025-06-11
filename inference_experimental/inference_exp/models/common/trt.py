from typing import List

import tensorrt as trt
import torch
from inference_exp.errors import ModelRuntimeError
from inference_exp.models.common.roboflow.model_packages import TRTConfig

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


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
        return infer_from_trt_engine_with_static_batch_size(
            pre_processed_images=pre_processed_images,
            trt_config=trt_config,
            engine=engine,
            context=context,
            device=device,
            input_name=input_name,
            outputs=outputs,
        )
    return infer_from_trt_engine_with_dynamic_batch_size(
        pre_processed_images=pre_processed_images,
        trt_config=trt_config,
        engine=engine,
        context=context,
        device=device,
        input_name=input_name,
        outputs=outputs,
    )


def infer_from_trt_engine_with_static_batch_size(
    pre_processed_images: torch.Tensor,
    trt_config: TRTConfig,
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    device: torch.device,
    input_name: str,
    outputs: List[str],
) -> List[torch.Tensor]:
    batch_pad_reminder = 0
    if pre_processed_images.shape[0] < trt_config.static_batch_size:
        batch_pad_reminder = (
            trt_config.static_batch_size - pre_processed_images.shape[0]
        )
        pre_processed_images = torch.cat(
            (
                pre_processed_images,
                torch.zeros(
                    (batch_pad_reminder,) + pre_processed_images.shape[1:],
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
    if not batch_pad_reminder:
        return results
    return [r[:-batch_pad_reminder] for r in results]


def infer_from_trt_engine_with_dynamic_batch_size(
    pre_processed_images: torch.Tensor,
    trt_config: TRTConfig,
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    device: torch.device,
    input_name: str,
    outputs: List[str],
) -> List[torch.Tensor]:
    if pre_processed_images.shape[0] <= trt_config.dynamic_batch_size_max:
        reminder = trt_config.dynamic_batch_size_min - pre_processed_images.shape[0]
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
    for i in range(0, pre_processed_images.shape[0], trt_config.dynamic_batch_size_max):
        batch = pre_processed_images[
            i : i + trt_config.dynamic_batch_size_max
        ].contiguous()
        reminder = trt_config.dynamic_batch_size_min - batch.shape[0]
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


def get_output_tensor_names(engine: trt.ICudaEngine) -> List[str]:
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
    return output_names
