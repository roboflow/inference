from typing import List

import torch
from inference_exp.errors import MissingDependencyError
from inference_exp.models.common.roboflow.model_packages import TRTConfig
from packaging.version import Version

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

TRT_VERSION = Version(trt.__version__)

if TRT_VERSION.major == 8:
    from inference_exp.models.common.trt.trt_8 import (
        execute_trt_engine,
        get_engine_inputs_and_outputs,
        load_model,
    )
elif TRT_VERSION.major == 10:
    from inference_exp.models.common.trt.trt_10 import (
        execute_trt_engine,
        get_engine_inputs_and_outputs,
        load_model,
    )
else:
    raise MissingDependencyError(
        message=f"Unsupported TRT version: {TRT_VERSION}.",
        help_url="https://todo",
    )


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
