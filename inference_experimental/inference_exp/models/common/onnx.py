from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from inference_exp.errors import MissingDependencyError, ModelRuntimeError

try:
    import onnxruntime
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import onnx tools required to run models with ONNX backend - this error means that some additional "
        f"dependencies are not installed in the environment. If you run the `inference-exp` library directly in your "
        f"Python program, make sure the following extras of the package are installed: \n"
        f"\t* `onnx-cpu` - when you wish to use library with CPU support only\n"
        f"\t* `onnx-cu12` - for running on GPU with Cuda 12 installed\n"
        f"\t* `onnx-cu118` - for running on GPU with Cuda 11.8 installed\n"
        f"\t* `onnx-jp6-cu126` - for running on Jetson with Jetpack 6\n"
        f"If you see this error using Roboflow infrastructure, make sure the service you use does support the model. "
        f"You can also contact Roboflow to get support.",
        help_url="https://todo",
    ) from import_error


TORCH_TYPES_MAPPING = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.uint8: np.uint8,
}

ORT_TYPES_TO_TORCH_TYPES_MAPPING = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
    "tensor(int16)": torch.int16,
    "tensor(int8)": torch.int8,
    "tensor(uint8)": torch.uint8,
    "tensor(uint16)": torch.uint16,
    "tensor(uint32)": torch.uint32,
    "tensor(uint64)": torch.uint64,
    "tensor(bool)": torch.bool,
}

MODEL_INPUT_CASTING = {
    torch.float16: {torch.float16, torch.float32, torch.float64},
    torch.float32: {torch.float16, torch.float32, torch.float64},
    torch.int8: {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float64,
        torch.float32,
        torch.float16,
    },
    torch.int16: {
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    },
    torch.int32: {
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    },
    torch.uint8: {
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
    },
    torch.bool: {torch.uint8, torch.int8, torch.float16, torch.float32, torch.float64},
}


def set_execution_provider_defaults(
    providers: List[Union[str, tuple]],
    model_package_path: str,
    device: torch.device,
    enable_fp16: bool = True,
    default_onnx_trt_options: bool = True,
) -> List[Union[str, tuple[str, dict[str, Any]]]]:
    result = []
    device_id_options = {}
    if device.index is not None:
        device_id_options["device_id"] = device.index
    for provider in providers:
        if provider == "TensorrtExecutionProvider" and default_onnx_trt_options:
            provider = (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": model_package_path,
                    "trt_fp16_enable": enable_fp16,
                    **device_id_options,
                },
            )
        if provider == "CUDAExecutionProvider":
            provider = ("CUDAExecutionProvider", device_id_options)
        result.append(provider)
    return result


def run_session_with_batch_size_limit(
    session: onnxruntime.InferenceSession,
    inputs: Dict[str, torch.Tensor],
    output_shape_mapping: Optional[Dict[str, tuple]] = None,
    max_batch_size: Optional[int] = None,
    min_batch_size: Optional[int] = None,
) -> List[torch.Tensor]:
    if max_batch_size is None:
        return run_session_via_iobinding(
            session=session,
            inputs=inputs,
            output_shape_mapping=output_shape_mapping,
        )
    input_batch_sizes = set()
    for input_tensor in inputs.values():
        input_batch_sizes.add(input_tensor.shape[0])
    if len(input_batch_sizes) != 1:
        raise ModelRuntimeError(
            message="When running forward pass through ONNX model detected inputs with different batch sizes. "
            "This is the error with the model you run. If the model was trained or exported "
            "on Roboflow platform - contact us to get help. Otherwise, verify your model package or "
            "implementation of the model class.",
            help_url="https://todo",
        )
    input_batch_size = input_batch_sizes.pop()
    if min_batch_size is None and input_batch_size <= max_batch_size:
        # no point iterating
        return run_session_via_iobinding(
            session=session,
            inputs=inputs,
            output_shape_mapping=output_shape_mapping,
        )
    all_results = []
    for _ in session.get_outputs():
        all_results.append([])
    for i in range(0, input_batch_size, max_batch_size):
        batch_inputs = {}
        reminder = 0
        for name, value in inputs.items():
            batched_value = value[i : i + max_batch_size]
            if min_batch_size is not None:
                reminder = min_batch_size - batched_value.shape[0]
            if reminder > 0:
                batched_value = torch.cat(
                    (
                        batched_value,
                        torch.zeros(
                            (reminder,) + batched_value.shape[1:],
                            dtype=batched_value.dtype,
                            device=batched_value.device,
                        ),
                    ),
                    dim=0,
                )
            batched_value = batched_value.contiguous()
            batch_inputs[name] = batched_value
        batch_output_shape_mapping = None
        if output_shape_mapping:
            batch_output_shape_mapping = {}
            for name, shape in output_shape_mapping.items():
                batch_output_shape_mapping[name] = (max_batch_size,) + shape[1:]
        batch_results = run_session_via_iobinding(
            session=session,
            inputs=batch_inputs,
            output_shape_mapping=batch_output_shape_mapping,
        )
        if reminder > 0:
            batch_results = [r[:-reminder] for r in batch_results]
        for partial_result, all_result_element in zip(batch_results, all_results):
            all_result_element.append(partial_result)
    return [torch.cat(e, dim=0).contiguous() for e in all_results]


def run_session_via_iobinding(
    session: onnxruntime.InferenceSession,
    inputs: Dict[str, torch.Tensor],
    output_shape_mapping: Optional[Dict[str, tuple]] = None,
) -> List[torch.Tensor]:
    inputs = auto_cast_session_inputs(
        session=session,
        inputs=inputs,
    )
    device = get_input_device(inputs=inputs)
    if device.type != "cuda":
        inputs_np = {name: value.cpu().numpy() for name, value in inputs.items()}
        results = session.run(None, inputs_np)
        return [torch.from_numpy(element).to(device=device) for element in results]
    try:
        import pycuda.driver as cuda
        from inference_exp.models.common.cuda import use_primary_cuda_context
    except ImportError as import_error:
        raise MissingDependencyError(
            message="TODO", help_url="https://todo"
        ) from import_error
    cuda.init()
    cuda_device = cuda.Device(device.index or 0)
    with use_primary_cuda_context(cuda_device=cuda_device):
        if output_shape_mapping is None:
            output_shape_mapping = {}
        binding = session.io_binding()
        pre_allocated_outputs: List[Optional[torch.Tensor]] = []
        some_outputs_dynamically_allocated = False
        for output in session.get_outputs():
            if is_tensor_shape_dynamic(output.shape):
                if output.name in output_shape_mapping:
                    torch_output_type = ort_tensor_type_to_torch_tensor_type(
                        output.type
                    )
                    pre_allocated_output = torch.empty(
                        output_shape_mapping[output.name],
                        dtype=torch_output_type,
                        device=device,
                    )
                    binding.bind_output(
                        name=output.name,
                        device_type="cuda",
                        device_id=device.index or 0,
                        element_type=torch_tensor_type_to_onnx_type(torch_output_type),
                        shape=tuple(pre_allocated_output.shape),
                        buffer_ptr=pre_allocated_output.data_ptr(),
                    )
                    pre_allocated_outputs.append(pre_allocated_output)
                else:
                    binding.bind_output(
                        name=output.name,
                        device_type="cuda",
                        device_id=device.index or 0,
                    )
                    some_outputs_dynamically_allocated = True
                    pre_allocated_outputs.append(None)
            else:
                torch_output_type = ort_tensor_type_to_torch_tensor_type(output.type)
                pre_allocated_output = torch.empty(
                    output.shape,
                    dtype=torch_output_type,
                    device=device,
                )
                binding.bind_output(
                    name=output.name,
                    device_type="cuda",
                    device_id=device.index or 0,
                    element_type=torch_tensor_type_to_onnx_type(torch_output_type),
                    shape=tuple(pre_allocated_output.shape),
                    buffer_ptr=pre_allocated_output.data_ptr(),
                )
                pre_allocated_outputs.append(pre_allocated_output)
        for ort_input in session.get_inputs():
            input_tensor = inputs[ort_input.name].contiguous()
            input_type = torch_tensor_type_to_onnx_type(tensor_dtype=input_tensor.dtype)
            binding.bind_input(
                name=ort_input.name,
                device_type=input_tensor.device.type,
                device_id=input_tensor.device.index or 0,
                element_type=input_type,
                shape=input_tensor.shape,
                buffer_ptr=input_tensor.data_ptr(),
            )
        binding.synchronize_inputs()
        session.run_with_iobinding(binding)
        if not some_outputs_dynamically_allocated:
            return pre_allocated_outputs
        bound_outputs = binding.get_outputs()
        result = []
        for pre_allocated_output, bound_output in zip(
            pre_allocated_outputs, bound_outputs
        ):
            if pre_allocated_output is not None:
                result.append(pre_allocated_output)
                continue
            # This is added for the sake of true compatibility with older builds of onnxruntime
            # which do not support zero-copy OrtValue -> torch.Tensor thanks top dlpack
            if not hasattr(bound_output._ortvalue, "to_dlpack"):
                # slower but needed :(
                out_tensor = torch.from_numpy(bound_output._ortvalue.numpy()).to(device)
            else:
                dlpack_tensor = bound_output._ortvalue.to_dlpack()
                out_tensor = torch.utils.dlpack.from_dlpack(dlpack_tensor)
            result.append(out_tensor)
        return result


def auto_cast_session_inputs(
    session: onnxruntime.InferenceSession, inputs: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    for ort_input in session.get_inputs():
        expected_type = ort_tensor_type_to_torch_tensor_type(ort_input.type)
        if ort_input.name not in inputs:
            raise ModelRuntimeError(
                message="While performing forward pass through the model, library bug was discovered - "
                f"required model input named '{ort_input.name}' is missing. Submit "
                f"issue to help us solving this problem: https://github.com/roboflow/inference/issues",
                help_url="https://todo",
            )
        actual_type = inputs[ort_input.name].dtype
        if actual_type == expected_type:
            continue
        if not can_model_input_be_casted(source=actual_type, target=expected_type):
            raise ModelRuntimeError(
                message="While performing forward pass through the model, library bug was discovered - "
                f"model requires the input type to be {expected_type}, but the actual input type is {actual_type} - "
                f"this is a bug in model implementation. Submit issue to help us solving this problem: "
                f"https://github.com/roboflow/inference/issues",
                help_url="https://todo",
            )
        inputs[ort_input.name] = inputs[ort_input.name].to(dtype=expected_type)
    return inputs


def torch_tensor_type_to_onnx_type(tensor_dtype: torch.dtype) -> Union[np.dtype, int]:
    if tensor_dtype not in TORCH_TYPES_MAPPING:
        raise ModelRuntimeError(
            message=f"While performing forward pass through the model, library discovered tensor of type {tensor_dtype} "
            f"which needs to be passed to onnxruntime session. Conversion of this type is currently not "
            f"supported in inference. At the moment you shall assume your model incompatible with the library. "
            f"To change that state - please submit new issue: https://github.com/roboflow/inference/issues",
            help_url="https://todo",
        )
    return TORCH_TYPES_MAPPING[tensor_dtype]


def ort_tensor_type_to_torch_tensor_type(ort_dtype: str) -> torch.dtype:
    if ort_dtype not in ORT_TYPES_TO_TORCH_TYPES_MAPPING:
        raise ModelRuntimeError(
            message=f"While performing forward pass through the model, library discovered ORT tensor of type {ort_dtype} "
            f"which needs to be casted into torch.Tensor. Conversion of this type is currently not "
            f"supported in inference. At the moment you shall assume your model incompatible with the library. "
            f"To change that state - please submit new issue: https://github.com/roboflow/inference/issues",
            help_url="https://todo",
        )
    return ORT_TYPES_TO_TORCH_TYPES_MAPPING[ort_dtype]


def is_tensor_shape_dynamic(shape: tuple) -> bool:
    return any(isinstance(dim, str) for dim in shape)


def can_model_input_be_casted(source: torch.dtype, target: torch.dtype) -> bool:
    if source not in MODEL_INPUT_CASTING:
        return False
    return target in MODEL_INPUT_CASTING[source]


def get_input_device(inputs: Dict[str, torch.Tensor]) -> torch.device:
    device = None
    for input_name, input_tensor in inputs.items():
        if device is None:
            device = input_tensor.device
        elif input_tensor.device != device:
            raise ModelRuntimeError(
                message="While performing forward pass through the model, library discovered the input tensor which is "
                f"wrongly allocated on a different device that rest of the inputs - input named '{input_name}' "
                f"is allocated on {input_tensor.device}, whereas rest of the inputs are allocated on {device}. "
                f"This is a bug in model implementation. To help us fixing that, please submit new issue: "
                f"https://github.com/roboflow/inference/issues",
                help_url="https://todo",
            )
    if device is None:
        raise ModelRuntimeError(
            message="No inputs detected for the model. Raise new issue to help us fixing the problem: "
            "https://github.com/roboflow/inference/issues",
            help_url="https://todo",
        )
    return device
