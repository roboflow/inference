from typing import List, Union, Optional, Dict

import numpy as np
import onnxruntime
import torch

from inference_exp.errors import ModelRuntimeError


TORCH_TYPES_MAPPING = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.uint8: np.uint8
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


def set_execution_provider_defaults(
    providers: List[Union[str, tuple]],
    model_package_path: str,
    device: torch.device,
    enable_fp16: bool = True,
    default_onnx_trt_options: bool = True,
) -> List[Union[str, tuple]]:
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


def run_session_via_iobinding(
    session: onnxruntime.InferenceSession,
    input_name: str,
    input_tensor: torch.Tensor,
    output_shape_mapping: Optional[Dict[str, tuple]] = None,
) -> List[torch.Tensor]:
    if input_tensor.device.type != "cuda":
        results = session.run(None, {input_name: input_tensor.numpy()})
        return [torch.from_numpy(element) for element in results]
    if output_shape_mapping is None:
        output_shape_mapping = {}
    binding = session.io_binding()
    pre_allocated_outputs: List[Optional[torch.Tensor]] = []
    some_outputs_dynamically_allocated = False
    for output in session.get_outputs():
        if is_tensor_shape_dynamic(output.shape):
            if output.name in output_shape_mapping:
                torch_output_type = ort_tensor_type_to_torch_tensor_type(output.type)
                pre_allocated_output = torch.empty(
                    output_shape_mapping[output.name],
                    dtype=torch.float32,
                    device=input_tensor.device,
                )
                binding.bind_output(
                    name=output.name,
                    device_type="cuda",
                    device_id=input_tensor.device.index or 0,
                    element_type=torch_output_type,
                    shape=tuple(pre_allocated_output.shape),
                    buffer_ptr=pre_allocated_output.data_ptr(),
                )
                pre_allocated_outputs.append(pre_allocated_output)
            else:
                some_outputs_dynamically_allocated = True
                pre_allocated_outputs.append(None)
        else:
            torch_output_type = ort_tensor_type_to_torch_tensor_type(output.type)
            pre_allocated_output = torch.empty(
                output.shape,
                dtype=torch.float32,
                device=input_tensor.device,
            )
            binding.bind_output(
                name=output.name,
                device_type="cuda",
                device_id=input_tensor.device.index or 0,
                element_type=torch_output_type,
                shape=tuple(pre_allocated_output.shape),
                buffer_ptr=pre_allocated_output.data_ptr(),
            )
            pre_allocated_outputs.append(pre_allocated_output)
    input_type = torch_tensor_type_to_onnx_type(tensor_dtype=input_tensor.dtype)
    input_data = input_tensor.contiguous()
    binding.bind_input(
        name=input_name,
        device_type=input_data.device.type,
        device_id=input_data.device.index or 0,
        element_type=input_type,
        shape=input_data.shape,
        buffer_ptr=input_data.data_ptr(),
    )
    binding.synchronize_inputs()
    session.run_with_iobinding(binding)
    if not some_outputs_dynamically_allocated:
        return pre_allocated_outputs
    bound_outputs = binding.get_outputs()
    result = []
    for pre_allocated_output, bound_output in zip(pre_allocated_outputs, bound_outputs):
        if pre_allocated_output is not None:
            result.append(pre_allocated_output)
            continue
        dlpack_tensor = bound_output._ortvalue.to_dlpack()
        out_tensor = torch.utils.dlpack.from_dlpack(dlpack_tensor)
        result.append(out_tensor)
    return result



def torch_tensor_type_to_onnx_type(tensor_dtype: torch.dtype) -> Union[np.dtype, int]:
    if tensor_dtype not in TORCH_TYPES_MAPPING:
        raise ModelRuntimeError(
            f"While performing forward pass through the model, library discovered tensor of type {tensor_dtype} "
            f"which needs to be passed to onnxruntime session. Conversion of this type is currently not "
            f"supported in inference. At the moment you shall assume your model incompatible with the library. "
            f"To change that state - please submit new issue: https://github.com/roboflow/inference/issues"
        )
    return TORCH_TYPES_MAPPING[tensor_dtype]


def ort_tensor_type_to_torch_tensor_type(ort_dtype: str) -> torch.dtype:
    if ort_dtype not in ORT_TYPES_TO_TORCH_TYPES_MAPPING:
        raise ModelRuntimeError(
            f"While performing forward pass through the model, library discovered ORT tensor of type {ort_dtype} "
            f"which needs to be casted into torch.Tensor. Conversion of this type is currently not "
            f"supported in inference. At the moment you shall assume your model incompatible with the library. "
            f"To change that state - please submit new issue: https://github.com/roboflow/inference/issues"
        )
    return ORT_TYPES_TO_TORCH_TYPES_MAPPING[ort_dtype]


def is_tensor_shape_dynamic(shape: tuple) -> bool:
    return any(isinstance(dim, str) for dim in shape)
