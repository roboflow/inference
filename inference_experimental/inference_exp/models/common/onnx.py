from typing import List, Union

import numpy as np
import onnxruntime
import torch


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
    inputs: torch.Tensor,
) -> List[torch.Tensor]:
    if inputs.device.type == "cpu":
        print(inputs)
        results = session.run(None, {input_name: inputs.numpy()})
        return [torch.from_numpy(element) for element in results]
    binding = session.io_binding()
    predictions = []
    dtype = None
    for output in session.get_outputs():
        # assemble numpy-based output buffers for the ONNX runtime to write to
        if dtype is None:
            dtype = np.float16 if "16" in output.type else np.float32
        prediction = np.empty(output.shape, dtype=dtype)
        binding.bind_output(
            name=output.name,
            device_type="cpu",
            device_id=0,
            element_type=dtype,
            shape=output.shape,
            buffer_ptr=prediction.ctypes.data,
        )
        predictions.append(prediction)
    input_data = inputs.contiguous()
    binding.bind_input(
        name=input_name,
        device_type=input_data.device.type,
        device_id=(
            input_data.device.index if input_data.device.index is not None else 0
        ),
        element_type=dtype,
        shape=input_data.shape,
        buffer_ptr=input_data.data_ptr(),
    )
    binding.synchronize_inputs()
    session.run_with_iobinding(binding)
    return [torch.from_numpy(prediction).float() for prediction in predictions]
