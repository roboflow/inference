from typing import List, Union

import onnxruntime as ort
import numpy as np

import torch


def get_onnxruntime_execution_providers(value: str) -> List[str]:
    """Extracts the ONNX runtime execution providers from the given string.

    The input string is expected to be a comma-separated list, possibly enclosed
    within square brackets and containing single quotes.

    Args:
        value (str): The string containing the list of ONNX runtime execution providers.

    Returns:
        List[str]: A list of strings representing each execution provider.
    """
    if len(value) == 0:
        return []
    value = value.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    return value.split(",")


def run_session_via_iobinding(session: ort.InferenceSession, input_name: str, input_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    binding = session.io_binding()

    output_metadata = session.get_outputs()[0]

    print(output_metadata.type)
    print(type(output_metadata.type))

    if "16" in output_metadata.type:
        dtype = np.float16
    else:
        dtype = np.float32
    
    if isinstance(input_data, np.ndarray):
        binding.bind_input(
            name=input_name,
            device_type="cpu",
            device_id=0,
            element_type=dtype,
            shape=input_data.shape,
            buffer_ptr=input_data.ctypes.data,
        )
        
        predictions = np.empty(output_metadata.shape, dtype=dtype)
        binding.bind_output(
            name=output_metadata.name,
            device_type="cpu",
            device_id=0,
            element_type=dtype,
            shape=output_metadata.shape,
            buffer_ptr=predictions.ctypes.data,
        )
    elif isinstance(input_data, torch.Tensor):
        input_data = input_data.contiguous()
        binding.bind_input(
            name=input_name,
            device_type=input_data.device.type,
            device_id=input_data.device.index,
            element_type=dtype,
            shape=input_data.shape,
            buffer_ptr=input_data.data_ptr(),
        )

        predictions = torch.empty(output_metadata.shape, dtype=torch.float16 if "16" in output_metadata.type else torch.float32, device=input_data.device).contiguous()
        binding.bind_output(
            name=output_metadata.name,
            device_type=input_data.device.type,
            device_id=input_data.device.index,
            element_type=dtype,
            shape=output_metadata.shape,
            buffer_ptr=predictions.data_ptr(),
        )

    session.run_with_iobinding(binding)

    if isinstance(input_data, np.ndarray):
        return predictions.astype(np.float32)
    elif isinstance(input_data, torch.Tensor):
        return predictions.float().cpu().numpy()
