from typing import TYPE_CHECKING, List, Union

import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    import torch

ImageMetaType = Union[np.ndarray, "torch.Tensor"]


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


def run_session_via_iobinding(
    session: ort.InferenceSession, input_name: str, input_data: ImageMetaType
) -> List[np.ndarray]:
    binding = session.io_binding()

    if "CUDAExecutionProvider" not in session.get_providers():
        # ensure we're using CPU because the ONNX runtime used doesn't support CUDA
        if not isinstance(input_data, np.ndarray):
            # data is a torch tensor that is potentially on the GPU, so we just move it to the CPU
            input_data = input_data.cpu()

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

    if isinstance(input_data, np.ndarray):
        input_data = np.ascontiguousarray(input_data.astype(dtype))
        binding.bind_input(
            name=input_name,
            device_type="cpu",
            device_id=0,
            element_type=dtype,
            shape=input_data.shape,
            buffer_ptr=input_data.ctypes.data,
        )
    else:
        # we assume that the input data is a torch tensor
        # but don't explicitly check for torch here so we don't require a torch import
        input_data = input_data.contiguous()
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

    session.run_with_iobinding(binding)

    # convert the output buffers to float32 as we may run mixed precision inference in the future
    predictions = [prediction.astype(np.float32) for prediction in predictions]

    return predictions
