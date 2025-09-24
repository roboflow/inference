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
    if isinstance(input_data, (np.ndarray, list)):
        # skip the iobinding and just run the session
        # we likely won't get any gains by pointing to the input data directly
        predictions = session.run(None, {input_name: input_data})
    elif "CUDAExecutionProvider" not in session.get_providers():
        # no point in doing iobinding as the input must live on CPU anyway
        input_data = (
            input_data.cpu().numpy()
        )  # since we must be a tensor but ONNX needs a numpy array
        predictions = session.run(None, {input_name: input_data})
    else:
        # we live on GPU and we can use CUDA ONNX, so point to the input data directly
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

        binding.synchronize_inputs()

        session.run_with_iobinding(binding)

        # convert the output buffers to float32 as we may run mixed precision inference in the future
        predictions = [prediction.astype(np.float32) for prediction in predictions]

    return predictions
