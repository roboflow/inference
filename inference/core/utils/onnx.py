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
    """Run an ONNX session, binding tensor inputs when CUDA is available.

    Args:
        session (ort.InferenceSession): Initialized ONNX session.
        input_name (str): Name of the model's image input.
        input_data (ImageMetaType): Preprocessed image array or tensor.

    Returns:
        List[np.ndarray]: Outputs in model order. The tensor binding path returns
            CPU arrays converted to float32.
    """
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
        predictions = _run_session_with_tensor_iobinding(
            session, input_name=input_name, input_data=input_data
        )

    return predictions


def _run_session_with_tensor_iobinding(
    session: ort.InferenceSession, *, input_name: str, input_data: "torch.Tensor"
) -> List[np.ndarray]:
    """Bind a tensor directly and let ONNX Runtime resolve output dimensions."""
    binding = session.io_binding()

    for output in session.get_outputs():
        # Symbolic output dimensions are only known after execution.
        binding.bind_output(name=output.name, device_type="cpu")

    input_data = input_data.contiguous()
    binding.bind_input(
        name=input_name,
        device_type=input_data.device.type,
        device_id=input_data.device.index if input_data.device.index is not None else 0,
        element_type=np.dtype(str(input_data.dtype).removeprefix("torch.")),
        shape=input_data.shape,
        buffer_ptr=input_data.data_ptr(),
    )

    binding.synchronize_inputs()

    session.run_with_iobinding(binding)

    # Outputs are already on CPU. Keep only the final copy so returned arrays
    # own their memory independently of the binding and its OrtValues.
    outputs = binding.get_outputs()
    predictions = [output.numpy().astype(np.float32, copy=True) for output in outputs]

    return predictions
