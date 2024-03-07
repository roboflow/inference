from inference.core.utils.onnx import get_onnxruntime_execution_providers


def test_get_onnxruntime_execution_providers_when_empty_input_provided() -> None:
    # when
    result = get_onnxruntime_execution_providers(value="")

    # then
    assert result == []


def test_get_onnxruntime_execution_providers_when_valid_input_provided() -> None:
    # when
    result = get_onnxruntime_execution_providers(
        value="[CUDAExecutionProvider,'OpenVINOExecutionProvider',CPUExecutionProvider]",
    )

    # then
    assert result == [
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
