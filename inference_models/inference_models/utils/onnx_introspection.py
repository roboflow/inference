from functools import cache
from typing import List

from inference_models.configuration import ONNXRUNTIME_EXECUTION_PROVIDERS


@cache
def get_selected_onnx_execution_providers() -> List[str]:
    """Get the list of ONNX execution providers that are both requested and available.

    Checks which ONNX Runtime execution providers are available on the system and
    filters them against the requested providers from the `ONNXRUNTIME_EXECUTION_PROVIDERS`
    environment variable. This is used internally by ONNX-based models to determine
    which execution providers to use.

    The function is cached, so subsequent calls return the same result without
    re-checking the environment.

    Returns:
        List of execution provider names that are both requested (via environment
        variable) and available on the system. Returns empty list if ONNX Runtime
        is not installed.

    Environment Variables:
        ONNXRUNTIME_EXECUTION_PROVIDERS: Comma-separated list of requested execution
            providers. Example: "CUDAExecutionProvider,CPUExecutionProvider"

    Examples:
        Check available ONNX execution providers:

        >>> from inference_models.developer_tools import get_selected_onnx_execution_providers
        >>>
        >>> providers = get_selected_onnx_execution_providers()
        >>> print(f"Available providers: {providers}")
        >>> # ['CUDAExecutionProvider', 'CPUExecutionProvider']

        Use in custom ONNX model:

        >>> from inference_models.developer_tools import get_selected_onnx_execution_providers
        >>> import onnxruntime as ort
        >>>
        >>> providers = get_selected_onnx_execution_providers()
        >>> if not providers:
        ...     raise RuntimeError("No ONNX execution providers available")
        >>>
        >>> session = ort.InferenceSession("model.onnx", providers=providers)

    Note:
        - Common execution providers: "CUDAExecutionProvider", "CPUExecutionProvider",
          "TensorrtExecutionProvider", "OpenVINOExecutionProvider"
        - The function only returns providers that are both requested AND available
        - If ONNX Runtime is not installed, returns an empty list

    See Also:
        - `x_ray_runtime_environment()`: Get comprehensive runtime information including
          all available ONNX execution providers
    """
    try:
        import onnxruntime

        available_providers = set(onnxruntime.get_available_providers())
        return [
            ep for ep in ONNXRUNTIME_EXECUTION_PROVIDERS if ep in available_providers
        ]
    except ImportError:
        return []
