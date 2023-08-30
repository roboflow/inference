from inference.core.env import TENSORRT_CACHE_PATH


def get_onnxruntime_execution_providers(value):
    """Extracts the ONNX runtime execution providers from the given string.

    The input string is expected to be a comma-separated list, possibly enclosed
    within square brackets and containing single quotes.

    Args:
        value (str): The string containing the list of ONNX runtime execution providers.

    Returns:
        List[str]: A list of strings representing each execution provider.
    """
    value = value.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    return value.split(",")
