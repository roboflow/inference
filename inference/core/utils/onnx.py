from inference.core.env import TENSORRT_CACHE_PATH


def get_onnxruntime_execution_providers(value):
    value = value.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    return value.split(",")
