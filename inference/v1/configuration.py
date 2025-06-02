import os

import torch

from inference.v1.utils.environment import parse_comma_separated_values

ONNXRUNTIME_EXECUTION_PROVIDERS = parse_comma_separated_values(
    values=os.getenv(
        "ONNXRUNTIME_EXECUTION_PROVIDERS",
        "CUDAExecutionProvider,OpenVINOExecutionProvider,CoreMLExecutionProvider,CPUExecutionProvider",
    )
)
DEFAULT_DEVICE_STR = os.getenv(
    "DEFAULT_DEVICE",
    (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ),
)
DEFAULT_DEVICE = torch.device(DEFAULT_DEVICE_STR)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_API_CALLS_TIMEOUT = int(os.getenv("ROBOFLOW_API_CALLS_TIMEOUT", "5"))
ROBOFLOW_API_CALLS_MAX_RETRIES = int(os.getenv("ROBOFLOW_API_CALLS_MAX_RETRIES", "3"))
ROBOFLOW_API_CODES_TO_RETRY = set(
    int(e.strip())
    for e in os.getenv("ROBOFLOW_API_CODES_TO_RETRY", "408,429,502,503,504").split(",")
)
ROBOFLOW_ENVIRONMENT = os.getenv("ROBOFLOW_ENVIRONMENT", "prod")
ROBOFLOW_API_HOST = os.getenv(
    "ROBOFLOW_API_HOST",
    (
        "https://api.roboflow.com"
        if ROBOFLOW_ENVIRONMENT.lower() == "prod"
        else "https://api.roboflow.one"
    ),
)
RUNNING_ON_JETSON = os.getenv("RUNNING_ON_JETSON")
JETPACK_VERSION = os.getenv("JETSON_L4T")
