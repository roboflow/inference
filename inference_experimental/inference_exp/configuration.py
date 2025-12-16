import os

import torch
from inference_exp.utils.environment import parse_comma_separated_values, str2bool

ONNXRUNTIME_EXECUTION_PROVIDERS = parse_comma_separated_values(
    values=os.getenv(
        "ONNXRUNTIME_EXECUTION_PROVIDERS",
        "CUDAExecutionProvider,OpenVINOExecutionProvider,CoreMLExecutionProvider,CPUExecutionProvider",
    )
    .strip("[")
    .strip("]")
)
DEFAULT_DEVICE_STR = os.getenv(
    "DEFAULT_DEVICE",
    ("cuda" if torch.cuda.is_available() else "cpu"),
)
DEFAULT_DEVICE = torch.device(DEFAULT_DEVICE_STR)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
API_CALLS_TIMEOUT = int(os.getenv("API_CALLS_TIMEOUT", "5"))
API_CALLS_MAX_TRIES = int(os.getenv("API_CALLS_MAX_TRIES", "3"))
IDEMPOTENT_API_REQUEST_CODES_TO_RETRY = set(
    int(e.strip())
    for e in os.getenv(
        "IDEMPOTENT_API_REQUEST_CODES_TO_RETRY", "408,429,502,503,504"
    ).split(",")
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
L4T_VERSION = os.getenv("L4T_VERSION")
INFERENCE_HOME = os.getenv("INFERENCE_HOME", "/tmp/cache")
DISABLE_INTERACTIVE_PROGRESS_BARS = str2bool(
    os.getenv("DISABLE_INTERACTIVE_PROGRESS_BARS", "False")
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
VERBOSE_LOG_LEVEL = os.getenv("VERBOSE_LOG_LEVEL", "INFO")
DISABLE_VERBOSE_LOGGER = str2bool(os.getenv("DISABLE_VERBOSE_LOGGER", "False"))
AUTO_LOADER_CACHE_EXPIRATION_MINUTES = int(
    os.getenv("AUTO_LOADER_CACHE_EXPIRATION_MINUTES", "1440")
)
ALLOW_URL_INPUT = str2bool(os.getenv("ALLOW_URL_INPUT", True))
ALLOW_NON_HTTPS_URL_INPUT = str2bool(os.getenv("ALLOW_NON_HTTPS_URL_INPUT", False))
ALLOW_URL_INPUT_WITHOUT_FQDN = str2bool(
    os.getenv("ALLOW_URL_INPUT_WITHOUT_FQDN", False)
)
WHITELISTED_DESTINATIONS_FOR_URL_INPUT = os.getenv(
    "WHITELISTED_DESTINATIONS_FOR_URL_INPUT"
)
if WHITELISTED_DESTINATIONS_FOR_URL_INPUT is not None:
    WHITELISTED_DESTINATIONS_FOR_URL_INPUT = parse_comma_separated_values(
        WHITELISTED_DESTINATIONS_FOR_URL_INPUT
    )

BLACKLISTED_DESTINATIONS_FOR_URL_INPUT = os.getenv(
    "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT"
)
if BLACKLISTED_DESTINATIONS_FOR_URL_INPUT is not None:
    BLACKLISTED_DESTINATIONS_FOR_URL_INPUT = parse_comma_separated_values(
        BLACKLISTED_DESTINATIONS_FOR_URL_INPUT
    )
ALLOW_LOCAL_STORAGE_ACCESS_FOR_REFERENCE_DATA = os.getenv(
    "ALLOW_LOCAL_STORAGE_ACCESS_FOR_REFERENCE_DATA"
)
