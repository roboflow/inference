import os

import torch

from inference_models.utils.environment import (
    get_boolean_from_env,
    get_comma_separated_list_of_integers_from_env,
    get_integer_from_env,
    parse_comma_separated_values,
)

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
API_CALLS_TIMEOUT = get_integer_from_env(variable_name="API_CALLS_TIMEOUT", default=5)
API_CALLS_MAX_TRIES = get_integer_from_env(
    variable_name="API_CALLS_MAX_TRIES", default=3
)
IDEMPOTENT_API_REQUEST_CODES_TO_RETRY = set(
    get_comma_separated_list_of_integers_from_env(
        variable_name="IDEMPOTENT_API_REQUEST_CODES_TO_RETRY",
        default=[408, 429, 502, 503, 504],
    )
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
DISABLE_INTERACTIVE_PROGRESS_BARS = get_boolean_from_env(
    variable_name="DISABLE_INTERACTIVE_PROGRESS_BARS",
    default=False,
)
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
VERBOSE_LOG_LEVEL = os.getenv("VERBOSE_LOG_LEVEL", "INFO")
DISABLE_VERBOSE_LOGGER = get_boolean_from_env(
    variable_name="DISABLE_VERBOSE_LOGGER", default=False
)
AUTO_LOADER_CACHE_EXPIRATION_MINUTES = get_integer_from_env(
    variable_name="AUTO_LOADER_CACHE_EXPIRATION_MINUTES", default=1440
)
ALLOW_URL_INPUT = get_boolean_from_env(variable_name="ALLOW_URL_INPUT", default=True)
ALLOW_NON_HTTPS_URL_INPUT = get_boolean_from_env(
    variable_name="ALLOW_NON_HTTPS_URL_INPUT", default=False
)
ALLOW_URL_INPUT_WITHOUT_FQDN = get_boolean_from_env(
    variable_name="ALLOW_URL_INPUT_WITHOUT_FQDN", default=False
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

USE_CUDA_GRAPHS_FOR_TRT_BACKEND = get_boolean_from_env(
    variable_name="USE_CUDA_GRAPHS_FOR_TRT_BACKEND",
    default=True,
)
