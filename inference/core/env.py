import os
import uuid

from dotenv import load_dotenv

load_dotenv(os.getcwd() + "/.env")

from inference.core.exceptions import InvalidEnvironmentVariableError


def bool_env(val):
    """
    Converts an environment variable to a boolean value.

    Args:
        val (str or bool): The environment variable value to be converted.

    Returns:
        bool: The converted boolean value.

    Raises:
        InvalidEnvironmentVariableError: If the value is not 'true', 'false', or a boolean.
    """
    if isinstance(val, bool):
        return val
    if val.lower() == "true":
        return True
    elif val.lower() == "false":
        return False
    else:
        raise InvalidEnvironmentVariableError(
            f"Expected a boolean environment variable (true or false) but got '{val}'"
        )


def required_providers_env(val):
    """
    Splits a comma-separated environment variable into a list.

    Args:
        val (str): The environment variable value to be split.

    Returns:
        list or None: The split values as a list, or None if the input is None.
    """
    if val is None:
        return None
    else:
        return val.split(",")


# The project name, default is "roboflow-platform"
PROJECT = os.getenv("PROJECT", "roboflow-platform")

# Allow numpy input, default is True
ALLOW_NUMPY_INPUT = bool_env(os.getenv("ALLOW_NUMPY_INPUT", True))

# List of allowed origins
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "")
ALLOW_ORIGINS = ALLOW_ORIGINS.split(",")

# Base URL for the API
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://api.roboflow.com"
    if PROJECT == "roboflow-platform"
    else "https://api.roboflow.one",
)

# Debug flag for the API, default is False
API_DEBUG = os.getenv("API_DEBUG", False)

# API key, default is None
API_KEY = os.getenv("ROBOFLOW_API_KEY", None) or os.getenv("API_KEY", None)

# AWS access key ID, default is None
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", None)

# AWS secret access key, default is None
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", None)

# CLIP version ID, default is "ViT-B-16"
CLIP_VERSION_ID = os.getenv("CLIP_VERSION_ID", "ViT-B-16")

# CLIP model ID
CLIP_MODEL_ID = f"clip/{CLIP_VERSION_ID}"

# Gaze version ID, default is "L2CS"
GAZE_VERSION_ID = os.getenv("GAZE_VERSION_ID", "L2CS")

# Gaze model ID
GAZE_MODEL_ID = f"gaze/{CLIP_VERSION_ID}"

# Maximum batch size for GAZE, default is 8
GAZE_MAX_BATCH_SIZE = int(os.getenv("GAZE_MAX_BATCH_SIZE", 8))

# Maximum batch size for CLIP, default is 8
CLIP_MAX_BATCH_SIZE = int(os.getenv("CLIP_MAX_BATCH_SIZE", 8))

# Class agnostic NMS flag, default is False
CLASS_AGNOSTIC_NMS = bool_env(os.getenv("CLASS_AGNOSTIC_NMS", False))

# Confidence threshold, default is 50%
CONFIDENCE = float(os.getenv("CONFIDENCE", 0.5))

# Flag to enable core models, default is True
CORE_MODELS_ENABLED = bool_env(os.getenv("CORE_MODELS_ENABLED", True))

# Flag to enable CLIP core model, default is True
CORE_MODEL_CLIP_ENABLED = bool_env(os.getenv("CORE_MODEL_CLIP_ENABLED", True))

# Flag to enable SAM core model, default is True
CORE_MODEL_SAM_ENABLED = bool_env(os.getenv("CORE_MODEL_SAM_ENABLED", True))

# Flag to enable GAZE core model, default is True
CORE_MODEL_GAZE_ENABLED = bool_env(os.getenv("CORE_MODEL_GAZE_ENABLED", True))

# ID of host device, default is None
DEVICE_ID = os.getenv("DEVICE_ID", None)

# Flag to disable inference cache, default is False
DISABLE_INFERENCE_CACHE = bool_env(os.getenv("DISABLE_INFERENCE_CACHE", False))

# Flag to disable auto-orientation preprocessing, default is False
DISABLE_PREPROC_AUTO_ORIENT = bool_env(os.getenv("DISABLE_PREPROC_AUTO_ORIENT", False))

# Flag to disable contrast preprocessing, default is False
DISABLE_PREPROC_CONTRAST = bool_env(os.getenv("DISABLE_PREPROC_CONTRAST", False))

# Flag to disable grayscale preprocessing, default is False
DISABLE_PREPROC_GRAYSCALE = bool_env(os.getenv("DISABLE_PREPROC_GRAYSCALE", False))

# Flag to disable static crop preprocessing, default is False
DISABLE_PREPROC_STATIC_CROP = bool_env(os.getenv("DISABLE_PREPROC_STATIC_CROP", False))

# Flag to disable version check, default is False
DISABLE_VERSION_CHECK = bool_env(os.getenv("DISABLE_VERSION_CHECK", False))

# ElastiCache endpoint
ELASTICACHE_ENDPOINT = os.environ.get(
    "ELASTICACHE_ENDPOINT",
    "roboflow-infer-prod.ljzegl.cfg.use2.cache.amazonaws.com:11211"
    if PROJECT == "roboflow-platform"
    else "roboflow-infer.ljzegl.cfg.use2.cache.amazonaws.com:11211",
)

# Flag to enable byte track, default is False
ENABLE_BYTE_TRACK = bool_env(os.getenv("ENABLE_BYTE_TRACK", False))

# Flag to enforce FPS, default is False
ENFORCE_FPS = bool_env(os.getenv("ENFORCE_FPS", False))

# Flag to fix batch size, default is False
FIX_BATCH_SIZE = bool_env(os.getenv("FIX_BATCH_SIZE", False))

# Host, default is "0.0.0.0"
HOST = os.getenv("HOST", "0.0.0.0")

# IoU threshold, default is 0.5
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.5))

# IP broadcast address, default is "127.0.0.1"
IP_BROADCAST_ADDR = os.getenv("IP_BROADCAST_ADDR", "127.0.0.1")

# IP broadcast port, default is 37020
IP_BROADCAST_PORT = int(os.getenv("IP_BROADCAST_PORT", 37020))

# Flag to enable JSON response, default is True
JSON_RESPONSE = bool_env(os.getenv("JSON_RESPONSE", True))

# Lambda flag, default is False
LAMBDA = bool_env(os.getenv("LAMBDA", False))

# Flag to enable legacy route, default is True
LEGACY_ROUTE_ENABLED = bool_env(os.getenv("LEGACY_ROUTE_ENABLED", True))

# License server, default is None
LICENSE_SERVER = os.getenv("LICENSE_SERVER", None)

# Log level, default is "INFO"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Maximum number of active models, default is 8
MAX_ACTIVE_MODELS = int(os.getenv("MAX_ACTIVE_MODELS", 8))

# Maximum batch size, default is 8
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 8))

# Maximum number of candidates, default is 3000
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", 3000))

# Maximum number of detections, default is 300
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", 300))

# Loop interval for expiration of memory cache, default is 5
MEMORY_CACHE_EXPIRE_INTERVAL = int(os.getenv("MEMORY_CACHE_EXPIRE_INTERVAL", 5))

# Metrics enabled flag, default is True
METRICS_ENABLED = bool_env(os.getenv("METRICS_ENABLED", True))
if LAMBDA:
    METRICS_ENABLED = False

# Interval for metrics aggregation, default is 60
METRICS_INTERVAL = int(os.getenv("METRICS_INTERVAL", 60))

# URL for posting metrics to Roboflow API, default is "{API_BASE_URL}/device-stats"
METRICS_URL = os.getenv("METRICS_URL", f"{API_BASE_URL}/device-healthcheck")

# Model cache directory, default is "/tmp/cache"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/cache")

# Model ID, default is None
MODEL_ID = os.getenv("MODEL_ID")

# Number of workers, default is 1
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 1))

ONNXRUNTIME_EXECUTION_PROVIDERS = os.getenv(
    "ONNXRUNTIME_EXECUTION_PROVIDERS", "[CUDAExecutionProvider,CPUExecutionProvider]"
)

# Port, default is 9001
PORT = int(os.getenv("PORT", 9001))

# Profile flag, default is False
PROFILE = bool_env(os.getenv("PROFILE", False))

# Redis host, default is None
REDIS_HOST = os.getenv("REDIS_HOST", None)

# Redis port, default is 6379
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Required ONNX providers, default is None
REQUIRED_ONNX_PROVIDERS = required_providers_env(
    os.getenv("REQUIRED_ONNX_PROVIDERS", None)
)

# Roboflow server UUID
ROBOFLOW_SERVER_UUID = os.getenv("ROBOFLOW_SERVER_UUID", str(uuid.uuid4()))

# Roboflow service secret, default is None
ROBOFLOW_SERVICE_SECRET = os.getenv("ROBOFLOW_SERVICE_SECRET", None)

# Maximum embedding cache size for SAM, default is 10
SAM_MAX_EMBEDDING_CACHE_SIZE = int(os.getenv("SAM_MAX_EMBEDDING_CACHE_SIZE", 10))

# SAM version ID, default is "vit_h"
SAM_VERSION_ID = os.getenv("SAM_VERSION_ID", "vit_h")

# Device ID, default is "sample-device-id"
INFERENCE_SERVER_ID = os.getenv("INFERENCE_SERVER_ID", None)

# Stream ID, default is None
STREAM_ID = os.getenv("STREAM_ID")
try:
    STREAM_ID = int(STREAM_ID)
except (TypeError, ValueError):
    pass

# Tags used for device management
TAGS = os.getenv("TAGS", "")
TAGS = [t for t in TAGS.split(",") if t]

# TensorRT cache path, default is MODEL_CACHE_DIR
TENSORRT_CACHE_PATH = os.getenv("TENSORRT_CACHE_PATH", MODEL_CACHE_DIR)

# Set TensorRT cache path
os.environ["ORT_TENSORRT_CACHE_PATH"] = TENSORRT_CACHE_PATH

# Version check mode, one of "once" or "continuous", default is "once"
VERSION_CHECK_MODE = os.getenv("VERSION_CHECK_MODE", "once")

# Metlo key, default is None
METLO_KEY = os.getenv("METLO_KEY", None)

# Core model bucket
CORE_MODEL_BUCKET = os.getenv(
    "CORE_MODEL_BUCKET",
    "roboflow-core-model-prod"
    if PROJECT == "roboflow-platform"
    else "roboflow-core-model-staging",
)

# Inference bucket
INFER_BUCKET = os.getenv(
    "INFER_BUCKET",
    "roboflow-infer-prod"
    if PROJECT == "roboflow-platform"
    else "roboflow-infer-staging",
)
