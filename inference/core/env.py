import os
import uuid
import warnings
from typing import Optional

from dotenv import load_dotenv

from inference.core.utils.environment import safe_split_value, str2bool
from inference.core.warnings import InferenceDeprecationWarning, ModelDependencyMissing

load_dotenv(os.getcwd() + "/.env")

# The project name, default is "roboflow-platform"
PROJECT = os.getenv("PROJECT", "roboflow-platform")

# Allow numpy input, default is False
ALLOW_NUMPY_INPUT = str2bool(os.getenv("ALLOW_NUMPY_INPUT", False))
ALLOW_URL_INPUT = str2bool(os.getenv("ALLOW_URL_INPUT", True))
ALLOW_NON_HTTPS_URL_INPUT = str2bool(os.getenv("ALLOW_NON_HTTPS_URL_INPUT", False))
ALLOW_URL_INPUT_WITHOUT_FQDN = str2bool(
    os.getenv("ALLOW_URL_INPUT_WITHOUT_FQDN", False)
)
WHITELISTED_DESTINATIONS_FOR_URL_INPUT = os.getenv(
    "WHITELISTED_DESTINATIONS_FOR_URL_INPUT"
)
if WHITELISTED_DESTINATIONS_FOR_URL_INPUT is not None:
    WHITELISTED_DESTINATIONS_FOR_URL_INPUT = set(
        WHITELISTED_DESTINATIONS_FOR_URL_INPUT.split(",")
    )
BLACKLISTED_DESTINATIONS_FOR_URL_INPUT = os.getenv(
    "BLACKLISTED_DESTINATIONS_FOR_URL_INPUT"
)
if BLACKLISTED_DESTINATIONS_FOR_URL_INPUT is not None:
    BLACKLISTED_DESTINATIONS_FOR_URL_INPUT = set(
        BLACKLISTED_DESTINATIONS_FOR_URL_INPUT.split(",")
    )

# List of allowed origins
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
ALLOW_ORIGINS = ALLOW_ORIGINS.split(",")

# Base URL for the API
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    (
        "https://api.roboflow.com"
        if PROJECT == "roboflow-platform"
        else "https://api.roboflow.one"
    ),
)
# Base URL for metrics collector
METRICS_COLLECTOR_BASE_URL = os.getenv(
    "METRICS_COLLECTOR_BASE_URL",
    API_BASE_URL,
)

# extra headers expected to be serialised json
ROBOFLOW_API_EXTRA_HEADERS = os.getenv("ROBOFLOW_API_EXTRA_HEADERS")

# Debug flag for the API, default is False
API_DEBUG = os.getenv("API_DEBUG", False)

# API key, default is None
API_KEY_ENV_NAMES = ["ROBOFLOW_API_KEY", "API_KEY"]
API_KEY = os.getenv(API_KEY_ENV_NAMES[0], None) or os.getenv(API_KEY_ENV_NAMES[1], None)

# AWS access key ID, default is None
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", None)

# AWS secret access key, default is None
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", None)

PALIGEMMA_VERSION_ID = os.getenv("PALIGEMMA_VERSION_ID", "paligemma-3b-mix-224")
# CLIP version ID, default is "ViT-B-16"
CLIP_VERSION_ID = os.getenv("CLIP_VERSION_ID", "ViT-B-16")

# CLIP model ID
CLIP_MODEL_ID = f"clip/{CLIP_VERSION_ID}"

# Gaze version ID, default is "L2CS"
GAZE_VERSION_ID = os.getenv("GAZE_VERSION_ID", "L2CS")

# Gaze model ID
GAZE_MODEL_ID = f"gaze/{GAZE_VERSION_ID}"

# OWLv2 version ID, default is "owlv2-large-patch14-ensemble"
OWLV2_VERSION_ID = os.getenv("OWLV2_VERSION_ID", "owlv2-large-patch14-ensemble")

# OWLv2 image cache size, default is 1000 since each image has max <MAX_DETECTIONS> boxes at ~4kb each
OWLV2_IMAGE_CACHE_SIZE = int(os.getenv("OWLV2_IMAGE_CACHE_SIZE", 10000))

# OWLv2 model cache size, default is 100 as memory is num_prompts * ~4kb and num_prompts is rarely above 1000 (but could be much higher)
OWLV2_MODEL_CACHE_SIZE = int(os.getenv("OWLV2_MODEL_CACHE_SIZE", 100))

# OWLv2 CPU image cache size, default is 10000
OWLV2_CPU_IMAGE_CACHE_SIZE = int(os.getenv("OWLV2_CPU_IMAGE_CACHE_SIZE", 1000))

# OWLv2 compile model, default is True
OWLV2_COMPILE_MODEL = str2bool(os.getenv("OWLV2_COMPILE_MODEL", True))

# Maximum batch size for GAZE, default is 8
GAZE_MAX_BATCH_SIZE = int(os.getenv("GAZE_MAX_BATCH_SIZE", 8))

# If true, this will store a non-verbose version of the inference request and repsonse in the cache
TINY_CACHE = str2bool(os.getenv("TINY_CACHE", True))

# Maximum batch size for CLIP, default is 8
CLIP_MAX_BATCH_SIZE = int(os.getenv("CLIP_MAX_BATCH_SIZE", 8))

# Class agnostic NMS flag, default is False
CLASS_AGNOSTIC_NMS_ENV = "CLASS_AGNOSTIC_NMS"
DEFAULT_CLASS_AGNOSTIC_NMS = False
CLASS_AGNOSTIC_NMS = str2bool(
    os.getenv(CLASS_AGNOSTIC_NMS_ENV, DEFAULT_CLASS_AGNOSTIC_NMS)
)

# Confidence threshold, default is 50%
CONFIDENCE_ENV = "CONFIDENCE"
DEFAULT_CONFIDENCE = 0.4
CONFIDENCE = float(os.getenv(CONFIDENCE_ENV, DEFAULT_CONFIDENCE))

# Flag to enable core models, default is True
CORE_MODELS_ENABLED = str2bool(os.getenv("CORE_MODELS_ENABLED", True))

# Flag to enable CLIP core model, default is True
CORE_MODEL_CLIP_ENABLED = str2bool(os.getenv("CORE_MODEL_CLIP_ENABLED", True))

# Flag to enable SAM core model, default is True
CORE_MODEL_SAM_ENABLED = str2bool(os.getenv("CORE_MODEL_SAM_ENABLED", True))
CORE_MODEL_SAM2_ENABLED = str2bool(os.getenv("CORE_MODEL_SAM2_ENABLED", True))

CORE_MODEL_OWLV2_ENABLED = str2bool(os.getenv("CORE_MODEL_OWLV2_ENABLED", False))

# Flag to enable GAZE core model, default is True
CORE_MODEL_GAZE_ENABLED = str2bool(os.getenv("CORE_MODEL_GAZE_ENABLED", True))

# Flag to enable DocTR core model, default is True
CORE_MODEL_DOCTR_ENABLED = str2bool(os.getenv("CORE_MODEL_DOCTR_ENABLED", True))

# Flag to enable TrOCR core model, default is True
CORE_MODEL_TROCR_ENABLED = str2bool(os.getenv("CORE_MODEL_TROCR_ENABLED", True))

# Flag to enable GROUNDINGDINO core model, default is True
CORE_MODEL_GROUNDINGDINO_ENABLED = str2bool(
    os.getenv("CORE_MODEL_GROUNDINGDINO_ENABLED", True)
)

LMM_ENABLED = str2bool(os.getenv("LMM_ENABLED", False))

# Flag to enable YOLO-World core model, default is True
CORE_MODEL_YOLO_WORLD_ENABLED = str2bool(
    os.getenv("CORE_MODEL_YOLO_WORLD_ENABLED", True)
)

# ID of host device, default is None
DEVICE_ID = os.getenv("DEVICE_ID", None)

# Whether or not to use PyTorch for preprocessing, default is False
USE_PYTORCH_FOR_PREPROCESSING = str2bool(
    os.getenv("USE_PYTORCH_FOR_PREPROCESSING", False)
)

# Flag to disable inference cache, default is False
DISABLE_INFERENCE_CACHE = str2bool(os.getenv("DISABLE_INFERENCE_CACHE", False))

# Flag to disable auto-orientation preprocessing, default is False
DISABLE_PREPROC_AUTO_ORIENT = str2bool(os.getenv("DISABLE_PREPROC_AUTO_ORIENT", False))

# Flag to disable contrast preprocessing, default is False
DISABLE_PREPROC_CONTRAST = str2bool(os.getenv("DISABLE_PREPROC_CONTRAST", False))

# Flag to disable grayscale preprocessing, default is False
DISABLE_PREPROC_GRAYSCALE = str2bool(os.getenv("DISABLE_PREPROC_GRAYSCALE", False))

# Flag to disable static crop preprocessing, default is False
DISABLE_PREPROC_STATIC_CROP = str2bool(os.getenv("DISABLE_PREPROC_STATIC_CROP", False))

# Flag to disable version check, default is False
DISABLE_VERSION_CHECK = str2bool(os.getenv("DISABLE_VERSION_CHECK", False))

# ElastiCache endpoint
ELASTICACHE_ENDPOINT = os.environ.get(
    "ELASTICACHE_ENDPOINT",
    (
        "roboflow-infer-prod.ljzegl.cfg.use2.cache.amazonaws.com:11211"
        if PROJECT == "roboflow-platform"
        else "roboflow-infer.ljzegl.cfg.use2.cache.amazonaws.com:11211"
    ),
)

# Flag to enable byte track, default is False
ENABLE_BYTE_TRACK = str2bool(os.getenv("ENABLE_BYTE_TRACK", False))

ENABLE_PROMETHEUS = str2bool(os.getenv("ENABLE_PROMETHEUS", False))

# Flag to enforce FPS, default is False
ENFORCE_FPS = str2bool(os.getenv("ENFORCE_FPS", False))
MAX_FPS = os.getenv("MAX_FPS")
if MAX_FPS is not None:
    MAX_FPS = int(MAX_FPS)

# Flag to fix batch size, default is False
FIX_BATCH_SIZE = str2bool(os.getenv("FIX_BATCH_SIZE", False))

# Host, default is "0.0.0.0"
HOST = os.getenv("HOST", "0.0.0.0")

# IoU threshold, default is 0.3
IOU_THRESHOLD_ENV = "IOU_THRESHOLD"
DEFAULT_IOU_THRESHOLD = 0.3
IOU_THRESHOLD = float(os.getenv(IOU_THRESHOLD_ENV, DEFAULT_IOU_THRESHOLD))

# IP broadcast address, default is "127.0.0.1"
IP_BROADCAST_ADDR = os.getenv("IP_BROADCAST_ADDR", "127.0.0.1")

# IP broadcast port, default is 37020
IP_BROADCAST_PORT = int(os.getenv("IP_BROADCAST_PORT", 37020))

# Flag to enable JSON response, default is True
JSON_RESPONSE = str2bool(os.getenv("JSON_RESPONSE", True))

# Lambda flag, default is False
LAMBDA = str2bool(os.getenv("LAMBDA", False))

# Whether is's GCP serverless service
GCP_SERVERLESS = str2bool(os.getenv("GCP_SERVERLESS", "False"))

# Flag to enable legacy route, default is True
LEGACY_ROUTE_ENABLED = str2bool(os.getenv("LEGACY_ROUTE_ENABLED", True))

# License server, default is None
LICENSE_SERVER = os.getenv("LICENSE_SERVER", None)

# Log level, default is "INFO"
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")

# Maximum number of active models, default is 8
MAX_ACTIVE_MODELS = int(os.getenv("MAX_ACTIVE_MODELS", 8))

# Maximum batch size, default is infinite
MAX_BATCH_SIZE = os.getenv("MAX_BATCH_SIZE", None)
if MAX_BATCH_SIZE is not None:
    MAX_BATCH_SIZE = int(MAX_BATCH_SIZE)
else:
    MAX_BATCH_SIZE = float("inf")

# Maximum number of candidates, default is 3000
MAX_CANDIDATES_ENV = "MAX_CANDIDATES"
DEFAULT_MAX_CANDIDATES = 3000
MAX_CANDIDATES = int(os.getenv(MAX_CANDIDATES_ENV, DEFAULT_MAX_CANDIDATES))

# Maximum number of detections, default is 300
MAX_DETECTIONS_ENV = "MAX_DETECTIONS"
DEFAULT_MAX_DETECTIONS = 300
MAX_DETECTIONS = int(os.getenv(MAX_DETECTIONS_ENV, DEFAULT_MAX_DETECTIONS))

# Loop interval for expiration of memory cache, default is 5
MEMORY_CACHE_EXPIRE_INTERVAL = int(os.getenv("MEMORY_CACHE_EXPIRE_INTERVAL", 5))

# Enable models cache auth
MODELS_CACHE_AUTH_ENABLED = str2bool(os.getenv("MODELS_CACHE_AUTH_ENABLED", False))

# Models cache auth cache ttl, default is 15 minutes
MODELS_CACHE_AUTH_CACHE_TTL = int(os.getenv("MODELS_CACHE_AUTH_CACHE_TTL", 15 * 60))

# Models cache auth cache max size, default is 0 (unlimited)
MODELS_CACHE_AUTH_CACHE_MAX_SIZE = int(os.getenv("MODELS_CACHE_AUTH_CACHE_MAX_SIZE", 0))

# Metrics enabled flag, default is True
METRICS_ENABLED = str2bool(os.getenv("METRICS_ENABLED", True))
if LAMBDA or GCP_SERVERLESS:
    METRICS_ENABLED = False

# Interval for metrics aggregation, default is 60
METRICS_INTERVAL = int(os.getenv("METRICS_INTERVAL", 60))

# URL for posting metrics to Roboflow API, default is "{API_BASE_URL}/inference-stats"
METRICS_URL = os.getenv("METRICS_URL", f"{API_BASE_URL}/inference-stats")

# Model cache directory, default is "/tmp/cache"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/cache")

# Model ID, default is None
MODEL_ID = os.getenv("MODEL_ID")

# Enable the builder, default is False
ENABLE_BUILDER = str2bool(os.getenv("ENABLE_BUILDER", False))
BUILDER_ORIGIN = os.getenv("BUILDER_ORIGIN", "https://app.roboflow.com")

# Enable jupyter notebook server route, default is False
NOTEBOOK_ENABLED = str2bool(os.getenv("NOTEBOOK_ENABLED", False))

# Jupyter notebook password, default is "roboflow"
NOTEBOOK_PASSWORD = os.getenv("NOTEBOOK_PASSWORD", "roboflow")

# Jupyter notebook port, default is 9002
NOTEBOOK_PORT = int(os.getenv("NOTEBOOK_PORT", 9002))

# Number of workers, default is 1
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 1))

ONNXRUNTIME_EXECUTION_PROVIDERS = os.getenv(
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "[CUDAExecutionProvider,OpenVINOExecutionProvider,CoreMLExecutionProvider,CPUExecutionProvider]",
)

# Port, default is 9001
PORT = int(os.getenv("PORT", 9001))

# Profile flag, default is False
PROFILE = str2bool(os.getenv("PROFILE", False))

# Redis host, default is None
REDIS_HOST = os.getenv("REDIS_HOST", None)

# Redis port, default is 6379
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_SSL = str2bool(os.getenv("REDIS_SSL", False))
REDIS_TIMEOUT = float(os.getenv("REDIS_TIMEOUT", 2.0))

# Required ONNX providers, default is None
REQUIRED_ONNX_PROVIDERS = safe_split_value(os.getenv("REQUIRED_ONNX_PROVIDERS", None))

# Roboflow server UUID
ROBOFLOW_SERVER_UUID = os.getenv("ROBOFLOW_SERVER_UUID", str(uuid.uuid4()))

# Roboflow service secret, default is None
ROBOFLOW_SERVICE_SECRET = os.getenv("ROBOFLOW_SERVICE_SECRET", None)

# Maximum embedding cache size for SAM, default is 10
SAM_MAX_EMBEDDING_CACHE_SIZE = int(os.getenv("SAM_MAX_EMBEDDING_CACHE_SIZE", 10))

SAM2_MAX_EMBEDDING_CACHE_SIZE = int(os.getenv("SAM2_MAX_EMBEDDING_CACHE_SIZE", 100))
SAM2_MAX_LOGITS_CACHE_SIZE = int(os.getenv("SAM2_MAX_LOGITS_CACHE_SIZE", 1000))
DISABLE_SAM2_LOGITS_CACHE = str2bool(os.getenv("DISABLE_SAM2_LOGITS_CACHE", False))

# SAM version ID, default is "vit_h"
SAM_VERSION_ID = os.getenv("SAM_VERSION_ID", "vit_h")
SAM2_VERSION_ID = os.getenv("SAM2_VERSION_ID", "hiera_large")


# Device ID, default is "sample-device-id"
INFERENCE_SERVER_ID = os.getenv("INFERENCE_SERVER_ID", None)

# Stream ID, default is None
STREAM_ID = os.getenv("STREAM_ID")
try:
    STREAM_ID = int(STREAM_ID)
except (TypeError, ValueError):
    pass

# Tags used for device management
TAGS = safe_split_value(os.getenv("TAGS", ""))

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
    (
        "roboflow-core-model-prod"
        if PROJECT == "roboflow-platform"
        else "roboflow-core-model-staging"
    ),
)

# Inference bucket
INFER_BUCKET = os.getenv(
    "INFER_BUCKET",
    (
        "roboflow-infer-prod"
        if PROJECT == "roboflow-platform"
        else "roboflow-infer-staging"
    ),
)

ACTIVE_LEARNING_ENABLED = str2bool(os.getenv("ACTIVE_LEARNING_ENABLED", True))
ACTIVE_LEARNING_TAGS = safe_split_value(os.getenv("ACTIVE_LEARNING_TAGS", None))

# Number inflight async tasks for async model manager
NUM_PARALLEL_TASKS = int(os.getenv("NUM_PARALLEL_TASKS", 512))
STUB_CACHE_SIZE = int(os.getenv("STUB_CACHE_SIZE", 256))
# New stream interface variables
PREDICTIONS_QUEUE_SIZE = int(
    os.getenv("INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE", 512)
)
RESTART_ATTEMPT_DELAY = int(os.getenv("INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY", 1))
DEFAULT_BUFFER_SIZE = int(os.getenv("VIDEO_SOURCE_BUFFER_SIZE", "64"))
DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE = float(
    os.getenv("VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE", "0.1")
)
DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE = float(
    os.getenv("VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE", "5.0")
)
DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES = int(
    os.getenv("VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES", "10")
)
DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW = int(
    os.getenv("VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW", "16")
)

ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING = str2bool(
    os.getenv("ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING", "False")
)

NUM_CELERY_WORKERS = os.getenv("NUM_CELERY_WORKERS", 4)
CELERY_LOG_LEVEL = os.getenv("CELERY_LOG_LEVEL", "WARNING")


LOCAL_INFERENCE_API_URL = os.getenv("LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
HOSTED_DETECT_URL = (
    "https://detect.roboflow.com"
    if PROJECT == "roboflow-platform"
    else "https://lambda-object-detection.staging.roboflow.com"
)
HOSTED_INSTANCE_SEGMENTATION_URL = (
    "https://outline.roboflow.com"
    if PROJECT == "roboflow-platform"
    else "https://lambda-instance-segmentation.staging.roboflow.com"
)
HOSTED_CLASSIFICATION_URL = (
    "https://classify.roboflow.com"
    if PROJECT == "roboflow-platform"
    else "https://lambda-classification.staging.roboflow.com"
)
HOSTED_CORE_MODEL_URL = (
    "https://infer.roboflow.com"
    if PROJECT == "roboflow-platform"
    else "https://3hkaykeh3j.execute-api.us-east-1.amazonaws.com"
)

DISABLE_WORKFLOW_ENDPOINTS = str2bool(os.getenv("DISABLE_WORKFLOW_ENDPOINTS", False))
WORKFLOWS_STEP_EXECUTION_MODE = os.getenv("WORKFLOWS_STEP_EXECUTION_MODE", "local")
WORKFLOWS_REMOTE_API_TARGET = os.getenv("WORKFLOWS_REMOTE_API_TARGET", "hosted")
WORKFLOWS_MAX_CONCURRENT_STEPS = int(os.getenv("WORKFLOWS_MAX_CONCURRENT_STEPS", "8"))
WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE = int(
    os.getenv("WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE", "1")
)
WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS = int(
    os.getenv("WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", "8")
)
ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS = str2bool(
    os.getenv("ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", True)
)

MODEL_VALIDATION_DISABLED = str2bool(os.getenv("MODEL_VALIDATION_DISABLED", "False"))

INFERENCE_WARNINGS_DISABLED = str2bool(
    os.getenv("INFERENCE_WARNINGS_DISABLED", "False")
)

if INFERENCE_WARNINGS_DISABLED:
    warnings.simplefilter("ignore", InferenceDeprecationWarning)

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DEVICE = os.getenv("DEVICE")

DEDICATED_DEPLOYMENT_WORKSPACE_URL = os.environ.get(
    "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None
)

ENABLE_STREAM_API = str2bool(os.getenv("ENABLE_STREAM_API", "False"))
STREAM_API_PRELOADED_PROCESSES = int(os.getenv("STREAM_API_PRELOADED_PROCESSES", "0"))

RUNS_ON_JETSON = str2bool(os.getenv("RUNS_ON_JETSON", "False"))

DOCKER_SOCKET_PATH: Optional[str] = os.getenv("DOCKER_SOCKET_PATH")

ENABLE_WORKFLOWS_PROFILING = str2bool(os.getenv("ENABLE_WORKFLOWS_PROFILING", "False"))
WORKFLOWS_PROFILER_BUFFER_SIZE = int(os.getenv("WORKFLOWS_PROFILER_BUFFER_SIZE", "64"))
WORKFLOWS_DEFINITION_CACHE_EXPIRY = int(
    os.getenv("WORKFLOWS_DEFINITION_CACHE_EXPIRY", 15 * 60)
)
USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS = str2bool(
    os.getenv("USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS", "True")
)
ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE = str2bool(
    os.getenv("ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE", "True")
)
ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES = str2bool(
    os.getenv("ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES", "True")
)
ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM = str2bool(
    os.getenv("ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", "True")
)
WORKFLOW_BLOCKS_WRITE_DIRECTORY = os.getenv("WORKFLOW_BLOCKS_WRITE_DIRECTORY")

DEDICATED_DEPLOYMENT_ID = os.getenv("DEDICATED_DEPLOYMENT_ID")

ROBOFLOW_INTERNAL_SERVICE_SECRET = os.getenv("ROBOFLOW_INTERNAL_SERVICE_SECRET")
ROBOFLOW_INTERNAL_SERVICE_NAME = os.getenv("ROBOFLOW_INTERNAL_SERVICE_NAME")

# Preload Models
PRELOAD_MODELS = (
    os.getenv("PRELOAD_MODELS").split(",") if os.getenv("PRELOAD_MODELS") else None
)

LOAD_ENTERPRISE_BLOCKS = str2bool(os.getenv("LOAD_ENTERPRISE_BLOCKS", "False"))
TRANSIENT_ROBOFLOW_API_ERRORS = set(
    int(e)
    for e in os.getenv("TRANSIENT_ROBOFLOW_API_ERRORS", "").split(",")
    if len(e) > 0
)
RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API = str2bool(
    os.getenv("RETRY_CONNECTION_ERRORS_TO_ROBOFLOW_API", "False")
)
TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES = int(
    os.getenv("TRANSIENT_ROBOFLOW_API_ERRORS_RETRIES", "3")
)
TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL = int(
    os.getenv("TRANSIENT_ROBOFLOW_API_ERRORS_RETRY_INTERVAL", "1")
)
ROBOFLOW_API_REQUEST_TIMEOUT = os.getenv("ROBOFLOW_API_REQUEST_TIMEOUT")
if ROBOFLOW_API_REQUEST_TIMEOUT:
    ROBOFLOW_API_REQUEST_TIMEOUT = int(ROBOFLOW_API_REQUEST_TIMEOUT)


IGNORE_MODEL_DEPENDENCIES_WARNINGS = str2bool(
    os.getenv("IGNORE_MODEL_DEPENDENCIES_WARNINGS", "False")
)
if IGNORE_MODEL_DEPENDENCIES_WARNINGS:
    warnings.simplefilter("ignore", ModelDependencyMissing)

DISK_CACHE_CLEANUP = str2bool(os.getenv("DISK_CACHE_CLEANUP", "True"))

# Stream manager configuration
try:
    STREAM_MANAGER_MAX_RAM_MB: Optional[float] = abs(
        float(os.getenv("STREAM_MANAGER_MAX_RAM_MB"))
    )
except:
    STREAM_MANAGER_MAX_RAM_MB: Optional[float] = None

try:
    STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE: int = abs(
        int(os.getenv("STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE"))
    )
except:
    STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE = 10
