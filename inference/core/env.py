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

# Suffix path to be appended to API_BASE_URL for endpoints that serve model weights.
# This is only expected to be used in Roboflow internal hosting environments.
INTERNAL_WEIGHTS_URL_SUFFIX = os.getenv("INTERNAL_WEIGHTS_URL_SUFFIX", "")

MD5_VERIFICATION_ENABLED = str2bool(os.getenv("MD5_VERIFICATION_ENABLED", False))

ATOMIC_CACHE_WRITES_ENABLED = str2bool(os.getenv("ATOMIC_CACHE_WRITES_ENABLED", False))

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

# Perception Encoder version ID, default is "PE-Core-L14-336"
PERCEPTION_ENCODER_VERSION_ID = os.getenv(
    "PERCEPTION_ENCODER_VERSION_ID", "PE-Core-L14-336"
)

# Perception Encoder model ID
PERCEPTION_ENCODER_MODEL_ID = f"perception_encoder/{PERCEPTION_ENCODER_VERSION_ID}"

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

# Preload comma separated list of Huggingface IDs for OWLv2 models
# NOTE: this will result in ALL inference processes to preload the models
#       (e.g. InferencePipelineManager, InferencePipeline, etc.)
#       Ensure NUM_WORKERS environmental variable is set to 1
#       and also ENABLE_STREAM_API environmental variable is set to False
PRELOAD_HF_IDS = os.getenv("PRELOAD_HF_IDS")
if PRELOAD_HF_IDS:
    PRELOAD_HF_IDS = [m.strip() for m in PRELOAD_HF_IDS.split(",")]

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

# Flag to enable PE core model, default is True
CORE_MODEL_PE_ENABLED = str2bool(os.getenv("CORE_MODEL_PE_ENABLED", True))

# Flag to enable SAM core model, default is True
CORE_MODEL_SAM_ENABLED = str2bool(os.getenv("CORE_MODEL_SAM_ENABLED", True))
CORE_MODEL_SAM2_ENABLED = str2bool(os.getenv("CORE_MODEL_SAM2_ENABLED", True))
CORE_MODEL_SAM3_ENABLED = str2bool(os.getenv("CORE_MODEL_SAM3_ENABLED", True))

CORE_MODEL_OWLV2_ENABLED = str2bool(os.getenv("CORE_MODEL_OWLV2_ENABLED", False))

# Maximum prompt batch size for SAM3 PCS requests
SAM3_MAX_PROMPT_BATCH_SIZE = int(os.getenv("SAM3_MAX_PROMPT_BATCH_SIZE", 16))
SAM3_EXEC_MODE = os.getenv("SAM3_EXEC_MODE", "local")
SAM3_EXEC_MODE = SAM3_EXEC_MODE.lower()
if SAM3_EXEC_MODE not in ["local", "remote"]:
    raise ValueError(
        f"Invalid SAM3 execution mode in ENVIRONMENT var SAM3_EXEC_MODE (local or remote): {SAM3_EXEC_MODE}"
    )

# Flag to enable GAZE core model, default is True
CORE_MODEL_GAZE_ENABLED = str2bool(os.getenv("CORE_MODEL_GAZE_ENABLED", True))

# Flag to enable DocTR core model, default is True
CORE_MODEL_DOCTR_ENABLED = str2bool(os.getenv("CORE_MODEL_DOCTR_ENABLED", True))

# Flag to enable EasyOCR core model, default is True
CORE_MODEL_EASYOCR_ENABLED = str2bool(os.getenv("CORE_MODEL_EASYOCR_ENABLED", True))

# Flag to enable TrOCR core model, default is True
CORE_MODEL_TROCR_ENABLED = str2bool(os.getenv("CORE_MODEL_TROCR_ENABLED", True))

# Flag to enable GROUNDINGDINO core model, default is True
CORE_MODEL_GROUNDINGDINO_ENABLED = str2bool(
    os.getenv("CORE_MODEL_GROUNDINGDINO_ENABLED", True)
)

LMM_ENABLED = str2bool(os.getenv("LMM_ENABLED", False))

QWEN_2_5_ENABLED = str2bool(os.getenv("QWEN_2_5_ENABLED", True))

QWEN_3_ENABLED = str2bool(os.getenv("QWEN_3_ENABLED", True))

DEPTH_ESTIMATION_ENABLED = str2bool(os.getenv("DEPTH_ESTIMATION_ENABLED", True))

SMOLVLM2_ENABLED = str2bool(os.getenv("SMOLVLM2_ENABLED", True))

MOONDREAM2_ENABLED = str2bool(os.getenv("MOONDREAM2_ENABLED", True))

PALIGEMMA_ENABLED = str2bool(os.getenv("PALIGEMMA_ENABLED", True))

FLORENCE2_ENABLED = str2bool(os.getenv("FLORENCE2_ENABLED", True))

SAM3_3D_OBJECTS_ENABLED = str2bool(os.getenv("SAM3_3D_OBJECTS_ENABLED", False))

# Flag to enable YOLO-World core model, default is True
CORE_MODEL_YOLO_WORLD_ENABLED = str2bool(
    os.getenv("CORE_MODEL_YOLO_WORLD_ENABLED", True)
)

# Enable experimental RFDETR backend (inference_models) rollout, default is True
USE_INFERENCE_EXP_MODELS = str2bool(os.getenv("USE_INFERENCE_EXP_MODELS", "False"))
ALLOW_INFERENCE_EXP_UNTRUSTED_MODELS = str2bool(
    os.getenv("ALLOW_INFERENCE_EXP_UNTRUSTED_MODELS", "False")
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

GET_MODEL_REGISTRY_ENABLED = str2bool(os.getenv("GET_MODEL_REGISTRY_ENABLED", "True"))

# Flag to enable API logging, default is False
API_LOGGING_ENABLED = str2bool(os.getenv("API_LOGGING_ENABLED", "False"))

# Header where correlaction ID for logging is stored
CORRELATION_ID_HEADER = os.getenv("CORRELATION_ID_HEADER", "X-Request-ID")

# Header where correlaction ID for logging is stored
CORRELATION_ID_LOG_KEY = os.getenv("CORRELATION_ID_LOG_KEY", "request_id")

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

# Models cache auth cache max size, default is 100_000_000 (0 DOES NOT MAKE IT UNLIMITED)
MODELS_CACHE_AUTH_CACHE_MAX_SIZE = int(
    os.getenv("MODELS_CACHE_AUTH_CACHE_MAX_SIZE", 100_000_000)
)

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

# Enable in-memory logs, default is False
ENABLE_IN_MEMORY_LOGS = str2bool(os.getenv("ENABLE_IN_MEMORY_LOGS", False))

# Enable dashboard page
ENABLE_DASHBOARD = str2bool(os.getenv("ENABLE_DASHBOARD", False))

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
# SAM3_CHECKPOINT_PATH = os.getenv("SAM3_CHECKPOINT_PATH")
# SAM3_BPE_PATH = os.getenv("SAM3_BPE_PATH", "/home/hansent/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
SAM3_IMAGE_SIZE = int(os.getenv("SAM3_IMAGE_SIZE", 1008))
# SAM3_REPO_PATH = os.getenv("SAM3_REPO_PATH", "/home/hansent/sam3")
SAM3_MAX_EMBEDDING_CACHE_SIZE = int(os.getenv("SAM3_MAX_EMBEDDING_CACHE_SIZE", 100))
SAM3_MAX_LOGITS_CACHE_SIZE = int(os.getenv("SAM3_MAX_LOGITS_CACHE_SIZE", 1000))
DISABLE_SAM3_LOGITS_CACHE = str2bool(os.getenv("DISABLE_SAM3_LOGITS_CACHE", False))

# EasyOCR version ID, default is "english_g2"
EASYOCR_VERSION_ID = os.getenv("EASYOCR_VERSION_ID", "english_g2")

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

DEBUG_AIORTC_QUEUES = str2bool(os.getenv("DEBUG_AIORTC_QUEUES", "False"))
DEBUG_WEBRTC_PROCESSING_LATENCY = str2bool(
    os.getenv("DEBUG_WEBRTC_PROCESSING_LATENCY", "False")
)
WEBRTC_REALTIME_PROCESSING = str2bool(os.getenv("WEBRTC_REALTIME_PROCESSING", "True"))

NUM_CELERY_WORKERS = os.getenv("NUM_CELERY_WORKERS", 4)
CELERY_LOG_LEVEL = os.getenv("CELERY_LOG_LEVEL", "WARNING")


LOCAL_INFERENCE_API_URL = os.getenv("LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
HOSTED_DETECT_URL = os.getenv(
    "HOSTED_DETECT_URL",
    (
        "https://detect.roboflow.com"
        if PROJECT == "roboflow-platform"
        else "https://lambda-object-detection.staging.roboflow.com"
    ),
)
HOSTED_INSTANCE_SEGMENTATION_URL = os.getenv(
    "HOSTED_INSTANCE_SEGMENTATION_URL",
    (
        "https://outline.roboflow.com"
        if PROJECT == "roboflow-platform"
        else "https://lambda-instance-segmentation.staging.roboflow.com"
    ),
)
HOSTED_CLASSIFICATION_URL = os.getenv(
    "HOSTED_CLASSIFICATION_URL",
    (
        "https://classify.roboflow.com"
        if PROJECT == "roboflow-platform"
        else "https://lambda-classification.staging.roboflow.com"
    ),
)
HOSTED_CORE_MODEL_URL = os.getenv(
    "HOSTED_CORE_MODEL_URL",
    (
        "https://infer.roboflow.com"
        if PROJECT == "roboflow-platform"
        else "https://3hkaykeh3j.execute-api.us-east-1.amazonaws.com"
    ),
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

# Modal configuration for Custom Python Blocks
WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE = os.getenv(
    "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local"
).lower()  # "local" or "modal"

# Strip quotes from Modal credentials in case users include them
_modal_token_id = os.getenv("MODAL_TOKEN_ID")
_modal_token_secret = os.getenv("MODAL_TOKEN_SECRET")

# Remove common quote characters that users might accidentally include
MODAL_TOKEN_ID = _modal_token_id.strip("\"'") if _modal_token_id else None
MODAL_TOKEN_SECRET = _modal_token_secret.strip("\"'") if _modal_token_secret else None
MODAL_WORKSPACE_NAME = os.getenv("MODAL_WORKSPACE_NAME", "roboflow")

# Control whether anonymous Modal execution is allowed (when no api_key is available)
MODAL_ALLOW_ANONYMOUS_EXECUTION = str2bool(
    os.getenv("MODAL_ALLOW_ANONYMOUS_EXECUTION", "False")
)

MODAL_ANONYMOUS_WORKSPACE_NAME = os.getenv(
    "MODAL_ANONYMOUS_WORKSPACE_NAME", "anonymous"
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


# Control SSL certificate verification for requests to the Roboflow API
# Default is True (verify SSL). Set ROBOFLOW_API_VERIFY_SSL=false to disable in local dev.
ROBOFLOW_API_VERIFY_SSL = str2bool(os.getenv("ROBOFLOW_API_VERIFY_SSL", "True"))

IGNORE_MODEL_DEPENDENCIES_WARNINGS = str2bool(
    os.getenv("IGNORE_MODEL_DEPENDENCIES_WARNINGS", "False")
)
if IGNORE_MODEL_DEPENDENCIES_WARNINGS:
    warnings.simplefilter("ignore", ModelDependencyMissing)

DISK_CACHE_CLEANUP = str2bool(os.getenv("DISK_CACHE_CLEANUP", "True"))
MEMORY_FREE_THRESHOLD = float(
    os.getenv("MEMORY_FREE_THRESHOLD", "0.0")
)  # percentage of free memory, 0 disables memory pressure detection

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

# Cache metadata lock timeout in seconds, default is 1.0
CACHE_METADATA_LOCK_TIMEOUT = float(os.getenv("CACHE_METADATA_LOCK_TIMEOUT", 1.0))
MODEL_LOCK_ACQUIRE_TIMEOUT = float(os.getenv("MODEL_LOCK_ACQUIRE_TIMEOUT", "60.0"))
HOT_MODELS_QUEUE_LOCK_ACQUIRE_TIMEOUT = float(
    os.getenv("HOT_MODELS_QUEUE_LOCK_ACQUIRE_TIMEOUT", "5.0")
)

# RFDETR input resolution limit for models loaded through onnx runtime
# 1280 -> ~3.5G
# 1440 -> ~5G
# 1600 -> ~10G
# 2048 -> ~22G
RFDETR_ONNX_MAX_RESOLUTION = int(os.getenv("RFDETR_ONNX_MAX_RESOLUTION", "1600"))

# Confidence lower bound to prevent OOM when inferring on instance segmentation models
CONFIDENCE_LOWER_BOUND_OOM_PREVENTION = float(
    os.getenv("CONFIDENCE_LOWER_BOUND_OOM_PREVENTION", "0.01")
)

WEBRTC_WORKER_ENABLED: bool = str2bool(os.getenv("WEBRTC_WORKER_ENABLED", "True"))

# Strip quotes from Modal WebRTC worker credentials in case users include them
_webrtc_modal_token_id = os.getenv("WEBRTC_MODAL_TOKEN_ID")
_webrtc_modal_token_secret = os.getenv("WEBRTC_MODAL_TOKEN_SECRET")

# Remove common quote characters that users might accidentally include
WEBRTC_MODAL_TOKEN_ID = (
    _webrtc_modal_token_id.strip("\"'") if _webrtc_modal_token_id else None
)
WEBRTC_MODAL_TOKEN_SECRET = (
    _webrtc_modal_token_secret.strip("\"'") if _webrtc_modal_token_secret else None
)
WEBRTC_MODAL_APP_NAME = os.getenv(
    "WEBRTC_MODAL_APP_NAME", f"inference-webrtc-{PROJECT}"
)
# seconds
WEBRTC_MODAL_RESPONSE_TIMEOUT = int(os.getenv("WEBRTC_MODAL_RESPONSE_TIMEOUT", "60"))
# seconds
WEBRTC_MODAL_WATCHDOG_TIMEMOUT = int(os.getenv("WEBRTC_MODAL_WATCHDOG_TIMEMOUT", "60"))
# seconds
WEBRTC_MODAL_FUNCTION_TIME_LIMIT = int(
    os.getenv("WEBRTC_MODAL_FUNCTION_TIME_LIMIT", "3600")
)
# seconds
WEBRTC_MODAL_FUNCTION_MAX_TIME_LIMIT = int(
    os.getenv("WEBRTC_MODAL_FUNCTION_MAX_TIME_LIMIT", "604800")  # 7 days
)
# seconds
WEBRTC_MODAL_SHUTDOWN_RESERVE = int(os.getenv("WEBRTC_MODAL_SHUTDOWN_RESERVE", "1"))
WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT = str2bool(
    os.getenv("WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT", "True")
)
# not set (to use CPU), any, or https://modal.com/docs/guide/gpu#specifying-gpu-type
WEBRTC_MODAL_FUNCTION_GPU = os.getenv("WEBRTC_MODAL_FUNCTION_GPU")
try:
    WEBRTC_MODAL_FUNCTION_MAX_INPUTS = int(
        os.getenv("WEBRTC_MODAL_FUNCTION_MAX_INPUTS")
    )
except (ValueError, TypeError):
    WEBRTC_MODAL_FUNCTION_MAX_INPUTS = None
WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS = int(
    os.getenv("WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS", "0")
)
WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS = int(
    os.getenv("WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS", "0")
)
# seconds
WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW = int(
    os.getenv("WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW", "15")
)
WEBRTC_MODAL_IMAGE_NAME = os.getenv(
    "WEBRTC_MODAL_IMAGE_NAME", "roboflow/roboflow-inference-server-gpu"
)
WEBRTC_MODAL_IMAGE_TAG = os.getenv("WEBRTC_MODAL_IMAGE_TAG")
WEBRTC_MODAL_ROBOFLOW_INTERNAL_SERVICE_NAME = os.getenv(
    "WEBRTC_MODAL_ROBOFLOW_INTERNAL_SERVICE_NAME", "webrtc-modal"
)
WEBRTC_MODAL_RTSP_PLACEHOLDER = os.getenv("WEBRTC_MODAL_RTSP_PLACEHOLDER")
WEBRTC_MODAL_RTSP_PLACEHOLDER_URL = os.getenv("WEBRTC_MODAL_RTSP_PLACEHOLDER_URL")
WEBRTC_MODAL_GCP_SECRET_NAME = os.getenv("WEBRTC_MODAL_GCP_SECRET_NAME")
WEBRTC_MODAL_MODELS_PRELOAD_API_KEY = os.getenv("WEBRTC_MODAL_MODELS_PRELOAD_API_KEY")
WEBRTC_MODAL_PRELOAD_MODELS = os.getenv("WEBRTC_MODAL_PRELOAD_MODELS")
WEBRTC_MODAL_PRELOAD_HF_IDS = os.getenv("WEBRTC_MODAL_PRELOAD_HF_IDS")
try:
    WEBRTC_MODAL_MIN_CPU_CORES = int(os.getenv("WEBRTC_MODAL_MIN_CPU_CORES"))
except (ValueError, TypeError):
    WEBRTC_MODAL_MIN_CPU_CORES = None
try:
    WEBRTC_MODAL_MIN_RAM_MB = int(os.getenv("WEBRTC_MODAL_MIN_RAM_MB"))
except (ValueError, TypeError):
    WEBRTC_MODAL_MIN_RAM_MB = None
WEBRTC_MODAL_PUBLIC_STUN_SERVERS = os.getenv(
    "WEBRTC_MODAL_PUBLIC_STUN_SERVERS",
    "stun:stun.l.google.com:19302,stun:stun1.l.google.com:19302,stun:stun2.l.google.com:19302,stun:stun3.l.google.com:19302,stun:stun4.l.google.com:19302",
)
WEBRTC_MODAL_USAGE_QUOTA_ENABLED = str2bool(
    os.getenv("WEBRTC_MODAL_USAGE_QUOTA_ENABLED", "False")
)
WEBRTC_DATA_CHANNEL_BUFFER_DRAINING_DELAY = float(
    os.getenv("WEBRTC_DATA_CHANNEL_BUFFER_DRAINING_DELAY", "0.1")
)
WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT = int(
    os.getenv("WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT", str(1024 * 1024))  # 1MB
)

# Maximum number of frames the server is allowed to be ahead of the last client ACK
# when ACK-based pacing is enabled on the WebRTC control datachannel.
#
# Example: if ack=1 and window=4, server may produce/send up to frame 5.
try:
    WEBRTC_DATA_CHANNEL_ACK_WINDOW = int(
        os.getenv("WEBRTC_DATA_CHANNEL_ACK_WINDOW", "20")
    )
except (ValueError, TypeError):
    WEBRTC_DATA_CHANNEL_ACK_WINDOW = 20
if WEBRTC_DATA_CHANNEL_ACK_WINDOW < 0:
    WEBRTC_DATA_CHANNEL_ACK_WINDOW = 0

HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_ENABLED = str2bool(
    os.getenv("HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_ENABLED", "True")
)
HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_WORKERS = int(
    os.getenv("HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_WORKERS", "16")
)

# Workflow block filtering configuration
# Comma-separated list of block type categories to disable (e.g., "sink,model")
WORKFLOW_DISABLED_BLOCK_TYPES = os.getenv("WORKFLOW_DISABLED_BLOCK_TYPES", "")
if WORKFLOW_DISABLED_BLOCK_TYPES:
    WORKFLOW_DISABLED_BLOCK_TYPES = [
        t.strip().lower() for t in WORKFLOW_DISABLED_BLOCK_TYPES.split(",") if t.strip()
    ]
else:
    WORKFLOW_DISABLED_BLOCK_TYPES = []

# Comma-separated list of block identifier patterns to disable
WORKFLOW_DISABLED_BLOCK_PATTERNS = os.getenv("WORKFLOW_DISABLED_BLOCK_PATTERNS", "")
if WORKFLOW_DISABLED_BLOCK_PATTERNS:
    WORKFLOW_DISABLED_BLOCK_PATTERNS = [
        p.strip().lower()
        for p in WORKFLOW_DISABLED_BLOCK_PATTERNS.split(",")
        if p.strip()
    ]
else:
    WORKFLOW_DISABLED_BLOCK_PATTERNS = []
