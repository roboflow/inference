import os
import warnings
from typing import Optional

import torch

from inference_models.errors import InvalidEnvVariable
from inference_models.utils.environment import (
    get_boolean_from_env,
    get_comma_separated_list_of_integers_from_env,
    get_float_from_env,
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
_legacy_license_server = os.getenv("LICENSE_SERVER")
SECURE_GATEWAY = os.getenv("SECURE_GATEWAY") or _legacy_license_server or None
if _legacy_license_server and not os.getenv("SECURE_GATEWAY"):
    warnings.warn(
        "`LICENSE_SERVER` env variable is deprecated, use `SECURE_GATEWAY` instead. "
        "`LICENSE_SERVER` will be removed end of Q3 2026.",
        DeprecationWarning,
        stacklevel=1,
    )
RUNNING_ON_JETSON = os.getenv("RUNNING_ON_JETSON")
L4T_VERSION = os.getenv("L4T_VERSION")
# Fall back to the inference server's MODEL_CACHE_DIR so that both cache
# layouts live on the same (typically mounted) volume without relying on
# import order between `inference` and `inference_models`.
INFERENCE_HOME = (
    os.getenv("INFERENCE_HOME") or os.getenv("MODEL_CACHE_DIR") or "/tmp/cache"
)
# Offline mode - disables all outbound network requests. Models are loaded
# exclusively from local cache. Designed for air-gapped deployments.
OFFLINE_MODE = get_boolean_from_env(variable_name="OFFLINE_MODE", default=False)
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
SAM3_IMAGE_SIZE = get_integer_from_env(variable_name="SAM3_IMAGE_SIZE", default=1008)
INFERENCE_MODELS_SAM3_MASK_PROCESSING_CHUNK_SIZE = get_integer_from_env(
    variable_name="INFERENCE_MODELS_SAM3_MASK_PROCESSING_CHUNK_SIZE", default=8
)
if INFERENCE_MODELS_SAM3_MASK_PROCESSING_CHUNK_SIZE < 1:
    raise InvalidEnvVariable(
        message=(
            "Expected environment variable `INFERENCE_MODELS_SAM3_MASK_PROCESSING_CHUNK_SIZE` "
            f"to be >= 1 but got '{INFERENCE_MODELS_SAM3_MASK_PROCESSING_CHUNK_SIZE}'"
        ),
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
    )
CHUNK_DOWNLOAD_CONNECT_TIMEOUT = get_float_from_env(
    variable_name="CHUNK_DOWNLOAD_CONNECT_TIMEOUT",
    default=30.0,
)
CHUNK_DOWNLOAD_READ_TIMEOUT = get_float_from_env(
    variable_name="CHUNK_DOWNLOAD_READ_TIMEOUT",
    default=60.0,
)
CHUNK_DOWNLOAD_MAX_ATTEMPTS = get_integer_from_env(
    variable_name="CHUNK_DOWNLOAD_MAX_ATTEMPTS",
    default=60,
)
FILE_LOCK_ACQUIRE_TIMEOUT = get_integer_from_env(
    variable_name="INFERENCE_MODELS_FILE_LOCK_ACQUIRE_TIMEOUT", default=20
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

# General model parameters defaults

INFERENCE_MODELS_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_DEFAULT_CONFIDENCE",
    default=0.4,
)
INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD",
    default=0.3,
)
INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS",
    default=300,
)
INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS",
    default=False,
)
INFERENCE_MODELS_DEFAULT_MAX_NEW_TOKENS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_DEFAULT_MAX_NEW_TOKENS",
    default=4096,
)
INFERENCE_MODELS_DEFAULT_NUM_BEAMS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_DEFAULT_NUM_BEAMS",
    default=3,
)
INFERENCE_MODELS_DEFAULT_DO_SAMPLE = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_DEFAULT_DO_SAMPLE",
    default=False,
)
INFERENCE_MODELS_DEFAULT_SKIP_SPECIAL_TOKENS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_DEFAULT_SKIP_SPECIAL_TOKENS",
    default=False,
)


# Model-specific parameters defaults
INFERENCE_MODELS_DEEP_LAB_V3_PLUS_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_DEEP_LAB_V3_PLUS_DEFAULT_CONFIDENCE",
    default=0.5,
)
INFERENCE_MODELS_DINOV3_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_DINOV3_DEFAULT_CONFIDENCE",
    default=0.5,
)
INFERENCE_MODELS_EASYOCR_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_EASYOCR_DEFAULT_CONFIDENCE",
    default=0.3,
)
INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS",
    default=INFERENCE_MODELS_DEFAULT_MAX_NEW_TOKENS,
)
INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS",
    default=INFERENCE_MODELS_DEFAULT_NUM_BEAMS,
)
INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE",
    default=INFERENCE_MODELS_DEFAULT_DO_SAMPLE,
)
INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_BOX_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_BOX_CONFIDENCE",
    default=0.5,
)
INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_MAX_DETECTIONS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_MAX_DETECTIONS",
    default=INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS,
)
INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_IOU_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_IOU_THRESHOLD",
    default=0.5,
)
INFERENCE_MODELS_MOONDREAM2_DEFAULT_MAX_NEW_TOKENS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_MOONDREAM2_DEFAULT_MAX_NEW_TOKENS",
    default=700,
)
INFERENCE_MODELS_OWLV2_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_OWLV2_DEFAULT_CONFIDENCE",
    default=0.99,
)
INFERENCE_MODELS_OWLV2_DEFAULT_IOU_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_OWLV2_DEFAULT_IOU_THRESHOLD",
    default=INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD,
)
INFERENCE_MODELS_OWLV2_DEFAULT_MAX_DETECTIONS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_OWLV2_DEFAULT_MAX_DETECTIONS",
    default=INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS,
)
INFERENCE_MODELS_OWLV2_DEFAULT_CLASS_AGNOSTIC_NMS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_OWLV2_DEFAULT_CLASS_AGNOSTIC_NMS",
    default=INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS,
)
INFERENCE_MODELS_PALIGEMMA_DEFAULT_MAX_NEW_TOKENS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_PALIGEMMA_DEFAULT_MAX_NEW_TOKENS",
    default=400,
)
INFERENCE_MODELS_PALIGEMMA_DEFAULT_DO_SAMPLE = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_PALIGEMMA_DEFAULT_DO_SAMPLE",
    default=INFERENCE_MODELS_DEFAULT_DO_SAMPLE,
)
INFERENCE_MODELS_PALIGEMMA_DEFAULT_SKIP_SPECIAL_TOKENS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_PALIGEMMA_DEFAULT_SKIP_SPECIAL_TOKENS",
    default=True,
)
INFERENCE_MODELS_QWEN3_VL_DEFAULT_MAX_NEW_TOKENS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_QWEN3_VL_DEFAULT_MAX_NEW_TOKENS",
    default=512,
)
INFERENCE_MODELS_QWEN3_VL_DEFAULT_DO_SAMPLE = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_QWEN3_VL_DEFAULT_DO_SAMPLE",
    default=INFERENCE_MODELS_DEFAULT_DO_SAMPLE,
)
INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS",
    default=512,
)
INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE",
    default=INFERENCE_MODELS_DEFAULT_DO_SAMPLE,
)
INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS",
    default=8192,
)
INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_GLM_OCR_DEFAULT_DO_SAMPLE",
    default=False,
)
INFERENCE_MODELS_QWEN3_5_DEFAULT_MAX_NEW_TOKENS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_QWEN3_5_DEFAULT_MAX_NEW_TOKENS",
    default=512,
)
INFERENCE_MODELS_QWEN3_5_DEFAULT_DO_SAMPLE = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_QWEN3_5_DEFAULT_DO_SAMPLE",
    default=INFERENCE_MODELS_DEFAULT_DO_SAMPLE,
)
INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS",
    default=512,
)
INFERENCE_MODELS_QWEN25_VL_DEFAULT_DO_SAMPLE = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_QWEN25_VL_DEFAULT_DO_SAMPLE",
    default=INFERENCE_MODELS_DEFAULT_DO_SAMPLE,
)
INFERENCE_MODELS_QWEN25_VL_DEFAULT_SKIP_SPECIAL_TOKENS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_QWEN25_VL_DEFAULT_SKIP_SPECIAL_TOKENS",
    default=True,
)
INFERENCE_MODELS_GEMMA4_DEFAULT_MAX_NEW_TOKENS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_GEMMA4_DEFAULT_MAX_NEW_TOKENS",
    default=512,
)
INFERENCE_MODELS_GEMMA4_DEFAULT_DO_SAMPLE = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_GEMMA4_DEFAULT_DO_SAMPLE",
    default=INFERENCE_MODELS_DEFAULT_DO_SAMPLE,
)
INFERENCE_MODELS_GEMMA4_DEFAULT_ENABLE_THINKING = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_GEMMA4_DEFAULT_ENABLE_THINKING",
    default=False,
)
INFERENCE_MODELS_GEMMA4_DEFAULT_SKIP_SPECIAL_TOKENS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_GEMMA4_DEFAULT_SKIP_SPECIAL_TOKENS",
    default=True,
)
# Official Gemma 4 sampling recommendations when ``do_sample`` is True (HF model cards).
INFERENCE_MODELS_GEMMA4_DEFAULT_TEMPERATURE = get_float_from_env(
    variable_name="INFERENCE_MODELS_GEMMA4_DEFAULT_TEMPERATURE",
    default=1.0,
)
INFERENCE_MODELS_GEMMA4_DEFAULT_TOP_P = get_float_from_env(
    variable_name="INFERENCE_MODELS_GEMMA4_DEFAULT_TOP_P",
    default=0.95,
)
INFERENCE_MODELS_GEMMA4_DEFAULT_TOP_K = get_integer_from_env(
    variable_name="INFERENCE_MODELS_GEMMA4_DEFAULT_TOP_K",
    default=64,
)
INFERENCE_MODELS_RESNET_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_RESNET_DEFAULT_CONFIDENCE",
    default=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
)
INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE",
    default=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
)
DEFAULT_INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED = False
INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED",
    default=DEFAULT_INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED,
)
INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_MAX_PIXELS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_MAX_PIXELS",
    default=4096 * 2160,
)
INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_MAX_RUNS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_MAX_RUNS",
    default=32768,
)
INFERENCE_MODELS_RFDETR_DEFAULT_KEY_POINTS_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_DETR_DEFAULT_KEY_POINTS_THRESHOLD",
    default=0.3,
)
DEFAULT_INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED = False
INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED",
    default=DEFAULT_INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED,
)
RFDETR_PIPELINE_DEPTH_ENV_NAME = "RFDETR_PIPELINE_DEPTH"
DEFAULT_RFDETR_PIPELINE_DEPTH = 1
MIN_RFDETR_PIPELINE_DEPTH = 1
MAX_RFDETR_PIPELINE_DEPTH = 2


def parse_rfdetr_pipeline_depth(value: Optional[str]) -> int:
    """Parse and validate the RF-DETR streaming pipeline depth.

    Depth is the number of in-flight CPU/GPU stages the stream adapter may keep
    alive. ``1`` preserves the original synchronous behavior; values greater
    than one enable delayed response finalization. Values above the supported
    maximum are normalized to ``2``. Zero, negative, and non-integer values are
    rejected instead of being silently clamped.
    """
    if value is None:
        return DEFAULT_RFDETR_PIPELINE_DEPTH
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise InvalidEnvVariable(
            message=(
                f"Expected environment variable `{RFDETR_PIPELINE_DEPTH_ENV_NAME}` "
                f"to be an integer but got '{value}'"
            ),
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
        )
    if parsed < MIN_RFDETR_PIPELINE_DEPTH:
        raise InvalidEnvVariable(
            message=(
                f"Expected environment variable `{RFDETR_PIPELINE_DEPTH_ENV_NAME}` "
                f"to be >= {MIN_RFDETR_PIPELINE_DEPTH} but got '{value}'"
            ),
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#invalidenvvariable",
        )
    return min(parsed, MAX_RFDETR_PIPELINE_DEPTH)


def get_rfdetr_pipeline_depth() -> int:
    """Read and validate ``RFDETR_PIPELINE_DEPTH`` from the environment."""
    return parse_rfdetr_pipeline_depth(os.getenv(RFDETR_PIPELINE_DEPTH_ENV_NAME))


RFDETR_PIPELINE_DEPTH = get_rfdetr_pipeline_depth()
INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_CONFIDENCE",
    default=0.99,
)
INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_IOU_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_IOU_THRESHOLD",
    default=0.3,
)
INFERENCE_MODELS_ROBOFLOW_INSTANT_MAX_DETECTIONS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_ROBOFLOW_INSTANT_MAX_DETECTIONS",
    default=INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS,
)
INFERENCE_MODELS_SMOL_VLM_DEFAULT_MAX_NEW_TOKENS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_SMOL_VLM_DEFAULT_MAX_NEW_TOKENS",
    default=400,
)
INFERENCE_MODELS_GEMMA4_DEFAULT_IMAGE_PROMPT = os.getenv(
    "INFERENCE_MODELS_GEMMA4_DEFAULT_IMAGE_PROMPT",
    "Describe what you see in this image.",
)
INFERENCE_MODELS_GEMMA4_DEFAULT_SYSTEM_PROMPT = os.getenv(
    "INFERENCE_MODELS_GEMMA4_DEFAULT_SYSTEM_PROMPT",
    "You are Gemma 4, a helpful multimodal assistant. Answer clearly and accurately.",
)
INFERENCE_MODELS_SMOL_VLM_DEFAULT_DO_SAMPLE = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_SMOL_VLM_DEFAULT_DO_SAMPLE",
    default=INFERENCE_MODELS_DEFAULT_DO_SAMPLE,
)
INFERENCE_MODELS_SMOL_VLM_DEFAULT_SKIP_SPECIAL_TOKENS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_SMOL_VLM_DEFAULT_SKIP_SPECIAL_TOKENS",
    default=True,
)
INFERENCE_MODELS_VIT_CLASSIFIER_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_VIT_CLASSIFIER_DEFAULT_CONFIDENCE",
    default=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
)
INFERENCE_MODELS_YOLACT_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLACT_DEFAULT_CONFIDENCE",
    default=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
)
INFERENCE_MODELS_YOLACT_DEFAULT_IOU_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLACT_DEFAULT_IOU_THRESHOLD",
    default=INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD,
)
INFERENCE_MODELS_YOLACT_DEFAULT_MAX_DETECTIONS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_YOLACT_DEFAULT_MAX_DETECTIONS",
    default=INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS,
)
INFERENCE_MODELS_YOLACT_DEFAULT_CLASS_AGNOSTIC_NMS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_YOLACT_DEFAULT_CLASS_AGNOSTIC_NMS",
    default=INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS,
)
INFERENCE_MODELS_YOLONAS_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLONAS_DEFAULT_CONFIDENCE",
    default=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
)
INFERENCE_MODELS_YOLONAS_DEFAULT_IOU_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLONAS_DEFAULT_IOU_THRESHOLD",
    default=INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD,
)
INFERENCE_MODELS_YOLONAS_DEFAULT_MAX_DETECTIONS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_YOLONAS_DEFAULT_MAX_DETECTIONS",
    default=INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS,
)
INFERENCE_MODELS_YOLONAS_DEFAULT_CLASS_AGNOSTIC_NMS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_YOLONAS_DEFAULT_CLASS_AGNOSTIC_NMS",
    default=INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS,
)
INFERENCE_MODELS_YOLOV5_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLOV5_DEFAULT_CONFIDENCE",
    default=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
)
INFERENCE_MODELS_YOLOV5_DEFAULT_IOU_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLOV5_DEFAULT_IOU_THRESHOLD",
    default=INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD,
)
INFERENCE_MODELS_YOLOV5_DEFAULT_MAX_DETECTIONS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_YOLOV5_DEFAULT_MAX_DETECTIONS",
    default=INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS,
)
INFERENCE_MODELS_YOLOV5_DEFAULT_CLASS_AGNOSTIC_NMS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_YOLOV5_DEFAULT_CLASS_AGNOSTIC_NMS",
    default=INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS,
)
INFERENCE_MODELS_YOLOV7_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLOV7_DEFAULT_CONFIDENCE",
    default=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
)
INFERENCE_MODELS_YOLOV7_DEFAULT_IOU_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLOV7_DEFAULT_IOU_THRESHOLD",
    default=INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD,
)
INFERENCE_MODELS_YOLOV7_DEFAULT_MAX_DETECTIONS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_YOLOV7_DEFAULT_MAX_DETECTIONS",
    default=INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS,
)
INFERENCE_MODELS_YOLOV7_DEFAULT_CLASS_AGNOSTIC_NMS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_YOLOV7_DEFAULT_CLASS_AGNOSTIC_NMS",
    default=INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS,
)
INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE",
    default=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
)
INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD",
    default=INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD,
)
INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS",
    default=INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS,
)
INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS",
    default=INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS,
)
INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_SMOOTHING_ENABLED = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_SMOOTHING_ENABLED",
    default=True,
)
INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_BINARIZATION_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_BINARIZATION_THRESHOLD",
    default=(
        0.5
        if INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_SMOOTHING_ENABLED
        else 0.0
    ),
)
INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_KEY_POINTS_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_KEY_POINTS_THRESHOLD",
    default=0.0,
)
INFERENCE_MODELS_YOLOV10_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLOV10_DEFAULT_CONFIDENCE",
    default=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
)
INFERENCE_MODELS_YOLOV10_DEFAULT_MAX_DETECTIONS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_YOLOV10_DEFAULT_MAX_DETECTIONS",
    default=INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS,
)
INFERENCE_MODELS_YOLO26_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLO26_DEFAULT_CONFIDENCE",
    default=0.25,
)
INFERENCE_MODELS_YOLO26_DEFAULT_KEY_POINTS_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLO26_DEFAULT_KEY_POINTS_THRESHOLD",
    default=0.3,
)
INFERENCE_MODELS_YOLOLITE_DEFAULT_CONFIDENCE = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLOLITE_DEFAULT_CONFIDENCE",
    default=0.25,
)
INFERENCE_MODELS_YOLOLITE_DEFAULT_IOU_THRESHOLD = get_float_from_env(
    variable_name="INFERENCE_MODELS_YOLOLITE_DEFAULT_IOU_THRESHOLD",
    default=INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD,
)
INFERENCE_MODELS_YOLOLITE_DEFAULT_MAX_DETECTIONS = get_integer_from_env(
    variable_name="INFERENCE_MODELS_YOLOLITE_DEFAULT_MAX_DETECTIONS",
    default=INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS,
)
INFERENCE_MODELS_YOLOLITE_DEFAULT_CLASS_AGNOSTIC_NMS = get_boolean_from_env(
    variable_name="INFERENCE_MODELS_YOLOLITE_DEFAULT_CLASS_AGNOSTIC_NMS",
    default=INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS,
)

ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND_ENV_NAME = (
    "ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND"
)
DEFAULT_ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND = False
