import contextvars
import os

from inference_sdk.utils.environment import str2bool

execution_id = contextvars.ContextVar("execution_id", default=None)

WORKFLOW_RUN_RETRIES_ENABLED = str2bool(
    os.getenv("WORKFLOW_RUN_RETRIES_ENABLED", "True")
)
EXECUTION_ID_HEADER = os.getenv("EXECUTION_ID_HEADER", "execution_id")
PROCESSING_TIME_HEADER = os.getenv("PROCESSING_TIME_HEADER", "X-Processing-Time")
INFERENCE_INTERNAL_USERNAME = os.getenv("INFERENCE_INTERNAL_USERNAME", None)
INFERENCE_INTERNAL_PASSWORD = os.getenv("INFERENCE_INTERNAL_PASSWORD", None)


class InferenceSDKDeprecationWarning(Warning):
    """Class used for warning of deprecated features in the Inference SDK"""

    pass
