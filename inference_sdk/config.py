import contextvars
import os

from inference_sdk.utils.environment import str2bool

execution_id = contextvars.ContextVar("execution_id", default=None)

WORKFLOW_RUN_RETRIES_ENABLED = str2bool(
    os.getenv("WORKFLOW_RUN_RETRIES_ENABLED", "True")
)
EXECUTION_ID_HEADER = os.getenv("EXECUTION_ID_HEADER", "execution_id")


class InferenceSDKDeprecationWarning(Warning):
    """Class used for warning of deprecated features in the Inference SDK"""

    pass
