import contextvars
import os

from inference_sdk.utils.environment import str2bool

execution_id = contextvars.ContextVar("execution_id", default=None)

WORKFLOW_RUN_RETRIES_ENABLED = str2bool(
    os.getenv("WORKFLOW_RUN_RETRIES_ENABLED", "True")
)
EXECUTION_ID_HEADER = os.getenv("EXECUTION_ID_HEADER", "execution_id")
PROCESSING_TIME_HEADER = os.getenv("PROCESSING_TIME_HEADER", "X-Processing-Time")

# WebRTC configuration
WEBRTC_INITIAL_FRAME_TIMEOUT = float(os.getenv("WEBRTC_INITIAL_FRAME_TIMEOUT", "90.0"))
WEBRTC_VIDEO_QUEUE_MAX_SIZE = int(os.getenv("WEBRTC_VIDEO_QUEUE_MAX_SIZE", "8"))
WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT = float(
    os.getenv("WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT", "2.0")
)


class InferenceSDKDeprecationWarning(Warning):
    """Class used for warning of deprecated features in the Inference SDK"""

    pass
