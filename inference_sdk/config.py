import contextvars
import os

from inference_sdk.utils.environment import str2bool

execution_id = contextvars.ContextVar("execution_id", default=None)

WORKFLOW_RUN_RETRIES_ENABLED = str2bool(
    os.getenv("WORKFLOW_RUN_RETRIES_ENABLED", "True")
)
EXECUTION_ID_HEADER = os.getenv("EXECUTION_ID_HEADER", "execution_id")
PROCESSING_TIME_HEADER = os.getenv("PROCESSING_TIME_HEADER", "X-Processing-Time")


ALL_ROBOFLOW_API_URLS = {
    "https://detect.roboflow.com",
    "https://outline.roboflow.com",
    "https://classify.roboflow.com",
    "https://infer.roboflow.com",
    "https://serverless.roboflow.com",
    "https://serverless.roboflow.one",
}


# WebRTC configuration
WEBRTC_INITIAL_FRAME_TIMEOUT = float(os.getenv("WEBRTC_INITIAL_FRAME_TIMEOUT", "90.0"))
WEBRTC_VIDEO_QUEUE_MAX_SIZE = int(os.getenv("WEBRTC_VIDEO_QUEUE_MAX_SIZE", "8"))
WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT = float(
    os.getenv("WEBRTC_EVENT_LOOP_SHUTDOWN_TIMEOUT", "2.0")
)

# Video file upload via datachannel
WEBRTC_VIDEO_UPLOAD_CHUNK_SIZE = int(
    os.getenv("WEBRTC_VIDEO_UPLOAD_CHUNK_SIZE", "49152")
)  # 48KB - safe for WebRTC
WEBRTC_VIDEO_UPLOAD_BUFFER_LIMIT = int(
    os.getenv("WEBRTC_VIDEO_UPLOAD_BUFFER_LIMIT", "262144")
)  # 256KB max buffered before backpressure

# Roboflow API base URL for TURN config and other services
RF_API_BASE_URL = os.getenv("RF_API_BASE_URL", "https://api.roboflow.com")


class InferenceSDKDeprecationWarning(Warning):
    """Class used for warning of deprecated features in the Inference SDK"""

    pass
