"""Run Workflows on live video with Roboflow Cloud Streaming.

from inference_sdk.cloud_streaming import CloudStreamingClient, WebcamSource

client = CloudStreamingClient(api_key="...")
with client.start_stream(WebcamSource(), workflow="my-workflow") as stream:
    for result in stream.results():
        print(result.frame_id, result.outputs)
"""

from inference_sdk.cloud_streaming.api import CloudStreamingError
from inference_sdk.cloud_streaming.client import CloudStreamingClient
from inference_sdk.cloud_streaming.results import ResultGap, StreamResult
from inference_sdk.cloud_streaming.sources import (
    LocalSource,
    ManualSource,
    RTSPSource,
    VideoFileSource,
    WebcamSource,
)
from inference_sdk.cloud_streaming.stream import CloudStream

__all__ = [
    "CloudStream",
    "CloudStreamingClient",
    "CloudStreamingError",
    "LocalSource",
    "ManualSource",
    "RTSPSource",
    "ResultGap",
    "StreamResult",
    "VideoFileSource",
    "WebcamSource",
]
