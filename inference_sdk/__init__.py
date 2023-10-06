from inference_sdk.http.client import InferenceHTTPClient
from inference_sdk.http.entities import (
    InferenceConfiguration,
    VisualisationResponseFormat,
)

try:
    from inference_sdk.version import __version__
except ImportError:
    __version__ = "development"
