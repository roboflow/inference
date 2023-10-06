from inference_client.http.client import InferenceHTTPClient
from inference_client.http.entities import (
    InferenceConfiguration,
    VisualisationResponseFormat,
)

try:
    from inference_client.version import __version__
except ImportError:
    __version__ = "development"
