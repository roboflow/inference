import os
import warnings

from inference_sdk.config import InferenceSDKDeprecationWarning
from inference_sdk.http.client import InferenceHTTPClient
from inference_sdk.http.entities import (
    InferenceConfiguration,
    VisualisationResponseFormat,
)
from inference_sdk.utils.environment import str2bool

# Environment variable to control whether SDK warnings are disabled.
# Set to "true" to disable all SDK-specific warnings, "false" to enable them.
# Default is "false" (warnings enabled).
INFERENCE_WARNINGS_DISABLED = str2bool(
    os.getenv("INFERENCE_WARNINGS_DISABLED", "False")
)

if INFERENCE_WARNINGS_DISABLED:
    warnings.simplefilter("ignore", InferenceSDKDeprecationWarning)

try:
    from inference_sdk.version import __version__
except ImportError:
    __version__ = "development"
