import os
import sys

CLI_LOG_LEVEL = os.getenv("CLI_LOG_LEVEL", "INFO")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
PROJECT = os.getenv("PROJECT", "roboflow-platform")

ROBOFLOW_REGION = os.getenv("ROBOFLOW_REGION", "us").strip().lower()
if ROBOFLOW_REGION not in {"us", "eu"}:
    print(
        f"Warning: unknown Roboflow region {ROBOFLOW_REGION!r}; falling back to 'us'.",
        file=sys.stderr,
    )
    ROBOFLOW_REGION = "us"

if ROBOFLOW_REGION == "eu":
    _DEFAULT_API_BASE_URL = "https://api.roboflow.eu"
elif PROJECT == "roboflow-platform":
    _DEFAULT_API_BASE_URL = "https://api.roboflow.com"
else:
    _DEFAULT_API_BASE_URL = "https://api.roboflow.one"

API_BASE_URL = os.getenv("API_BASE_URL", _DEFAULT_API_BASE_URL)
