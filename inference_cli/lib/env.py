import os

from inference_sdk.regions import get_roboflow_region, resolve_roboflow_service_url

CLI_LOG_LEVEL = os.getenv("CLI_LOG_LEVEL", "INFO")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
PROJECT = os.getenv("PROJECT", "roboflow-platform")
ROBOFLOW_REGION = get_roboflow_region()
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    resolve_roboflow_service_url("api", region=ROBOFLOW_REGION, project=PROJECT),
)
