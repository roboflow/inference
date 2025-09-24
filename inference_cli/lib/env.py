import os

CLI_LOG_LEVEL = os.getenv("CLI_LOG_LEVEL", "INFO")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
PROJECT = os.getenv("PROJECT", "roboflow-platform")
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    (
        "https://api.roboflow.com"
        if PROJECT == "roboflow-platform"
        else "https://api.roboflow.one"
    ),
)
