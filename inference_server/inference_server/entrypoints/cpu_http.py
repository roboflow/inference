from inference_server.legacy_env import apply_legacy_env

apply_legacy_env()

from inference_server.app import app  # noqa: E402,F401
