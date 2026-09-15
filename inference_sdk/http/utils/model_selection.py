from typing import Mapping

from inference_sdk.http.errors import InvalidParameterError

MODEL_SELECTION_HEADER = "X-Roboflow-Model-Selection"


def ensure_model_selection_applied(headers: Mapping[str, str], selectors: dict) -> None:
    if any(
        selectors.get(name) is not None
        for name in ("model_package_id", "backend", "quantization")
    ):
        if headers.get(MODEL_SELECTION_HEADER) != "applied":
            raise InvalidParameterError(
                "The server did not confirm model package selection. Upgrade the server "
                "and enable USE_INFERENCE_MODELS before requesting a specific package."
            )
