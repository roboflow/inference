from typing import Mapping

from inference_sdk.http.errors import ModelSelectionNotConfirmedError

MODEL_SELECTION_HEADER = "X-Roboflow-Model-Selection"


def ensure_model_selection_applied(headers: Mapping[str, str], selectors: dict) -> None:
    if any(
        selectors.get(name) is not None
        for name in ("model_package_id", "backend", "quantization")
    ):
        if headers.get(MODEL_SELECTION_HEADER) != "applied":
            raise ModelSelectionNotConfirmedError(
                "The server did not confirm model package selection. Upgrade the server "
                "and enable USE_INFERENCE_MODELS, or check whether a proxy removed the "
                "X-Roboflow-Model-Selection response header."
            )
