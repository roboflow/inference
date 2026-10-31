import numpy as np

from inference_cli.lib.enterprise.inference_compiler.errors import (
    ModelInferenceError,
    ModelLoadingError,
)
from inference_models import AutoModel


def verify_auto_model(package_dir: str) -> None:
    image = np.zeros((512, 513, 3), dtype=np.uint8)
    try:
        model = AutoModel.from_pretrained(model_id_or_path=package_dir)
    except Exception as error:
        raise ModelLoadingError("Could not load compiled model") from error
    try:
        _ = model(image)
    except Exception as error:
        raise ModelInferenceError(
            "Could not perform inference from compiled model"
        ) from error
