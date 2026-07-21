import os

from inference_models.errors import ModelRuntimeError
from inference_models.utils.imports import import_class_from_file


def load_runtime_from_package(
    model_name_or_path: str,
    runtime_module_file: str,
    runtime_class_name: str,
) -> type:
    """Load a generator runtime class shipped inside a model package.

    The Cosmos 3 generator towers are not loadable through plain transformers:
    their denoising loops come from NVIDIA's cosmos stack. Instead of making
    that stack an `inference_models` dependency, the model package ships a
    self-contained runtime module which this helper imports (same mechanism as
    moondream2's package-local code).
    """
    runtime_path = os.path.join(model_name_or_path, runtime_module_file)
    if not os.path.exists(runtime_path):
        raise ModelRuntimeError(
            message=f"Model package at {model_name_or_path} does not contain the "
            f"expected runtime module `{runtime_module_file}`. The package may be "
            "corrupted or predate generator support.",
            help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
        )
    return import_class_from_file(
        file_path=runtime_path,
        class_name=runtime_class_name,
    )
