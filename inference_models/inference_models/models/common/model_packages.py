import os.path
from typing import Dict, List

from inference_models.errors import CorruptedModelPackageError


def get_model_package_contents(
    model_package_dir: str,
    elements: List[str],
) -> Dict[str, str]:
    """Get absolute paths to files within a model package directory.

    This function is primarily used when implementing custom models that need to
    load files from a model package. It validates that all required files exist
    and returns their absolute paths.

    Args:
        model_package_dir: Absolute path to the model package directory.
        elements: List of file names (relative to package directory) to retrieve.

    Returns:
        Dictionary mapping each element name to its absolute path.

    Raises:
        CorruptedModelPackageError: If any of the requested elements don't exist
            in the package directory.

    Examples:
        Load model weights and config from a package:

        >>> from inference_models.developer_tools import get_model_package_contents
        >>>
        >>> package_contents = get_model_package_contents(
        ...     model_package_dir="/path/to/model/package",
        ...     elements=["weights.pt", "config.json", "class_names.txt"]
        ... )
        >>>
        >>> # Access file paths
        >>> weights_path = package_contents["weights.pt"]
        >>> config_path = package_contents["config.json"]
        >>> class_names_path = package_contents["class_names.txt"]

        Use in custom model implementation:

        >>> from inference_models import ObjectDetectionModel
        >>> from inference_models.developer_tools import get_model_package_contents
        >>> import torch
        >>>
        >>> class MyDetector(ObjectDetectionModel):
        ...     @classmethod
        ...     def from_pretrained(cls, model_name_or_path: str, **kwargs):
        ...         # Get required files from package
        ...         contents = get_model_package_contents(
        ...             model_package_dir=model_name_or_path,
        ...             elements=["model.onnx", "config.json"]
        ...         )
        ...
        ...         # Load model using the paths
        ...         model = load_onnx_model(contents["model.onnx"])
        ...         config = load_config(contents["config.json"])
        ...
        ...         return cls(model=model, config=config)

    See Also:
        - Custom model implementation guide: https://inference.roboflow.com/inference_models/how-to/local-packages/
    """
    result = {}
    for element in elements:
        element_path = os.path.join(model_package_dir, element)
        if not os.path.exists(element_path):
            raise CorruptedModelPackageError(
                message=f"Model package is incomplete. Could not find element {element}. "
                f"If you attempt to run `inference-models` locally - inspect the contents of local directory to check for "
                f"completeness of model package download - lack of files may indicate network issues. Verification "
                f"of connectivity may be a good first step. If you prepared the model package manually - examine the "
                f"correctness of the setup. If you run on managed serving - contact support if the issue is "
                f"not ephemeral.",
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        result[element] = element_path
    return result
