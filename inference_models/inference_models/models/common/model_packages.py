import os.path
from typing import Dict, List

from inference_models.errors import CorruptedModelPackageError


def get_model_package_contents(
    model_package_dir: str,
    elements: List[str],
) -> Dict[str, str]:
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
                help_url="https://todo",
            )
        result[element] = element_path
    return result
