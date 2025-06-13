from typing import Dict, List

from inference_exp.errors import CorruptedModelPackageError
from inference_exp.utils.file_system import index_directory


def get_model_package_contents(
    model_package_dir: str,
    elements: List[str],
) -> Dict[str, str]:
    model_package_content = index_directory(path=model_package_dir)
    result = {}
    for element in elements:
        if element not in model_package_content:
            raise CorruptedModelPackageError(
                f"Model package saved under path {model_package_dir} is incomplete. Could not find element {element}. "
                f"If you attempt to run `inference` locally - inspect the contents of local directory to check for "
                f"completeness of model package download - lack of files may indicate network issues. Verification "
                f"of connectivity may be a good first step. If you prepared the model package manually - examine the "
                f"correctness of the setup. If you run on managed serving - contact support if the issue is "
                f"not ephemeral."
            )
        result[element] = model_package_content[element]
    return result
