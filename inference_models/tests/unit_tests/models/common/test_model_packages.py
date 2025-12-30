import os.path

import pytest

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.model_packages import get_model_package_contents


def test_get_model_package_contents_when_all_enlisted_content_is_present(
    example_package_dir: str,
) -> None:
    # when
    result = get_model_package_contents(
        model_package_dir=example_package_dir, elements=["file_1.txt", "file_2.txt"]
    )

    # then
    assert result == {
        "file_1.txt": os.path.join(example_package_dir, "file_1.txt"),
        "file_2.txt": os.path.join(example_package_dir, "file_2.txt"),
    }


def test_get_model_package_contents_when_not_all_enlisted_content_is_present(
    example_package_dir: str,
) -> None:
    # when
    with pytest.raises(CorruptedModelPackageError):
        _ = get_model_package_contents(
            model_package_dir=example_package_dir,
            elements=["file_1.txt", "file_2.txt", "non_existing.txt"],
        )
