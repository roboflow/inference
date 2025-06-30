import hashlib
import os.path

import pytest
from inference_exp.utils.download import download_files_to_directory


@pytest.mark.timeout(10)
def test_download_files_to_directory_small_files(empty_local_dir: str) -> None:
    # given
    some_path = os.path.join(empty_local_dir, "some.jpg")
    other_path = os.path.join(empty_local_dir, "some.jpg")
    files_specs = [
        ("some.jpg", "https://media.roboflow.com/dog.jpeg"),
        ("other.jpg", "https://media.roboflow.com/dog.jpeg"),
    ]

    # when
    download_files_to_directory(
        target_path=empty_local_dir,
        files_specs=files_specs,
    )

    # then
    assert os.path.isfile(some_path)
    assert os.path.isfile(other_path)
    assert calculate_md5(file=some_path) == "fc3f17e4aa70acf5bfc385aaa344d275"
    assert calculate_md5(file=other_path) == "fc3f17e4aa70acf5bfc385aaa344d275"


@pytest.mark.timeout(120)
@pytest.mark.slow
def test_download_files_to_directory_large_files(empty_local_dir: str) -> None:
    # given
    yolonas_path = os.path.join(empty_local_dir, "yolonas.zip")
    yolov10_path = os.path.join(empty_local_dir, "yolov10_det.zip")
    files_specs = [
        (
            "yolonas.zip",
            "https://storage.googleapis.com/roboflow-tests-assets/yolonas.zip",
        ),
        (
            "yolov10_det.zip",
            "https://storage.googleapis.com/roboflow-tests-assets/yolov10_det.zip",
        ),
    ]

    # when
    download_files_to_directory(
        target_path=empty_local_dir,
        files_specs=files_specs,
    )

    # then
    assert os.path.isfile(yolonas_path)
    assert os.path.isfile(yolov10_path)
    assert calculate_md5(file=yolonas_path) == "8d91e76f3da85ff3df8f4a5c3a420b72"
    assert calculate_md5(file=yolov10_path) == "d04b35026fb68111be4b57b2518aaace"


def calculate_md5(file: str) -> str:
    hash_object = hashlib.md5()
    with open(file, "rb") as f:
        while True:
            chunk = f.read(1024)
            if len(chunk) == 0:
                break
            hash_object.update(chunk)
    return hash_object.hexdigest()
