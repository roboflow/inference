import hashlib
import os.path

import pytest
from inference_exp.errors import FileHashSumMissmatch, UntrustedFileError
from inference_exp.utils.download import download_files_to_directory, get_content_length


@pytest.mark.timeout(10)
@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_download_files_to_directory_small_files(empty_local_dir: str) -> None:
    # given
    some_path = os.path.join(empty_local_dir, "some.jpg")
    other_path = os.path.join(empty_local_dir, "other.jpg")
    files_specs = [
        (
            "some.jpg",
            "https://media.roboflow.com/dog.jpeg",
            "fc3f17e4aa70acf5bfc385aaa344d275",
        ),
        (
            "other.jpg",
            "https://media.roboflow.com/dog.jpeg",
            "fc3f17e4aa70acf5bfc385aaa344d275",
        ),
    ]

    # when
    result = download_files_to_directory(
        target_dir=empty_local_dir,
        files_specs=files_specs,
    )

    # then
    assert os.path.isfile(some_path)
    assert os.path.isfile(other_path)
    assert calculate_md5(file=some_path) == "fc3f17e4aa70acf5bfc385aaa344d275"
    assert calculate_md5(file=other_path) == "fc3f17e4aa70acf5bfc385aaa344d275"


@pytest.mark.timeout(10)
@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_download_files_to_directory_small_files_with_nested_file_handles_and_event_handlers(
    empty_local_dir: str,
) -> None:
    # given
    some_path = os.path.join(empty_local_dir, "nested_1", "some.jpg")
    other_path = os.path.join(empty_local_dir, "other.jpg")
    files_specs = [
        (
            "nested_1/some.jpg",
            "https://media.roboflow.com/dog.jpeg",
            "fc3f17e4aa70acf5bfc385aaa344d275",
        ),
        (
            "other.jpg",
            "https://media.roboflow.com/dog.jpeg",
            "fc3f17e4aa70acf5bfc385aaa344d275",
        ),
    ]
    file_renames = []
    file_creations = []

    # when
    result = download_files_to_directory(
        target_dir=empty_local_dir,
        files_specs=files_specs,
        on_file_created=lambda e: file_creations.append(e),
        on_file_renamed=lambda e, f: file_renames.append((e, f)),
    )

    # then
    assert result["nested_1/some.jpg"] == some_path
    assert result["other.jpg"] == other_path
    assert len(file_renames) == 2
    assert len(file_creations) == 2
    file_creations = sorted(file_creations)
    file_renames = sorted(file_renames, key=lambda e: e[1])
    assert os.path.basename(file_creations[0]).startswith("some.jpg.")
    assert os.path.basename(file_creations[1]).startswith("other.jpg.")
    assert os.path.basename(file_renames[0][0]).startswith("some.jpg.")
    assert os.path.basename(file_renames[1][0]).startswith("other.jpg.")
    assert file_renames[0][1] == some_path
    assert file_renames[1][1] == other_path
    assert os.path.isfile(some_path)
    assert os.path.isfile(other_path)
    assert calculate_md5(file=some_path) == "fc3f17e4aa70acf5bfc385aaa344d275"
    assert calculate_md5(file=other_path) == "fc3f17e4aa70acf5bfc385aaa344d275"


@pytest.mark.timeout(10)
@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_download_files_to_directory_when_invalid_hash_sum_provided(
    empty_local_dir: str,
) -> None:
    # given
    files_specs = [
        (
            "some.jpg",
            "https://media.roboflow.com/dog.jpeg",
            "fc3f17e4aa70acf5bfc385aaa344d275",
        ),
        ("other.jpg", "https://media.roboflow.com/dog.jpeg", "invalid"),
    ]

    # when
    with pytest.raises(FileHashSumMissmatch):
        download_files_to_directory(
            target_dir=empty_local_dir,
            files_specs=files_specs,
        )


@pytest.mark.timeout(10)
@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_download_files_to_directory_when_invalid_hash_sum_provided_but_verification_disabled(
    empty_local_dir: str,
) -> None:
    # given
    some_path = os.path.join(empty_local_dir, "some.jpg")
    other_path = os.path.join(empty_local_dir, "other.jpg")
    files_specs = [
        (
            "some.jpg",
            "https://media.roboflow.com/dog.jpeg",
            "fc3f17e4aa70acf5bfc385aaa344d275",
        ),
        ("other.jpg", "https://media.roboflow.com/dog.jpeg", "invalid"),
    ]

    # when
    result = download_files_to_directory(
        target_dir=empty_local_dir,
        files_specs=files_specs,
        verify_hash_while_download=False,
    )

    # then
    assert result["some.jpg"] == some_path
    assert result["other.jpg"] == other_path
    assert os.path.isfile(some_path)
    assert os.path.isfile(other_path)
    assert calculate_md5(file=some_path) == "fc3f17e4aa70acf5bfc385aaa344d275"
    assert calculate_md5(file=other_path) == "fc3f17e4aa70acf5bfc385aaa344d275"


@pytest.mark.timeout(10)
@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_download_files_to_directory_when_hash_sum_not_provided(
    empty_local_dir: str,
) -> None:
    # given
    files_specs = [
        (
            "some.jpg",
            "https://media.roboflow.com/dog.jpeg",
            "fc3f17e4aa70acf5bfc385aaa344d275",
        ),
        ("other.jpg", "https://media.roboflow.com/dog.jpeg", None),
    ]

    # when
    with pytest.raises(UntrustedFileError):
        download_files_to_directory(
            target_dir=empty_local_dir,
            files_specs=files_specs,
        )


@pytest.mark.timeout(10)
@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_download_files_to_directory_when_hash_sum_not_provided_but_untrusted_allowed(
    empty_local_dir: str,
) -> None:
    # given
    some_path = os.path.join(empty_local_dir, "some.jpg")
    other_path = os.path.join(empty_local_dir, "other.jpg")
    files_specs = [
        (
            "some.jpg",
            "https://media.roboflow.com/dog.jpeg",
            "fc3f17e4aa70acf5bfc385aaa344d275",
        ),
        ("other.jpg", "https://media.roboflow.com/dog.jpeg", None),
    ]

    # when
    result = download_files_to_directory(
        target_dir=empty_local_dir,
        files_specs=files_specs,
        download_files_without_hash=True,
    )

    # then
    assert result["some.jpg"] == some_path
    assert result["other.jpg"] == other_path
    assert os.path.isfile(some_path)
    assert os.path.isfile(other_path)
    assert calculate_md5(file=some_path) == "fc3f17e4aa70acf5bfc385aaa344d275"
    assert calculate_md5(file=other_path) == "fc3f17e4aa70acf5bfc385aaa344d275"


@pytest.mark.timeout(120)
@pytest.mark.slow
@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_download_files_to_directory_large_files(empty_local_dir: str) -> None:
    # given
    yolonas_path = os.path.join(empty_local_dir, "yolonas.zip")
    yolov10_path = os.path.join(empty_local_dir, "yolov10_det.zip")
    files_specs = [
        (
            "yolonas.zip",
            "https://storage.googleapis.com/roboflow-tests-assets/yolonas.zip",
            "8d91e76f3da85ff3df8f4a5c3a420b72",
        ),
        (
            "yolov10_det.zip",
            "https://storage.googleapis.com/roboflow-tests-assets/yolov10_det.zip",
            "d04b35026fb68111be4b57b2518aaace",
        ),
    ]

    # when
    download_files_to_directory(
        target_dir=empty_local_dir,
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


@pytest.mark.timeout(10)
@pytest.mark.torch_models
@pytest.mark.cpu_only
def test_get_content_length_against_real_asset() -> None:
    # when
    result = get_content_length("https://media.roboflow.com/dog.jpeg")

    # then
    assert result == 106055
