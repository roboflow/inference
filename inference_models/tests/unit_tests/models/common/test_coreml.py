import json
import os
import shutil
import zipfile
from types import SimpleNamespace
from typing import List

import pytest
from filelock import FileLock, Timeout

from inference_models.errors import CorruptedModelPackageError, InvalidEnvVariable
from inference_models.models.common import coreml
from inference_models.models.common.model_packages import COREML_CACHE_DIR_NAME


def _write_bundle(directory: str, weights: bytes = b"weights") -> None:
    os.makedirs(os.path.join(directory, "Data", "com.apple.CoreML"), exist_ok=True)
    with open(os.path.join(directory, "Manifest.json"), "w") as f:
        json.dump({"fileFormatVersion": "1.0.0"}, f)
    with open(
        os.path.join(directory, "Data", "com.apple.CoreML", "weights.bin"), "wb"
    ) as f:
        f.write(weights)


def _zip_bundle(bundle_dir: str, archive_path: str, top_level_folder: str = "") -> None:
    with zipfile.ZipFile(archive_path, "w") as archive:
        for root, _, files in os.walk(bundle_dir):
            for name in files:
                full_path = os.path.join(root, name)
                relative_path = os.path.relpath(full_path, bundle_dir)
                archive.write(full_path, os.path.join(top_level_folder, relative_path))


def _read_weights(bundle_path: str) -> bytes:
    with open(
        os.path.join(bundle_path, "Data", "com.apple.CoreML", "weights.bin"), "rb"
    ) as f:
        return f.read()


@pytest.fixture
def loaded_paths(monkeypatch) -> List[str]:
    """Record the bundle each load reads (instead of loading it with coremltools)."""
    paths = []

    def fake_load(mlpackage_path: str, compute_units: str, **kwargs) -> str:
        paths.append(mlpackage_path)
        return mlpackage_path

    monkeypatch.setattr(coreml, "load_coreml_model", fake_load)
    return paths


def _zipped_package(tmp_path, top_level_folder: str = "") -> str:
    source = tmp_path / "source.mlpackage"
    _write_bundle(str(source))
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    _zip_bundle(
        str(source), str(package_dir / "weights.mlpackage.zip"), top_level_folder
    )
    return str(package_dir)


def test_load_coreml_package_prefers_directory_bundle(tmp_path, loaded_paths) -> None:
    bundle = tmp_path / "weights.mlpackage"
    _write_bundle(str(bundle))

    assert coreml.load_coreml_package(str(tmp_path)) == str(bundle)
    assert not (tmp_path / COREML_CACHE_DIR_NAME).exists()


@pytest.mark.parametrize("top_level_folder", ["", "weights.mlpackage"])
def test_load_coreml_package_extracts_zipped_bundle_into_coreml_cache(
    tmp_path, loaded_paths, top_level_folder: str
) -> None:
    package_dir = _zipped_package(tmp_path, top_level_folder)

    result = coreml.load_coreml_package(package_dir)

    native_root = os.path.join(package_dir, COREML_CACHE_DIR_NAME, "native")
    assert os.path.dirname(os.path.dirname(result)) == native_root
    assert os.path.basename(result) == "weights.mlpackage"
    assert _read_weights(result) == b"weights"


def test_load_coreml_package_reuses_extracted_bundle(
    tmp_path, loaded_paths, monkeypatch
) -> None:
    package_dir = _zipped_package(tmp_path)
    first = coreml.load_coreml_package(package_dir)

    def fail_extraction(**kwargs) -> None:
        raise AssertionError("bundle should not be extracted twice")

    monkeypatch.setattr(coreml, "_extract_bundle", fail_extraction)

    assert coreml.load_coreml_package(package_dir) == first


def test_load_coreml_package_re_extracts_replaced_archive(
    tmp_path, loaded_paths
) -> None:
    package_dir = _zipped_package(tmp_path)
    first = coreml.load_coreml_package(package_dir)
    replacement = tmp_path / "replacement.mlpackage"
    _write_bundle(str(replacement), weights=b"retrained")
    archive = os.path.join(package_dir, "weights.mlpackage.zip")
    _zip_bundle(str(replacement), archive)
    os.utime(archive, ns=(1, 1))

    second = coreml.load_coreml_package(package_dir)

    assert second != first
    assert _read_weights(second) == b"retrained"
    assert not os.path.exists(first)


def test_load_coreml_package_re_extracts_partially_deleted_bundle(
    tmp_path, loaded_paths
) -> None:
    package_dir = _zipped_package(tmp_path)
    extracted = coreml.load_coreml_package(package_dir)
    os.remove(os.path.join(extracted, "Manifest.json"))

    result = coreml.load_coreml_package(package_dir)

    assert os.path.isfile(os.path.join(result, "Manifest.json"))
    assert _read_weights(result) == b"weights"


def test_load_coreml_package_extracts_and_loads_under_the_package_coreml_cache_lock(
    tmp_path, monkeypatch
) -> None:
    package_dir = _zipped_package(tmp_path)
    lock_path = os.path.join(package_dir, ".coreml_cache.lock")
    held_during_load = []

    def fake_load(mlpackage_path: str, compute_units: str, **kwargs) -> str:
        probe = FileLock(lock_path, timeout=0)
        try:
            probe.acquire()
            probe.release()
            held_during_load.append(False)
        except Timeout:
            held_during_load.append(True)
        return mlpackage_path

    monkeypatch.setattr(coreml, "load_coreml_model", fake_load)

    coreml.load_coreml_package(package_dir)

    assert held_during_load == [True]


def test_load_coreml_package_survives_a_cache_purge_right_before_the_lock(
    tmp_path, loaded_paths, monkeypatch
) -> None:
    package_dir = _zipped_package(tmp_path)
    coreml.load_coreml_package(package_dir)
    real_file_lock = coreml.FileLock

    def purge_then_lock(path, *args, **kwargs):
        # The watchdog purges coreml_cache after the loader decided to use it, before it holds the lock.
        shutil.rmtree(
            os.path.join(package_dir, COREML_CACHE_DIR_NAME), ignore_errors=True
        )
        return real_file_lock(path, *args, **kwargs)

    monkeypatch.setattr(coreml, "FileLock", purge_then_lock)

    result = coreml.load_coreml_package(package_dir)

    assert _read_weights(result) == b"weights"


def test_load_coreml_package_extracts_outside_package_in_offline_mode(
    tmp_path, loaded_paths, monkeypatch
) -> None:
    package_dir = _zipped_package(tmp_path)
    monkeypatch.setattr(coreml, "OFFLINE_MODE", True)
    monkeypatch.setattr(coreml.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    result = coreml.load_coreml_package(package_dir)

    assert result.startswith(str(tmp_path / "tmp" / "inference-models-coreml"))
    assert _read_weights(result) == b"weights"
    assert not os.path.exists(os.path.join(package_dir, COREML_CACHE_DIR_NAME))


def test_load_coreml_package_rejects_archive_entries_outside_bundle(
    tmp_path, loaded_paths
) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    with zipfile.ZipFile(package_dir / "weights.mlpackage.zip", "w") as archive:
        archive.writestr("Manifest.json", "{}")
        archive.writestr("../escaped.txt", "oops")

    with pytest.raises(CorruptedModelPackageError):
        coreml.load_coreml_package(str(package_dir))

    assert not (tmp_path / "escaped.txt").exists()
    assert loaded_paths == []


def test_load_coreml_package_rejects_archive_without_bundle(
    tmp_path, loaded_paths
) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    with zipfile.ZipFile(package_dir / "weights.mlpackage.zip", "w") as archive:
        archive.writestr("readme.txt", "not a bundle")

    with pytest.raises(CorruptedModelPackageError):
        coreml.load_coreml_package(str(package_dir))


def test_load_coreml_package_requires_bundle_or_archive(tmp_path, loaded_paths) -> None:
    with pytest.raises(CorruptedModelPackageError):
        coreml.load_coreml_package(str(tmp_path))


def _spec(input_type: str, outputs=("boxes", "scores", "labels")) -> SimpleNamespace:
    model_input_type = SimpleNamespace(
        WhichOneof=lambda _: input_type,
        imageType=SimpleNamespace(height=384, width=512),
        multiArrayType=SimpleNamespace(shape=[1, 3, 312, 312]),
    )
    return SimpleNamespace(
        description=SimpleNamespace(
            input=[SimpleNamespace(name="image", type=model_input_type)],
            output=[SimpleNamespace(name=name) for name in outputs],
        )
    )


def test_read_signature_for_image_input() -> None:
    signature = coreml.read_signature(_spec("imageType"))

    assert signature == coreml.CoreMLModelSignature(
        input_name="image",
        image_input=True,
        input_height=384,
        input_width=512,
        output_names=("boxes", "scores", "labels"),
    )


def test_read_signature_for_tensor_input() -> None:
    signature = coreml.read_signature(_spec("multiArrayType", ("dets", "labels")))

    assert signature.image_input is False
    assert (signature.input_height, signature.input_width) == (312, 312)
    assert signature.output_names == ("dets", "labels")


def test_read_signature_rejects_unsupported_input() -> None:
    with pytest.raises(CorruptedModelPackageError):
        coreml.read_signature(_spec("stringType"))


def test_load_coreml_model_rejects_unknown_compute_units(tmp_path) -> None:
    with pytest.raises(InvalidEnvVariable):
        coreml.load_coreml_model(str(tmp_path), compute_units="GPUOnly")
