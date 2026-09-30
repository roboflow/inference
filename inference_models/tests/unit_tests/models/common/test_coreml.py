import json
import os
import zipfile
from types import SimpleNamespace

import pytest

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


def test_resolve_mlpackage_prefers_directory_bundle(tmp_path) -> None:
    bundle = tmp_path / "weights.mlpackage"
    _write_bundle(str(bundle))

    assert coreml.resolve_mlpackage(str(tmp_path)) == str(bundle)
    assert not (tmp_path / COREML_CACHE_DIR_NAME).exists()


@pytest.mark.parametrize("top_level_folder", ["", "weights.mlpackage"])
def test_resolve_mlpackage_extracts_zipped_bundle_into_coreml_cache(
    tmp_path, top_level_folder: str
) -> None:
    source = tmp_path / "source.mlpackage"
    _write_bundle(str(source))
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    _zip_bundle(
        str(source), str(package_dir / "weights.mlpackage.zip"), top_level_folder
    )

    result = coreml.resolve_mlpackage(str(package_dir))

    assert result == str(
        package_dir / COREML_CACHE_DIR_NAME / "native" / "weights.mlpackage"
    )
    assert os.path.isfile(os.path.join(result, "Manifest.json"))
    assert _read_weights(result) == b"weights"


def test_resolve_mlpackage_reuses_extracted_bundle(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mlpackage"
    _write_bundle(str(source))
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    _zip_bundle(str(source), str(package_dir / "weights.mlpackage.zip"))
    first = coreml.resolve_mlpackage(str(package_dir))

    def fail_extraction(**kwargs) -> None:
        raise AssertionError("bundle should not be extracted twice")

    monkeypatch.setattr(coreml, "_extract_bundle", fail_extraction)

    assert coreml.resolve_mlpackage(str(package_dir)) == first


def test_resolve_mlpackage_re_extracts_partially_deleted_bundle(tmp_path) -> None:
    source = tmp_path / "source.mlpackage"
    _write_bundle(str(source))
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    _zip_bundle(str(source), str(package_dir / "weights.mlpackage.zip"))
    extracted = coreml.resolve_mlpackage(str(package_dir))
    os.remove(os.path.join(extracted, "Manifest.json"))

    result = coreml.resolve_mlpackage(str(package_dir))

    assert os.path.isfile(os.path.join(result, "Manifest.json"))
    assert _read_weights(result) == b"weights"


def test_resolve_mlpackage_extracts_outside_package_in_offline_mode(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.mlpackage"
    _write_bundle(str(source))
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    _zip_bundle(str(source), str(package_dir / "weights.mlpackage.zip"))
    monkeypatch.setattr(coreml, "OFFLINE_MODE", True)
    monkeypatch.setattr(coreml.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))

    result = coreml.resolve_mlpackage(str(package_dir))

    assert result.startswith(str(tmp_path / "tmp" / "inference-models-coreml"))
    assert _read_weights(result) == b"weights"
    assert not (package_dir / COREML_CACHE_DIR_NAME).exists()


def test_resolve_mlpackage_rejects_archive_entries_outside_bundle(tmp_path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    with zipfile.ZipFile(package_dir / "weights.mlpackage.zip", "w") as archive:
        archive.writestr("Manifest.json", "{}")
        archive.writestr("../escaped.txt", "oops")

    with pytest.raises(CorruptedModelPackageError):
        coreml.resolve_mlpackage(str(package_dir))

    assert not (tmp_path / "escaped.txt").exists()


def test_resolve_mlpackage_rejects_archive_without_bundle(tmp_path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    with zipfile.ZipFile(package_dir / "weights.mlpackage.zip", "w") as archive:
        archive.writestr("readme.txt", "not a bundle")

    with pytest.raises(CorruptedModelPackageError):
        coreml.resolve_mlpackage(str(package_dir))


def test_resolve_mlpackage_requires_bundle_or_archive(tmp_path) -> None:
    with pytest.raises(CorruptedModelPackageError):
        coreml.resolve_mlpackage(str(tmp_path))


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
