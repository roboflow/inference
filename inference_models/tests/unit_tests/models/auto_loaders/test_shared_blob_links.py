import errno
import hashlib
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.auto_loaders import core


@pytest.fixture
def shared_blob(tmp_path: Path, monkeypatch):
    shared = tmp_path / "shared-blobs"
    shared.mkdir()
    payload = b"model weights"
    digest = hashlib.md5(payload).hexdigest()
    blob = shared / digest
    blob.write_bytes(payload)
    package = tmp_path / "package"
    package.mkdir()
    monkeypatch.setattr(core, "generate_shared_blobs_path", lambda: str(shared))
    return blob, package, digest


def _identities(package: Path, digest: str):
    return core._materialize_package_artifact_identities(
        package_dir=str(package),
        declarations=[
            {"file_handle": "weights.bin", "storage": "shared_blob", "md5_hash": digest}
        ],
    )


def _missing_symlink_privilege(*args, **kwargs):
    error = OSError(errno.EPERM, "A required privilege is not held by the client")
    error.winerror = 1314
    raise error


def test_windows_symlink_privilege_falls_back_to_a_verified_hardlink(
    shared_blob, monkeypatch
) -> None:
    blob, package, digest = shared_blob
    created = Mock()
    monkeypatch.setattr(core.os, "symlink", _missing_symlink_privilege)

    result = core.create_symlinks_to_shared_blobs(
        model_dir=str(package),
        shared_files_mapping={"weights.bin": str(blob)},
        on_symlink_created=created,
    )

    destination = package / "weights.bin"
    assert result == {"weights.bin": str(destination)}
    assert not destination.is_symlink()
    assert os.path.samefile(destination, blob)
    assert destination.read_bytes() == b"model weights"
    assert _identities(package, digest)[0]["md5_hash"] == digest
    created.assert_called_once_with(str(blob), str(destination))

    # Reusing the cache must neither recreate nor replace the hardlink.
    monkeypatch.setattr(core.os, "link", Mock(side_effect=AssertionError("relinked")))
    core.create_symlinks_to_shared_blobs(
        model_dir=str(package), shared_files_mapping={"weights.bin": str(blob)}
    )
    assert os.path.samefile(destination, blob)


@pytest.mark.parametrize("link_type", ["symlink", "hardlink"])
def test_shared_blob_identity_accepts_links_to_the_expected_blob(
    shared_blob, link_type: str
) -> None:
    blob, package, digest = shared_blob
    destination = package / "weights.bin"
    if link_type == "hardlink":
        os.link(blob, destination)
    else:
        try:
            destination.symlink_to(blob)
        except OSError as error:
            pytest.skip(f"Symlinks are unavailable: {error}")

    assert _identities(package, digest) == [
        {
            "file_handle": "weights.bin",
            "storage": "shared_blob",
            "md5_hash": digest,
            "sha256_hash": None,
        }
    ]


@pytest.mark.parametrize("invalid_entry", ["copy", "unrelated", "missing", "corrupted"])
def test_shared_blob_identity_rejects_an_unverified_file(
    shared_blob, invalid_entry: str
) -> None:
    blob, package, digest = shared_blob
    destination = package / "weights.bin"
    if invalid_entry == "copy":
        destination.write_bytes(blob.read_bytes())
    elif invalid_entry == "unrelated":
        unrelated = package / "other.bin"
        unrelated.write_bytes(b"other weights")
        os.link(unrelated, destination)
    elif invalid_entry == "corrupted":
        os.link(blob, destination)
        blob.write_bytes(b"changed weights")

    with pytest.raises(CorruptedModelPackageError, match="expected shared blob"):
        _identities(package, digest)


def test_shared_blob_identity_rejects_a_symlinked_shared_blob(
    shared_blob, tmp_path: Path
) -> None:
    blob, package, digest = shared_blob
    outside = tmp_path / "outside.bin"
    blob.rename(outside)
    try:
        blob.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Symlinks are unavailable: {error}")
    (package / "weights.bin").symlink_to(blob)

    with pytest.raises(CorruptedModelPackageError, match="expected shared blob"):
        _identities(package, digest)


def test_non_privilege_symlink_error_is_not_hidden(shared_blob, monkeypatch) -> None:
    blob, package, _ = shared_blob
    failure = PermissionError(errno.EACCES, "directory is read-only")
    link = Mock()
    monkeypatch.setattr(core.os, "symlink", Mock(side_effect=failure))
    monkeypatch.setattr(core.os, "link", link)

    with pytest.raises(PermissionError) as raised:
        core.handle_symlink_creation(str(blob), str(package / "weights.bin"))

    assert raised.value is failure
    link.assert_not_called()


def test_hardlink_failure_is_not_reported_as_success(shared_blob, monkeypatch) -> None:
    blob, package, _ = shared_blob
    failure = OSError(errno.EXDEV, "different filesystems")
    created = Mock()
    monkeypatch.setattr(core.os, "symlink", _missing_symlink_privilege)
    monkeypatch.setattr(core.os, "link", Mock(side_effect=failure))

    with pytest.raises(OSError) as raised:
        core.handle_symlink_creation(
            str(blob), str(package / "weights.bin"), on_symlink_created=created
        )

    assert raised.value is failure
    created.assert_not_called()
    assert not (package / "weights.bin").exists()


def test_hardlink_race_preserves_the_other_process_file(
    shared_blob, monkeypatch
) -> None:
    blob, package, digest = shared_blob
    destination = package / "weights.bin"
    real_link = os.link
    created = Mock()
    monkeypatch.setattr(core.os, "symlink", _missing_symlink_privilege)

    def competing_link(target, name):
        real_link(target, name)
        raise FileExistsError(errno.EEXIST, "created by another process")

    monkeypatch.setattr(core.os, "link", competing_link)

    core.handle_symlink_creation(
        str(blob), str(destination), on_symlink_created=created
    )

    assert os.path.samefile(destination, blob)
    assert _identities(package, digest)[0]["md5_hash"] == digest
    created.assert_not_called()
