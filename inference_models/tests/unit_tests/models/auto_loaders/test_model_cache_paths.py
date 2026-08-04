import json
import os
import re

import pytest

from inference_models.errors import InsecureModelIdentifierError
from inference_models.models.auto_loaders import model_cache_paths
from inference_models.models.auto_loaders.model_cache_paths import (
    generate_model_package_cache_path,
    generate_model_package_cache_path_candidates,
    resolve_existing_model_package_cache_path,
    slugify_model_id_to_os_safe_format_v1,
    slugify_model_id_to_os_safe_format_v2,
)


@pytest.fixture(autouse=True)
def isolate_inference_home(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))


def _write_model_config(package_path: str, model_id: str) -> None:
    os.makedirs(package_path, exist_ok=True)
    with open(os.path.join(package_path, "model_config.json"), "w") as config_file:
        json.dump({"model_id": model_id}, config_file)


def test_v2_slug_separates_known_v1_32_bit_collision():
    first_model_id = f"{'a' * 48}/36371"
    second_model_id = f"{'a' * 48}/72629"

    assert slugify_model_id_to_os_safe_format_v1(
        first_model_id
    ) == slugify_model_id_to_os_safe_format_v1(second_model_id)
    assert (
        slugify_model_id_to_os_safe_format_v1(first_model_id) == f"{'a' * 48}-d9e80196"
    )

    first_v2_slug = slugify_model_id_to_os_safe_format_v2(first_model_id)
    second_v2_slug = slugify_model_id_to_os_safe_format_v2(second_model_id)
    assert first_v2_slug != second_v2_slug
    assert re.fullmatch(r"v2-[A-Za-z0-9_-]+-[0-9a-f]{32}", first_v2_slug)
    assert re.fullmatch(r"v2-[A-Za-z0-9_-]+-[0-9a-f]{32}", second_v2_slug)


def test_new_package_path_uses_v2_slug():
    model_id = "workspace/project/1"

    package_path = generate_model_package_cache_path(
        model_id=model_id,
        package_id="package1",
    )

    assert os.path.basename(os.path.dirname(package_path)) == (
        slugify_model_id_to_os_safe_format_v2(model_id)
    )


@pytest.mark.parametrize(
    "package_id",
    ["", "unsafe-package", "../package", "a" * 256, "CON"],
)
def test_package_id_must_be_nonempty_ascii_alphanumeric(package_id):
    with pytest.raises(InsecureModelIdentifierError):
        generate_model_package_cache_path(
            model_id="workspace/project/1",
            package_id=package_id,
        )

    with pytest.raises(InsecureModelIdentifierError):
        resolve_existing_model_package_cache_path(
            model_id="workspace/project/1",
            package_id=package_id,
        )


def test_rejects_case_insensitive_package_id_alias():
    model_id = "workspace/project/1"
    v2_package_path, _ = generate_model_package_cache_path_candidates(
        model_id=model_id,
        package_id="Package1",
    )
    _write_model_config(package_path=v2_package_path, model_id=model_id)

    with pytest.raises(InsecureModelIdentifierError):
        generate_model_package_cache_path(
            model_id=model_id,
            package_id="package1",
        )
    assert (
        resolve_existing_model_package_cache_path(
            model_id=model_id,
            package_id="package1",
        )
        is None
    )


def test_resolves_exactly_attributed_legacy_v1_package():
    model_id = "workspace/project/1"
    _, legacy_package_path = generate_model_package_cache_path_candidates(
        model_id=model_id,
        package_id="package1",
    )
    _write_model_config(package_path=legacy_package_path, model_id=model_id)

    assert (
        resolve_existing_model_package_cache_path(
            model_id=model_id,
            package_id="package1",
        )
        == legacy_package_path
    )


def test_rejects_wrong_legacy_v1_owner_even_for_colliding_slug():
    requested_model_id = f"{'a' * 48}/36371"
    colliding_model_id = f"{'a' * 48}/72629"
    _, legacy_package_path = generate_model_package_cache_path_candidates(
        model_id=requested_model_id,
        package_id="package1",
    )
    _write_model_config(
        package_path=legacy_package_path,
        model_id=colliding_model_id,
    )

    assert (
        resolve_existing_model_package_cache_path(
            model_id=requested_model_id,
            package_id="package1",
        )
        is None
    )
    assert (
        resolve_existing_model_package_cache_path(
            model_id=colliding_model_id,
            package_id="package1",
        )
        == legacy_package_path
    )


def test_unattributed_v2_package_requires_explicit_local_cache_opt_in():
    model_id = "workspace/project/1"
    v2_package_path, _ = generate_model_package_cache_path_candidates(
        model_id=model_id,
        package_id="localtrt1",
    )
    os.makedirs(v2_package_path)

    assert (
        resolve_existing_model_package_cache_path(
            model_id=model_id,
            package_id="localtrt1",
        )
        is None
    )
    assert (
        resolve_existing_model_package_cache_path(
            model_id=model_id,
            package_id="localtrt1",
            allow_unattributed_local_cache=True,
        )
        == v2_package_path
    )


def test_unattributed_legacy_v1_package_is_rejected_with_local_cache_opt_in():
    model_id = "workspace/project/1"
    _, legacy_package_path = generate_model_package_cache_path_candidates(
        model_id=model_id,
        package_id="localtrt1",
    )
    os.makedirs(legacy_package_path)

    assert (
        resolve_existing_model_package_cache_path(
            model_id=model_id,
            package_id="localtrt1",
            allow_unattributed_local_cache=True,
        )
        is None
    )


@pytest.mark.parametrize("invalid_model_id", ["", None, [], {}])
def test_invalid_manifest_owner_is_rejected_even_with_local_cache_opt_in(
    invalid_model_id,
):
    model_id = "workspace/project/1"
    _, legacy_package_path = generate_model_package_cache_path_candidates(
        model_id=model_id,
        package_id="localtrt1",
    )
    _write_model_config(
        package_path=legacy_package_path,
        model_id=invalid_model_id,
    )

    assert (
        resolve_existing_model_package_cache_path(
            model_id=model_id,
            package_id="localtrt1",
            allow_unattributed_local_cache=True,
        )
        is None
    )


def test_v2_package_is_preferred_over_legacy_v1():
    model_id = "workspace/project/1"
    v2_package_path, legacy_package_path = generate_model_package_cache_path_candidates(
        model_id=model_id,
        package_id="package1",
    )
    _write_model_config(package_path=legacy_package_path, model_id=model_id)
    _write_model_config(package_path=v2_package_path, model_id=model_id)

    assert (
        resolve_existing_model_package_cache_path(
            model_id=model_id,
            package_id="package1",
        )
        == v2_package_path
    )


def test_exactly_attributed_legacy_package_beats_unattributed_v2_fallback():
    model_id = "workspace/project/1"
    v2_package_path, legacy_package_path = generate_model_package_cache_path_candidates(
        model_id=model_id,
        package_id="localtrt1",
    )
    os.makedirs(v2_package_path)
    _write_model_config(package_path=legacy_package_path, model_id=model_id)

    assert (
        resolve_existing_model_package_cache_path(
            model_id=model_id,
            package_id="localtrt1",
            allow_unattributed_local_cache=True,
        )
        == legacy_package_path
    )


def test_rejects_symlinked_package_directory(tmp_path):
    model_id = "workspace/project/1"
    v2_package_path, _ = generate_model_package_cache_path_candidates(
        model_id=model_id,
        package_id="package1",
    )
    external_package = tmp_path / "external-package"
    _write_model_config(package_path=str(external_package), model_id=model_id)
    os.makedirs(os.path.dirname(v2_package_path))
    os.symlink(external_package, v2_package_path, target_is_directory=True)

    assert (
        resolve_existing_model_package_cache_path(
            model_id=model_id,
            package_id="package1",
        )
        is None
    )


def test_rejects_symlinked_model_config(tmp_path):
    model_id = "workspace/project/1"
    v2_package_path, _ = generate_model_package_cache_path_candidates(
        model_id=model_id,
        package_id="package1",
    )
    os.makedirs(v2_package_path)
    external_config = tmp_path / "external-config.json"
    external_config.write_text(json.dumps({"model_id": model_id}))
    os.symlink(external_config, os.path.join(v2_package_path, "model_config.json"))

    assert (
        resolve_existing_model_package_cache_path(
            model_id=model_id,
            package_id="package1",
            allow_unattributed_local_cache=True,
        )
        is None
    )


def test_closes_manifest_descriptor_when_fdopen_fails(monkeypatch):
    model_id = "workspace/project/1"
    v2_package_path, _ = generate_model_package_cache_path_candidates(
        model_id=model_id,
        package_id="package1",
    )
    _write_model_config(package_path=v2_package_path, model_id=model_id)
    closed_descriptors = []
    original_close = os.close

    def record_close(file_descriptor):
        closed_descriptors.append(file_descriptor)
        original_close(file_descriptor)

    def fail_fdopen(*args, **kwargs):
        raise OSError("fdopen failed")

    monkeypatch.setattr(model_cache_paths.os, "close", record_close)
    monkeypatch.setattr(model_cache_paths.os, "fdopen", fail_fdopen)

    assert (
        resolve_existing_model_package_cache_path(
            model_id=model_id,
            package_id="package1",
        )
        is None
    )
    assert len(closed_descriptors) == 1
