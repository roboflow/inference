import os
import re

import pytest

from inference_models.errors import InsecureModelIdentifierError
from inference_models.models.auto_loaders import model_cache_paths
from inference_models.models.auto_loaders.model_cache_paths import (
    generate_model_cache_root_for_model_id,
    generate_model_package_cache_path,
    resolve_existing_model_package_cache_path,
    slugify_model_id_to_os_safe_format,
)


@pytest.fixture(autouse=True)
def isolate_inference_home(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache_paths, "INFERENCE_HOME", str(tmp_path))


def test_slug_format_is_stable() -> None:
    # given / when
    slug = slugify_model_id_to_os_safe_format("workspace/model/1")

    # then: the on-disk format written since 0.32.0 — prefix + 128-bit digest
    assert re.fullmatch(r"v2-[A-Za-z0-9_-]+-[0-9a-f]{32}", slug)
    assert slug == slugify_model_id_to_os_safe_format("workspace/model/1")


def test_slug_separates_known_32_bit_collision() -> None:
    # given: a model-id pair whose legacy 32-bit digests collided
    first_model_id = f"{'a' * 48}/36371"
    second_model_id = f"{'a' * 48}/64545"

    # when / then
    assert slugify_model_id_to_os_safe_format(
        first_model_id
    ) != slugify_model_id_to_os_safe_format(second_model_id)


def test_slug_strips_path_separators() -> None:
    # given a hostile model id
    slug = slugify_model_id_to_os_safe_format("../../../etc/passwd")

    # then
    assert "/" not in slug
    assert ".." not in slug


def test_package_path_is_deduced_from_slug_and_package_id() -> None:
    # when
    package_path = generate_model_package_cache_path(
        model_id="workspace/model/1", package_id="pkg1"
    )

    # then
    expected_root = generate_model_cache_root_for_model_id(model_id="workspace/model/1")
    assert package_path == os.path.join(expected_root, "pkg1")
    assert expected_root.endswith(
        slugify_model_id_to_os_safe_format("workspace/model/1")
    )


@pytest.mark.parametrize(
    "package_id",
    ["", "pkg-1", "pkg/1", "pkg.1", "..", "NUL", "COM1", "a" * 256, "pkg 1"],
)
def test_package_id_must_be_nonempty_ascii_alphanumeric(package_id) -> None:
    with pytest.raises(InsecureModelIdentifierError):
        generate_model_package_cache_path(
            model_id="workspace/model/1", package_id=package_id
        )


def test_resolve_returns_existing_package_dir() -> None:
    # given
    package_path = generate_model_package_cache_path(
        model_id="workspace/model/1", package_id="pkg1"
    )
    os.makedirs(package_path)

    # when / then
    assert (
        resolve_existing_model_package_cache_path(
            model_id="workspace/model/1", package_id="pkg1"
        )
        == package_path
    )


def test_resolve_returns_none_for_missing_package() -> None:
    assert (
        resolve_existing_model_package_cache_path(
            model_id="workspace/model/1", package_id="pkg1"
        )
        is None
    )


def test_resolve_returns_none_when_package_path_is_a_file() -> None:
    # given
    package_path = generate_model_package_cache_path(
        model_id="workspace/model/1", package_id="pkg1"
    )
    os.makedirs(os.path.dirname(package_path))
    open(package_path, "w").close()

    # when / then
    assert (
        resolve_existing_model_package_cache_path(
            model_id="workspace/model/1", package_id="pkg1"
        )
        is None
    )


def test_resolve_rejects_invalid_package_id() -> None:
    with pytest.raises(InsecureModelIdentifierError):
        resolve_existing_model_package_cache_path(
            model_id="workspace/model/1", package_id="pkg/../1"
        )
