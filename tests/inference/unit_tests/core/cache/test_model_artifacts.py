import json
import os
import os.path
from unittest import mock
from unittest.mock import MagicMock, call

import pytest
from humanfriendly.testing import touch

from inference.core.cache import model_artifacts
from inference.core.cache.model_artifacts import (
    are_all_files_cached,
    clear_cache,
    get_cache_dir,
    get_cache_dir_for_read,
    get_cache_file_path,
    initialise_cache,
    is_file_cached,
    load_json_from_cache,
    load_text_file_from_cache,
    save_bytes_in_cache,
    save_json_in_cache,
    save_text_lines_in_cache,
    slugify_model_id_to_cache_key,
)
from inference.core.exceptions import ModelArtefactError
from inference.core.utils.file_system import MAX_PATH_SEGMENT_BYTES
from tests.inference.unit_tests.core.utils.test_file_system import (
    assert_bytes_file_content_correct,
    assert_text_file_content_correct,
)


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_initialise_cache(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "3")
    get_cache_dir_mock.return_value = cache_dir

    # when
    initialise_cache(model_id="some/3")

    # then
    assert os.path.isdir(cache_dir)
    get_cache_dir_mock.assert_called_once_with(model_id="some/3")


def test_initialise_cache_never_creates_missing_model_directory_offline(
    tmp_path,
) -> None:
    missing_cache_dir = tmp_path / "missing" / "1"
    with (
        mock.patch.object(model_artifacts, "OFFLINE_MODE", True),
        mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", str(tmp_path)),
    ):
        initialise_cache(model_id="missing/1")

    assert not missing_cache_dir.exists()


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_are_all_files_cached_when_all_files_exists(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "3")
    get_cache_dir_mock.return_value = cache_dir
    touch(os.path.join(cache_dir, "a.txt"))
    touch(os.path.join(cache_dir, "b", "c.txt"))

    # when
    result = are_all_files_cached(
        files=["a.txt", "b/c.txt"],
        model_id="some/3",
    )

    # then
    assert result is True
    get_cache_dir_mock.assert_has_calls([call(model_id="some/3")] * 2)


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_are_all_files_cached_when_not_all_files_exists(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "3")
    get_cache_dir_mock.return_value = cache_dir
    touch(os.path.join(cache_dir, "a.txt"))

    # when
    result = are_all_files_cached(
        files=["a.txt", "b/c.txt"],
        model_id="some/3",
    )

    # then
    assert result is False
    get_cache_dir_mock.assert_has_calls([call(model_id="some/3")] * 2)


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_is_file_cached_when_file_exists(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "3")
    get_cache_dir_mock.return_value = cache_dir
    touch(os.path.join(cache_dir, "b", "c.txt"))

    # when
    result = is_file_cached(
        file="b/c.txt",
        model_id="some/3",
    )

    # then
    assert result is True
    get_cache_dir_mock.assert_called_once_with(model_id="some/3")


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_is_file_cached_when_file_does_not_exist(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "3")
    get_cache_dir_mock.return_value = cache_dir

    # when
    result = is_file_cached(
        file="a.txt",
        model_id="some/3",
    )

    # then
    assert result is False
    get_cache_dir_mock.assert_called_once_with(model_id="some/3")


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_load_text_file_from_cache(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "3")
    get_cache_dir_mock.return_value = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "a.txt"), "w") as f:
        f.write("\n".join(["", "A", "", "B", ""]))

    # when
    result = load_text_file_from_cache(
        file="a.txt",
        model_id="some/3",
        split_lines=True,
        strip_white_chars=True,
    )

    # then
    assert result == ["A", "B"]
    get_cache_dir_mock.assert_called_once_with(model_id="some/3")


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_load_json_from_cache(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "3")
    get_cache_dir_mock.return_value = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "a.json"), "w") as f:
        json.dump({"some": "key"}, f)

    # when
    result = load_json_from_cache(file="a.json", model_id="some/3")

    # then
    assert result == {"some": "key"}
    get_cache_dir_mock.assert_called_once_with(model_id="some/3")


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_save_bytes_in_cache(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    expected_file_path = os.path.join(cache_dir, "file.dat")
    touch(os.path.join(cache_dir, "file.dat"))

    # when
    save_bytes_in_cache(content=b"SOME CONTENT", file="file.dat", model_id="some/2")

    # then
    assert_bytes_file_content_correct(
        file_path=expected_file_path, content=b"SOME CONTENT"
    )
    get_cache_dir_mock.assert_called_once_with(model_id="some/2")


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_save_json_in_cache(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    expected_file_path = os.path.join(cache_dir, "file.json")
    touch(os.path.join(cache_dir, "file.json"))

    # when
    save_json_in_cache(
        content=["a", "b"],
        file="file.json",
        model_id="some/2",
        indent=4,
    )

    # then
    assert_text_file_content_correct(
        file_path=expected_file_path, content=json.dumps(["a", "b"], indent=4)
    )
    get_cache_dir_mock.assert_called_once_with(model_id="some/2")


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_save_text_lines_in_cache(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    expected_file_path = os.path.join(cache_dir, "file.txt")
    touch(os.path.join(cache_dir, "file.txt"))

    # when
    save_text_lines_in_cache(
        content=["a", "b"],
        file="file.txt",
        model_id="some/2",
    )

    # then
    assert_text_file_content_correct(
        file_path=expected_file_path,
        content="a\nb",
    )
    get_cache_dir_mock.assert_called_once_with(model_id="some/2")


@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_get_cache_file_path_when_model_id_given() -> None:
    # when
    result = get_cache_file_path(file="some.txt", model_id="yolo/3")

    # then
    assert result == "/some/cache/yolo/3/some.txt"


@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_get_cache_file_path_when_model_id_not_given() -> None:
    # when
    result = get_cache_file_path(file="sub_dir/some.txt")

    # then
    assert result == "/some/cache/sub_dir/some.txt"


@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_get_cache_dir_when_model_id_given() -> None:
    # when
    result = get_cache_dir(model_id="yolo/3")

    # then
    assert result == "/some/cache/yolo/3"


@pytest.mark.parametrize(
    "model_id",
    [
        "thermal dogs and people/18",
        "model.v1/2",
    ],
)
@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_get_cache_dir_preserves_portable_ordinary_model_ids(model_id: str) -> None:
    assert get_cache_dir(model_id=model_id) == os.path.join("/some/cache", model_id)


@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_get_cache_dir_when_model_id_has_long_segment() -> None:
    # given
    long_model_slug = "find-" + ("class-" * 60) + "instant-1"
    model_id = f"workspace/{long_model_slug}"

    # when
    result = get_cache_dir(model_id=model_id)

    # then
    assert result == os.path.join(
        "/some/cache", slugify_model_id_to_cache_key(model_id=model_id)
    )
    assert len(os.fsencode(os.path.basename(result))) <= MAX_PATH_SEGMENT_BYTES


@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_get_cache_dir_when_model_id_has_too_many_segments() -> None:
    # given
    model_id = "/".join(["segment"] * 700)

    # when
    result = get_cache_dir(model_id=model_id)

    # then
    assert result == os.path.join(
        "/some/cache", slugify_model_id_to_cache_key(model_id=model_id)
    )


@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_get_cache_dir_when_model_id_points_outside_cache_root() -> None:
    # given
    model_id = "../outside"

    # when / then
    with pytest.raises(ValueError, match="unsafe or ambiguous path segment"):
        get_cache_dir(model_id=model_id)


@pytest.mark.parametrize(
    "safe_model_id, ambiguous_model_id",
    [
        ("victim", "victim/."),
        ("victim", "victim/child/.."),
        ("victim/1", "victim//1"),
        ("victim", r"victim\."),
        ("victim", r"victim\child\.."),
        (r"victim\1", r"victim\\1"),
        ("victim/1", r"victim/\1"),
    ],
)
@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_get_cache_dir_rejects_ambiguous_model_id_aliases(
    safe_model_id: str, ambiguous_model_id: str
) -> None:
    assert get_cache_dir(model_id=safe_model_id).startswith("/some/cache/")

    with pytest.raises(ValueError, match="unsafe or ambiguous path segment"):
        get_cache_dir(model_id=ambiguous_model_id)


@pytest.mark.parametrize(
    "model_id",
    [
        "",
        "/victim",
        "victim/",
        r"\victim",
        "victim\\",
    ],
)
@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_get_cache_dir_rejects_empty_model_id_path_segments(model_id: str) -> None:
    with pytest.raises(ValueError, match="unsafe or ambiguous path segment"):
        get_cache_dir(model_id=model_id)


def test_v2_model_id_slugs_do_not_share_known_v1_digest_collision() -> None:
    first_model_id = f"{'a' * 48}/36371"
    second_model_id = f"{'a' * 48}/72629"

    first_legacy_slug = model_artifacts._slugify_model_id_to_cache_key(
        model_id=first_model_id,
        digest_size=model_artifacts.LEGACY_MODEL_ID_CACHE_SLUG_HASH_BYTES,
        namespace_prefix="",
    )
    second_legacy_slug = model_artifacts._slugify_model_id_to_cache_key(
        model_id=second_model_id,
        digest_size=model_artifacts.LEGACY_MODEL_ID_CACHE_SLUG_HASH_BYTES,
        namespace_prefix="",
    )
    first_v2_slug = slugify_model_id_to_cache_key(model_id=first_model_id)
    second_v2_slug = slugify_model_id_to_cache_key(model_id=second_model_id)

    assert first_legacy_slug == second_legacy_slug
    assert first_v2_slug != second_v2_slug
    assert len(first_v2_slug.rsplit("-", maxsplit=1)[-1]) >= 32


@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_v2_slug_namespace_cannot_be_reused_as_a_raw_model_id() -> None:
    overlong_model_id = f"workspace/{'x' * 300}"
    generated_slug = slugify_model_id_to_cache_key(model_id=overlong_model_id)

    original_cache_dir = get_cache_dir(model_id=overlong_model_id)
    generated_slug_model_cache_dir = get_cache_dir(model_id=generated_slug)

    assert generated_slug.startswith(
        model_artifacts.MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX
    )
    assert original_cache_dir != generated_slug_model_cache_dir
    assert generated_slug_model_cache_dir != os.path.join("/some/cache", generated_slug)


@pytest.mark.parametrize(
    "model_id",
    [
        "workflow",
        "models-cache",
        "auto-resolution-cache",
        "shared-blobs",
        "_file_locks",
        "huggingface",
        "hf_home",
        "lora-bases",
        "owl-v2-serialized-data",
        "usage.db",
    ],
)
@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_cache_infrastructure_namespaces_are_not_raw_model_paths(
    model_id: str,
) -> None:
    result = get_cache_dir(model_id=model_id)

    assert result != os.path.join("/some/cache", model_id)
    assert os.path.basename(result).startswith(
        model_artifacts.MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "con",
        "con .txt",
        "nul.txt",
        "aux ",
        "prn.",
        "com1",
        "lpt9",
        "model.",
        "model ",
    ],
)
@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_windows_ambiguous_segments_are_not_raw_model_paths(model_id: str) -> None:
    assert get_cache_dir(model_id=model_id) != os.path.join("/some/cache", model_id)


@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_case_and_unicode_ambiguous_ids_have_distinct_v2_paths() -> None:
    lowercase_path = get_cache_dir(model_id="model/1")
    uppercase_path = get_cache_dir(model_id="Model/1")
    composed_unicode_path = get_cache_dir(model_id="caf\u00e9/1")
    decomposed_unicode_path = get_cache_dir(model_id="cafe\u0301/1")

    assert lowercase_path == "/some/cache/model/1"
    assert uppercase_path != lowercase_path
    assert composed_unicode_path != decomposed_unicode_path
    assert all(
        os.path.basename(path).startswith(
            model_artifacts.MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX
        )
        for path in (
            uppercase_path,
            composed_unicode_path,
            decomposed_unicode_path,
        )
    )


@pytest.mark.parametrize(
    "model_id",
    [
        "Model/1",
        "caf\u00e9/1",
        "cafe\u0301/1",
        r"victim\1",
    ],
)
@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_legacy_raw_cache_path_is_retained_as_exact_owner_fallback(
    model_id: str,
) -> None:
    assert (
        model_artifacts.get_model_id_cache_path(
            model_id=model_id,
            cache_dir_root="/some/cache",
        )
        != model_id
    )
    assert (
        model_artifacts.get_legacy_model_id_cache_path(
            model_id=model_id,
            cache_dir_root="/some/cache",
        )
        == model_id
    )


@mock.patch.object(model_artifacts, "OFFLINE_MODE", True)
def test_offline_cache_reads_exact_owned_legacy_raw_artifacts(tmp_path) -> None:
    model_id = "Workspace/Model/1"
    legacy_cache_dir = tmp_path / "Workspace" / "Model" / "1"
    legacy_cache_dir.mkdir(parents=True)
    (legacy_cache_dir / "model_type.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "project_task_type": "object-detection",
                "model_type": "yolov8n",
            }
        )
    )
    (legacy_cache_dir / "weights.bin").write_bytes(b"legacy-weights")

    assert get_cache_dir_for_read(
        model_id=model_id,
        cache_dir_root=str(tmp_path),
    ) == str(legacy_cache_dir)
    with mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", str(tmp_path)):
        assert model_artifacts.get_cache_file_path_for_read(
            file="weights.bin",
            model_id=model_id,
        ) == str(legacy_cache_dir / "weights.bin")
        assert is_file_cached(file="weights.bin", model_id=model_id)


@mock.patch.object(model_artifacts, "OFFLINE_MODE", True)
def test_offline_cache_rejects_unowned_or_symlinked_legacy_raw_artifacts(
    tmp_path,
) -> None:
    model_id = "Workspace/Model/1"
    expected_v2_cache_dir = tmp_path / slugify_model_id_to_cache_key(model_id)
    legacy_cache_dir = tmp_path / "Workspace" / "Model" / "1"
    legacy_cache_dir.mkdir(parents=True)
    metadata_path = legacy_cache_dir / "model_type.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": "different/model/1",
                "project_task_type": "object-detection",
                "model_type": "yolov8n",
            }
        )
    )

    assert get_cache_dir_for_read(
        model_id=model_id,
        cache_dir_root=str(tmp_path),
    ) == str(expected_v2_cache_dir)

    metadata_path.unlink()
    external_metadata = tmp_path / "external-model-type.json"
    external_metadata.write_text(json.dumps({"model_id": model_id}))
    metadata_path.symlink_to(external_metadata)

    assert get_cache_dir_for_read(
        model_id=model_id,
        cache_dir_root=str(tmp_path),
    ) == str(expected_v2_cache_dir)

    metadata_path.unlink()
    metadata_path.write_text(json.dumps({"model_id": model_id}))
    external_weights = tmp_path / "external-weights.bin"
    external_weights.write_bytes(b"outside")
    (legacy_cache_dir / "weights.bin").symlink_to(external_weights)

    assert get_cache_dir_for_read(
        model_id=model_id,
        cache_dir_root=str(tmp_path),
    ) == str(expected_v2_cache_dir)


@mock.patch.object(model_artifacts, "OFFLINE_MODE", True)
def test_offline_cache_rejects_symlinked_artifact_in_current_tree(tmp_path) -> None:
    model_id = "workspace/model/1"
    cache_dir = tmp_path / "workspace" / "model" / "1"
    cache_dir.mkdir(parents=True)
    (cache_dir / "model_type.json").write_text(json.dumps({"model_id": model_id}))
    external_weights = tmp_path / "external-weights.bin"
    external_weights.write_bytes(b"outside")
    (cache_dir / "weights.bin").symlink_to(external_weights)

    with pytest.raises(ModelArtefactError, match="unsafe offline cache tree"):
        get_cache_dir_for_read(
            model_id=model_id,
            cache_dir_root=str(tmp_path),
        )


@mock.patch.object(model_artifacts, "OFFLINE_MODE", True)
def test_offline_cache_prefers_exact_owned_v2_over_legacy_artifacts(tmp_path) -> None:
    model_id = "Workspace/Model/1"
    v2_cache_dir = tmp_path / slugify_model_id_to_cache_key(model_id)
    legacy_cache_dir = tmp_path / "Workspace" / "Model" / "1"
    for cache_dir in (v2_cache_dir, legacy_cache_dir):
        cache_dir.mkdir(parents=True)
        (cache_dir / "model_type.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "project_task_type": "object-detection",
                    "model_type": "yolov8n",
                }
            )
        )

    assert get_cache_dir_for_read(
        model_id=model_id,
        cache_dir_root=str(tmp_path),
    ) == str(v2_cache_dir)


@mock.patch.object(model_artifacts, "OFFLINE_MODE", True)
def test_offline_legacy_fallback_is_not_used_by_writes_or_cleanup(tmp_path) -> None:
    model_id = "Workspace/Model/1"
    legacy_cache_dir = tmp_path / "Workspace" / "Model" / "1"
    legacy_cache_dir.mkdir(parents=True)
    (legacy_cache_dir / "model_type.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "project_task_type": "object-detection",
                "model_type": "yolov8n",
            }
        )
    )
    legacy_artifact = legacy_cache_dir / "weights.bin"
    legacy_artifact.write_bytes(b"legacy-weights")
    v2_cache_dir = tmp_path / slugify_model_id_to_cache_key(model_id)

    with mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", str(tmp_path)):
        initialise_cache(model_id=model_id)
        assert not v2_cache_dir.exists()

        save_bytes_in_cache(
            content=b"new-v2-weights",
            file="weights.bin",
            model_id=model_id,
        )
        assert (v2_cache_dir / "weights.bin").read_bytes() == b"new-v2-weights"
        assert legacy_artifact.read_bytes() == b"legacy-weights"

        clear_cache(model_id=model_id)

    assert not v2_cache_dir.exists()
    assert legacy_artifact.read_bytes() == b"legacy-weights"


@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_windows_separator_form_cannot_alias_canonical_model_id() -> None:
    canonical_path = get_cache_dir(model_id="victim/1")
    windows_separator_path = get_cache_dir(model_id=r"victim\1")

    assert canonical_path == "/some/cache/victim/1"
    assert windows_separator_path != canonical_path
    assert os.path.basename(windows_separator_path).startswith(
        model_artifacts.MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX
    )


@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_legacy_v1_slug_shape_is_reserved_from_raw_model_ids() -> None:
    legacy_looking_model_id = "ordinary-model-deadbeef"

    result = get_cache_dir(model_id=legacy_looking_model_id)

    assert result != os.path.join("/some/cache", legacy_looking_model_id)
    assert os.path.basename(result).startswith(
        model_artifacts.MODEL_ID_CACHE_SLUG_NAMESPACE_PREFIX
    )


@mock.patch.object(model_artifacts, "MODEL_CACHE_DIR", "/some/cache")
def test_get_cache_dir_when_model_id_not_given() -> None:
    # when
    result = get_cache_dir()

    # then
    assert result == "/some/cache"


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_clear_cache(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    touch(os.path.join(cache_dir, "file.txt"))
    touch(os.path.join(cache_dir, "other", "file.txt"))
    touch(os.path.join(empty_local_dir, "some", "1", "file.txt"))
    # when
    clear_cache(model_id="some/2")

    # then
    get_cache_dir_mock.assert_called_once_with(model_id="some/2")
    assert os.listdir(empty_local_dir) == ["some"]
    assert os.listdir(os.path.join(empty_local_dir, "some")) == ["1"]


@mock.patch.object(model_artifacts, "get_cache_dir")
def test_clear_cache_when_nothing_to_delete(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    touch(os.path.join(empty_local_dir, "some", "1", "file.txt"))
    # when
    clear_cache(model_id="some/2")

    # then
    get_cache_dir_mock.assert_called_once_with(model_id="some/2")
    assert os.listdir(empty_local_dir) == ["some"]
    assert os.listdir(os.path.join(empty_local_dir, "some")) == ["1"]


# Tests for atomic cache writes feature


@mock.patch.object(model_artifacts, "ATOMIC_CACHE_WRITES_ENABLED", True)
@mock.patch.object(model_artifacts, "get_cache_dir")
@mock.patch.object(model_artifacts, "dump_bytes_atomic")
@mock.patch.object(model_artifacts, "dump_bytes")
def test_save_bytes_in_cache_uses_atomic_when_enabled(
    dump_bytes_mock: MagicMock,
    dump_bytes_atomic_mock: MagicMock,
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    content = b"test content"
    expected_path = os.path.join(cache_dir, "file.dat")

    # when
    save_bytes_in_cache(content=content, file="file.dat", model_id="some/2")

    # then
    dump_bytes_atomic_mock.assert_called_once_with(
        path=expected_path, content=content, allow_override=True
    )
    dump_bytes_mock.assert_not_called()


@mock.patch.object(model_artifacts, "ATOMIC_CACHE_WRITES_ENABLED", False)
@mock.patch.object(model_artifacts, "get_cache_dir")
@mock.patch.object(model_artifacts, "dump_bytes_atomic")
@mock.patch.object(model_artifacts, "dump_bytes")
def test_save_bytes_in_cache_uses_regular_when_disabled(
    dump_bytes_mock: MagicMock,
    dump_bytes_atomic_mock: MagicMock,
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    content = b"test content"
    expected_path = os.path.join(cache_dir, "file.dat")

    # when
    save_bytes_in_cache(content=content, file="file.dat", model_id="some/2")

    # then
    dump_bytes_mock.assert_called_once_with(
        path=expected_path, content=content, allow_override=True
    )
    dump_bytes_atomic_mock.assert_not_called()


@mock.patch.object(model_artifacts, "ATOMIC_CACHE_WRITES_ENABLED", True)
@mock.patch.object(model_artifacts, "get_cache_dir")
@mock.patch.object(model_artifacts, "dump_json_atomic")
@mock.patch.object(model_artifacts, "dump_json")
def test_save_json_in_cache_uses_atomic_when_enabled(
    dump_json_mock: MagicMock,
    dump_json_atomic_mock: MagicMock,
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    content = {"key": "value"}
    expected_path = os.path.join(cache_dir, "file.json")

    # when
    save_json_in_cache(content=content, file="file.json", model_id="some/2", indent=2)

    # then
    dump_json_atomic_mock.assert_called_once_with(
        path=expected_path, content=content, allow_override=True, indent=2
    )
    dump_json_mock.assert_not_called()


@mock.patch.object(model_artifacts, "ATOMIC_CACHE_WRITES_ENABLED", False)
@mock.patch.object(model_artifacts, "get_cache_dir")
@mock.patch.object(model_artifacts, "dump_json_atomic")
@mock.patch.object(model_artifacts, "dump_json")
def test_save_json_in_cache_uses_regular_when_disabled(
    dump_json_mock: MagicMock,
    dump_json_atomic_mock: MagicMock,
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    content = {"key": "value"}
    expected_path = os.path.join(cache_dir, "file.json")

    # when
    save_json_in_cache(content=content, file="file.json", model_id="some/2", indent=2)

    # then
    dump_json_mock.assert_called_once_with(
        path=expected_path, content=content, allow_override=True, indent=2
    )
    dump_json_atomic_mock.assert_not_called()


@mock.patch.object(model_artifacts, "ATOMIC_CACHE_WRITES_ENABLED", True)
@mock.patch.object(model_artifacts, "get_cache_dir")
@mock.patch.object(model_artifacts, "dump_text_lines_atomic")
@mock.patch.object(model_artifacts, "dump_text_lines")
def test_save_text_lines_in_cache_uses_atomic_when_enabled(
    dump_text_lines_mock: MagicMock,
    dump_text_lines_atomic_mock: MagicMock,
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    content = ["line1", "line2"]
    expected_path = os.path.join(cache_dir, "file.txt")

    # when
    save_text_lines_in_cache(content=content, file="file.txt", model_id="some/2")

    # then
    dump_text_lines_atomic_mock.assert_called_once_with(
        path=expected_path, content=content, allow_override=True
    )
    dump_text_lines_mock.assert_not_called()


@mock.patch.object(model_artifacts, "ATOMIC_CACHE_WRITES_ENABLED", False)
@mock.patch.object(model_artifacts, "get_cache_dir")
@mock.patch.object(model_artifacts, "dump_text_lines_atomic")
@mock.patch.object(model_artifacts, "dump_text_lines")
def test_save_text_lines_in_cache_uses_regular_when_disabled(
    dump_text_lines_mock: MagicMock,
    dump_text_lines_atomic_mock: MagicMock,
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    content = ["line1", "line2"]
    expected_path = os.path.join(cache_dir, "file.txt")

    # when
    save_text_lines_in_cache(content=content, file="file.txt", model_id="some/2")

    # then
    dump_text_lines_mock.assert_called_once_with(
        path=expected_path, content=content, allow_override=True
    )
    dump_text_lines_atomic_mock.assert_not_called()


# Integration test with actual atomic writes
@mock.patch.object(model_artifacts, "ATOMIC_CACHE_WRITES_ENABLED", True)
@mock.patch.object(model_artifacts, "get_cache_dir")
def test_save_json_in_cache_atomic_integration(
    get_cache_dir_mock: MagicMock,
    empty_local_dir: str,
) -> None:
    # given
    cache_dir = os.path.join(empty_local_dir, "some", "2")
    get_cache_dir_mock.return_value = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    content = {"test": "data", "number": 42}

    # when
    save_json_in_cache(
        content=content,
        file="test.json",
        model_id="some/2",
        indent=2,
    )

    # then
    expected_file = os.path.join(cache_dir, "test.json")
    assert os.path.exists(expected_file)
    with open(expected_file) as f:
        loaded = json.load(f)
    assert loaded == content
    # Verify no temp files remain
    temp_files = [f for f in os.listdir(cache_dir) if f.startswith(".tmp_")]
    assert len(temp_files) == 0
