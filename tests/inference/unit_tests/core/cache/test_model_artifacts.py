import json
import os.path
from unittest import mock
from unittest.mock import MagicMock, call

from humanfriendly.testing import touch

from inference.core.cache import model_artifacts
from inference.core.cache.model_artifacts import (
    are_all_files_cached,
    clear_cache,
    get_cache_dir,
    get_cache_file_path,
    initialise_cache,
    is_file_cached,
    load_json_from_cache,
    load_text_file_from_cache,
    save_bytes_in_cache,
    save_json_in_cache,
    save_text_lines_in_cache,
)
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
