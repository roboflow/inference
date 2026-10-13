"""Unit tests of ``v2/local_file_sink@v1`` against a temporary directory."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest
from roboflow_workflows.v2.blocks.sinks.local_file import (
    FileSystemAccessError,
    LocalFileSinkBlock,
)

CSV_ROW_1 = "name,count\na,1\n"
CSV_ROW_2 = "name,count\nb,2"
SAVED = {"error_status": False, "message": "Data saved successfully"}


def _fixed_clock(tz: Any) -> datetime:
    return datetime(2026, 1, 2, 3, 4, 5, 678, tzinfo=timezone.utc)


def _block(**resources: Any) -> LocalFileSinkBlock:
    resources.setdefault("allow_access_to_file_system", True)
    block = LocalFileSinkBlock(**resources)

    return block


def _save(block: LocalFileSinkBlock, content: str, directory: Path, **params: Any):
    arguments: Dict[str, Any] = {
        "file_type": "csv",
        "output_mode": "append_log",
        "target_directory": str(directory),
        "file_name_prefix": "log",
        "max_entries_per_file": 1024,
    }
    arguments.update(params)
    result = block.run(content=content, **arguments)

    return result


def _texts(directory: Path, pattern: str = "*") -> List[str]:
    texts = [path.read_text() for path in sorted(directory.glob(pattern))]

    return texts


def test_append_log_writes_csv_header_once(tmp_path: Path) -> None:
    block = _block()

    results = [_save(block, row, tmp_path) for row in (CSV_ROW_1, CSV_ROW_2)]

    assert results == [SAVED, SAVED]
    assert _texts(tmp_path, "log_*.csv") == ["name,count\na,1\nb,2\n"]


def test_append_log_rotates_after_max_entries(tmp_path: Path) -> None:
    block = _block()

    for text in ("one", "two", "three"):
        _save(block, text, tmp_path, file_type="txt", max_entries_per_file=2)

    assert sorted(_texts(tmp_path)) == ["one\ntwo\n", "three\n"]


def test_append_log_json_becomes_one_line_per_entry(tmp_path: Path) -> None:
    block = _block()

    _save(block, '{\n  "a": 1\n}', tmp_path, file_type="json")
    _save(block, '{"a": 2}', tmp_path, file_type="json")

    assert _texts(tmp_path, "*.jsonl") == ['{"a": 1}\n{"a": 2}\n']


def test_append_log_reports_invalid_json_without_writing(tmp_path: Path) -> None:
    result = _save(_block(), "{not json", tmp_path, file_type="json")

    assert result == {"error_status": True, "message": "Invalid JSON content"}
    assert list(tmp_path.iterdir()) == []


def test_changed_target_starts_a_new_append_file(tmp_path: Path) -> None:
    block = _block()

    _save(block, "a", tmp_path / "x", file_type="txt")
    _save(block, "b", tmp_path / "y", file_type="txt")

    assert _texts(tmp_path / "x") == ["a\n"]
    assert _texts(tmp_path / "y") == ["b\n"]


def test_separate_files_never_overwrite_in_the_same_microsecond(tmp_path: Path) -> None:
    block = _block(wall_clock=_fixed_clock)

    for text in ("first", "second"):
        _save(block, text, tmp_path, output_mode="separate_files", file_type="txt")

    names = sorted(path.name for path in tmp_path.iterdir())
    stamp = _fixed_clock(None).astimezone().strftime("%Y_%m_%d_%H_%M_%S_%f")
    assert names == [f"log_{stamp}.txt", f"log_{stamp}_1.txt"]
    assert sorted(_texts(tmp_path)) == ["first\n", "second\n"]


def test_file_access_needs_explicit_host_permission(tmp_path: Path) -> None:
    block = LocalFileSinkBlock()

    with pytest.raises(FileSystemAccessError, match="allow_access_to_file_system"):
        _save(block, "a", tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_disabled_sinks_write_nothing(tmp_path: Path) -> None:
    result = _save(_block(disable_sinks=True), "a", tmp_path)

    assert result == {
        "error_status": False,
        "message": "Sink was disabled by workflow execution policy",
    }
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "target, prefix",
    [("../outside", "log"), (".", "../escaped"), ("link", "log")],
    ids=["parent-directory", "prefix-escape", "symlink-escape"],
)
def test_writes_stay_inside_the_allowed_directory(
    tmp_path: Path, target: str, prefix: str
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "link").symlink_to(tmp_path, target_is_directory=True)
    block = _block(allowed_write_directory=str(allowed))

    with pytest.raises(FileSystemAccessError, match="allowed write directory"):
        _save(block, "a", allowed / target, file_name_prefix=prefix)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["allowed"]


def test_subdirectory_of_the_allowed_directory_is_accepted(tmp_path: Path) -> None:
    block = _block(allowed_write_directory=str(tmp_path))

    result = _save(block, "a", tmp_path / "nested" / "deeper", file_type="txt")

    assert result == SAVED
    assert _texts(tmp_path / "nested" / "deeper") == ["a\n"]


@pytest.mark.parametrize("output_mode", ["append_log", "separate_files"])
def test_unwritable_target_is_reported_in_outputs(
    tmp_path: Path, output_mode: str
) -> None:
    occupied = tmp_path / "occupied"
    occupied.write_text("a file, not a directory")

    result = _save(_block(), "a", occupied, output_mode=output_mode)

    assert result["error_status"] is True
    assert result["message"]


@pytest.mark.parametrize("output_mode", ["append_log", "separate_files"])
def test_non_ascii_is_written_as_utf8(tmp_path: Path, output_mode: str) -> None:
    result = _save(
        _block(), "zażółć 🚗", tmp_path, file_type="txt", output_mode=output_mode
    )

    assert result == SAVED
    [path] = tmp_path.glob("log_*.txt")
    assert path.read_bytes() == "zażółć 🚗\n".encode("utf-8")


@pytest.mark.parametrize("output_mode", ["append_log", "separate_files"])
def test_unencodable_text_is_a_failed_write_not_a_crash(
    tmp_path: Path, output_mode: str
) -> None:
    # A lone surrogate, e.g. from a decoded file name, has no UTF-8 encoding.
    result = _save(
        _block(), "bad \udc80", tmp_path, file_type="txt", output_mode=output_mode
    )

    assert result["error_status"] is True
    assert result["message"]
