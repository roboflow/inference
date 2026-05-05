import os
import time
from datetime import datetime, timedelta
from typing import Optional, Set

import pytest

from inference.core.managers.inference_models_cache_watchdog import (
    MODELS_CACHE_DIR,
    SHARED_BLOBS_DIR,
    FileInfo,
    list_files,
    nominate_files_for_deletion,
    purge_files,
    purge_inference_models_cache,
    rank_for_deletion,
    summarize_disk_size,
)


def test_purge_files_returns_zero_for_empty_list() -> None:
    result = purge_files(files=[])
    assert result == 0.0


def test_purge_files_deletes_single_file_and_returns_its_size(
    empty_local_dir: str,
) -> None:
    # given
    file = _create_file(empty_local_dir, "model.bin", size_mb=1.0)

    # when
    result = purge_files(files=[file])

    # then
    assert result == 1.0
    assert not os.path.exists(file.path)


def test_purge_files_deletes_multiple_files_and_returns_total_size(
    empty_local_dir: str,
) -> None:
    # given
    files = [
        _create_file(empty_local_dir, "a.bin", size_mb=0.5),
        _create_file(empty_local_dir, "b.bin", size_mb=1.0),
        _create_file(empty_local_dir, "c.bin", size_mb=2.0),
    ]

    # when
    result = purge_files(files=files)

    # then
    assert result == 3.5
    for file in files:
        assert not os.path.exists(file.path)


def test_purge_files_silently_skips_missing_file(
    empty_local_dir: str,
) -> None:
    # given
    existing = _create_file(empty_local_dir, "exists.bin", size_mb=1.0)
    missing = FileInfo(
        path=os.path.join(empty_local_dir, "gone.bin"),
        size_mb=5.0,
        modified_at=datetime.now(),
    )

    # when
    result = purge_files(files=[missing, existing])

    # then
    assert result == 1.0
    assert not os.path.exists(existing.path)


def test_purge_files_handles_directory_permission_error_without_crashing(
    empty_local_dir: str,
) -> None:
    # given
    files = [
        _create_file(empty_local_dir, "a.bin", size_mb=1.0),
        _create_file(empty_local_dir, "b.bin", size_mb=2.0),
    ]
    os.chmod(empty_local_dir, 0o555)

    # when
    try:
        result = purge_files(files=files)
    finally:
        os.chmod(empty_local_dir, 0o755)

    # then
    assert result == 0.0
    for file in files:
        assert os.path.exists(file.path)


def test_purge_files_creates_lock_file_adjacent_to_target(
    empty_local_dir: str,
) -> None:
    # given
    file = _create_file(empty_local_dir, "model.bin", size_mb=1.0)

    # when
    purge_files(files=[file])

    # then
    assert not os.path.exists(file.path)


def test_purge_files_deletes_file_in_nested_subdirectory(
    empty_local_dir: str,
) -> None:
    # given
    subdir = os.path.join(empty_local_dir, "models", "v1")
    os.makedirs(subdir)
    file = _create_file(subdir, "weights.bin", size_mb=2.0)

    # when
    result = purge_files(files=[file])

    # then
    assert result == 2.0
    assert not os.path.exists(file.path)


def test_purge_files_skips_file_when_lock_cannot_be_acquired(
    empty_local_dir: str,
) -> None:
    # given
    from filelock import FileLock

    file = _create_file(empty_local_dir, "locked.bin", size_mb=5.0)
    lock_path = os.path.join(empty_local_dir, f".locked.bin.lock")

    # when
    with FileLock(lock_path, timeout=1):
        result = purge_files(files=[file], file_lock_acquire_timeout=0)

    # then
    assert result == 0.0
    assert os.path.exists(file.path)


def test_purge_files_continues_processing_after_failure(
    empty_local_dir: str,
) -> None:
    # given
    missing = FileInfo(
        path=os.path.join(empty_local_dir, "missing.bin"),
        size_mb=3.0,
        modified_at=datetime.now(),
    )
    ok = _create_file(empty_local_dir, "ok.bin", size_mb=2.0)

    # when
    result = purge_files(files=[missing, ok])

    # then
    assert result == 2.0
    assert not os.path.exists(ok.path)


def test_nominate_returns_empty_list_when_no_files_given() -> None:
    result = nominate_files_for_deletion(files=[], to_be_reclaimed=100.0)
    assert result == []


def test_nominate_returns_empty_list_when_nothing_to_reclaim() -> None:
    files = [_create_file_info("a.bin", 10.0), _create_file_info("b.bin", 20.0)]
    result = nominate_files_for_deletion(files=files, to_be_reclaimed=0.0)
    assert result == []


def test_nominate_returns_single_file_when_it_covers_target() -> None:
    files = [_create_file_info("big.bin", 50.0), _create_file_info("small.bin", 1.0)]
    result = nominate_files_for_deletion(files=files, to_be_reclaimed=30.0)
    assert len(result) == 1
    assert result[0].path == "/fake/big.bin"


def test_nominate_accumulates_files_until_target_is_reached() -> None:
    files = [
        _create_file_info("a.bin", 3.0),
        _create_file_info("b.bin", 4.0),
        _create_file_info("c.bin", 5.0),
    ]
    result = nominate_files_for_deletion(files=files, to_be_reclaimed=6.0)
    assert len(result) == 2
    assert result[0].path == "/fake/a.bin"
    assert result[1].path == "/fake/b.bin"


def test_nominate_returns_all_files_when_total_size_below_target() -> None:
    files = [_create_file_info("a.bin", 1.0), _create_file_info("b.bin", 2.0)]
    result = nominate_files_for_deletion(files=files, to_be_reclaimed=100.0)
    assert len(result) == 2


def test_nominate_stops_exactly_when_target_is_met() -> None:
    files = [
        _create_file_info("a.bin", 5.0),
        _create_file_info("b.bin", 5.0),
        _create_file_info("c.bin", 5.0),
    ]
    result = nominate_files_for_deletion(files=files, to_be_reclaimed=10.0)
    assert len(result) == 2


def test_nominate_preserves_input_order() -> None:
    files = [
        _create_file_info("first.bin", 1.0),
        _create_file_info("second.bin", 1.0),
        _create_file_info("third.bin", 1.0),
    ]
    result = nominate_files_for_deletion(files=files, to_be_reclaimed=2.0)
    assert [f.path for f in result] == ["/fake/first.bin", "/fake/second.bin"]


def test_nominate_handles_fractional_sizes() -> None:
    files = [
        _create_file_info("a.bin", 0.3),
        _create_file_info("b.bin", 0.3),
        _create_file_info("c.bin", 0.3),
        _create_file_info("d.bin", 0.3),
    ]
    result = nominate_files_for_deletion(files=files, to_be_reclaimed=0.5)
    assert len(result) == 2


def test_nominate_includes_file_that_pushes_past_target() -> None:
    files = [
        _create_file_info("a.bin", 1.0),
        _create_file_info("b.bin", 100.0),
        _create_file_info("c.bin", 1.0),
    ]
    result = nominate_files_for_deletion(files=files, to_be_reclaimed=5.0)
    assert len(result) == 2
    assert result[1].path == "/fake/b.bin"


def test_rank_returns_empty_list_for_empty_input() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    result = rank_for_deletion(files=[], now=now)
    assert result == []


def test_rank_raises_when_thresholds_not_ascending() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    files = [_create_file_info("a.bin", 1.0, modified_at=_days_ago(now, 5))]
    with pytest.raises(ValueError, match="ascending order"):
        rank_for_deletion(
            files=files,
            now=now,
            recent_threshold_days=10,
            warm_threshold_days=5,
            stale_threshold_days=30,
        )


def test_rank_raises_when_warm_equals_stale() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    files = [_create_file_info("a.bin", 1.0, modified_at=_days_ago(now, 5))]
    with pytest.raises(ValueError, match="ascending order"):
        rank_for_deletion(
            files=files,
            now=now,
            recent_threshold_days=1,
            warm_threshold_days=7,
            stale_threshold_days=7,
        )


def test_abandoned_files_ranked_before_stale() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    stale = _create_file_info("stale.bin", 100.0, modified_at=_days_ago(now, 15))
    abandoned = _create_file_info("abandoned.bin", 1.0, modified_at=_days_ago(now, 60))
    result = rank_for_deletion(files=[stale, abandoned], now=now)
    assert result[0].path == "/fake/abandoned.bin"
    assert result[1].path == "/fake/stale.bin"


def test_stale_files_ranked_before_warm() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    warm = _create_file_info("warm.bin", 100.0, modified_at=_days_ago(now, 3))
    stale = _create_file_info("stale.bin", 1.0, modified_at=_days_ago(now, 15))
    result = rank_for_deletion(files=[warm, stale], now=now)
    assert result[0].path == "/fake/stale.bin"
    assert result[1].path == "/fake/warm.bin"


def test_warm_files_ranked_before_recent() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    recent = _create_file_info("recent.bin", 100.0, modified_at=_days_ago(now, 0.5))
    warm = _create_file_info("warm.bin", 1.0, modified_at=_days_ago(now, 3))
    result = rank_for_deletion(files=[recent, warm], now=now)
    assert result[0].path == "/fake/warm.bin"
    assert result[1].path == "/fake/recent.bin"


def test_full_group_ordering() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    recent = _create_file_info("recent.bin", 1.0, modified_at=_days_ago(now, 0.5))
    warm = _create_file_info("warm.bin", 1.0, modified_at=_days_ago(now, 3))
    stale = _create_file_info("stale.bin", 1.0, modified_at=_days_ago(now, 15))
    abandoned = _create_file_info("abandoned.bin", 1.0, modified_at=_days_ago(now, 60))
    result = rank_for_deletion(files=[recent, warm, stale, abandoned], now=now)
    assert [f.path for f in result] == [
        "/fake/abandoned.bin",
        "/fake/stale.bin",
        "/fake/warm.bin",
        "/fake/recent.bin",
    ]


def test_bigger_files_ranked_first_within_same_group() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    small = _create_file_info("small.bin", 1.0, modified_at=_days_ago(now, 3))
    medium = _create_file_info("medium.bin", 10.0, modified_at=_days_ago(now, 3))
    large = _create_file_info("large.bin", 50.0, modified_at=_days_ago(now, 3))
    result = rank_for_deletion(files=[small, large, medium], now=now)
    assert [f.path for f in result] == [
        "/fake/large.bin",
        "/fake/medium.bin",
        "/fake/small.bin",
    ]


def test_size_ordering_within_each_group() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    abandoned_small = _create_file_info("ab_s.bin", 2.0, modified_at=_days_ago(now, 60))
    abandoned_large = _create_file_info(
        "ab_l.bin", 20.0, modified_at=_days_ago(now, 60)
    )
    recent_small = _create_file_info("rc_s.bin", 1.0, modified_at=_days_ago(now, 0.5))
    recent_large = _create_file_info("rc_l.bin", 50.0, modified_at=_days_ago(now, 0.5))
    result = rank_for_deletion(
        files=[recent_large, abandoned_small, recent_small, abandoned_large],
        now=now,
    )
    assert [f.path for f in result] == [
        "/fake/ab_l.bin",
        "/fake/ab_s.bin",
        "/fake/rc_l.bin",
        "/fake/rc_s.bin",
    ]


def test_group_takes_precedence_over_size() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    huge_recent = _create_file_info(
        "huge_recent.bin", 999.0, modified_at=_days_ago(now, 0.5)
    )
    tiny_abandoned = _create_file_info(
        "tiny_abandoned.bin", 0.001, modified_at=_days_ago(now, 60)
    )
    result = rank_for_deletion(files=[huge_recent, tiny_abandoned], now=now)
    assert result[0].path == "/fake/tiny_abandoned.bin"
    assert result[1].path == "/fake/huge_recent.bin"


def test_files_on_group_boundary_fall_into_correct_group() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    at_recent = _create_file_info("at_recent.bin", 1.0, modified_at=_days_ago(now, 1.0))
    just_over_recent = _create_file_info(
        "over_recent.bin", 1.0, modified_at=_days_ago(now, 1.001)
    )
    result = rank_for_deletion(files=[at_recent, just_over_recent], now=now)
    assert result[0].path == "/fake/over_recent.bin"
    assert result[1].path == "/fake/at_recent.bin"


def test_equal_size_within_group_produces_stable_output() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    a = _create_file_info("a.bin", 5.0, modified_at=_days_ago(now, 3))
    b = _create_file_info("b.bin", 5.0, modified_at=_days_ago(now, 3))
    result = rank_for_deletion(files=[a, b], now=now)
    assert result[0].path == "/fake/a.bin"
    assert result[1].path == "/fake/b.bin"


def test_custom_thresholds_are_respected() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    file_2d = _create_file_info("2d.bin", 1.0, modified_at=_days_ago(now, 2))
    file_6d = _create_file_info("6d.bin", 1.0, modified_at=_days_ago(now, 6))
    result = rank_for_deletion(
        files=[file_2d, file_6d],
        now=now,
        recent_threshold_days=1,
        warm_threshold_days=3,
        stale_threshold_days=10,
    )
    assert result[0].path == "/fake/6d.bin"
    assert result[1].path == "/fake/2d.bin"


def test_single_file_returns_that_file() -> None:
    now = datetime(2026, 3, 19, 12, 0, 0)
    file = _create_file_info("only.bin", 5.0, modified_at=_days_ago(now, 10))
    result = rank_for_deletion(files=[file], now=now)
    assert len(result) == 1
    assert result[0].path == "/fake/only.bin"


def test_summarize_returns_zero_for_empty_list() -> None:
    result = summarize_disk_size(files_info=[])
    assert result == 0.0


def test_summarize_returns_size_of_single_file() -> None:
    files = [_create_file_info("a.bin", size_mb=5.0)]
    result = summarize_disk_size(files_info=files)
    assert result == 5.0


def test_summarize_returns_total_of_multiple_files() -> None:
    files = [
        _create_file_info("a.bin", size_mb=1.5),
        _create_file_info("b.bin", size_mb=2.5),
        _create_file_info("c.bin", size_mb=6.0),
    ]
    result = summarize_disk_size(files_info=files)
    assert result == 10.0


def test_summarize_handles_fractional_sizes() -> None:
    files = [
        _create_file_info("a.bin", size_mb=0.1),
        _create_file_info("b.bin", size_mb=0.2),
        _create_file_info("c.bin", size_mb=0.3),
    ]
    result = summarize_disk_size(files_info=files)
    assert abs(result - 0.6) < 1e-9


def test_summarize_handles_zero_size_files() -> None:
    files = [
        _create_file_info("empty.bin", size_mb=0.0),
        _create_file_info("real.bin", size_mb=3.0),
    ]
    result = summarize_disk_size(files_info=files)
    assert result == 3.0


def test_list_files_returns_empty_for_nonexistent_path() -> None:
    result = list_files("/does/not/exist")
    assert result == []


def test_list_files_returns_single_file_when_path_is_file(
    empty_local_dir: str,
) -> None:
    file = _create_file(empty_local_dir, "model.bin", size_mb=2.0)

    result = list_files(file.path)

    assert len(result) == 1
    assert result[0].path == os.path.abspath(file.path)
    assert result[0].size_mb == 2.0


def test_list_files_returns_empty_for_empty_directory(
    empty_local_dir: str,
) -> None:
    result = list_files(empty_local_dir)
    assert result == []


def test_list_files_returns_all_files_in_flat_directory(
    empty_local_dir: str,
) -> None:
    _create_file(empty_local_dir, "a.bin", size_mb=0.001)
    _create_file(empty_local_dir, "b.bin", size_mb=0.002)

    result = list_files(empty_local_dir)

    assert len(result) == 2
    assert _names(result) == {"a.bin", "b.bin"}


def test_list_files_traverses_nested_directories(
    empty_local_dir: str,
) -> None:
    level1 = os.path.join(empty_local_dir, "models")
    level2 = os.path.join(level1, "v1")
    level3 = os.path.join(level2, "weights")
    os.makedirs(level3)
    root = _create_file(empty_local_dir, "root.bin", size_mb=0.001)
    l1 = _create_file(level1, "l1.bin", size_mb=0.001)
    l2 = _create_file(level2, "l2.bin", size_mb=0.001)
    l3 = _create_file(level3, "l3.bin", size_mb=0.001)

    result = list_files(empty_local_dir)

    assert len(result) == 4
    assert _paths(result) == {
        os.path.abspath(root.path),
        os.path.abspath(l1.path),
        os.path.abspath(l2.path),
        os.path.abspath(l3.path),
    }


def test_list_files_includes_hidden_files(
    empty_local_dir: str,
) -> None:
    hidden_dir = os.path.join(empty_local_dir, ".hidden_dir")
    os.makedirs(hidden_dir)
    _create_file(empty_local_dir, ".hidden_config", size_mb=0.001)
    _create_file(empty_local_dir, "visible.bin", size_mb=0.001)
    _create_file(hidden_dir, ".deeply_hidden", size_mb=0.001)

    result = list_files(empty_local_dir)

    assert _names(result) == {"visible.bin", ".hidden_config", ".deeply_hidden"}


def test_list_files_excludes_lock_files_in_directory(
    empty_local_dir: str,
) -> None:
    _create_file(empty_local_dir, "model.bin", size_mb=0.001)
    _create_file(empty_local_dir, f"model.bin.lock", size_mb=0.001)
    _create_file(empty_local_dir, "other.lock", size_mb=0.001)

    result = list_files(empty_local_dir)

    assert len(result) == 1
    assert _names(result) == {"model.bin"}


def test_list_files_returns_empty_when_path_is_lock_file(
    empty_local_dir: str,
) -> None:
    lock = _create_file(empty_local_dir, f"model.lock", size_mb=0.001)

    result = list_files(lock.path)

    assert result == []


def test_list_files_skips_symlinked_file_in_directory(
    empty_local_dir: str,
) -> None:
    real = _create_file(empty_local_dir, "real.bin", size_mb=0.001)
    os.symlink(real.path, os.path.join(empty_local_dir, "link.bin"))

    result = list_files(empty_local_dir)

    assert len(result) == 1
    assert _names(result) == {"real.bin"}


def test_list_files_returns_empty_when_path_is_symlink(
    empty_local_dir: str,
) -> None:
    real_dir = os.path.join(empty_local_dir, "real")
    os.makedirs(real_dir)
    _create_file(real_dir, "file.bin", size_mb=0.001)
    link_path = os.path.join(empty_local_dir, "link_to_dir")
    os.symlink(real_dir, link_path)

    result = list_files(link_path)

    assert result == []


def test_list_files_does_not_follow_symlinked_directories(
    empty_local_dir: str,
) -> None:
    real_subdir = os.path.join(empty_local_dir, "real_sub")
    os.makedirs(real_subdir)
    _create_file(real_subdir, "nested.bin", size_mb=0.001)
    os.symlink(real_subdir, os.path.join(empty_local_dir, "linked_sub"))
    _create_file(empty_local_dir, "root.bin", size_mb=0.001)

    result = list_files(empty_local_dir)

    assert _names(result) == {"root.bin", "nested.bin"}


def test_list_files_skips_files_in_permission_restricted_directory(
    empty_local_dir: str,
) -> None:
    accessible_dir = os.path.join(empty_local_dir, "accessible")
    restricted_dir = os.path.join(empty_local_dir, "restricted")
    os.makedirs(accessible_dir)
    os.makedirs(restricted_dir)
    _create_file(accessible_dir, "ok.bin", size_mb=0.001)
    _create_file(restricted_dir, "secret.bin", size_mb=0.001)
    os.chmod(restricted_dir, 0o000)

    try:
        result = list_files(empty_local_dir)
    finally:
        os.chmod(restricted_dir, 0o755)

    assert "ok.bin" in _names(result)
    assert "secret.bin" not in _names(result)


def test_list_files_reports_correct_size(
    empty_local_dir: str,
) -> None:
    _create_file(empty_local_dir, "big.bin", size_mb=3.0)

    result = list_files(empty_local_dir)

    assert len(result) == 1
    assert result[0].size_mb == 3.0


def test_list_files_reports_modification_time(
    empty_local_dir: str,
) -> None:
    file = _create_file(empty_local_dir, "file.bin", size_mb=0.001)

    result = list_files(empty_local_dir)

    assert len(result) == 1
    stat_mtime = os.stat(file.path).st_mtime
    assert abs(result[0].modified_at.timestamp() - stat_mtime) < 1e-3


def test_list_files_handles_multiple_sibling_directories(
    empty_local_dir: str,
) -> None:
    for name in ["alpha", "beta", "gamma"]:
        subdir = os.path.join(empty_local_dir, name)
        os.makedirs(subdir)
        _create_file(subdir, f"{name}.bin", size_mb=0.001)

    result = list_files(empty_local_dir)

    assert len(result) == 3
    assert _names(result) == {"alpha.bin", "beta.bin", "gamma.bin"}


def test_list_files_mixed_real_symlink_lock_and_hidden(
    empty_local_dir: str,
) -> None:
    subdir = os.path.join(empty_local_dir, "sub")
    os.makedirs(subdir)
    nested = _create_file(subdir, "nested.bin", size_mb=0.001)
    _create_file(empty_local_dir, "real.bin", size_mb=0.001)
    _create_file(empty_local_dir, ".hidden.bin", size_mb=0.001)
    _create_file(empty_local_dir, f"cache.lock", size_mb=0.001)
    os.symlink(nested.path, os.path.join(empty_local_dir, "link.bin"))

    result = list_files(empty_local_dir)

    assert _names(result) == {"real.bin", ".hidden.bin", "nested.bin"}


def test_purge_skips_when_cache_is_under_limit(
    empty_local_dir: str,
) -> None:
    # given
    shared, models = _setup_inference_home(empty_local_dir)
    _create_file(shared, "blob_a.bin", size_mb=1.0)
    _create_file(models, "model_a.bin", size_mb=1.0)

    # when
    purge_inference_models_cache(
        inference_home=empty_local_dir,
        max_cache_size_mb=10,
    )

    # then — nothing deleted
    assert os.path.exists(os.path.join(shared, "blob_a.bin"))
    assert os.path.exists(os.path.join(models, "model_a.bin"))


def test_purge_skips_when_cache_exactly_at_limit(
    empty_local_dir: str,
) -> None:
    # given
    shared, models = _setup_inference_home(empty_local_dir)
    _create_file(shared, "blob.bin", size_mb=3.0)
    _create_file(models, "model.bin", size_mb=2.0)

    # when
    purge_inference_models_cache(
        inference_home=empty_local_dir,
        max_cache_size_mb=5,
    )

    # then — nothing deleted
    assert os.path.exists(os.path.join(shared, "blob.bin"))
    assert os.path.exists(os.path.join(models, "model.bin"))


def test_purge_deletes_oldest_file_first(
    empty_local_dir: str,
) -> None:
    # given — 3MB total, limit 2MB, need to reclaim 1MB
    shared, models = _setup_inference_home(empty_local_dir)
    old = _create_file(shared, "old.bin", size_mb=1.0)
    new = _create_file(models, "new.bin", size_mb=2.0)
    _touch_with_age(old.path, days_old=60)
    _touch_with_age(new.path, days_old=0.5)

    # when
    purge_inference_models_cache(
        inference_home=empty_local_dir,
        max_cache_size_mb=2,
    )

    # then — old file deleted, new file kept
    assert not os.path.exists(old.path)
    assert os.path.exists(new.path)


def test_purge_deletes_bigger_file_first_within_same_staleness(
    empty_local_dir: str,
) -> None:
    # given — 6MB total, limit 4MB, need 2MB, all same age group
    shared, models = _setup_inference_home(empty_local_dir)
    small = _create_file(shared, "small.bin", size_mb=1.0)
    medium = _create_file(shared, "medium.bin", size_mb=2.0)
    big = _create_file(models, "big.bin", size_mb=3.0)
    for f in [small, medium, big]:
        _touch_with_age(f.path, days_old=3)

    # when
    purge_inference_models_cache(
        inference_home=empty_local_dir,
        max_cache_size_mb=4,
    )

    # then — big file deleted (3MB > 2MB needed), others kept
    assert not os.path.exists(big.path)
    assert os.path.exists(small.path)
    assert os.path.exists(medium.path)


def test_purge_deletes_across_both_directories(
    empty_local_dir: str,
) -> None:
    # given — 4MB total, limit 1MB, need 3MB
    shared, models = _setup_inference_home(empty_local_dir)
    blob = _create_file(shared, "blob.bin", size_mb=2.0)
    model = _create_file(models, "model.bin", size_mb=2.0)
    _touch_with_age(blob.path, days_old=60)
    _touch_with_age(model.path, days_old=60)

    # when
    purge_inference_models_cache(
        inference_home=empty_local_dir,
        max_cache_size_mb=1,
    )

    # then — both deleted
    assert not os.path.exists(blob.path)
    assert not os.path.exists(model.path)


def test_purge_deletes_only_enough_to_meet_budget(
    empty_local_dir: str,
) -> None:
    # given — 5MB total, limit 3MB, need 2MB
    # abandoned 2MB file should be enough
    shared, models = _setup_inference_home(empty_local_dir)
    abandoned = _create_file(shared, "abandoned.bin", size_mb=2.0)
    stale = _create_file(shared, "stale.bin", size_mb=1.0)
    recent = _create_file(models, "recent.bin", size_mb=2.0)
    _touch_with_age(abandoned.path, days_old=60)
    _touch_with_age(stale.path, days_old=15)
    _touch_with_age(recent.path, days_old=0.5)

    # when
    purge_inference_models_cache(
        inference_home=empty_local_dir,
        max_cache_size_mb=3,
    )

    # then — only abandoned deleted
    assert not os.path.exists(abandoned.path)
    assert os.path.exists(stale.path)
    assert os.path.exists(recent.path)


def test_purge_handles_missing_cache_directories(
    empty_local_dir: str,
) -> None:
    # given — no shared-blobs or models-cache created

    # when / then — no crash
    purge_inference_models_cache(
        inference_home=empty_local_dir,
        max_cache_size_mb=10,
    )


def test_purge_handles_only_one_directory_existing(
    empty_local_dir: str,
) -> None:
    # given — only models-cache exists, over budget
    models = os.path.join(empty_local_dir, MODELS_CACHE_DIR)
    os.makedirs(models)
    old = _create_file(models, "old.bin", size_mb=3.0)
    _touch_with_age(old.path, days_old=60)

    # when
    purge_inference_models_cache(
        inference_home=empty_local_dir,
        max_cache_size_mb=1,
    )

    # then
    assert not os.path.exists(old.path)


def test_purge_preserves_files_in_nested_subdirectories_when_under_budget(
    empty_local_dir: str,
) -> None:
    # given
    shared, models = _setup_inference_home(empty_local_dir)
    subdir = os.path.join(models, "rfdetr", "v1")
    os.makedirs(subdir)
    _create_file(subdir, "weights.bin", size_mb=1.0)
    _create_file(shared, "blob.bin", size_mb=1.0)

    # when
    purge_inference_models_cache(
        inference_home=empty_local_dir,
        max_cache_size_mb=10,
    )

    # then
    assert os.path.exists(os.path.join(subdir, "weights.bin"))
    assert os.path.exists(os.path.join(shared, "blob.bin"))


def test_purge_deletes_files_from_nested_subdirectories_when_over_budget(
    empty_local_dir: str,
) -> None:
    # given — 3MB total, limit 1MB
    shared, models = _setup_inference_home(empty_local_dir)
    subdir = os.path.join(models, "yolov8", "v2")
    os.makedirs(subdir)
    nested = _create_file(subdir, "weights.bin", size_mb=2.0)
    root = _create_file(shared, "blob.bin", size_mb=1.0)
    _touch_with_age(nested.path, days_old=60)
    _touch_with_age(root.path, days_old=0.5)

    # when
    purge_inference_models_cache(
        inference_home=empty_local_dir,
        max_cache_size_mb=1,
    )

    # then — oldest (nested) deleted first
    assert not os.path.exists(nested.path)
    assert os.path.exists(root.path)


def test_purge_with_undeletable_file_reclaims_partial_amount(
    empty_local_dir: str,
) -> None:
    # given — 4MB total, limit 1MB, need 3MB
    # protected file (2MB) can't be deleted, only deletable (2MB) is reclaimed
    shared, models = _setup_inference_home(empty_local_dir)
    protected_dir = os.path.join(shared, "protected")
    os.makedirs(protected_dir)
    protected = _create_file(protected_dir, "locked.bin", size_mb=2.0)
    deletable = _create_file(models, "deletable.bin", size_mb=2.0)
    _touch_with_age(protected.path, days_old=60)
    _touch_with_age(deletable.path, days_old=30)
    os.chmod(protected_dir, 0o555)

    # when
    try:
        purge_inference_models_cache(
            inference_home=empty_local_dir,
            max_cache_size_mb=1,
        )
    finally:
        os.chmod(protected_dir, 0o755)

    # then — deletable removed, protected survives
    assert not os.path.exists(deletable.path)
    assert os.path.exists(protected.path)


def _days_ago(from_date: datetime, days: float) -> datetime:
    return from_date - timedelta(days=days)


def _create_file_info(
    name: str, size_mb: float, modified_at: Optional[datetime] = None
) -> FileInfo:
    if modified_at is None:
        modified_at = datetime.now()
    return FileInfo(path=f"/fake/{name}", size_mb=size_mb, modified_at=modified_at)


def _create_file(directory: str, name: str, size_mb: float) -> FileInfo:
    path = os.path.join(directory, name)
    with open(path, "wb") as f:
        f.write(b"\x00" * int(size_mb * 1024 * 1024))
    modified_at = datetime.fromtimestamp(os.stat(path).st_mtime)
    return FileInfo(
        path=path,
        size_mb=size_mb,
        modified_at=modified_at,
    )


def _paths(result) -> Set[str]:
    return {f.path for f in result}


def _names(result) -> Set[str]:
    return {os.path.basename(f.path) for f in result}


def _setup_inference_home(base_dir: str):
    shared = os.path.join(base_dir, SHARED_BLOBS_DIR)
    models = os.path.join(base_dir, MODELS_CACHE_DIR)
    os.makedirs(shared, exist_ok=True)
    os.makedirs(models, exist_ok=True)
    return shared, models


def _touch_with_age(path: str, days_old: float) -> None:
    mtime = time.time() - (days_old * 86400)
    os.utime(path, (mtime, mtime))
