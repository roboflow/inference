from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from inference.core.managers.base import ModelManager
from inference.core.managers.decorators import fixed_size_cache
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache

GIB = 1024**3
HostMemory = namedtuple("HostMemory", ["total", "available"])


@pytest.fixture
def cpu_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(fixed_size_cache, "_cuda_is_available", lambda: False)
    monkeypatch.setattr(fixed_size_cache, "CGROUP_ROOT", str(tmp_path))
    monkeypatch.setattr(fixed_size_cache, "MEMORY_FREE_THRESHOLD", 0.15)
    monkeypatch.setattr(
        fixed_size_cache.psutil,
        "virtual_memory",
        lambda: HostMemory(total=32 * GIB, available=16 * GIB),
    )
    return tmp_path


def write_cgroup_v2(root: Path, limit: str, current: int, inactive_file: int) -> None:
    (root / "memory.max").write_text(f"{limit}\n")
    (root / "memory.current").write_text(f"{current}\n")
    (root / "memory.stat").write_text(
        f"anon {current - inactive_file}\nfile {inactive_file}\n"
        f"active_file 0\ninactive_file {inactive_file}\n"
    )


def write_cgroup_v1(root: Path, limit: int, usage: int, inactive_file: int) -> None:
    v1_dir = root / "memory"
    v1_dir.mkdir()
    (v1_dir / "memory.limit_in_bytes").write_text(f"{limit}\n")
    (v1_dir / "memory.usage_in_bytes").write_text(f"{usage}\n")
    (v1_dir / "memory.stat").write_text(
        f"inactive_file {inactive_file}\ntotal_inactive_file {inactive_file}\n"
    )


def detect_pressure() -> bool:
    decorator = WithFixedSizeCache(ModelManager(model_registry=MagicMock()))
    return decorator.memory_pressure_detected()


def test_cgroup_v2_under_threshold_detects_pressure(cpu_host: Path) -> None:
    write_cgroup_v2(cpu_host, limit=str(22 * GIB), current=20 * GIB, inactive_file=0)

    assert fixed_size_cache.container_memory_free_fraction() == pytest.approx(2 / 22)
    assert detect_pressure() is True


def test_cgroup_v2_over_threshold_detects_no_pressure(cpu_host: Path) -> None:
    write_cgroup_v2(cpu_host, limit=str(22 * GIB), current=10 * GIB, inactive_file=0)

    assert detect_pressure() is False


def test_cgroup_v2_ignores_reclaimable_inactive_file_cache(cpu_host: Path) -> None:
    write_cgroup_v2(
        cpu_host, limit=str(22 * GIB), current=21 * GIB, inactive_file=11 * GIB
    )

    assert fixed_size_cache.container_memory_free_fraction() == pytest.approx(12 / 22)
    assert detect_pressure() is False


def test_cgroup_v2_without_limit_uses_host_memory(cpu_host: Path) -> None:
    write_cgroup_v2(cpu_host, limit="max", current=31 * GIB, inactive_file=0)

    assert fixed_size_cache.container_memory_free_fraction() == pytest.approx(0.5)
    assert detect_pressure() is False


def test_cgroup_v1_under_threshold_detects_pressure(cpu_host: Path) -> None:
    write_cgroup_v1(cpu_host, limit=26 * GIB, usage=25 * GIB, inactive_file=0)

    assert fixed_size_cache.container_memory_free_fraction() == pytest.approx(1 / 26)
    assert detect_pressure() is True


def test_cgroup_v1_unlimited_uses_host_memory(
    cpu_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_cgroup_v1(
        cpu_host, limit=9223372036854771712, usage=30 * GIB, inactive_file=0
    )
    monkeypatch.setattr(
        fixed_size_cache.psutil,
        "virtual_memory",
        lambda: HostMemory(total=32 * GIB, available=2 * GIB),
    )

    assert fixed_size_cache.container_memory_free_fraction() == pytest.approx(2 / 32)
    assert detect_pressure() is True


def test_no_cgroup_uses_host_memory(cpu_host: Path) -> None:
    assert fixed_size_cache.container_memory_free_fraction() == pytest.approx(0.5)
    assert detect_pressure() is False


def test_unreadable_cgroup_detects_no_pressure(cpu_host: Path) -> None:
    (cpu_host / "memory.max").write_text("not-a-number\n")

    assert detect_pressure() is False


def test_zero_threshold_skips_memory_check(
    cpu_host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fixed_size_cache, "MEMORY_FREE_THRESHOLD", 0.0)
    decorator = WithFixedSizeCache(ModelManager(model_registry=MagicMock()), max_size=8)
    pressure_check = MagicMock(return_value=True)
    monkeypatch.setattr(decorator, "memory_pressure_detected", pressure_check)

    decorator.add_model(model_id="some/1", api_key="key")
    decorator.add_model(model_id="some/2", api_key="key")

    pressure_check.assert_not_called()
    assert set(decorator.keys()) == {"some/1", "some/2"}


def test_memory_pressure_evicts_least_recently_used_models_below_max_size(
    cpu_host: Path,
) -> None:
    decorator = WithFixedSizeCache(ModelManager(model_registry=MagicMock()), max_size=8)
    for model_id in ["some/1", "some/2", "some/3", "some/4"]:
        decorator.add_model(model_id=model_id, api_key="key")
    write_cgroup_v2(cpu_host, limit=str(22 * GIB), current=21 * GIB, inactive_file=0)
    remove_from_base = decorator.model_manager.remove

    def remove_and_free_memory(model_id: str, delete_from_disk: bool = True) -> None:
        write_cgroup_v2(
            cpu_host, limit=str(22 * GIB), current=10 * GIB, inactive_file=0
        )
        remove_from_base(model_id, delete_from_disk=delete_from_disk)

    decorator.model_manager.remove = remove_and_free_memory

    decorator.add_model(model_id="some/5", api_key="key")

    assert list(decorator._key_queue) == ["some/4", "some/5"]
    assert set(decorator.keys()) == {"some/4", "some/5"}


def test_memory_pressure_keeps_pinned_models(cpu_host: Path) -> None:
    decorator = WithFixedSizeCache(ModelManager(model_registry=MagicMock()), max_size=8)
    decorator.add_model(model_id="pinned/1", api_key="key")
    decorator.pin_model("pinned/1")
    decorator.add_model(model_id="some/2", api_key="key")
    write_cgroup_v2(cpu_host, limit=str(22 * GIB), current=21 * GIB, inactive_file=0)

    decorator.add_model(model_id="some/3", api_key="key")

    assert set(decorator.keys()) == {"pinned/1", "some/3"}
