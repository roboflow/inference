import io
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

LORA_BASE_URL = "https://example.com/base.tar.gz?signature=1"


def _model(transformers_module):
    model = transformers_module.LoRATransformerModel.__new__(
        transformers_module.LoRATransformerModel
    )
    model.api_key = "key"
    model.device_id = "device"

    return model


def _prepare(
    monkeypatch,
    tmp_path,
    transformers_module,
    *,
    offline=False,
    fail_first_download=False,
):
    from inference.core.cache import model_artifacts

    monkeypatch.setattr(model_artifacts, "MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(transformers_module, "MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(transformers_module, "OFFLINE_MODE", offline)
    cache_dir = Path(transformers_module.get_cache_dir("lora-bases/org/base/main"))
    api_calls = []
    monkeypatch.setattr(
        transformers_module,
        "get_roboflow_base_lora",
        lambda *args: api_calls.append(args) or {"weights": {"model": LORA_BASE_URL}},
    )

    download_calls = []

    def fake_stream_url_to_cache(url, filename, model_id):
        download_calls.append(url)
        download_dir = Path(transformers_module.get_cache_dir(model_id))
        download_dir.mkdir(parents=True, exist_ok=True)
        if fail_first_download and len(download_calls) == 1:
            raise RuntimeError("download failed")
        content = b"{}"
        member = tarfile.TarInfo(name="config.json")
        member.size = len(content)
        with tarfile.open(download_dir / filename, "w:gz") as tar:
            tar.addfile(member, io.BytesIO(content))

    monkeypatch.setattr(
        transformers_module, "stream_url_to_cache", fake_stream_url_to_cache
    )

    return cache_dir, api_calls


def test_incomplete_lora_base_is_rebuilt(monkeypatch, tmp_path) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    cache_dir.mkdir(parents=True)
    (cache_dir / "partial.bin").write_bytes(b"partial")

    result = _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert result == str(cache_dir)
    assert len(api_calls) == 1
    assert not (cache_dir / "partial.bin").exists()
    assert (cache_dir / "config.json").exists()
    assert (cache_dir / ".complete").exists()


@pytest.mark.parametrize("failure", ["metadata", "download", "extraction"])
def test_failed_rebuild_keeps_existing_lora_base(
    monkeypatch, tmp_path, failure
) -> None:
    from inference.models.transformers import transformers

    cache_dir, _ = _prepare(
        monkeypatch,
        tmp_path,
        transformers,
        fail_first_download=(failure == "download"),
    )
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_bytes(b"old")
    model = _model(transformers)
    expected_exception = {
        "metadata": ConnectionError,
        "download": RuntimeError,
        "extraction": tarfile.ReadError,
    }[failure]

    def _unreachable(*args):
        raise ConnectionError("api unreachable")

    def _fail_extraction(self, path):
        raise tarfile.ReadError("interrupted extraction")

    with monkeypatch.context() as patch:
        if failure == "metadata":
            patch.setattr(transformers, "get_roboflow_base_lora", _unreachable)
        if failure == "extraction":
            patch.setattr(tarfile.TarFile, "extractall", _fail_extraction)
        with pytest.raises(expected_exception):
            model.get_lora_base_from_roboflow("org/base", "main")

        assert (cache_dir / "config.json").read_bytes() == b"old"
        assert not (cache_dir / ".complete").exists()

    if failure == "metadata":
        return

    result = model.get_lora_base_from_roboflow("org/base", "main")

    assert result == str(cache_dir)
    assert (cache_dir / "config.json").read_bytes() == b"{}"
    assert (cache_dir / ".complete").exists()
    assert not Path(
        transformers.get_cache_dir("lora-bases/org/base/main~staging")
    ).exists()
    assert not Path(
        transformers.get_cache_dir("lora-bases/org/base/main~backup")
    ).exists()


def test_rebuild_leaves_other_revision_cache_untouched(monkeypatch, tmp_path) -> None:
    from inference.models.transformers import transformers

    cache_dir, _ = _prepare(monkeypatch, tmp_path, transformers)
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_bytes(b"old")
    other = Path(transformers.get_cache_dir("lora-bases/org/base/main.staging"))
    other.mkdir(parents=True)
    (other / "config.json").write_bytes(b"other")
    (other / ".complete").touch()

    result = _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert result == str(cache_dir)
    assert (cache_dir / "config.json").read_bytes() == b"{}"
    assert (cache_dir / ".complete").exists()
    assert (other / "config.json").read_bytes() == b"other"
    assert (other / ".complete").exists()


def test_failed_staging_cleanup_keeps_existing_lora_base(monkeypatch, tmp_path) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_bytes(b"old")
    staging = Path(transformers.get_cache_dir("lora-bases/org/base/main~staging"))
    staging.mkdir(parents=True)
    (staging / "stale.bin").write_bytes(b"stale")
    original_rmtree = transformers.shutil.rmtree

    def _refuse_staging(path, *args, **kwargs):
        if Path(path) != staging:
            removed = original_rmtree(path, *args, **kwargs)

            return removed

        if kwargs.get("ignore_errors"):
            return None

        raise OSError("busy")

    monkeypatch.setattr(transformers.shutil, "rmtree", _refuse_staging)

    with pytest.raises(OSError):
        _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert (cache_dir / "config.json").read_bytes() == b"old"
    assert not (cache_dir / ".complete").exists()
    assert (staging / "stale.bin").exists()
    assert len(api_calls) == 1


@pytest.mark.parametrize("existing_cache", [False, True])
def test_failed_promotion_restores_old_cache(monkeypatch, tmp_path, existing_cache):
    from inference.models.transformers import transformers

    cache_dir, _ = _prepare(monkeypatch, tmp_path, transformers)
    staging = Path(transformers.get_cache_dir("lora-bases/org/base/main~staging"))
    backup = Path(transformers.get_cache_dir("lora-bases/org/base/main~backup"))
    if existing_cache:
        cache_dir.mkdir(parents=True)
        (cache_dir / "config.json").write_bytes(b"old")
    original_replace = transformers.os.replace

    def _fail_promotion(source, destination):
        if Path(source) == staging:
            assert (staging / ".complete").is_file()
            assert not cache_dir.exists()
            if existing_cache:
                assert (backup / "config.json").read_bytes() == b"old"
            raise OSError("promotion failed")

        original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(transformers.os, "replace", _fail_promotion)
        with pytest.raises(OSError, match="promotion failed"):
            _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    if existing_cache:
        assert (cache_dir / "config.json").read_bytes() == b"old"
    else:
        assert not cache_dir.exists()
    assert not (cache_dir / ".complete").exists()
    assert not backup.exists()

    _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert (cache_dir / "config.json").read_bytes() == b"{}"
    assert (cache_dir / ".complete").is_file()
    assert not staging.exists()
    assert not backup.exists()


def test_failed_rollback_can_be_recovered_offline(monkeypatch, tmp_path):
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    staging = Path(transformers.get_cache_dir("lora-bases/org/base/main~staging"))
    backup = Path(transformers.get_cache_dir("lora-bases/org/base/main~backup"))
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_bytes(b"old")
    original_replace = transformers.os.replace

    def _fail_promotion_and_rollback(source, destination):
        if Path(source) in {staging, backup}:
            raise OSError("rename unavailable")

        original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(transformers.os, "replace", _fail_promotion_and_rollback)
        with pytest.raises(OSError, match="rename unavailable"):
            _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert not cache_dir.exists()
    assert (backup / "config.json").read_bytes() == b"old"
    monkeypatch.setattr(transformers, "OFFLINE_MODE", True)

    result = _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert result == str(cache_dir)
    assert (cache_dir / "config.json").read_bytes() == b"old"
    assert not (cache_dir / ".complete").exists()
    assert not backup.exists()
    assert len(api_calls) == 1


@pytest.mark.parametrize("crash_after", ["backup", "promotion"])
@pytest.mark.parametrize("offline", [False, True])
def test_interrupted_replacement_recovers_on_next_load(
    monkeypatch, tmp_path, crash_after, offline
):
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    staging = Path(transformers.get_cache_dir("lora-bases/org/base/main~staging"))
    backup = Path(transformers.get_cache_dir("lora-bases/org/base/main~backup"))
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_bytes(b"old")
    original_replace = transformers.os.replace

    class SimulatedCrash(BaseException):
        pass

    def _crash_after_rename(source, destination):
        original_replace(source, destination)
        if (crash_after == "backup" and Path(destination) == backup) or (
            crash_after == "promotion" and Path(source) == staging
        ):
            raise SimulatedCrash()

    with monkeypatch.context() as patch:
        patch.setattr(transformers.os, "replace", _crash_after_rename)
        with pytest.raises(SimulatedCrash):
            _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert (backup / "config.json").read_bytes() == b"old"
    monkeypatch.setattr(transformers, "OFFLINE_MODE", offline)

    result = _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    rebuilds = crash_after == "backup" and not offline
    keeps_old = crash_after == "backup" and offline
    assert result == str(cache_dir)
    assert (cache_dir / "config.json").read_bytes() == (b"old" if keeps_old else b"{}")
    assert (cache_dir / ".complete").is_file() == (not keeps_old)
    assert not backup.exists()
    assert len(api_calls) == (2 if rebuilds else 1)


def test_failed_backup_cleanup_preserves_promoted_cache(monkeypatch, tmp_path):
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    backup = Path(transformers.get_cache_dir("lora-bases/org/base/main~backup"))
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_bytes(b"old")
    original_rmtree = transformers.shutil.rmtree

    def _fail_cleanup(path, *args, **kwargs):
        if Path(path) == backup:
            raise OSError("backup busy")

        original_rmtree(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(transformers.shutil, "rmtree", _fail_cleanup)
        with pytest.raises(OSError, match="backup busy"):
            _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert (cache_dir / "config.json").read_bytes() == b"{}"
    assert (cache_dir / ".complete").is_file()
    assert (backup / "config.json").read_bytes() == b"old"

    _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert (cache_dir / "config.json").read_bytes() == b"{}"
    assert not backup.exists()
    assert len(api_calls) == 1


def test_ambiguous_recovery_keeps_both_caches(monkeypatch, tmp_path):
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    backup = Path(transformers.get_cache_dir("lora-bases/org/base/main~backup"))
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_bytes(b"current")
    backup.mkdir(parents=True)
    (backup / "config.json").write_bytes(b"old")

    with pytest.raises(transformers.ModelArtefactError, match="both an unmarked cache"):
        _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert (cache_dir / "config.json").read_bytes() == b"current"
    assert (backup / "config.json").read_bytes() == b"old"
    assert api_calls == []


def test_fresh_lora_base_is_downloaded_and_marked(monkeypatch, tmp_path) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)

    _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert len(api_calls) == 1
    assert (cache_dir / "config.json").exists()
    assert (cache_dir / ".complete").exists()


def test_complete_lora_base_is_reused(monkeypatch, tmp_path) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_bytes(b"{}")
    (cache_dir / ".complete").touch()

    _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert api_calls == []
    assert (cache_dir / "config.json").exists()


def test_offline_mode_trusts_existing_lora_base_without_marker(
    monkeypatch, tmp_path
) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers, offline=True)
    cache_dir.mkdir(parents=True)
    (cache_dir / "config.json").write_bytes(b"{}")

    result = _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert result == str(cache_dir)
    assert api_calls == []
    assert (cache_dir / "config.json").exists()
    assert not (cache_dir / ".complete").exists()


def test_offline_mode_miss_raises_before_touching_the_cache(
    monkeypatch, tmp_path
) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers, offline=True)

    with pytest.raises(transformers.ModelArtefactError):
        _model(transformers).get_lora_base_from_roboflow("org/base", "main")

    assert api_calls == []
    assert not cache_dir.exists()
    assert not (tmp_path / "_file_locks").exists()


def test_failed_download_leaves_no_marker_and_next_call_retries(
    monkeypatch, tmp_path
) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(
        monkeypatch, tmp_path, transformers, fail_first_download=True
    )
    model = _model(transformers)

    with pytest.raises(RuntimeError):
        model.get_lora_base_from_roboflow("org/base", "main")

    assert not (cache_dir / ".complete").exists()

    result = model.get_lora_base_from_roboflow("org/base", "main")

    assert result == str(cache_dir)
    assert len(api_calls) == 2
    assert (cache_dir / "config.json").exists()
    assert (cache_dir / ".complete").exists()


def test_failed_extraction_leaves_no_marker_and_next_call_rebuilds(
    monkeypatch, tmp_path
) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    model = _model(transformers)
    original_extractall = tarfile.TarFile.extractall

    def _fail_extraction(self, path):
        (Path(path) / "partial.bin").write_bytes(b"partial")
        raise tarfile.ReadError("interrupted extraction")

    with monkeypatch.context() as patch:
        patch.setattr(tarfile.TarFile, "extractall", _fail_extraction)
        with pytest.raises(tarfile.ReadError, match="interrupted extraction"):
            model.get_lora_base_from_roboflow("org/base", "main")

    assert not (cache_dir / ".complete").exists()
    assert tarfile.TarFile.extractall is original_extractall

    model.get_lora_base_from_roboflow("org/base", "main")

    assert len(api_calls) == 2
    assert not (cache_dir / "partial.bin").exists()
    assert (cache_dir / "config.json").exists()
    assert (cache_dir / ".complete").exists()


def test_concurrent_loads_wait_for_complete_lora_base(monkeypatch, tmp_path) -> None:
    from inference.models.transformers import transformers

    cache_dir, api_calls = _prepare(monkeypatch, tmp_path, transformers)
    download_started = Event()
    finish_download = Event()
    second_lock_attempted = Event()
    original_download = transformers.stream_url_to_cache
    original_lock = transformers.FileLock

    def _download(url, filename, model_id):
        download_started.set()
        assert finish_download.wait(timeout=10)
        original_download(url, filename, model_id)

    class ObservedFileLock(original_lock):
        def acquire(self, *args, **kwargs):
            if download_started.is_set():
                second_lock_attempted.set()
            return super().acquire(*args, **kwargs)

    monkeypatch.setattr(transformers, "stream_url_to_cache", _download)
    monkeypatch.setattr(transformers, "FileLock", ObservedFileLock)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            _model(transformers).get_lora_base_from_roboflow, "org/base", "main"
        )
        try:
            assert download_started.wait(timeout=10)
            second = executor.submit(
                _model(transformers).get_lora_base_from_roboflow, "org/base", "main"
            )
            assert second_lock_attempted.wait(timeout=10)
            assert not second.done()
        finally:
            finish_download.set()

        assert first.result(timeout=10) == str(cache_dir)
        assert second.result(timeout=10) == str(cache_dir)

    assert len(api_calls) == 1
    assert (cache_dir / "config.json").exists()
    assert (cache_dir / ".complete").exists()
